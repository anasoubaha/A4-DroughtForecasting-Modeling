"""PyTorch LSTM forecaster for SPEI3 (Phase 12).

Architecture
------------
- One `nn.LSTMCell` unrolled manually so we can apply **variational dropout**
  (Gal & Ghahramani, 2016) — the same dropout mask is reused for every timestep
  within a sequence, on both the input vector and the recurrent hidden state.
  PyTorch's built-in `nn.LSTM` only exposes inter-layer dropout, which is not
  the "recurrent dropout between memory states" the v12 spec asks for.
- A single linear head maps the final hidden state to a scalar SPEI3(t + L)
  prediction.

Loss
----
Two options selected via `loss="mse" | "weighted_mse"` at construction:

    weighted_mse:   loss_i = w_i · (ŷ_i - y_i)²
                    w_i    = 1 + alpha · 1{|y_i| > threshold}

  Defaults: threshold = 1.0, alpha = 3.0 → extremes count 4× normal samples.
  This is the "Regression to the Mean"-defeating loss the spec calls for.

Public API
----------
The class follows the same xarray-in / xarray-out contract as the tabular
models in `droughtmodel.models._tabular`, BUT in addition exposes the lower-level
`fit_tensors(X_train, y_train, X_val, y_val)` and `predict_tensors(X)` calls
so the LSTM pipeline can pre-bake the (samples, T, features) tensors with
`droughtmodel.sequence.build_sequences`. The xarray entry points apply
`build_sequences` internally as a convenience for the unit tests.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np
import xarray as xr

from droughtmodel.models.base import BaseModel
from droughtmodel.sequence import build_sequences, predict_to_grid


# Try to import torch at module scope. If unavailable, set sentinels so the
# rest of the droughtmodel package keeps importing — instantiating LSTMModel
# will then raise a clear ImportError.
try:
    import torch
    import torch.nn as _nn
    _HAS_TORCH = True
except ImportError:                                   # pragma: no cover
    torch = None                                      # type: ignore[assignment]
    _nn = None                                        # type: ignore[assignment]
    _HAS_TORCH = False


def _require_torch():
    if not _HAS_TORCH:
        raise ImportError(
            "The LSTM model requires PyTorch. Install it with "
            "`pip install torch` in the project venv."
        )
    return torch


# ---------------------------------------------------------------------------
# Weighted MSE
# ---------------------------------------------------------------------------

def weighted_mse_loss(
    pred,
    target,
    *,
    threshold: float = 1.0,
    alpha: float = 3.0,
):
    """Per-sample-weighted MSE: ``w_i = 1 + alpha · 1{|y_i| > threshold}``.

    With defaults (threshold=1.0, alpha=3.0) every standardized SPEI3 value
    whose magnitude exceeds 1 σ contributes 4× the gradient of a near-zero
    sample, directly attacking the LSTM's tendency to predict 0.0 on noisy
    long-lead targets.
    """
    _require_torch()
    diff = pred - target
    weights = 1.0 + alpha * (target.abs() > threshold).to(pred.dtype)
    return (weights * diff * diff).mean()


# ---------------------------------------------------------------------------
# Variational-dropout LSTM cell stack
#
# Defined at MODULE level (not inside a factory function) so that joblib /
# pickle can locate the class by its dotted import path at unpickling time —
# nested-class instances are not picklable and silently produced 10-byte
# corrupt files during the v12 model-save step (fixed 2026-06-30).
# ---------------------------------------------------------------------------

if _HAS_TORCH:

    class VariationalLSTM(_nn.Module):
        """Single-layer LSTM with input AND recurrent variational dropout.

        At training time, two boolean masks are sampled once per sample (per
        forward pass) and reused unchanged across all T timesteps:
          - input mask  m_x ~ Bernoulli(1 - dropout)  applied to x_t  at every t
          - hidden mask m_h ~ Bernoulli(1 - dropout)  applied to h_{t-1} at every t
        At eval time the masks are disabled (identity), matching the standard
        dropout convention. This is the formulation in Gal & Ghahramani (2016).
        """

        def __init__(
            self,
            input_size: int,
            hidden_size: int,
            dropout: float = 0.0,
        ):
            super().__init__()
            if not (0.0 <= dropout < 1.0):
                raise ValueError(f"dropout must be in [0, 1); got {dropout}")
            self.input_size = input_size
            self.hidden_size = hidden_size
            self.dropout = float(dropout)
            self.cell = _nn.LSTMCell(input_size, hidden_size)
            self.head = _nn.Linear(hidden_size, 1)

        def forward(self, x):                           # x: (B, T, F)
            B, T, F = x.shape
            device = x.device
            dtype = x.dtype
            h = torch.zeros(B, self.hidden_size, device=device, dtype=dtype)
            c = torch.zeros(B, self.hidden_size, device=device, dtype=dtype)

            if self.training and self.dropout > 0.0:
                # Inverted dropout: mask scaled by 1 / (1 - p) so the expectation
                # at training time matches the eval-time pass-through.
                keep = 1.0 - self.dropout
                x_mask = (
                    torch.empty(B, F, device=device, dtype=dtype)
                    .bernoulli_(keep)
                    .div_(keep)
                )
                h_mask = (
                    torch.empty(B, self.hidden_size, device=device, dtype=dtype)
                    .bernoulli_(keep)
                    .div_(keep)
                )
            else:
                x_mask = None
                h_mask = None

            for t in range(T):
                xt = x[:, t, :]
                if x_mask is not None:
                    xt = xt * x_mask
                h_in = h if h_mask is None else h * h_mask
                h, c = self.cell(xt, (h_in, c))

            return self.head(h).squeeze(-1)             # (B,)

else:                                                 # pragma: no cover
    VariationalLSTM = None                            # type: ignore[assignment]


# ---------------------------------------------------------------------------
# Public model class
# ---------------------------------------------------------------------------

@dataclass
class _FitState:
    epochs_run: int
    best_val_loss: float
    best_epoch: int
    train_loss_history: list[float]
    val_loss_history: list[float]


class LSTMModel(BaseModel):
    """PyTorch LSTM for SPEI3(t + L) regression with variational dropout.

    Constructor parameters are read from `configs/models/lstm.yaml::params` AND
    can be overridden per-trial by the LSTM pipeline's grid search.
    """

    name = "lstm"

    def __init__(
        self,
        hidden_units: int = 16,
        dropout: float = 0.2,
        sequence_length: int = 6,
        learning_rate: float = 1e-3,
        batch_size: int = 256,
        max_epochs: int = 80,
        patience: int = 10,
        loss: str = "weighted_mse",
        weighted_mse_threshold: float = 1.0,
        weighted_mse_alpha: float = 3.0,
        weight_decay: float = 0.0,
        grad_clip_norm: float = 1.0,
        num_workers: int = 0,
        device: str = "auto",
        seed: int = 42,
        **_unused: Any,                  # tolerate extra YAML keys
    ):
        self.hidden_units = int(hidden_units)
        self.dropout = float(dropout)
        self.sequence_length = int(sequence_length)
        self.learning_rate = float(learning_rate)
        self.batch_size = int(batch_size)
        self.max_epochs = int(max_epochs)
        self.patience = int(patience)
        self.loss = str(loss)
        self.weighted_mse_threshold = float(weighted_mse_threshold)
        self.weighted_mse_alpha = float(weighted_mse_alpha)
        self.weight_decay = float(weight_decay)
        # Global L2 gradient-norm clip applied before each optimizer step.
        # Set to 0 (or any non-positive value) to disable. Default 1.0 was
        # chosen after a run with no clipping diverged to NaN at L=3 / L=6
        # under the weighted-MSE loss (2026-06-30 incident).
        self.grad_clip_norm = float(grad_clip_norm)
        self.num_workers = int(num_workers)
        self.device_str = str(device)
        self.seed = int(seed)

        self.feature_names_: list[str] | None = None
        self.module_: Any = None         # set after fit
        self.fit_state_: _FitState | None = None

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _resolve_device(self):
        _require_torch()
        if self.device_str == "auto":
            if torch.cuda.is_available():
                return torch.device("cuda")
            # NOTE: MPS is intentionally NOT picked up by `auto`. PyTorch's
            # Metal backend produces NaN on the first batch for our manually-
            # unrolled variational-dropout LSTMCell loop (verified on
            # 2026-06-30 with torch 2.x / macOS Darwin 25). If you want MPS,
            # set `device: mps` EXPLICITLY in the YAML and accept the risk;
            # the model may not train correctly.
            return torch.device("cpu")
        return torch.device(self.device_str)

    def _make_loss_fn(self):
        _require_torch()
        if self.loss == "mse":
            return torch.nn.functional.mse_loss
        if self.loss == "weighted_mse":
            thr = self.weighted_mse_threshold
            alpha = self.weighted_mse_alpha
            def _fn(pred, target):
                return weighted_mse_loss(pred, target, threshold=thr, alpha=alpha)
            return _fn
        raise ValueError(f"Unknown loss: {self.loss!r}")

    def _seed_torch(self):
        _require_torch()
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.seed)

    # ------------------------------------------------------------------
    # Core fit / predict on pre-baked tensors
    # ------------------------------------------------------------------
    def fit_tensors(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray | None = None,
        y_val: np.ndarray | None = None,
        feature_names: list[str] | None = None,
    ) -> "LSTMModel":
        """Fit on pre-baked `(N, T, F)` tensors. Returns self.

        Uses Adam + early stopping on val loss when (X_val, y_val) is provided.
        When validation data is absent we just train for `max_epochs` and keep
        the final weights.
        """
        _require_torch()
        from torch.utils.data import DataLoader, TensorDataset

        if X_train.ndim != 3:
            raise ValueError(f"X_train must be 3-D (N, T, F); got shape {X_train.shape}")
        if y_train.ndim != 1 or y_train.shape[0] != X_train.shape[0]:
            raise ValueError(
                f"y_train must be 1-D and length {X_train.shape[0]}; got shape {y_train.shape}"
            )

        self.feature_names_ = list(feature_names) if feature_names else None
        device = self._resolve_device()
        self._seed_torch()

        n_features = X_train.shape[-1]
        self.module_ = VariationalLSTM(
            input_size=n_features,
            hidden_size=self.hidden_units,
            dropout=self.dropout,
        ).to(device)

        opt = torch.optim.Adam(
            self.module_.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )
        loss_fn = self._make_loss_fn()
        mse_fn = torch.nn.functional.mse_loss   # plain MSE for val tracking

        Xt_t = torch.as_tensor(X_train, dtype=torch.float32)
        yt_t = torch.as_tensor(y_train, dtype=torch.float32)
        train_ds = TensorDataset(Xt_t, yt_t)
        train_loader = DataLoader(
            train_ds,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            drop_last=False,
        )

        has_val = X_val is not None and y_val is not None and len(X_val) > 0
        if has_val:
            Xv_t = torch.as_tensor(X_val, dtype=torch.float32).to(device)
            yv_t = torch.as_tensor(y_val, dtype=torch.float32).to(device)

        best_val = math.inf
        best_state = None
        best_epoch = 0
        stale = 0
        train_history: list[float] = []
        val_history: list[float] = []
        epoch = 0

        diverged = False
        for epoch in range(1, self.max_epochs + 1):
            self.module_.train()
            total = 0.0
            n_seen = 0
            for xb, yb in train_loader:
                xb = xb.to(device, non_blocking=True)
                yb = yb.to(device, non_blocking=True)
                opt.zero_grad()
                pred = self.module_(xb)
                loss = loss_fn(pred, yb)
                loss_val = float(loss.detach().cpu())
                if not math.isfinite(loss_val):
                    # NaN / inf loss → an earlier batch made the weights blow up.
                    # Without aborting, every later batch perpetuates the NaN and
                    # the saved model is useless. Break out and let the caller
                    # surface a clear failure (best_val_loss will stay = +inf).
                    diverged = True
                    break
                loss.backward()
                if self.grad_clip_norm and self.grad_clip_norm > 0:
                    torch.nn.utils.clip_grad_norm_(
                        self.module_.parameters(), max_norm=self.grad_clip_norm
                    )
                opt.step()
                total += loss_val * xb.size(0)
                n_seen += xb.size(0)
            if diverged:
                # Mark this epoch as NaN so the loss-history reflects what happened.
                train_history.append(float("nan"))
                val_history.append(float("nan"))
                break
            train_loss = total / max(n_seen, 1)
            train_history.append(train_loss)

            if has_val:
                self.module_.eval()
                with torch.no_grad():
                    pred_v = self.module_(Xv_t)
                    val_loss = float(mse_fn(pred_v, yv_t).cpu())
                val_history.append(val_loss)
                if not math.isfinite(val_loss):
                    diverged = True
                    break
                if val_loss < best_val - 1e-6:
                    best_val = val_loss
                    best_epoch = epoch
                    best_state = {
                        k: v.detach().cpu().clone()
                        for k, v in self.module_.state_dict().items()
                    }
                    stale = 0
                else:
                    stale += 1
                    if stale >= self.patience:
                        break
            else:
                val_history.append(float("nan"))

        # Restore best weights when early stopping was active.
        if has_val and best_state is not None:
            self.module_.load_state_dict(best_state)

        self.fit_state_ = _FitState(
            epochs_run=epoch,
            best_val_loss=best_val if has_val else float("nan"),
            best_epoch=best_epoch if has_val else epoch,
            train_loss_history=train_history,
            val_loss_history=val_history,
        )
        return self

    def predict_tensors(self, X: np.ndarray) -> np.ndarray:
        """Predict on a pre-baked `(N, T, F)` tensor. Returns 1-D float array."""
        _require_torch()
        if self.module_ is None:
            raise RuntimeError("LSTMModel must be `.fit_tensors()` before `.predict_tensors()`.")
        if X.ndim != 3:
            raise ValueError(f"X must be 3-D (N, T, F); got shape {X.shape}")
        if X.shape[0] == 0:
            return np.zeros((0,), dtype=np.float32)
        device = next(self.module_.parameters()).device
        self.module_.eval()
        out_chunks: list[np.ndarray] = []
        with torch.no_grad():
            Xt = torch.as_tensor(X, dtype=torch.float32)
            # Chunk to keep peak memory bounded
            bs = max(self.batch_size, 256)
            for i in range(0, Xt.shape[0], bs):
                chunk = Xt[i:i + bs].to(device, non_blocking=True)
                out_chunks.append(self.module_(chunk).cpu().numpy())
        return np.concatenate(out_chunks, axis=0)

    # ------------------------------------------------------------------
    # xarray-in / xarray-out (BaseModel contract — used by tests)
    # ------------------------------------------------------------------
    def fit(self, train: xr.Dataset, val: xr.Dataset | None = None) -> "LSTMModel":
        X_train, y_train, meta = build_sequences(train, self.sequence_length)
        feature_names = meta.feature_names
        X_val = y_val = None
        if val is not None:
            X_val, y_val, _ = build_sequences(val, self.sequence_length)
        return self.fit_tensors(X_train, y_train, X_val, y_val, feature_names=feature_names)

    def predict(self, dataset: xr.Dataset) -> xr.DataArray:
        if self.module_ is None:
            raise RuntimeError("LSTMModel must be `.fit()` before `.predict()`.")
        X, _, meta = build_sequences(dataset, self.sequence_length)
        preds = self.predict_tensors(X)
        return predict_to_grid(preds, meta, dataset["target"])

    def feature_importance(self) -> dict[str, float] | None:
        """LSTM has no native feature importance (use permutation importance
        via a post-hoc script if needed)."""
        return None