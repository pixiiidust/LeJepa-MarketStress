"""
Trainer: AdamW training loop with early stopping and collapse checks.

Trains on 2010-2016, validates on 2017-2018.
Monitors: validation prediction loss (MSE between z_pred and z_target).
Early stopping: patience=30, min_delta=1e-5.
Collapse checks per epoch: latent_std_min, latent_rank, covariance_condition_number.
Raises CollapseError if latent_std_min < 0.05 at end of training.

Public interface:
    Trainer(config).fit(model, train_windows, val_windows) -> best_model
"""
from __future__ import annotations

import copy

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from src.sigreg import SIGRegLoss
from src.types import Config, Window


class CollapseError(Exception):
    pass


def _windows_to_tensors(
    windows: list[Window],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return (X_context, X_target) tensors for a list of windows.

    X_context: (N, 180)  — flattened 30-day context window
    X_target:  (N, 180)  — flattened 30-day "future context" window
                            = concat(context[-20:], target_array) reshaped
    """
    ctx_list: list[np.ndarray] = []
    tgt_list: list[np.ndarray] = []
    for w in windows:
        ctx_list.append(w.context_array.flatten())
        # future-context window: last 20 context days + 10 target days
        future_ctx = np.concatenate([w.context_array[-20:], w.target_array], axis=0)
        tgt_list.append(future_ctx.flatten())
    X = torch.tensor(np.stack(ctx_list), dtype=torch.float32)
    Y = torch.tensor(np.stack(tgt_list), dtype=torch.float32)
    return X, Y


def _collapse_stats(
    model: nn.Module, X: torch.Tensor
) -> tuple[float, int, float]:
    """Compute latent_std_min, latent_rank, covariance_condition_number on X."""
    model.eval()
    with torch.no_grad():
        z = model.encode(X)                    # (N, 16)

    std = z.std(dim=0)
    latent_std_min = std.min().item()

    z_c = z - z.mean(dim=0)
    cov = (z_c.T @ z_c) / (max(len(z) - 1, 1))
    try:
        eig = torch.linalg.eigvalsh(cov)
        eig_max = eig.max().item()
        eig_min = eig.min().abs().item()
        cond = eig_max / max(eig_min, 1e-10)
    except Exception:
        cond = float("inf")

    rank = int(torch.linalg.matrix_rank(z, atol=1e-3).item())

    return latent_std_min, rank, cond


class Trainer:
    def __init__(self, config: Config):
        self.config = config

    def fit(
        self,
        model: nn.Module,
        train_windows: list[Window],
        val_windows: list[Window],
    ) -> nn.Module:
        cfg = self.config

        X_train, Y_train = _windows_to_tensors(train_windows)
        X_val, Y_val = _windows_to_tensors(val_windows)

        dataset = TensorDataset(X_train, Y_train)
        loader = DataLoader(dataset, batch_size=cfg.batch_size, shuffle=True)

        mse_fn = nn.MSELoss()
        sigreg_fn = SIGRegLoss(lam=cfg.sigreg_lambda)

        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=cfg.learning_rate,
            weight_decay=cfg.weight_decay,
        )

        best_val_loss = float("inf")
        best_weights = copy.deepcopy(model.state_dict())
        patience_counter = 0

        for epoch in range(cfg.max_epochs):
            model.train()
            for X_batch, Y_batch in loader:
                optimizer.zero_grad()
                z_context = model.encode(X_batch)
                z_target = model.encode(Y_batch)
                z_pred = model.predict(z_context)
                loss = mse_fn(z_pred, z_target) + sigreg_fn(z_context, z_target, z_pred)
                loss.backward()
                optimizer.step()

            # Validation loss (prediction MSE only — early-stopping monitor)
            model.eval()
            with torch.no_grad():
                z_ctx_val = model.encode(X_val)
                z_tgt_val = model.encode(Y_val)
                z_pred_val = model.predict(z_ctx_val)
                val_loss = mse_fn(z_pred_val, z_tgt_val).item()

            # Collapse diagnostics
            std_min, rank, cond = _collapse_stats(model, X_val)
            print(
                f"epoch={epoch+1:04d}  val_loss={val_loss:.6f}"
                f"  latent_std_min={std_min:.4f}"
                f"  latent_rank={rank}"
                f"  cov_cond={cond:.2e}"
            )

            # Best checkpoint
            if val_loss < best_val_loss - cfg.early_stopping_min_delta:
                best_val_loss = val_loss
                best_weights = copy.deepcopy(model.state_dict())
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= cfg.early_stopping_patience:
                    print(f"Early stopping at epoch {epoch+1}")
                    break

        # Restore best weights, set eval mode
        model.load_state_dict(best_weights)
        model.eval()

        # Collapse check on best model
        std_min, _, _ = _collapse_stats(model, X_val)
        if std_min < 0.05:
            raise CollapseError(
                f"Latent collapse detected: latent_std_min={std_min:.4f} < 0.05. "
                "Model embeddings have degenerated; training cannot produce valid scores."
            )

        return model
