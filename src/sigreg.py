"""
SIGRegLoss: Sketched Isotropic Gaussian Regularization adapter.

Adapted from galilai-group/lejepa for time-series batch shapes (B x 16).
Applied to z_context, z_target, and z_pred within each training step.
Lambda fixed at 0.02.

Public interface:
    SIGRegLoss(lam=0.02).forward(z_context, z_target, z_pred) -> loss: Tensor
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


def _sigreg_single(z: torch.Tensor) -> torch.Tensor:
    """SIGReg for one embedding population, shape (B, D).

    Variance term:   penalises dimensions whose std < 1 (collapse prevention).
    Covariance term: penalises off-diagonal cross-correlations (decorrelation).
    """
    B, D = z.shape
    eps = 1e-4

    # Variance term: sum over dims of max(0, 1 - std_d)^2
    std = (z.var(dim=0) + eps).sqrt()          # (D,)
    var_loss = F.relu(1.0 - std).pow(2).mean()

    # Covariance term: off-diagonal of sample covariance
    z_c = z - z.mean(dim=0)
    cov = (z_c.T @ z_c) / (B - 1)             # (D, D)
    off_diag = ~torch.eye(D, dtype=torch.bool, device=z.device)
    cov_loss = cov[off_diag].pow(2).sum() / D

    return var_loss + cov_loss


class SIGRegLoss(nn.Module):
    def __init__(self, lam: float = 0.02):
        super().__init__()
        self.lam = lam

    def forward(
        self,
        z_context: torch.Tensor,
        z_target: torch.Tensor,
        z_pred: torch.Tensor,
    ) -> torch.Tensor:
        reg = (
            _sigreg_single(z_context)
            + _sigreg_single(z_target)
            + _sigreg_single(z_pred)
        )
        return self.lam * reg
