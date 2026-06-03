"""
SIGRegLoss: Sketched Isotropic Gaussian Regularization adapter.

Adapted from galilai-group/lejepa for time-series batch shapes (B x 16).
Applied to z_context, z_target, and z_pred within each training step.
Lambda fixed at 0.02.

Public interface:
    SIGRegLoss(lam=0.02).forward(z_context, z_target, z_pred) -> loss: Tensor
"""
