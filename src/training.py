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
