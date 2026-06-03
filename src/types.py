"""
Shared types: Config, Window, WindowDataset, Verdict.

These are the cross-module contracts. Define them here so every module
imports from one place and callers never invent their own shapes.
"""
from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd


@dataclass
class Config:
    # --- reproducibility ---
    seed: int = 42

    # --- architecture ---
    latent_dim: int = 16
    encoder_dims: tuple[int, ...] = (180, 64, 16)
    predictor_dims: tuple[int, ...] = (16, 32, 16)

    # --- windows ---
    context_window: int = 30
    target_window: int = 10

    # --- optimiser ---
    learning_rate: float = 3e-4
    weight_decay: float = 1e-4
    batch_size: int = 64
    max_epochs: int = 500

    # --- early stopping ---
    early_stopping_patience: int = 30
    early_stopping_min_delta: float = 1e-5

    # --- regularisation ---
    sigreg_lambda: float = 0.02

    # --- calibration / thresholds ---
    threshold_percentile: int = 99

    # --- episode detection ---
    episode_cooldown_days: int = 10

    # --- crash onset ---
    drawdown_threshold: float = -0.20
    drawdown_window_days: int = 60
    drawdown_lookback_days: int = 252

    # --- split boundaries (inclusive) ---
    train_start: str = "2010-01-01"
    train_end: str = "2016-12-31"
    val_start: str = "2017-01-01"
    val_end: str = "2018-12-31"
    calib_start: str = "2019-01-01"
    calib_end: str = "2020-12-31"
    test_start: str = "2021-01-01"
    test_end: str = "2022-12-31"

    # --- data sources ---
    equity_ticker: str = "SOXX"
    vix_ticker: str = "^VIX"
    bond_series: str = "DGS10"

    # --- set at runtime ---
    git_commit: str = ""
    lejepa_threshold: float = float("nan")


@dataclass
class Window:
    """One (context, target) pair from the sliding window."""
    context_array: np.ndarray   # shape (context_window, n_features)
    target_array: np.ndarray    # shape (target_window, n_features)
    date: pd.Timestamp          # last day of the context window (day T)


@dataclass
class WindowDataset:
    """All four splits, ready for training / calibration / scoring."""
    train: list[Window]
    val: list[Window]
    calib: list[Window]
    test: list[Window]

    train_dates: pd.DatetimeIndex
    val_dates: pd.DatetimeIndex
    calib_dates: pd.DatetimeIndex
    test_dates: pd.DatetimeIndex


@dataclass
class Verdict:
    """Pass/fail result from Evaluator.evaluate()."""
    # crash oracle
    crash_onset_date: Optional[pd.Timestamp]

    # first breach dates
    lejepa_first_breach: Optional[pd.Timestamp]
    rv20_first_breach: Optional[pd.Timestamp]
    vix_first_breach: Optional[pd.Timestamp]

    # lead times (trading days; positive = before crash)
    lejepa_lead_days: Optional[int]
    rv20_lead_days: Optional[int]
    vix_lead_days: Optional[int]

    # 2021 episode / alert-day counts
    lejepa_episodes_2021: int
    rv20_episodes_2021: int
    vix_episodes_2021: int
    lejepa_alert_days_2021: int
    rv20_alert_days_2021: int
    vix_alert_days_2021: int

    # pass/fail per criterion
    criterion_1: bool   # lejepa_first_breach <= crash_onset_date
    criterion_2: bool   # lejepa beats BOTH baselines on timing
    criterion_3: bool   # lejepa 2021 episodes <= max(rv20, vix)
    criterion_4: bool   # all split boundary assertions passed

    partial_signal: bool    # lejepa beats exactly one baseline on timing
    all_pass: bool          # all four criteria true
