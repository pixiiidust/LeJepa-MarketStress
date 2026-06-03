"""
Tests for src/types.py: Config, Window, WindowDataset, Verdict.
"""
import dataclasses
import json

import numpy as np
import pandas as pd
import pytest

from src.types import Config, Verdict, Window, WindowDataset


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# WindowDataset
# ---------------------------------------------------------------------------

def test_window_dataset_holds_split_date_indices():
    dates = pd.bdate_range("2021-01-04", periods=5)
    ds = WindowDataset(
        train=[], val=[], calib=[], test=[],
        train_dates=dates, val_dates=dates,
        calib_dates=dates, test_dates=dates,
    )
    assert isinstance(ds.test_dates, pd.DatetimeIndex)
    assert len(ds.test_dates) == 5


def test_window_carries_context_target_and_date():
    ctx = np.zeros((30, 6))
    tgt = np.zeros((10, 6))
    date = pd.Timestamp("2021-06-01")
    w = Window(context_array=ctx, target_array=tgt, date=date)
    assert w.context_array.shape == (30, 6)
    assert w.target_array.shape == (10, 6)
    assert w.date == date


# ---------------------------------------------------------------------------
# Verdict
# ---------------------------------------------------------------------------

def test_verdict_all_pass_field():
    v = Verdict(
        crash_onset_date=pd.Timestamp("2022-01-03"),
        lejepa_first_breach=pd.Timestamp("2021-11-01"),
        rv20_first_breach=pd.Timestamp("2021-12-01"),
        vix_first_breach=pd.Timestamp("2021-12-15"),
        lejepa_lead_days=40, rv20_lead_days=20, vix_lead_days=10,
        lejepa_episodes_2021=1, rv20_episodes_2021=2, vix_episodes_2021=2,
        lejepa_alert_days_2021=5, rv20_alert_days_2021=8, vix_alert_days_2021=7,
        criterion_1=True, criterion_2=True, criterion_3=True, criterion_4=True,
        partial_signal=False, all_pass=True,
    )
    assert v.all_pass is True
    assert v.partial_signal is False


def test_config_serialises_to_json():
    cfg = Config()
    d = dataclasses.asdict(cfg)
    blob = json.dumps(d)           # must not raise
    recovered = json.loads(blob)
    assert recovered["seed"] == 42
    assert recovered["latent_dim"] == 16


def test_config_has_preregistered_hyperparameters():
    cfg = Config()
    assert cfg.seed == 42
    assert cfg.latent_dim == 16
    assert cfg.context_window == 30
    assert cfg.target_window == 10
    assert cfg.learning_rate == 3e-4
    assert cfg.weight_decay == 1e-4
    assert cfg.batch_size == 64
    assert cfg.max_epochs == 500
    assert cfg.early_stopping_patience == 30
    assert cfg.early_stopping_min_delta == 1e-5
    assert cfg.sigreg_lambda == 0.02
    assert cfg.threshold_percentile == 99
    assert cfg.episode_cooldown_days == 10
    assert cfg.drawdown_threshold == -0.20
    assert cfg.drawdown_window_days == 60
    assert cfg.drawdown_lookback_days == 252
