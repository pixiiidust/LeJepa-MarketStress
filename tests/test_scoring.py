"""
Tests for Scorer (episode detection).

Cover:
- 9-day gap in breaches: two episodes detected (gap < 10-day cooldown; doesn't split ep2 into ep3)
- 10-day gap in breaches: two episodes detected (gap == cooldown; second breach starts ep2)
- pred_error_valid_date = T+10 trading days for mid-period rows
- pred_error = NaN for last 10 rows of test period
- lejepa_episode_id increments correctly across multi-episode sequences
"""
import math

import numpy as np
import pandas as pd
import pytest
import torch
from unittest.mock import MagicMock

from src.model import LeJEPAModel
from src.calibration import MahalanobisCalibrator
from src.scoring import Scorer
from src.types import Window

# ---------------------------------------------------------------------------
# Constants / helpers
# ---------------------------------------------------------------------------

_N_FEATURES = 6
_CONTEXT = 30
_TARGET = 10
_LATENT = 16
_THRESHOLD = 1.0
_HIGH = 2.0   # above threshold → breach
_LOW = 0.1    # below threshold → no breach


def _make_window(val: float = 0.0) -> Window:
    ctx = np.full((_CONTEXT, _N_FEATURES), val, dtype=np.float32)
    tgt = np.full((_TARGET, _N_FEATURES), val, dtype=np.float32)
    return Window(context_array=ctx, target_array=tgt, date=pd.Timestamp("2021-01-04"))


def _mock_model() -> MagicMock:
    model = MagicMock()
    model.encode.return_value = torch.zeros(_LATENT, dtype=torch.float32)
    model.predict.return_value = torch.zeros(_LATENT, dtype=torch.float32)
    return model


def _mock_target_cal(scores: list[float]) -> MagicMock:
    """target_cal with controlled per-window scores."""
    cal = MagicMock()
    cal.threshold = _THRESHOLD
    cal.score.side_effect = list(scores)
    return cal


def _mock_context_cal(n: int) -> MagicMock:
    cal = MagicMock()
    cal.threshold = _THRESHOLD
    cal.score.return_value = _LOW
    return cal


def _days(n: int) -> pd.DatetimeIndex:
    return pd.bdate_range("2021-01-04", periods=n)


def _run(breach_pattern: list[float]) -> pd.DataFrame:
    n = len(breach_pattern)
    scorer = Scorer(
        model=_mock_model(),
        target_cal=_mock_target_cal(breach_pattern),
        context_cal=_mock_context_cal(n),
        trading_days=_days(n),
    )
    return scorer.score([_make_window() for _ in range(n)])


# ---------------------------------------------------------------------------
# Test 1 — 9-day gap between two breach events → still in same episode
# ---------------------------------------------------------------------------

def test_nine_day_gap_two_episodes_detected():
    # ep1: breach at 0 | 10 quiet (1-10) | ep2: breach at 11 | 9 quiet (12-20) | ep2 continues at 21
    # 9-day internal gap inside ep2 does NOT open ep3 → total: 2 episodes
    pattern = [_HIGH] + [_LOW] * 10 + [_HIGH] + [_LOW] * 9 + [_HIGH]
    df = _run(pattern)
    ids = df["lejepa_episode_id"]

    assert ids.max() == 2, "9-day gap keeps second breach inside ep2 → only 2 episodes"
    assert ids.iloc[0] == 1
    assert ids.iloc[11] == 2
    assert ids.iloc[21] == 2, "breach after 9-day gap must remain in ep2"


# ---------------------------------------------------------------------------
# Test 2 — 10-day gap between two breach events → creates second episode
# ---------------------------------------------------------------------------

def test_ten_day_gap_two_episodes_detected():
    # ep1: breach at 0 | exactly 10 quiet (1-10) | ep2: breach at 11
    pattern = [_HIGH] + [_LOW] * 10 + [_HIGH]
    df = _run(pattern)
    ids = df["lejepa_episode_id"]

    assert ids.max() == 2, "10-day gap resets cooldown → second breach opens ep2"
    assert ids.iloc[0] == 1
    assert ids.iloc[-1] == 2


# ---------------------------------------------------------------------------
# Test 3 — pred_error_valid_date = T+10 trading days for a mid-period row
# ---------------------------------------------------------------------------

def test_pred_error_valid_date_is_t_plus_ten_trading_days():
    n = 25
    days = _days(n)
    scorer = Scorer(
        model=_mock_model(),
        target_cal=_mock_target_cal([_LOW] * n),
        context_cal=_mock_context_cal(n),
        trading_days=days,
    )
    df = scorer.score([_make_window() for _ in range(n)])

    # Row at position 5 (well before last-10 cutoff): valid_date must be days[15]
    assert df["pred_error_valid_date"].iloc[5] == days[15]
    # Sanity: valid_date uses the trading calendar, not calendar arithmetic
    assert df["pred_error_valid_date"].iloc[0] == days[10]


# ---------------------------------------------------------------------------
# Test 4 — pred_error = NaN for exactly the last 10 rows
# ---------------------------------------------------------------------------

def test_pred_error_nan_for_last_ten_rows():
    n = 25
    days = _days(n)
    scorer = Scorer(
        model=_mock_model(),
        target_cal=_mock_target_cal([_LOW] * n),
        context_cal=_mock_context_cal(n),
        trading_days=days,
    )
    df = scorer.score([_make_window() for _ in range(n)])

    assert df["pred_error"].iloc[-10:].isna().all(), "last 10 rows must have NaN pred_error"
    assert df["pred_error"].iloc[:-10].notna().all(), "first n-10 rows must have finite pred_error"

    # Also verify 32 latent columns are present
    assert all(f"z_pred_{j}" in df.columns for j in range(16))
    assert all(f"z_context_{j}" in df.columns for j in range(16))


# ---------------------------------------------------------------------------
# Test 5 — lejepa_episode_id increments correctly across multi-episode sequence
# ---------------------------------------------------------------------------

def test_episode_id_increments_across_multi_episode_sequence():
    # Three breach groups, each separated by exactly 10 quiet days → 3 episodes
    pattern = (
        [_HIGH] + [_LOW] * 10 +
        [_HIGH] + [_LOW] * 10 +
        [_HIGH]
    )
    df = _run(pattern)
    ids = df["lejepa_episode_id"]

    assert ids.max() == 3
    assert ids.iloc[0] == 1
    assert ids.iloc[11] == 2
    assert ids.iloc[22] == 3
    # Non-breach rows within cooldown carry id=0
    assert ids.iloc[1] == 0


# ---------------------------------------------------------------------------
# Tests for latent column count driven by model dim (#18)
# ---------------------------------------------------------------------------

def _mock_model_dim(latent: int) -> MagicMock:
    model = MagicMock()
    model.encode.return_value = torch.zeros(latent, dtype=torch.float32)
    model.predict.return_value = torch.zeros(latent, dtype=torch.float32)
    return model


def _run_dim(latent: int, n: int = 5) -> pd.DataFrame:
    scorer = Scorer(
        model=_mock_model_dim(latent),
        target_cal=_mock_target_cal([_LOW] * n),
        context_cal=_mock_context_cal(n),
        trading_days=_days(n),
    )
    return scorer.score([_make_window() for _ in range(n)])


def test_scorer_dim8_no_spurious_columns():
    df = _run_dim(8)
    assert all(f"z_pred_{j}" in df.columns for j in range(8))
    assert "z_pred_8" not in df.columns


def test_scorer_latent_columns_driven_by_model_dim():
    df = _run_dim(8)
    assert all(f"z_pred_{j}" in df.columns for j in range(8))
    assert all(f"z_context_{j}" in df.columns for j in range(8))
    assert all(f"z_pred_{j}" not in df.columns for j in range(8, 16))
    assert all(f"z_context_{j}" not in df.columns for j in range(8, 16))


# ---------------------------------------------------------------------------
# Acceptance criterion — eval mode and no_grad during scoring
# ---------------------------------------------------------------------------

def test_scorer_uses_eval_mode_and_no_grad():
    class GradTracker(LeJEPAModel):
        def __init__(self):
            super().__init__()
            self.grad_states: list[bool] = []

        def encode(self, x: torch.Tensor) -> torch.Tensor:
            self.grad_states.append(torch.is_grad_enabled())
            return super().encode(x)

    model = GradTracker()
    model.train()  # explicitly put in training mode before handing to Scorer

    n = 5
    rng = np.random.default_rng(42)
    z_fit = rng.standard_normal((50, _LATENT)).astype(np.float32)

    target_cal = MahalanobisCalibrator()
    target_cal.fit(z_fit)
    target_cal.set_threshold(z_fit)

    context_cal = MahalanobisCalibrator()
    context_cal.fit(z_fit)
    context_cal.set_threshold(z_fit)

    scorer = Scorer(
        model=model,
        target_cal=target_cal,
        context_cal=context_cal,
        trading_days=_days(n),
    )
    scorer.score([_make_window() for _ in range(n)])

    assert not model.training, "Scorer must call model.eval()"
    assert model.grad_states, "encode() must have been called"
    assert all(not g for g in model.grad_states), "encode() must run inside torch.no_grad()"
