"""
Tests for src/utils.py: detect_episodes and make_windows.
"""
import numpy as np
import pandas as pd
import pytest

from src.utils import detect_episodes, make_windows


# ---------------------------------------------------------------------------
# make_windows helpers
# ---------------------------------------------------------------------------

def _synthetic_features(n_days: int, n_features: int = 6) -> tuple[np.ndarray, pd.DatetimeIndex]:
    rng = np.random.default_rng(0)
    features = rng.standard_normal((n_days, n_features))
    dates = pd.bdate_range("2010-01-04", periods=n_days)
    return features, dates


# ---------------------------------------------------------------------------
# detect_episodes helpers
# ---------------------------------------------------------------------------

def _breach_series(flags: list[int]) -> pd.Series:
    dates = pd.bdate_range("2021-01-04", periods=len(flags))
    return pd.Series([bool(f) for f in flags], index=dates)


# ---------------------------------------------------------------------------
# detect_episodes
# ---------------------------------------------------------------------------

def test_no_breaches_returns_all_zeros():
    breach = _breach_series([0, 0, 0, 0, 0])
    ids = detect_episodes(breach)
    assert (ids == 0).all()


def test_single_breach_is_episode_one():
    breach = _breach_series([0, 1, 0, 0, 0])
    ids = detect_episodes(breach)
    assert ids.iloc[1] == 1
    assert ids.iloc[0] == 0
    assert ids.iloc[2] == 0


def test_nine_day_gap_is_one_episode():
    # gap < cooldown(10): second burst is still part of episode 1
    flags = [1] + [0] * 9 + [1]
    breach = _breach_series(flags)
    ids = detect_episodes(breach)
    assert ids.iloc[0] == 1
    assert ids.iloc[-1] == 1
    assert ids.max() == 1


def test_episode_ids_increment_across_three_episodes():
    # three bursts each separated by exactly 10 quiet days
    flags = [1] + [0] * 10 + [1] + [0] * 10 + [1]
    breach = _breach_series(flags)
    ids = detect_episodes(breach)
    # layout: [1, 0×10, 1, 0×10, 1] → indices 0, 11, 22
    assert ids.iloc[0] == 1
    assert ids.iloc[11] == 2
    assert ids.iloc[22] == 3


# ---------------------------------------------------------------------------
# make_windows
# ---------------------------------------------------------------------------

def test_make_windows_correct_count():
    n, ctx, tgt = 50, 30, 10
    features, dates = _synthetic_features(n)
    windows = make_windows(features, dates, context=ctx, target=tgt)
    assert len(windows) == n - ctx - tgt + 1


def test_make_windows_correct_shapes():
    features, dates = _synthetic_features(50)
    windows = make_windows(features, dates, context=30, target=10)
    w = windows[0]
    assert w.context_array.shape == (30, 6)
    assert w.target_array.shape == (10, 6)


def test_make_windows_date_is_last_context_day():
    features, dates = _synthetic_features(50)
    windows = make_windows(features, dates, context=30, target=10)
    # first window: context covers rows 0..29, date = dates[29]
    assert windows[0].date == dates[29]
    # second window: context covers rows 1..30, date = dates[30]
    assert windows[1].date == dates[30]


def test_make_windows_no_overlap_past_end():
    # exact fit: N = context + target → exactly 1 window
    n, ctx, tgt = 40, 30, 10
    features, dates = _synthetic_features(n)
    windows = make_windows(features, dates, context=ctx, target=tgt)
    assert len(windows) == 1
    # target of that window ends at row 39 (last valid row)
    last_w = windows[-1]
    assert last_w.target_array.shape[0] == tgt


def test_ten_day_gap_starts_new_episode():
    # gap == cooldown(10): cooldown expires, second burst is episode 2
    flags = [1] + [0] * 10 + [1]
    breach = _breach_series(flags)
    ids = detect_episodes(breach)
    assert ids.iloc[0] == 1
    assert ids.iloc[-1] == 2
    assert ids.max() == 2
