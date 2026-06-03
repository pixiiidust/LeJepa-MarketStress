"""
Tests for DataPipeline.

Cover:
- Scaler fitted on train slice only (stats match training-period ground truth)
- Alignment produces no NaN after forward-fill on synthetic series with gaps
- Split boundary assertion fires when a window spans two splits
- log_volume_ratio excludes today (t-20:t-1, not t-20:t)
"""
import numpy as np
import pandas as pd
import pytest

from src.types import Config
from src.data_pipeline import (
    compute_features,
    fit_and_apply_scaler,
    forward_fill_dgs10,
    assert_split_boundaries,
)


def _dates(n: int, start: str = "2015-01-01") -> pd.DatetimeIndex:
    return pd.bdate_range(start, periods=n)


def _raw_df(n: int, start: str = "2015-01-01", **col_overrides) -> pd.DataFrame:
    """Synthetic raw DataFrame; keyword args override individual columns."""
    idx = _dates(n, start)
    base = {
        "close": pd.Series(100.0, index=idx),
        "volume": pd.Series(1_000.0, index=idx),
        "vix": pd.Series(20.0, index=idx),
        "dgs10": pd.Series(2.0, index=idx),
    }
    base.update(col_overrides)
    return pd.DataFrame(base)


# ---------------------------------------------------------------------------
# log_volume_ratio
# ---------------------------------------------------------------------------

def test_log_volume_ratio_excludes_today():
    """log_volume_ratio window is mean(volume[t-20:t-1]), NOT including today."""
    n = 60
    idx = _dates(n)
    volume = pd.Series(1_000.0, index=idx)
    volume.iloc[29] = 1_000_000.0   # sharp spike on day 29

    features = compute_features(_raw_df(n, volume=volume))

    # At day 29 the rolling mean covers days 9..28 (all 1 000) → ratio = 1 000 000/1 000
    # If today were included the mean would be (19×1000 + 1e6)/20 ≈ 50 950, giving ~2.977
    expected = np.log(1_000_000.0 / 1_000.0)
    got = features["log_volume_ratio"].iloc[29]
    assert np.isclose(got, expected), (
        f"Expected {expected:.6f} (excludes today), got {got:.6f}. "
        "Rolling mean may be including the current day."
    )


# ---------------------------------------------------------------------------
# Scaler fitted on train slice only
# ---------------------------------------------------------------------------

def test_scaler_fitted_on_train_only():
    """RobustScaler center_/scale_ match training-period statistics only."""
    rng = np.random.default_rng(42)
    n_train, n_other = 60, 40
    n = n_train + n_other
    idx = _dates(n)
    train_end_date = idx[n_train - 1]

    # Training: N(0, 1); post-training: N(50, 1) — leak is obvious if scaler sees both
    train_vals = rng.normal(0.0, 1.0, (n_train, 2))
    other_vals = rng.normal(50.0, 1.0, (n_other, 2))
    vals = np.vstack([train_vals, other_vals])
    features_df = pd.DataFrame(vals, index=idx, columns=["f1", "f2"])

    _, scaler = fit_and_apply_scaler(features_df, str(train_end_date.date()))

    expected_center = np.median(train_vals, axis=0)
    assert np.allclose(scaler.center_, expected_center, atol=1e-10), (
        "center_ mismatch — scaler may have seen post-training data"
    )
    q75 = np.percentile(train_vals, 75, axis=0)
    q25 = np.percentile(train_vals, 25, axis=0)
    assert np.allclose(scaler.scale_, q75 - q25, atol=1e-10), (
        "scale_ mismatch — scaler may have seen post-training data"
    )


# ---------------------------------------------------------------------------
# Alignment / forward-fill
# ---------------------------------------------------------------------------

def test_alignment_no_nan_after_forward_fill():
    """A 2-day DGS10 gap leaves zero NaN after forward-fill (max_fill=3)."""
    idx = _dates(30)
    dgs10 = pd.Series(2.0, index=idx)
    dgs10.iloc[5] = np.nan
    dgs10.iloc[6] = np.nan   # 2-day gap — within the limit

    result = forward_fill_dgs10(dgs10, max_fill=3)

    assert result.isna().sum() == 0, "NaN values remain after forward-fill"
    assert result.iloc[5] == pytest.approx(2.0)
    assert result.iloc[6] == pytest.approx(2.0)


# ---------------------------------------------------------------------------
# Split boundary assertions
# ---------------------------------------------------------------------------

def test_split_boundary_train_too_late_raises():
    """AssertionError when train date range extends past train_end."""
    config = Config()   # train_end="2016-12-31", calib_start="2019-01-01"

    bad_train = pd.bdate_range("2010-01-04", "2017-01-10")   # ends 2017
    good_calib = pd.bdate_range("2019-01-02", "2019-06-28")

    with pytest.raises(AssertionError):
        assert_split_boundaries(bad_train, good_calib, config)


def test_split_boundary_calib_too_early_raises():
    """AssertionError when calibration dates start before calib_start."""
    config = Config()

    good_train = pd.bdate_range("2010-01-04", "2016-12-30")
    bad_calib = pd.bdate_range("2018-01-02", "2018-12-31")   # starts 2018

    with pytest.raises(AssertionError):
        assert_split_boundaries(good_train, bad_calib, config)
