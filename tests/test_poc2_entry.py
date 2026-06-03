"""
Tests for run_lejepa_soxx_poc2._apply_regime_filter.

Behaviors tested:
- Correct latent rows removed by window mask
- Correct calib_df rows removed by daily mask
- AssertionError when filtered sample < 50 rows
- Before/after counts logged to console
"""
import numpy as np
import pandas as pd
import pytest


def _make_dates(n: int, start: str = "2019-01-02") -> pd.DatetimeIndex:
    return pd.bdate_range(start=start, periods=n)


def _make_rv20(dates: pd.DatetimeIndex, high_indices: set, high_val: float = 5.0) -> pd.Series:
    values = np.ones(len(dates))
    for i in high_indices:
        values[i] = high_val
    return pd.Series(values, index=dates)


# ---------------------------------------------------------------------------
# Cycle 1: window mask removes correct latent rows
# ---------------------------------------------------------------------------

def test_regime_filter_removes_high_regime_latent_rows():
    """Latent arrays must have rows removed where window dates exceed regime threshold."""
    from run_lejepa_soxx_poc2 import _apply_regime_filter

    n = 60
    window_dates = _make_dates(n, start="2019-01-02")
    high_indices = {5, 20, 40}  # 3 windows above threshold
    rv20 = _make_rv20(window_dates, high_indices, high_val=5.0)

    # Daily calib_df: same dates, no high-regime days (so assertion passes)
    calib_df = pd.DataFrame({"rv20": np.ones(n), "vix": np.ones(n)}, index=window_dates)

    z_target = np.arange(n * 16, dtype=float).reshape(n, 16)
    z_pred = z_target + 100
    z_context = z_target + 200

    # threshold = 2.0 → z-scores of 5.0 (high_val with mean=0, std=1) exceed it
    z_t, z_p, z_c, _ = _apply_regime_filter(
        z_target, z_pred, z_context,
        window_dates, calib_df,
        rv20, rv20_mean=0.0, rv20_std=1.0,
        regime_threshold=2.0,
        min_rows=50,
    )

    kept = [i for i in range(n) if i not in high_indices]
    assert len(z_t) == len(kept)
    assert len(z_p) == len(kept)
    assert len(z_c) == len(kept)
    np.testing.assert_array_equal(z_t, z_target[kept])
    np.testing.assert_array_equal(z_p, z_pred[kept])
    np.testing.assert_array_equal(z_c, z_context[kept])


# ---------------------------------------------------------------------------
# Cycle 2: daily mask removes correct calib_df rows
# ---------------------------------------------------------------------------

def test_regime_filter_removes_high_regime_calib_df_rows():
    """calib_df must have rows removed where daily dates exceed regime threshold."""
    from run_lejepa_soxx_poc2 import _apply_regime_filter

    n_windows = 60
    n_daily = 80  # calib_df has more rows than window array

    window_dates = _make_dates(n_windows, start="2019-01-02")
    daily_dates = _make_dates(n_daily, start="2019-01-02")

    high_daily = {10, 30, 50, 70}  # 4 daily rows above threshold
    rv20_daily = _make_rv20(daily_dates, high_daily, high_val=5.0)

    # rv20 series covers both date ranges
    rv20_all = rv20_daily.reindex(daily_dates.union(window_dates), fill_value=1.0)

    calib_df = pd.DataFrame(
        {"rv20": rv20_daily.values, "vix": np.ones(n_daily)}, index=daily_dates
    )
    z_target = np.ones((n_windows, 16))
    z_pred = np.ones((n_windows, 16))
    z_context = np.ones((n_windows, 16))

    _, _, _, df_filtered = _apply_regime_filter(
        z_target, z_pred, z_context,
        window_dates, calib_df,
        rv20_all, rv20_mean=0.0, rv20_std=1.0,
        regime_threshold=2.0,
        min_rows=50,
    )

    expected_kept = [i for i in range(n_daily) if i not in high_daily]
    assert len(df_filtered) == len(expected_kept)
    pd.testing.assert_index_equal(df_filtered.index, daily_dates[expected_kept])


# ---------------------------------------------------------------------------
# Cycle 3: AssertionError when filtered sample < min_rows
# ---------------------------------------------------------------------------

def test_regime_filter_raises_when_below_min_rows():
    """AssertionError must be raised when filtered latent count < min_rows."""
    from run_lejepa_soxx_poc2 import _apply_regime_filter

    n = 60
    dates = _make_dates(n, start="2019-01-02")
    # All z-scores are 5.0, threshold is 2.0 → all excluded
    rv20 = pd.Series(np.full(n, 5.0), index=dates)
    calib_df = pd.DataFrame({"rv20": rv20.values, "vix": np.ones(n)}, index=dates)

    z_target = np.ones((n, 16))
    z_pred = np.ones((n, 16))
    z_context = np.ones((n, 16))

    with pytest.raises(AssertionError, match="50"):
        _apply_regime_filter(
            z_target, z_pred, z_context,
            dates, calib_df,
            rv20, rv20_mean=0.0, rv20_std=1.0,
            regime_threshold=2.0,
            min_rows=50,
        )


# ---------------------------------------------------------------------------
# Cycle 4: before/after counts logged
# ---------------------------------------------------------------------------

def test_regime_filter_logs_before_after_counts(capsys):
    """Console output must include before and after row counts."""
    from run_lejepa_soxx_poc2 import _apply_regime_filter

    n = 60
    high_indices = {1, 2, 3}
    dates = _make_dates(n, start="2019-01-02")
    rv20 = _make_rv20(dates, high_indices, high_val=5.0)
    calib_df = pd.DataFrame({"rv20": rv20.values, "vix": np.ones(n)}, index=dates)

    z_target = np.ones((n, 16))
    z_pred = np.ones((n, 16))
    z_context = np.ones((n, 16))

    _apply_regime_filter(
        z_target, z_pred, z_context,
        dates, calib_df,
        rv20, rv20_mean=0.0, rv20_std=1.0,
        regime_threshold=2.0,
        min_rows=50,
    )

    captured = capsys.readouterr()
    assert str(n) in captured.out          # before count
    assert str(n - len(high_indices)) in captured.out  # after count
