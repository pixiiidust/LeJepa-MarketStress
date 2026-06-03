"""
Tests for MahalanobisCalibrator and regime_filter_mask.

Cover:
- threshold equals 99th percentile of calibration scores on synthetic normal vectors
- score(z) increases monotonically as z moves away from the distribution mean
- Ledoit-Wolf shrinkage produces bounded condition number on low-sample synthetic data
- fit must be called before score/set_threshold
- regime_filter_mask: exact False positions, all-True, all-False, gap handling, end-to-end consistency
"""
import numpy as np
import pandas as pd
import pytest

from src.calibration import MahalanobisCalibrator, regime_filter_mask


def _make_dates(n: int, start: str = "2010-01-04") -> pd.DatetimeIndex:
    return pd.bdate_range(start=start, periods=n)


RNG = np.random.default_rng(42)


def _fit_calibrator(n: int = 200, d: int = 16) -> MahalanobisCalibrator:
    z_fit = RNG.standard_normal((n, d)).astype(np.float64)
    cal = MahalanobisCalibrator()
    cal.fit(z_fit)
    return cal, z_fit


def test_threshold_equals_99th_percentile():
    """threshold must be exactly np.percentile(scores, 99) — no interpolation fudge."""
    rng = np.random.default_rng(42)
    z_fit = rng.standard_normal((300, 16)).astype(np.float64)
    z_calib = rng.standard_normal((200, 16)).astype(np.float64)

    cal = MahalanobisCalibrator()
    cal.fit(z_fit)
    cal.set_threshold(z_calib, percentile=99)

    scores = np.array([cal.score(z_calib[i]) for i in range(len(z_calib))])
    expected = np.percentile(scores, 99)
    assert cal.threshold == expected


def test_score_increases_with_distance_from_mean():
    """score(z) must increase monotonically as z moves further from the fitted mean."""
    rng = np.random.default_rng(42)
    n, d = 300, 16
    z_fit = rng.standard_normal((n, d)).astype(np.float64)
    cal = MahalanobisCalibrator()
    cal.fit(z_fit)

    # Start at the mean; step along a fixed direction
    direction = rng.standard_normal(d)
    direction /= np.linalg.norm(direction)
    base = cal.mean_

    prev_score = cal.score(base)
    for step in [0.5, 1.0, 2.0, 4.0, 8.0]:
        z_step = base + step * direction
        s = cal.score(z_step)
        assert s > prev_score, f"score did not increase at step={step}"
        prev_score = s


def test_ledoit_wolf_bounded_condition_number():
    """LedoitWolf shrinkage must keep the covariance condition number finite and bounded."""
    rng = np.random.default_rng(42)
    # Low-sample regime: 30 samples, 16 dims (p ≈ n/2 — ill-conditioned without shrinkage)
    z_fit = rng.standard_normal((30, 16)).astype(np.float64)
    cal = MahalanobisCalibrator()
    cal.fit(z_fit)

    # Condition number of the precision matrix must be finite and < 1e6
    prec = cal.precision_matrix_
    eigs = np.linalg.eigvalsh(prec)
    cond = eigs.max() / max(eigs.min(), 1e-10)
    assert np.isfinite(cond), "Condition number is not finite"
    assert cond < 1e6, f"Condition number {cond:.2e} exceeds 1e6 — shrinkage may not be applied"


def test_score_before_fit_raises():
    """Calling score before fit must raise RuntimeError."""
    cal = MahalanobisCalibrator()
    with pytest.raises(RuntimeError):
        cal.score(np.zeros(16))


def test_set_threshold_before_fit_raises():
    """Calling set_threshold before fit must raise RuntimeError."""
    cal = MahalanobisCalibrator()
    with pytest.raises(RuntimeError):
        cal.set_threshold(np.zeros((10, 16)))


# ---------------------------------------------------------------------------
# regime_filter_mask tests
# ---------------------------------------------------------------------------

def test_exactly_five_above_threshold_are_false():
    """Exactly the five dates with z-score > threshold must be False; all others True."""
    dates = _make_dates(20)
    high_indices = [2, 5, 10, 14, 18]
    values = np.zeros(20)
    for i in high_indices:
        values[i] = 2.0  # z-score = 2.0 with mean=0, std=1
    rv20 = pd.Series(values, index=dates)

    mask = regime_filter_mask(dates, rv20, rv20_mean=0.0, rv20_std=1.0, pct_threshold=1.5)

    expected = np.ones(20, dtype=bool)
    for i in high_indices:
        expected[i] = False
    np.testing.assert_array_equal(mask, expected)


def test_threshold_above_max_zscore_returns_all_true():
    """When threshold exceeds every z-score, every date must be kept (all True)."""
    dates = _make_dates(10)
    values = np.arange(1.0, 11.0)  # z-scores 1..10 with mean=0, std=1
    rv20 = pd.Series(values, index=dates)

    mask = regime_filter_mask(dates, rv20, rv20_mean=0.0, rv20_std=1.0, pct_threshold=11.0)

    assert mask.all()


def test_threshold_below_min_zscore_returns_all_false():
    """When threshold is below every z-score, no date is kept (all False)."""
    dates = _make_dates(10)
    values = np.arange(1.0, 11.0)  # z-scores 1..10 with mean=0, std=1
    rv20 = pd.Series(values, index=dates)

    mask = regime_filter_mask(dates, rv20, rv20_mean=0.0, rv20_std=1.0, pct_threshold=0.5)

    assert not mask.any()


def test_gaps_in_rv20_series_map_to_false_and_length_matches():
    """Dates missing from rv20_series reindex to NaN; NaN z-scores must be False.
    Returned array length must equal len(dates) regardless of series coverage."""
    dates = _make_dates(10)
    gap_indices = {1, 3, 5, 7, 9}
    present_dates = dates[[i for i in range(10) if i not in gap_indices]]
    rv20 = pd.Series(np.zeros(len(present_dates)), index=present_dates)  # z-score = 0.0 for present

    mask = regime_filter_mask(dates, rv20, rv20_mean=0.0, rv20_std=1.0, pct_threshold=1.0)

    assert len(mask) == len(dates)
    for i in gap_indices:
        assert not mask[i], f"gap at index {i} should be False (NaN z-score)"
    for i in range(10):
        if i not in gap_indices:
            assert mask[i], f"present date at index {i} (z=0.0) should be True"


def test_mask_excludes_same_regime_dates_in_latents_and_dataframe():
    """Applying the mask to a latent array and a daily DataFrame must exclude identical rows.

    Synthetic setup: 10 dates, 3 marked as high-regime. After masking, both the latent
    array and the DataFrame must retain exactly the same 7 non-regime rows.
    """
    n = 10
    dates = _make_dates(n)
    high_indices = {2, 5, 8}
    values = np.where(np.isin(np.arange(n), list(high_indices)), 3.0, 0.0)
    rv20 = pd.Series(values, index=dates)

    mask = regime_filter_mask(dates, rv20, rv20_mean=0.0, rv20_std=1.0, pct_threshold=1.5)

    latents = np.arange(n * 16, dtype=float).reshape(n, 16)
    df = pd.DataFrame({"val": np.arange(n, dtype=float)}, index=dates)

    filtered_latents = latents[mask]
    filtered_df = df[mask]

    kept_indices = [i for i in range(n) if i not in high_indices]
    assert len(filtered_latents) == len(kept_indices)
    assert len(filtered_df) == len(kept_indices)
    np.testing.assert_array_equal(filtered_latents, latents[kept_indices])
    pd.testing.assert_index_equal(filtered_df.index, dates[kept_indices])
