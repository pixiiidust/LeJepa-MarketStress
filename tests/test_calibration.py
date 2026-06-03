"""
Tests for MahalanobisCalibrator.

Cover:
- threshold equals 99th percentile of calibration scores on synthetic normal vectors
- score(z) increases monotonically as z moves away from the distribution mean
- Ledoit-Wolf shrinkage produces bounded condition number on low-sample synthetic data
- fit must be called before score/set_threshold
"""
import numpy as np
import pytest

from src.calibration import MahalanobisCalibrator


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
