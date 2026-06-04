"""
Tests for PCAScorer and IsolationForestScorer (issue #21).

All tests use synthetic list[Window] data — no yfinance or FRED calls.

Coverage:
- PCAScorer.score() returns correct columns
- pca_breach is True when pca_score > pca_threshold
- Reconstruction error near-zero for in-distribution, positive for OOD
- pca_threshold = p99 of calibration reconstruction errors
- fit_calibration returns self (PCAScorer)
- IsolationForestScorer.score() returns correct columns
- OOD window scores above threshold from clean calibration set
- if_threshold = p99 of calibration anomaly scores
- fit_calibration returns self (IsolationForestScorer)
- Episode IDs match detect_episodes output
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.types import Config, Window
from src.utils import detect_episodes


# ---------------------------------------------------------------------------
# Lazy import (fails with ImportError until module exists — RED phase)
# ---------------------------------------------------------------------------

from src.poc3_baselines import PCAScorer, IsolationForestScorer


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _make_windows(
    n: int,
    *,
    scale: float = 1.0,
    seed: int = 42,
    start: str = "2010-01-04",
) -> list[Window]:
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range(start, periods=n)
    return [
        Window(
            context_array=(rng.standard_normal((30, 6)) * scale).astype(np.float32),
            target_array=rng.standard_normal((10, 6)).astype(np.float32),
            date=dates[i],
        )
        for i in range(n)
    ]


@pytest.fixture
def config():
    return Config()


@pytest.fixture
def train_windows():
    return _make_windows(200, seed=1, start="2010-01-04")


@pytest.fixture
def calib_windows():
    return _make_windows(100, seed=2, start="2019-01-02")


@pytest.fixture
def test_windows():
    return _make_windows(50, seed=3, start="2021-01-04")


@pytest.fixture
def ood_windows():
    return _make_windows(50, scale=20.0, seed=99, start="2021-01-04")


@pytest.fixture
def fitted_pca(train_windows, calib_windows, config):
    return PCAScorer(train_windows, config).fit_calibration(calib_windows)


@pytest.fixture
def fitted_if(train_windows, calib_windows, config):
    return IsolationForestScorer(train_windows, config).fit_calibration(calib_windows)


# ---------------------------------------------------------------------------
# Cycle 1 — PCAScorer output columns
# ---------------------------------------------------------------------------

PCA_EXPECTED_COLUMNS = {"pca_score", "pca_threshold", "pca_breach", "pca_episode_id"}


def test_pca_score_has_correct_columns(fitted_pca, test_windows):
    out = fitted_pca.score(test_windows)
    assert set(out.columns) == PCA_EXPECTED_COLUMNS


def test_pca_score_length_matches_input(fitted_pca, test_windows):
    out = fitted_pca.score(test_windows)
    assert len(out) == len(test_windows)


def test_pca_score_index_is_window_dates(fitted_pca, test_windows):
    out = fitted_pca.score(test_windows)
    expected_dates = pd.DatetimeIndex([w.date for w in test_windows])
    pd.testing.assert_index_equal(out.index, expected_dates)


# ---------------------------------------------------------------------------
# Cycle 2 — pca_breach logic
# ---------------------------------------------------------------------------

def test_pca_breach_is_boolean(fitted_pca, test_windows):
    out = fitted_pca.score(test_windows)
    assert out["pca_breach"].dtype == bool


def test_pca_breach_true_when_score_exceeds_threshold(fitted_pca, test_windows):
    out = fitted_pca.score(test_windows)
    above = out["pca_score"] > out["pca_threshold"]
    pd.testing.assert_series_equal(
        out["pca_breach"].reset_index(drop=True),
        above.reset_index(drop=True),
        check_names=False,
    )


def test_pca_threshold_column_is_constant(fitted_pca, test_windows):
    out = fitted_pca.score(test_windows)
    assert out["pca_threshold"].nunique() == 1


# ---------------------------------------------------------------------------
# Cycle 3 — Reconstruction error: OOD > in-distribution
# ---------------------------------------------------------------------------

def test_pca_ood_score_higher_than_indistribution(train_windows, calib_windows,
                                                   test_windows, ood_windows, config):
    scorer = PCAScorer(train_windows, config).fit_calibration(calib_windows)
    in_dist = scorer.score(test_windows)
    out_dist = scorer.score(ood_windows)
    assert out_dist["pca_score"].mean() > in_dist["pca_score"].mean()


def test_pca_ood_breaches_more_than_indistribution(train_windows, calib_windows,
                                                    test_windows, ood_windows, config):
    scorer = PCAScorer(train_windows, config).fit_calibration(calib_windows)
    in_breaches = scorer.score(test_windows)["pca_breach"].sum()
    ood_breaches = scorer.score(ood_windows)["pca_breach"].sum()
    assert ood_breaches > in_breaches


# ---------------------------------------------------------------------------
# Cycle 4 — pca_threshold = p99 of calibration errors
# ---------------------------------------------------------------------------

def test_pca_threshold_is_p99_of_calib_errors(train_windows, calib_windows, config):
    scorer = PCAScorer(train_windows, config)
    scorer.fit_calibration(calib_windows)
    calib_out = scorer.score(calib_windows)
    expected = float(np.percentile(calib_out["pca_score"], 99))
    assert scorer.pca_threshold == pytest.approx(expected, rel=1e-5)


# ---------------------------------------------------------------------------
# Cycle 5 — fit_calibration returns self (PCAScorer)
# ---------------------------------------------------------------------------

def test_pca_fit_calibration_returns_self(train_windows, calib_windows, config):
    scorer = PCAScorer(train_windows, config)
    result = scorer.fit_calibration(calib_windows)
    assert result is scorer


def test_pca_score_idempotent(fitted_pca, test_windows):
    out1 = fitted_pca.score(test_windows)
    out2 = fitted_pca.score(test_windows)
    pd.testing.assert_frame_equal(out1, out2)


# ---------------------------------------------------------------------------
# Cycle 6 — IsolationForestScorer output columns
# ---------------------------------------------------------------------------

IF_EXPECTED_COLUMNS = {"if_score", "if_threshold", "if_breach", "if_episode_id"}


def test_if_score_has_correct_columns(fitted_if, test_windows):
    out = fitted_if.score(test_windows)
    assert set(out.columns) == IF_EXPECTED_COLUMNS


def test_if_score_length_matches_input(fitted_if, test_windows):
    out = fitted_if.score(test_windows)
    assert len(out) == len(test_windows)


def test_if_score_index_is_window_dates(fitted_if, test_windows):
    out = fitted_if.score(test_windows)
    expected_dates = pd.DatetimeIndex([w.date for w in test_windows])
    pd.testing.assert_index_equal(out.index, expected_dates)


# ---------------------------------------------------------------------------
# Cycle 7 — OOD IF score above threshold
# ---------------------------------------------------------------------------

def test_if_ood_score_higher_than_indistribution(train_windows, calib_windows,
                                                  test_windows, ood_windows, config):
    scorer = IsolationForestScorer(train_windows, config).fit_calibration(calib_windows)
    in_dist = scorer.score(test_windows)
    out_dist = scorer.score(ood_windows)
    assert out_dist["if_score"].mean() > in_dist["if_score"].mean()


def test_if_breach_is_boolean(fitted_if, test_windows):
    out = fitted_if.score(test_windows)
    assert out["if_breach"].dtype == bool


def test_if_breach_true_when_score_exceeds_threshold(fitted_if, test_windows):
    out = fitted_if.score(test_windows)
    above = out["if_score"] > out["if_threshold"]
    pd.testing.assert_series_equal(
        out["if_breach"].reset_index(drop=True),
        above.reset_index(drop=True),
        check_names=False,
    )


def test_if_threshold_column_is_constant(fitted_if, test_windows):
    out = fitted_if.score(test_windows)
    assert out["if_threshold"].nunique() == 1


# ---------------------------------------------------------------------------
# Cycle 8 — if_threshold = p99 of calibration scores
# ---------------------------------------------------------------------------

def test_if_threshold_is_p99_of_calib_scores(train_windows, calib_windows, config):
    scorer = IsolationForestScorer(train_windows, config)
    scorer.fit_calibration(calib_windows)
    calib_out = scorer.score(calib_windows)
    expected = float(np.percentile(calib_out["if_score"], 99))
    assert scorer.if_threshold == pytest.approx(expected, rel=1e-5)


# ---------------------------------------------------------------------------
# Cycle 9 — fit_calibration returns self (IsolationForestScorer)
# ---------------------------------------------------------------------------

def test_if_fit_calibration_returns_self(train_windows, calib_windows, config):
    scorer = IsolationForestScorer(train_windows, config)
    result = scorer.fit_calibration(calib_windows)
    assert result is scorer


def test_if_score_idempotent(fitted_if, test_windows):
    out1 = fitted_if.score(test_windows)
    out2 = fitted_if.score(test_windows)
    pd.testing.assert_frame_equal(out1, out2)


# ---------------------------------------------------------------------------
# Cycle 10 — Episode IDs match detect_episodes
# ---------------------------------------------------------------------------

def test_pca_episode_ids_match_detect_episodes(fitted_pca, test_windows, config):
    out = fitted_pca.score(test_windows)
    expected = detect_episodes(out["pca_breach"], cooldown=config.episode_cooldown_days)
    pd.testing.assert_series_equal(
        out["pca_episode_id"].reset_index(drop=True),
        expected.reset_index(drop=True),
        check_names=False,
    )


def test_if_episode_ids_match_detect_episodes(fitted_if, test_windows, config):
    out = fitted_if.score(test_windows)
    expected = detect_episodes(out["if_breach"], cooldown=config.episode_cooldown_days)
    pd.testing.assert_series_equal(
        out["if_episode_id"].reset_index(drop=True),
        expected.reset_index(drop=True),
        check_names=False,
    )
