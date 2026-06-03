"""
Tests for BaselineScorer (issue #10).

All tests use seed=42 synthetic DataFrames — no yfinance or FRED calls.

Coverage:
- Normalisation stats come from train_df only and are never modified by fit_calibration or score
- rv20_threshold and vix_threshold are the 99th percentile of calibration z-scores
- score(calib_df) and score(test_df) callable independently in any order after fit_calibration
- Episode IDs in output match detect_episodes() output on the same breach series (identity check)
- Output columns are exactly the specified set
"""
import numpy as np
import pandas as pd
import pytest

from src.baselines import BaselineScorer
from src.types import Config
from src.utils import detect_episodes


RNG = np.random.default_rng(42)


def _make_df(n: int, rv20_mean: float = 0.15, vix_mean: float = 20.0, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2010-01-04", periods=n)
    rv20 = np.abs(rng.normal(rv20_mean, 0.05, n))
    vix = np.abs(rng.normal(vix_mean, 5.0, n))
    return pd.DataFrame({"rv20": rv20, "vix": vix}, index=dates)


@pytest.fixture
def train_df():
    return _make_df(500, seed=1)


@pytest.fixture
def calib_df():
    return _make_df(200, rv20_mean=0.20, vix_mean=22.0, seed=2)


@pytest.fixture
def test_df():
    return _make_df(150, rv20_mean=0.30, vix_mean=30.0, seed=3)


@pytest.fixture
def config():
    return Config()


@pytest.fixture
def fitted_scorer(train_df, calib_df, config):
    bs = BaselineScorer(train_df, config)
    bs.fit_calibration(calib_df)
    return bs


# ---------------------------------------------------------------------------
# Normalisation stats come exclusively from train_df
# ---------------------------------------------------------------------------

def test_normalisation_stats_from_train_only(train_df, config):
    bs = BaselineScorer(train_df, config)
    assert bs.rv20_mean == pytest.approx(train_df["rv20"].mean())
    assert bs.rv20_std == pytest.approx(train_df["rv20"].std())
    assert bs.vix_mean == pytest.approx(train_df["vix"].mean())
    assert bs.vix_std == pytest.approx(train_df["vix"].std())


def test_fit_calibration_does_not_change_norm_stats(train_df, calib_df, config):
    bs = BaselineScorer(train_df, config)
    rv20_mean_before = bs.rv20_mean
    rv20_std_before = bs.rv20_std
    vix_mean_before = bs.vix_mean
    vix_std_before = bs.vix_std

    bs.fit_calibration(calib_df)

    assert bs.rv20_mean == rv20_mean_before
    assert bs.rv20_std == rv20_std_before
    assert bs.vix_mean == vix_mean_before
    assert bs.vix_std == vix_std_before


def test_score_does_not_change_norm_stats(fitted_scorer, test_df):
    bs = fitted_scorer
    rv20_mean_before = bs.rv20_mean
    vix_mean_before = bs.vix_mean

    bs.score(test_df)

    assert bs.rv20_mean == rv20_mean_before
    assert bs.vix_mean == vix_mean_before


# ---------------------------------------------------------------------------
# Thresholds are 99th percentile of calibration z-scores
# ---------------------------------------------------------------------------

def test_rv20_threshold_is_99th_percentile_of_calib_scores(train_df, calib_df, config):
    bs = BaselineScorer(train_df, config)
    bs.fit_calibration(calib_df)

    calib_zscores = (calib_df["rv20"] - bs.rv20_mean) / bs.rv20_std
    expected = float(np.percentile(calib_zscores, 99))
    assert bs.rv20_threshold == pytest.approx(expected)


def test_vix_threshold_is_99th_percentile_of_calib_scores(train_df, calib_df, config):
    bs = BaselineScorer(train_df, config)
    bs.fit_calibration(calib_df)

    calib_zscores = (calib_df["vix"] - bs.vix_mean) / bs.vix_std
    expected = float(np.percentile(calib_zscores, 99))
    assert bs.vix_threshold == pytest.approx(expected)


# ---------------------------------------------------------------------------
# score() output columns
# ---------------------------------------------------------------------------

EXPECTED_COLUMNS = {
    "rv20_zscore", "rv20_threshold", "rv20_breach", "rv20_episode_id",
    "vix_zscore", "vix_threshold", "vix_breach", "vix_episode_id",
}


def test_score_output_has_correct_columns(fitted_scorer, test_df):
    out = fitted_scorer.score(test_df)
    assert set(out.columns) == EXPECTED_COLUMNS


def test_score_threshold_columns_are_constant(fitted_scorer, test_df):
    out = fitted_scorer.score(test_df)
    assert (out["rv20_threshold"] == fitted_scorer.rv20_threshold).all()
    assert (out["vix_threshold"] == fitted_scorer.vix_threshold).all()


# ---------------------------------------------------------------------------
# score() callable independently for calib and test splits
# ---------------------------------------------------------------------------

def test_score_calib_then_test_consistent(train_df, calib_df, test_df, config):
    bs = BaselineScorer(train_df, config)
    bs.fit_calibration(calib_df)

    out_calib = bs.score(calib_df)
    out_test = bs.score(test_df)

    # thresholds must be identical in both outputs
    assert (out_calib["rv20_threshold"] == out_test["rv20_threshold"].iloc[0]).all()
    assert (out_calib["vix_threshold"] == out_test["vix_threshold"].iloc[0]).all()


def test_score_test_then_calib_same_thresholds(train_df, calib_df, test_df, config):
    bs = BaselineScorer(train_df, config)
    bs.fit_calibration(calib_df)

    out_test = bs.score(test_df)
    out_calib = bs.score(calib_df)

    assert (out_test["rv20_threshold"] == out_calib["rv20_threshold"].iloc[0]).all()


def test_score_idempotent(fitted_scorer, test_df):
    out1 = fitted_scorer.score(test_df)
    out2 = fitted_scorer.score(test_df)
    pd.testing.assert_frame_equal(out1, out2)


# ---------------------------------------------------------------------------
# Episode IDs match detect_episodes() on the same breach series
# ---------------------------------------------------------------------------

def test_rv20_episode_ids_match_detect_episodes(fitted_scorer, test_df, config):
    out = fitted_scorer.score(test_df)
    expected_ids = detect_episodes(out["rv20_breach"], cooldown=config.episode_cooldown_days)
    pd.testing.assert_series_equal(
        out["rv20_episode_id"].reset_index(drop=True),
        expected_ids.reset_index(drop=True),
        check_names=False,
    )


def test_vix_episode_ids_match_detect_episodes(fitted_scorer, test_df, config):
    out = fitted_scorer.score(test_df)
    expected_ids = detect_episodes(out["vix_breach"], cooldown=config.episode_cooldown_days)
    pd.testing.assert_series_equal(
        out["vix_episode_id"].reset_index(drop=True),
        expected_ids.reset_index(drop=True),
        check_names=False,
    )


def test_episode_ids_on_calib_match_detect_episodes(train_df, calib_df, config):
    bs = BaselineScorer(train_df, config)
    bs.fit_calibration(calib_df)
    out = bs.score(calib_df)

    expected_rv20 = detect_episodes(out["rv20_breach"], cooldown=config.episode_cooldown_days)
    pd.testing.assert_series_equal(
        out["rv20_episode_id"].reset_index(drop=True),
        expected_rv20.reset_index(drop=True),
        check_names=False,
    )


# ---------------------------------------------------------------------------
# Breach logic
# ---------------------------------------------------------------------------

def test_breach_is_boolean(fitted_scorer, test_df):
    out = fitted_scorer.score(test_df)
    assert out["rv20_breach"].dtype == bool
    assert out["vix_breach"].dtype == bool


def test_breach_when_zscore_exceeds_threshold(train_df, config):
    """Construct a df where some rv20 values are known to breach."""
    # train with tight distribution
    rng = np.random.default_rng(42)
    n_train = 300
    dates_train = pd.bdate_range("2010-01-04", periods=n_train)
    rv20_train = rng.normal(0.10, 0.01, n_train)
    vix_train = rng.normal(15.0, 1.0, n_train)
    train = pd.DataFrame({"rv20": rv20_train, "vix": vix_train}, index=dates_train)

    # calib with same distribution → low threshold
    n_calib = 100
    dates_calib = pd.bdate_range("2019-01-02", periods=n_calib)
    rv20_calib = rng.normal(0.10, 0.01, n_calib)
    vix_calib = rng.normal(15.0, 1.0, n_calib)
    calib = pd.DataFrame({"rv20": rv20_calib, "vix": vix_calib}, index=dates_calib)

    bs = BaselineScorer(train, config)
    bs.fit_calibration(calib)

    # test df with extreme rv20 → must breach
    n_test = 5
    dates_test = pd.bdate_range("2021-01-04", periods=n_test)
    rv20_test = np.array([10.0, 10.0, 10.0, 10.0, 10.0])  # far above threshold
    vix_test = np.array([1.0, 1.0, 1.0, 1.0, 1.0])        # far below threshold
    test = pd.DataFrame({"rv20": rv20_test, "vix": vix_test}, index=dates_test)

    out = bs.score(test)
    assert out["rv20_breach"].all(), "All rows with extreme rv20 should breach"
    assert not out["vix_breach"].any(), "All rows with low vix should not breach"


def test_fit_calibration_returns_self(train_df, calib_df, config):
    bs = BaselineScorer(train_df, config)
    result = bs.fit_calibration(calib_df)
    assert result is bs
