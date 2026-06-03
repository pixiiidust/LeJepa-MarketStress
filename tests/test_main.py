"""Tests for run_lejepa_soxx_poc helper functions.

All tests use synthetic data — no network calls.
"""
import numpy as np
import pandas as pd
import pytest
import torch

from src.types import Config, Window
from src.model import LeJEPAModel


def _bdate_range(start, n):
    return pd.bdate_range(start=start, periods=n)


# ---------------------------------------------------------------------------
# Lazy imports — each raises ImportError until the function is implemented
# ---------------------------------------------------------------------------

def _get_split_df():
    from run_lejepa_soxx_poc import _split_df
    return _split_df


def _get_collect_calib_latents():
    from run_lejepa_soxx_poc import _collect_calib_latents
    return _collect_calib_latents


def _get_compute_forward_drawdown():
    from run_lejepa_soxx_poc import _compute_forward_drawdown
    return _compute_forward_drawdown


def _get_crash_onset_date_from_close():
    from run_lejepa_soxx_poc import _crash_onset_date_from_close
    return _crash_onset_date_from_close


def _get_build_scores_df():
    from run_lejepa_soxx_poc import _build_scores_df
    return _build_scores_df


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def tiny_model():
    torch.manual_seed(0)
    return LeJEPAModel(Config())


@pytest.fixture
def calib_windows():
    rng = np.random.default_rng(42)
    return [
        Window(
            context_array=rng.standard_normal((30, 6)).astype(np.float32),
            target_array=rng.standard_normal((10, 6)).astype(np.float32),
            date=pd.Timestamp("2019-01-02"),
        )
        for _ in range(20)
    ]


# ---------------------------------------------------------------------------
# Helpers shared by multiple test classes
# ---------------------------------------------------------------------------

def _make_lejepa_df(dates):
    n = len(dates)
    rng = np.random.default_rng(0)
    d = {
        "lejepa_score": rng.random(n),
        "lejepa_context_score": rng.random(n),
        "pred_error": rng.random(n),
        "pred_error_valid_date": [pd.NaT] * n,
        "lejepa_breach": [False] * n,
        "lejepa_episode_id": [0] * n,
    }
    for j in range(16):
        d[f"z_pred_{j}"] = rng.random(n)
        d[f"z_context_{j}"] = rng.random(n)
    return pd.DataFrame(d, index=dates)


def _make_baseline_df(dates):
    n = len(dates)
    return pd.DataFrame(
        {
            "rv20_zscore": np.zeros(n),
            "rv20_threshold": 2.0,
            "rv20_breach": [False] * n,
            "rv20_episode_id": [0] * n,
            "vix_zscore": np.zeros(n),
            "vix_threshold": 2.0,
            "vix_breach": [False] * n,
            "vix_episode_id": [0] * n,
        },
        index=dates,
    )


# ---------------------------------------------------------------------------
# _split_df
# ---------------------------------------------------------------------------

class TestSplitDf:
    def test_filters_by_date_range(self):
        _split_df = _get_split_df()
        dates = _bdate_range("2021-01-04", 60)
        df = pd.DataFrame({"x": np.arange(60)}, index=dates)
        result = _split_df(df, "2021-01-04", "2021-02-28")
        assert len(result) > 0
        assert all(result.index >= pd.Timestamp("2021-01-04"))
        assert all(result.index <= pd.Timestamp("2021-02-28"))

    def test_returns_empty_for_non_overlapping_range(self):
        _split_df = _get_split_df()
        dates = _bdate_range("2021-01-04", 60)
        df = pd.DataFrame({"x": np.arange(60)}, index=dates)
        result = _split_df(df, "2000-01-01", "2000-12-31")
        assert len(result) == 0

    def test_inclusive_endpoints(self):
        _split_df = _get_split_df()
        dates = _bdate_range("2021-01-04", 10)
        df = pd.DataFrame({"x": np.arange(10)}, index=dates)
        result = _split_df(df, str(dates[0].date()), str(dates[-1].date()))
        assert len(result) == 10


# ---------------------------------------------------------------------------
# _collect_calib_latents
# ---------------------------------------------------------------------------

class TestCollectCalibLatents:
    def test_output_shapes(self, tiny_model, calib_windows):
        _collect_calib_latents = _get_collect_calib_latents()
        z_tgt, z_pred, z_ctx = _collect_calib_latents(tiny_model, calib_windows)
        n = len(calib_windows)
        assert z_tgt.shape == (n, 16)
        assert z_pred.shape == (n, 16)
        assert z_ctx.shape == (n, 16)

    def test_model_in_eval_mode_after_call(self, tiny_model, calib_windows):
        _collect_calib_latents = _get_collect_calib_latents()
        tiny_model.train()
        _collect_calib_latents(tiny_model, calib_windows)
        assert not tiny_model.training

    def test_returns_numpy_arrays(self, tiny_model, calib_windows):
        _collect_calib_latents = _get_collect_calib_latents()
        z_tgt, z_pred, z_ctx = _collect_calib_latents(tiny_model, calib_windows)
        assert isinstance(z_tgt, np.ndarray)
        assert isinstance(z_pred, np.ndarray)
        assert isinstance(z_ctx, np.ndarray)


# ---------------------------------------------------------------------------
# _compute_forward_drawdown
# ---------------------------------------------------------------------------

class TestComputeForwardDrawdown:
    def test_nan_at_tail(self):
        _compute_forward_drawdown = _get_compute_forward_drawdown()
        dates = _bdate_range("2022-01-03", 70)
        close = pd.Series(np.ones(70) * 100.0, index=dates)
        result = _compute_forward_drawdown(close, window=60)
        # Last row must be NaN (no full 60-day window)
        assert pd.isna(result.iloc[-1])
        # First 10 rows have full windows (i+60 ≤ 70 for i ∈ 0..10)
        assert not result.iloc[:10].isna().any()

    def test_flat_series_zero_drawdown(self):
        _compute_forward_drawdown = _get_compute_forward_drawdown()
        dates = _bdate_range("2022-01-03", 100)
        close = pd.Series(np.ones(100) * 100.0, index=dates)
        result = _compute_forward_drawdown(close, window=60)
        non_nan = result.dropna()
        assert len(non_nan) > 0
        assert (non_nan.abs() < 1e-9).all()

    def test_declining_series_negative(self):
        _compute_forward_drawdown = _get_compute_forward_drawdown()
        dates = _bdate_range("2022-01-03", 100)
        close = pd.Series(np.linspace(100.0, 50.0, 100), index=dates)
        result = _compute_forward_drawdown(close, window=60)
        non_nan = result.dropna()
        assert (non_nan < 0).all()

    def test_known_value(self):
        _compute_forward_drawdown = _get_compute_forward_drawdown()
        dates = _bdate_range("2022-01-03", 65)
        vals = np.ones(65) * 100.0
        # At index 0, forward window [0:60] has a known minimum
        vals[59] = 80.0  # min of window [0:60] = 80
        close = pd.Series(vals, index=dates)
        result = _compute_forward_drawdown(close, window=60)
        expected = 80.0 / 100.0 - 1  # -0.20
        assert abs(result.iloc[0] - expected) < 1e-9

    def test_index_preserved(self):
        _compute_forward_drawdown = _get_compute_forward_drawdown()
        dates = _bdate_range("2022-01-03", 80)
        close = pd.Series(np.ones(80) * 100.0, index=dates)
        result = _compute_forward_drawdown(close, window=60)
        pd.testing.assert_index_equal(result.index, dates)


# ---------------------------------------------------------------------------
# _crash_onset_date_from_close
# ---------------------------------------------------------------------------

class TestCrashOnsetDateFromClose:
    def test_returns_none_for_flat_series(self):
        _crash_onset_date_from_close = _get_crash_onset_date_from_close()
        dates = _bdate_range("2022-01-03", 100)
        close = pd.Series(np.ones(100) * 100.0, index=dates)
        result = _crash_onset_date_from_close(close, window=60, threshold=-0.20)
        assert result is None

    def test_returns_none_when_series_shorter_than_window(self):
        _crash_onset_date_from_close = _get_crash_onset_date_from_close()
        dates = _bdate_range("2022-01-03", 5)
        close = pd.Series(np.ones(5) * 100.0, index=dates)
        result = _crash_onset_date_from_close(close, window=60, threshold=-0.20)
        assert result is None

    def test_returns_first_day_when_crash_in_first_window(self):
        _crash_onset_date_from_close = _get_crash_onset_date_from_close()
        vals = np.ones(100) * 100.0
        vals[1:] = 70.0  # 30% drop from day 1 → window [0:60] sees min=70
        dates = _bdate_range("2022-01-03", 100)
        close = pd.Series(vals, index=dates)
        result = _crash_onset_date_from_close(close, window=60, threshold=-0.20)
        assert result == dates[0]

    def test_returns_onset_at_correct_position(self):
        _crash_onset_date_from_close = _get_crash_onset_date_from_close()
        # Days 0..5: close=100; days 6+: close=70. Window [0:60] includes drop → onset at day 0.
        # But a window starting at day 6 would see close=70/70−1=0. The FIRST window to detect it
        # must be day 0 if that window includes any drop.
        # Use a crash starting right at the edge: day 60 (only day 0's window sees it).
        # days 0..59 = 100, day 60 = 70. Window [0:60] = days 0..59 → all 100 → NO crash.
        # Window [1:61] = days 1..60 → includes day 60 = 70 → crash at day 1.
        vals = np.ones(100) * 100.0
        vals[60] = 70.0
        dates = _bdate_range("2022-01-03", 100)
        close = pd.Series(vals, index=dates)
        result = _crash_onset_date_from_close(close, window=60, threshold=-0.20)
        assert result == dates[1]  # first window to see the crash starts at day 1


# ---------------------------------------------------------------------------
# _build_scores_df
# ---------------------------------------------------------------------------

class TestBuildScoresDf:
    def test_has_required_columns(self):
        _build_scores_df = _get_build_scores_df()
        dates = _bdate_range("2021-01-04", 50)
        result = _build_scores_df(
            _make_lejepa_df(dates),
            _make_baseline_df(dates),
            pd.Series(np.ones(50) * 100.0, index=dates),
            pd.Series(np.zeros(50), index=dates),
            pd.Series([False] * 50, index=dates),
        )
        for col in [
            "soxx_close", "forward_drawdown_60d", "crash_onset_flag",
            "lejepa_score", "rv20_zscore", "lejepa_breach", "rv20_breach",
        ]:
            assert col in result.columns, f"Missing: {col}"

    def test_soxx_close_values(self):
        _build_scores_df = _get_build_scores_df()
        dates = _bdate_range("2021-01-04", 30)
        soxx_close = pd.Series(np.arange(30, dtype=float), index=dates)
        result = _build_scores_df(
            _make_lejepa_df(dates),
            _make_baseline_df(dates),
            soxx_close,
            pd.Series(np.zeros(30), index=dates),
            pd.Series([False] * 30, index=dates),
        )
        np.testing.assert_array_equal(result["soxx_close"].values, np.arange(30, dtype=float))

    def test_index_preserved(self):
        _build_scores_df = _get_build_scores_df()
        dates = _bdate_range("2021-01-04", 40)
        result = _build_scores_df(
            _make_lejepa_df(dates),
            _make_baseline_df(dates),
            pd.Series(np.ones(40), index=dates),
            pd.Series(np.zeros(40), index=dates),
            pd.Series([False] * 40, index=dates),
        )
        pd.testing.assert_index_equal(result.index, dates)

    def test_crash_onset_flag_propagated(self):
        _build_scores_df = _get_build_scores_df()
        dates = _bdate_range("2021-01-04", 30)
        crash_flag = pd.Series([False] * 30, index=dates)
        crash_flag.iloc[10] = True
        result = _build_scores_df(
            _make_lejepa_df(dates),
            _make_baseline_df(dates),
            pd.Series(np.ones(30), index=dates),
            pd.Series(np.zeros(30), index=dates),
            crash_flag,
        )
        assert result["crash_onset_flag"].iloc[10] is True or result["crash_onset_flag"].iloc[10] == True
        assert int(result["crash_onset_flag"].sum()) == 1
