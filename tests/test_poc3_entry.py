"""
Tests for run_poc3_baselines helper functions and runner.

All tests use synthetic data — no network calls, no trained model weights.
"""
from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest

from src.types import Config, Verdict, Window, WindowDataset


# ---------------------------------------------------------------------------
# Lazy import (fails with ImportError until module exists)
# ---------------------------------------------------------------------------

def _get(name: str):
    import run_poc3_baselines as m
    return getattr(m, name)


# ---------------------------------------------------------------------------
# Shared test fixtures
# ---------------------------------------------------------------------------

def _make_verdict(
    all_pass: bool = False,
    lead_days: int | None = None,
    first_breach: pd.Timestamp | None = None,
    episodes_2021: int = 0,
    alert_days_2021: int = 0,
) -> Verdict:
    fb = first_breach if first_breach is not None else (
        pd.Timestamp("2021-01-04") if lead_days is not None else None
    )
    return Verdict(
        crash_onset_date=pd.Timestamp("2021-12-10"),
        lejepa_first_breach=fb,
        rv20_first_breach=None,
        vix_first_breach=None,
        lejepa_lead_days=lead_days,
        rv20_lead_days=None,
        vix_lead_days=None,
        lejepa_episodes_2021=episodes_2021,
        rv20_episodes_2021=0,
        vix_episodes_2021=0,
        lejepa_alert_days_2021=alert_days_2021,
        rv20_alert_days_2021=0,
        vix_alert_days_2021=0,
        criterion_1=all_pass,
        criterion_2=all_pass,
        criterion_3=all_pass,
        criterion_4=True,
        partial_signal=False,
        all_pass=all_pass,
    )


# ---------------------------------------------------------------------------
# Cycle 1 — _build_signal_row
# ---------------------------------------------------------------------------

class TestBuildSignalRow:
    def test_required_keys_present(self):
        fn = _get("_build_signal_row")
        row = fn("lejepa", _make_verdict())
        required = {"signal_name", "first_breach", "lead_days", "episodes_2021", "alert_days_2021", "all_pass"}
        assert required.issubset(set(row.keys()))

    def test_signal_name_stored(self):
        fn = _get("_build_signal_row")
        assert _get("_build_signal_row")("pca", _make_verdict())["signal_name"] == "pca"

    def test_lead_days_propagated(self):
        fn = _get("_build_signal_row")
        row = fn("if", _make_verdict(lead_days=15))
        assert row["lead_days"] == 15

    def test_all_pass_propagated(self):
        fn = _get("_build_signal_row")
        assert _get("_build_signal_row")("rv20", _make_verdict(all_pass=True))["all_pass"] is True

    def test_first_breach_propagated(self):
        fn = _get("_build_signal_row")
        ts = pd.Timestamp("2021-03-15")
        row = fn("vix", _make_verdict(first_breach=ts, lead_days=10))
        assert row["first_breach"] == ts

    def test_episodes_and_alert_days_propagated(self):
        fn = _get("_build_signal_row")
        row = fn("lejepa", _make_verdict(episodes_2021=3, alert_days_2021=12))
        assert row["episodes_2021"] == 3
        assert row["alert_days_2021"] == 12


# ---------------------------------------------------------------------------
# Cycle 2 — _print_console_summary
# ---------------------------------------------------------------------------

def _five_signal_rows(
    first_breaches: list | None = None,
    lead_days: list | None = None,
    all_pass_flags: list | None = None,
) -> list[dict]:
    signals = ["lejepa", "rv20", "vix", "pca", "if"]
    fbs = first_breaches or [pd.Timestamp("2021-01-04")] * 3 + [None, None]
    lds = lead_days or [30, 20, 10, None, None]
    aps = all_pass_flags or [True, False, False, False, False]
    return [
        {"signal_name": s, "first_breach": fbs[i], "lead_days": lds[i],
         "episodes_2021": i, "alert_days_2021": i * 2, "all_pass": aps[i]}
        for i, s in enumerate(signals)
    ]


class TestPrintConsoleSummary:
    def test_prints_all_five_signal_names(self, capsys):
        _get("_print_console_summary")(_five_signal_rows())
        out = capsys.readouterr().out
        for name in ["lejepa", "rv20", "vix", "pca", "if"]:
            assert name in out

    def test_prints_lead_days(self, capsys):
        _get("_print_console_summary")(_five_signal_rows(lead_days=[42, 20, 10, None, None]))
        out = capsys.readouterr().out
        assert "42" in out

    def test_handles_none_first_breach_without_error(self, capsys):
        rows = _five_signal_rows()  # pca and if have None first_breach
        _get("_print_console_summary")(rows)
        out = capsys.readouterr().out
        assert out  # something printed

    def test_prints_pass_or_fail(self, capsys):
        _get("_print_console_summary")(_five_signal_rows())
        out = capsys.readouterr().out
        assert "PASS" in out or "FAIL" in out


# ---------------------------------------------------------------------------
# Cycle 3 — _append_poc3_summary
# ---------------------------------------------------------------------------

class TestAppendPoc3Summary:
    def test_creates_file_containing_baselines(self, tmp_path):
        _get("_append_poc3_summary")(tmp_path, _five_signal_rows())
        content = (tmp_path / "poc3_summary.md").read_text()
        assert "baselines" in content

    def test_appends_not_overwrites(self, tmp_path):
        (tmp_path / "poc3_summary.md").write_text("previous line\n")
        _get("_append_poc3_summary")(tmp_path, _five_signal_rows())
        content = (tmp_path / "poc3_summary.md").read_text()
        assert "previous line" in content
        assert "baselines" in content

    def test_pass_count_in_file(self, tmp_path):
        _get("_append_poc3_summary")(tmp_path, _five_signal_rows())
        content = (tmp_path / "poc3_summary.md").read_text()
        assert "1/5" in content  # only lejepa passes in _five_signal_rows()


# ---------------------------------------------------------------------------
# Shared mock factories for integration tests
# ---------------------------------------------------------------------------

def _make_minimal_dataset() -> WindowDataset:
    rng = np.random.default_rng(0)

    def make_windows(n: int) -> list[Window]:
        return [
            Window(
                context_array=rng.standard_normal((30, 6)).astype(np.float32),
                target_array=rng.standard_normal((10, 6)).astype(np.float32),
                date=pd.Timestamp("2019-01-02"),
            )
            for _ in range(n)
        ]

    return WindowDataset(
        train=make_windows(20),
        val=make_windows(10),
        calib=make_windows(20),
        test=make_windows(20),
        train_dates=pd.bdate_range("2010-01-04", periods=20),
        val_dates=pd.bdate_range("2017-01-03", periods=10),
        calib_dates=pd.bdate_range("2019-01-02", periods=20),
        test_dates=pd.bdate_range("2021-01-04", periods=20),
    )


def _make_mock_lejepa_df(dates: pd.DatetimeIndex) -> pd.DataFrame:
    n = len(dates)
    return pd.DataFrame(
        {
            "lejepa_score": np.ones(n) * 0.5,
            "lejepa_context_score": np.ones(n) * 0.5,
            "pred_error": np.ones(n) * 0.1,
            "pred_error_valid_date": [pd.NaT] * n,
            "lejepa_breach": [False] * n,
            "lejepa_episode_id": [0] * n,
        },
        index=dates,
    )


def _make_mock_baseline_df(dates: pd.DatetimeIndex) -> pd.DataFrame:
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


def _make_mock_pca_df(dates: pd.DatetimeIndex) -> pd.DataFrame:
    n = len(dates)
    return pd.DataFrame(
        {
            "pca_score": np.ones(n) * 0.3,
            "pca_threshold": 1.5,
            "pca_breach": [False] * n,
            "pca_episode_id": [0] * n,
        },
        index=dates,
    )


def _make_mock_if_df(dates: pd.DatetimeIndex) -> pd.DataFrame:
    n = len(dates)
    return pd.DataFrame(
        {
            "if_score": np.ones(n) * 0.2,
            "if_threshold": 0.5,
            "if_breach": [False] * n,
            "if_episode_id": [0] * n,
        },
        index=dates,
    )


def _patch_baselines(monkeypatch, dataset: WindowDataset):
    """Monkeypatch all heavy deps in run_poc3_baselines. Returns module."""
    import run_poc3_baselines as m

    n = 5_000
    all_dates = pd.bdate_range("2010-01-04", periods=n)
    raw_df = pd.DataFrame({"close": np.ones(n) * 100.0, "vix": np.ones(n)}, index=all_dates)
    features_df = pd.DataFrame({"rv20": np.ones(n)}, index=all_dates)

    mock_pipeline = MagicMock()
    mock_pipeline.build.return_value = dataset
    mock_pipeline.raw_df = raw_df
    mock_pipeline.features_df = features_df
    monkeypatch.setattr(m, "DataPipeline", lambda config: mock_pipeline)

    mock_model = MagicMock()
    mock_model.parameters.return_value = []
    monkeypatch.setattr(m, "LeJEPAModel", lambda config: mock_model)

    mock_trainer = MagicMock()
    mock_trainer.fit.return_value = mock_model
    monkeypatch.setattr(m, "Trainer", lambda config: mock_trainer)

    rng = np.random.default_rng(99)
    latents = rng.standard_normal((20, 16)).astype(np.float64)
    monkeypatch.setattr(m, "_collect_calib_latents", lambda model, windows: (latents, latents, latents))

    mock_cal = MagicMock()
    mock_cal.threshold = 3.0
    monkeypatch.setattr(m, "MahalanobisCalibrator", lambda: mock_cal)

    def make_scorer(*a, **kw):
        s = MagicMock()
        s.score.return_value = _make_mock_lejepa_df(dataset.test_dates)
        return s

    monkeypatch.setattr(m, "Scorer", make_scorer)

    def make_baseline(*a, **kw):
        b = MagicMock()
        b.fit_calibration.return_value = b
        b.score.return_value = _make_mock_baseline_df(dataset.test_dates)
        return b

    monkeypatch.setattr(m, "BaselineScorer", make_baseline)

    def make_pca(*a, **kw):
        p = MagicMock()
        p.fit_calibration.return_value = p
        p.score.return_value = _make_mock_pca_df(dataset.test_dates)
        return p

    monkeypatch.setattr(m, "PCAScorer", make_pca)

    def make_if(*a, **kw):
        i = MagicMock()
        i.fit_calibration.return_value = i
        i.score.return_value = _make_mock_if_df(dataset.test_dates)
        return i

    monkeypatch.setattr(m, "IsolationForestScorer", make_if)

    def make_evaluator(*a, **kw):
        e = MagicMock()
        e.evaluate.return_value = _make_verdict()
        return e

    monkeypatch.setattr(m, "Evaluator", make_evaluator)

    return m


# ---------------------------------------------------------------------------
# Cycle 4 — run_baselines integration
# ---------------------------------------------------------------------------

class TestRunBaselines:
    def test_returns_5_rows(self, monkeypatch, tmp_path):
        m = _patch_baselines(monkeypatch, _make_minimal_dataset())
        df = m.run_baselines(outdir=tmp_path)
        assert len(df) == 5

    def test_signal_names_are_all_five(self, monkeypatch, tmp_path):
        m = _patch_baselines(monkeypatch, _make_minimal_dataset())
        df = m.run_baselines(outdir=tmp_path)
        assert set(df["signal_name"]) == {"lejepa", "rv20", "vix", "pca", "if"}

    def test_required_columns_present(self, monkeypatch, tmp_path):
        m = _patch_baselines(monkeypatch, _make_minimal_dataset())
        df = m.run_baselines(outdir=tmp_path)
        required = {"signal_name", "first_breach", "lead_days", "episodes_2021", "alert_days_2021", "all_pass"}
        assert required.issubset(set(df.columns))

    def test_csv_written_with_correct_columns(self, monkeypatch, tmp_path):
        m = _patch_baselines(monkeypatch, _make_minimal_dataset())
        m.run_baselines(outdir=tmp_path)
        csv = pd.read_csv(tmp_path / "baselines_results.csv")
        required = {"signal_name", "first_breach", "lead_days", "episodes_2021", "alert_days_2021", "all_pass"}
        assert required.issubset(set(csv.columns))
        assert len(csv) == 5

    def test_chart_written(self, monkeypatch, tmp_path):
        m = _patch_baselines(monkeypatch, _make_minimal_dataset())
        m.run_baselines(outdir=tmp_path)
        assert (tmp_path / "baselines_chart.png").exists()

    def test_poc3_summary_contains_baselines(self, monkeypatch, tmp_path):
        m = _patch_baselines(monkeypatch, _make_minimal_dataset())
        m.run_baselines(outdir=tmp_path)
        content = (tmp_path / "poc3_summary.md").read_text()
        assert "baselines" in content

    def test_evaluator_called_five_times(self, monkeypatch, tmp_path):
        import run_poc3_baselines as m_mod
        call_count = [0]
        original_patch = _patch_baselines(monkeypatch, _make_minimal_dataset())

        def counting_evaluator(*a, **kw):
            call_count[0] += 1
            e = MagicMock()
            e.evaluate.return_value = _make_verdict()
            return e

        monkeypatch.setattr(m_mod, "Evaluator", counting_evaluator)
        m_mod.run_baselines(outdir=tmp_path)
        assert call_count[0] == 5
