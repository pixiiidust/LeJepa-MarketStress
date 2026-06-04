"""
Tests for run_poc3_seed_sweep helper functions and runner.

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
# Lazy imports (fail with ImportError until module exists — RED phase)
# ---------------------------------------------------------------------------

def _get(name: str):
    import run_poc3_seed_sweep as m
    return getattr(m, name)


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _make_verdict(all_pass: bool = False, lead_days: int | None = None) -> Verdict:
    return Verdict(
        crash_onset_date=pd.Timestamp("2021-12-10"),
        lejepa_first_breach=(
            pd.Timestamp("2021-01-04") if lead_days is not None else None
        ),
        rv20_first_breach=None,
        vix_first_breach=None,
        lejepa_lead_days=lead_days,
        rv20_lead_days=None,
        vix_lead_days=None,
        lejepa_episodes_2021=0,
        rv20_episodes_2021=0,
        vix_episodes_2021=0,
        lejepa_alert_days_2021=0,
        rv20_alert_days_2021=0,
        vix_alert_days_2021=0,
        criterion_1=all_pass,
        criterion_2=all_pass,
        criterion_3=all_pass,
        criterion_4=True,
        partial_signal=False,
        all_pass=all_pass,
    )


def _make_minimal_dataset() -> WindowDataset:
    rng = np.random.default_rng(0)
    n_features = 6

    def make_windows(n: int) -> list[Window]:
        return [
            Window(
                context_array=rng.standard_normal((30, n_features)).astype(np.float32),
                target_array=rng.standard_normal((10, n_features)).astype(np.float32),
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


def _calibrator_factory(start_threshold: float = 3.0, step: float = 0.5):
    """Returns a fresh callable each call; each invocation gets a distinct threshold."""
    counter = [0]

    def factory():
        mock_cal = MagicMock()
        mock_cal.threshold = start_threshold + counter[0] * step
        counter[0] += 1
        return mock_cal

    return factory


def _patch_sweep(monkeypatch, dataset: WindowDataset, collapse: bool = False):
    """Monkeypatch all heavy dependencies in run_poc3_seed_sweep. Returns module."""
    import run_poc3_seed_sweep as m

    n = 5_000
    all_dates = pd.bdate_range("2010-01-04", periods=n)
    raw_df = pd.DataFrame(
        {"close": np.ones(n) * 100.0, "vix": np.ones(n)},
        index=all_dates,
    )
    features_df = pd.DataFrame({"rv20": np.ones(n)}, index=all_dates)

    mock_pipeline = MagicMock()
    mock_pipeline.build.return_value = dataset
    mock_pipeline.raw_df = raw_df
    mock_pipeline.features_df = features_df
    monkeypatch.setattr(m, "DataPipeline", lambda config: mock_pipeline)

    mock_model = MagicMock()
    mock_model.parameters.return_value = []
    monkeypatch.setattr(m, "LeJEPAModel", lambda config: mock_model)

    mock_trainer_inst = MagicMock()
    mock_trainer_inst.fit.return_value = mock_model
    monkeypatch.setattr(m, "Trainer", lambda config: mock_trainer_inst)

    if collapse:
        latents = np.full((20, 16), 0.0001)
    else:
        rng = np.random.default_rng(99)
        latents = rng.standard_normal((20, 16)).astype(np.float64)

    monkeypatch.setattr(
        m, "_collect_calib_latents",
        lambda model, windows: (latents, latents, latents),
    )

    monkeypatch.setattr(m, "MahalanobisCalibrator", _calibrator_factory())

    def make_scorer(*args, **kwargs):
        s = MagicMock()
        s.score.return_value = _make_mock_lejepa_df(dataset.test_dates)
        return s

    monkeypatch.setattr(m, "Scorer", make_scorer)

    def make_baseline(*args, **kwargs):
        b = MagicMock()
        b.fit_calibration.return_value = b
        b.score.return_value = _make_mock_baseline_df(dataset.test_dates)
        return b

    monkeypatch.setattr(m, "BaselineScorer", make_baseline)

    def make_evaluator(*args, **kwargs):
        e = MagicMock()
        e.evaluate.return_value = _make_verdict()
        return e

    monkeypatch.setattr(m, "Evaluator", make_evaluator)

    return m


# ---------------------------------------------------------------------------
# Cycle 1 — _compute_latent_std_min
# ---------------------------------------------------------------------------

class TestComputeLatentStdMin:
    def test_returns_minimum_std_across_dims(self):
        fn = _get("_compute_latent_std_min")
        rng = np.random.default_rng(0)
        z = rng.standard_normal((100, 16))
        z[:, 3] = 0.001  # near-constant dim
        assert fn(z) < 0.01

    def test_returns_float(self):
        fn = _get("_compute_latent_std_min")
        z = np.random.default_rng(1).standard_normal((50, 8))
        assert isinstance(fn(z), float)

    def test_uniform_columns_give_expected_range(self):
        fn = _get("_compute_latent_std_min")
        rng = np.random.default_rng(2)
        z = rng.standard_normal((200, 4))
        result = fn(z)
        assert 0.5 < result < 2.0


# ---------------------------------------------------------------------------
# Cycle 2 — _build_seed_row
# ---------------------------------------------------------------------------

class TestBuildSeedRow:
    def test_required_keys_present(self):
        fn = _get("_build_seed_row")
        config = Config(seed=3, lejepa_threshold=4.5)
        verdict = _make_verdict()
        row = fn(seed=3, config=config, verdict=verdict, latent_std_min=0.3)
        required = {
            "seed", "lejepa_threshold", "lejepa_first_breach",
            "crash_onset_date", "lejepa_lead_days", "all_pass",
            "criterion_1", "criterion_2", "criterion_3", "criterion_4",
        }
        assert required.issubset(set(row.keys()))

    def test_seed_value_matches(self):
        fn = _get("_build_seed_row")
        row = fn(seed=7, config=Config(seed=7, lejepa_threshold=3.2),
                 verdict=_make_verdict(), latent_std_min=0.5)
        assert row["seed"] == 7

    def test_threshold_comes_from_config(self):
        fn = _get("_build_seed_row")
        config = Config(seed=0, lejepa_threshold=6.33)
        row = fn(seed=0, config=config, verdict=_make_verdict(), latent_std_min=0.5)
        assert row["lejepa_threshold"] == pytest.approx(6.33)

    def test_verdict_fields_propagated(self):
        fn = _get("_build_seed_row")
        config = Config(seed=1, lejepa_threshold=2.5)
        verdict = _make_verdict(all_pass=True, lead_days=5)
        row = fn(seed=1, config=config, verdict=verdict, latent_std_min=0.4)
        assert row["all_pass"] is True
        assert row["lejepa_lead_days"] == 5


# ---------------------------------------------------------------------------
# Cycle 3 — _print_console_summary
# ---------------------------------------------------------------------------

class TestPrintConsoleSummary:
    def _rows(self, n_pass: int = 3, lead_days: list | None = None) -> list[dict]:
        if lead_days is None:
            lead_days = [10, 20, 30]
        rows = []
        for i in range(10):
            rows.append({
                "seed": i,
                "lejepa_threshold": 3.0 + i * 0.1,
                "lejepa_lead_days": lead_days[i] if i < len(lead_days) else None,
                "all_pass": i < n_pass,
            })
        return rows

    def test_prints_pass_fraction(self, capsys):
        fn = _get("_print_console_summary")
        fn(self._rows(n_pass=3))
        out = capsys.readouterr().out
        assert "3" in out and "10" in out

    def test_prints_mean_lead_days(self, capsys):
        fn = _get("_print_console_summary")
        fn(self._rows(n_pass=3, lead_days=[10, 20, 30]))
        out = capsys.readouterr().out
        assert "20.0" in out  # mean of [10, 20, 30]

    def test_handles_no_firing_seeds(self, capsys):
        fn = _get("_print_console_summary")
        rows = [{"seed": i, "lejepa_lead_days": None, "all_pass": False}
                for i in range(10)]
        fn(rows)  # must not raise
        out = capsys.readouterr().out
        assert "0" in out


# ---------------------------------------------------------------------------
# Cycle 4 — _append_poc3_summary
# ---------------------------------------------------------------------------

class TestAppendPoc3Summary:
    def _rows(self, n_pass: int = 2, lead_days: list | None = None) -> list[dict]:
        if lead_days is None:
            lead_days = [5, 8]
        rows = []
        for i in range(10):
            rows.append({
                "seed": i,
                "lejepa_lead_days": lead_days[i] if i < len(lead_days) else None,
                "all_pass": i < n_pass,
            })
        return rows

    def test_creates_file_and_contains_seed_sweep(self, tmp_path):
        fn = _get("_append_poc3_summary")
        fn(tmp_path, self._rows())
        content = (tmp_path / "poc3_summary.md").read_text()
        assert "seed_sweep" in content

    def test_pass_count_appears_in_file(self, tmp_path):
        fn = _get("_append_poc3_summary")
        fn(tmp_path, self._rows(n_pass=2))
        content = (tmp_path / "poc3_summary.md").read_text()
        assert "2/10" in content

    def test_appends_not_overwrites(self, tmp_path):
        fn = _get("_append_poc3_summary")
        (tmp_path / "poc3_summary.md").write_text("previous line\n")
        fn(tmp_path, self._rows())
        content = (tmp_path / "poc3_summary.md").read_text()
        assert "previous line" in content
        assert "seed_sweep" in content


# ---------------------------------------------------------------------------
# Cycle 5 — run_seed_sweep integration (all heavy deps mocked)
# ---------------------------------------------------------------------------

class TestRunSeedSweep:
    def test_returns_10_rows(self, monkeypatch, tmp_path):
        dataset = _make_minimal_dataset()
        m = _patch_sweep(monkeypatch, dataset)
        df = m.run_seed_sweep(outdir=tmp_path)
        assert len(df) == 10

    def test_seed_column_is_0_through_9(self, monkeypatch, tmp_path):
        dataset = _make_minimal_dataset()
        m = _patch_sweep(monkeypatch, dataset)
        df = m.run_seed_sweep(outdir=tmp_path)
        assert list(df["seed"]) == list(range(10))

    def test_required_columns_present(self, monkeypatch, tmp_path):
        dataset = _make_minimal_dataset()
        m = _patch_sweep(monkeypatch, dataset)
        df = m.run_seed_sweep(outdir=tmp_path)
        required = {
            "seed", "lejepa_threshold", "lejepa_first_breach",
            "crash_onset_date", "lejepa_lead_days", "all_pass",
            "criterion_1", "criterion_2", "criterion_3", "criterion_4",
        }
        assert required.issubset(set(df.columns))

    def test_thresholds_differ_across_seeds(self, monkeypatch, tmp_path):
        dataset = _make_minimal_dataset()
        m = _patch_sweep(monkeypatch, dataset)
        df = m.run_seed_sweep(outdir=tmp_path)
        assert df["lejepa_threshold"].nunique() > 1

    def test_csv_written_inside_poc3_dir(self, monkeypatch, tmp_path):
        dataset = _make_minimal_dataset()
        m = _patch_sweep(monkeypatch, dataset)
        m.run_seed_sweep(outdir=tmp_path)
        assert (tmp_path / "seed_sweep_results.csv").exists()

    def test_poc3_summary_md_written(self, monkeypatch, tmp_path):
        dataset = _make_minimal_dataset()
        m = _patch_sweep(monkeypatch, dataset)
        m.run_seed_sweep(outdir=tmp_path)
        assert (tmp_path / "poc3_summary.md").exists()
        assert "seed_sweep" in (tmp_path / "poc3_summary.md").read_text()

    def test_collapse_warning_printed(self, monkeypatch, tmp_path, capsys):
        dataset = _make_minimal_dataset()
        _patch_sweep(monkeypatch, dataset, collapse=True)
        import run_poc3_seed_sweep as m
        m.run_seed_sweep(outdir=tmp_path)
        out = capsys.readouterr().out
        assert "WARNING" in out.upper() or "collapse" in out.lower()
