"""
Tests for OutputWriter.

Cover:
- scores.csv has all required columns in order with no extras; header comment labels eval-only columns
- run_config.json is valid JSON; all config keys present; NaN threshold serialized as null
- soxx_lejepa_chart.png is created on synthetic data without errors
- Console summary prints all first-breach dates and lead times regardless of pass/fail
- Console summary disclaimer is printed verbatim
"""
import dataclasses
import json

import numpy as np
import pandas as pd
import pytest

from src.output import OutputWriter
from src.types import Config, Verdict

# ---------------------------------------------------------------------------
# Expected CSV column order (index column "date" first, then DataFrame cols)
# ---------------------------------------------------------------------------

_CSV_COLUMNS = [
    "date",
    "soxx_close",
    "lejepa_score",
    "lejepa_threshold",
    "lejepa_breach",
    "lejepa_episode_id",
    "pred_error",
    "pred_error_valid_date",
    "lejepa_context_score",
    *[f"z_pred_{j}" for j in range(16)],
    *[f"z_context_{j}" for j in range(16)],
    "rv20_zscore",
    "rv20_threshold",
    "rv20_breach",
    "rv20_episode_id",
    "vix_zscore",
    "vix_threshold",
    "vix_breach",
    "vix_episode_id",
    "forward_drawdown_60d",
    "crash_onset_flag",
]

_DISCLAIMER = (
    "This POC tests whether LeJEPA latent drift can function as an unsupervised "
    "early warning signal for market regime stress, and whether it beats simple "
    "volatility and VIX baselines in a leakage-controlled historical backtest."
)


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------

def _make_scores_df(n: int = 30) -> pd.DataFrame:
    """Synthetic scores_df with all required input columns (no lejepa_threshold — added by OutputWriter)."""
    dates = pd.bdate_range("2021-01-04", periods=n)
    rng = np.random.default_rng(42)
    data: dict = {
        "soxx_close": rng.uniform(90, 110, n),
        "lejepa_score": rng.uniform(0, 5, n),
        "lejepa_breach": [False] * n,
        "lejepa_episode_id": [0] * n,
        "pred_error": rng.uniform(0, 1, n),
        "pred_error_valid_date": dates,
        "lejepa_context_score": rng.uniform(0, 5, n),
        **{f"z_pred_{j}": rng.standard_normal(n) for j in range(16)},
        **{f"z_context_{j}": rng.standard_normal(n) for j in range(16)},
        "rv20_zscore": rng.uniform(-2, 3, n),
        "rv20_threshold": [2.0] * n,
        "rv20_breach": [False] * n,
        "rv20_episode_id": [0] * n,
        "vix_zscore": rng.uniform(-2, 3, n),
        "vix_threshold": [2.0] * n,
        "vix_breach": [False] * n,
        "vix_episode_id": [0] * n,
        "forward_drawdown_60d": rng.uniform(-0.1, 0, n),
        "crash_onset_flag": [False] * n,
    }
    return pd.DataFrame(data, index=dates)


def _make_verdict(all_pass: bool = True, partial_signal: bool = False) -> Verdict:
    dates = pd.bdate_range("2021-01-04", periods=30)
    return Verdict(
        crash_onset_date=dates[10],
        lejepa_first_breach=dates[5],
        rv20_first_breach=dates[7],
        vix_first_breach=dates[8],
        lejepa_lead_days=5,
        rv20_lead_days=3,
        vix_lead_days=2,
        lejepa_episodes_2021=1,
        rv20_episodes_2021=2,
        vix_episodes_2021=2,
        lejepa_alert_days_2021=3,
        rv20_alert_days_2021=5,
        vix_alert_days_2021=4,
        criterion_1=True,
        criterion_2=True,
        criterion_3=True,
        criterion_4=True,
        partial_signal=partial_signal,
        all_pass=all_pass,
    )


def _make_config(threshold: float = 3.5) -> Config:
    cfg = Config()
    cfg.lejepa_threshold = threshold
    return cfg


# ---------------------------------------------------------------------------
# scores.csv
# ---------------------------------------------------------------------------

def test_scores_csv_columns_in_exact_order(tmp_path):
    OutputWriter(tmp_path).write(_make_scores_df(), _make_verdict(), _make_config())
    df = pd.read_csv(tmp_path / "scores.csv", comment="#")
    assert list(df.columns) == _CSV_COLUMNS


def test_scores_csv_no_extra_columns(tmp_path):
    OutputWriter(tmp_path).write(_make_scores_df(), _make_verdict(), _make_config())
    df = pd.read_csv(tmp_path / "scores.csv", comment="#")
    assert len(df.columns) == len(_CSV_COLUMNS)


def test_scores_csv_eval_only_header_comment(tmp_path):
    OutputWriter(tmp_path).write(_make_scores_df(), _make_verdict(), _make_config())
    with open(tmp_path / "scores.csv") as f:
        first_line = f.readline()
    assert first_line.startswith("#")
    assert "forward_drawdown_60d" in first_line
    assert "crash_onset_flag" in first_line


def test_scores_csv_lejepa_threshold_from_config(tmp_path):
    cfg = _make_config(threshold=4.2)
    OutputWriter(tmp_path).write(_make_scores_df(), _make_verdict(), cfg)
    df = pd.read_csv(tmp_path / "scores.csv", comment="#")
    assert (df["lejepa_threshold"] == 4.2).all()


# ---------------------------------------------------------------------------
# run_config.json
# ---------------------------------------------------------------------------

def test_run_config_json_is_valid_json(tmp_path):
    OutputWriter(tmp_path).write(_make_scores_df(), _make_verdict(), _make_config())
    with open(tmp_path / "run_config.json") as f:
        payload = json.load(f)
    assert "config" in payload
    assert "verdict" in payload


def test_run_config_json_all_config_keys_present(tmp_path):
    cfg = _make_config()
    OutputWriter(tmp_path).write(_make_scores_df(), _make_verdict(), cfg)
    with open(tmp_path / "run_config.json") as f:
        payload = json.load(f)
    for key in dataclasses.asdict(cfg):
        assert key in payload["config"]


def test_run_config_json_contains_verdict_summary_keys(tmp_path):
    OutputWriter(tmp_path).write(_make_scores_df(), _make_verdict(), _make_config())
    with open(tmp_path / "run_config.json") as f:
        payload = json.load(f)
    for key in [
        "crash_onset_date", "lejepa_first_breach", "rv20_first_breach", "vix_first_breach",
        "lejepa_lead_days", "rv20_lead_days", "vix_lead_days",
        "lejepa_episodes_2021", "rv20_episodes_2021", "vix_episodes_2021",
        "criterion_1", "criterion_2", "criterion_3", "criterion_4",
        "all_pass",
    ]:
        assert key in payload["verdict"]


def test_run_config_json_nan_threshold_serialized_as_null(tmp_path):
    cfg = Config()  # lejepa_threshold = float("nan") by default
    OutputWriter(tmp_path).write(_make_scores_df(), _make_verdict(), cfg)
    with open(tmp_path / "run_config.json") as f:
        payload = json.load(f)
    assert payload["config"]["lejepa_threshold"] is None


# ---------------------------------------------------------------------------
# Chart
# ---------------------------------------------------------------------------

def test_chart_file_created(tmp_path):
    OutputWriter(tmp_path).write(_make_scores_df(), _make_verdict(), _make_config())
    assert (tmp_path / "soxx_lejepa_chart.png").exists()


def test_chart_created_with_none_crash_onset(tmp_path):
    verdict = Verdict(
        crash_onset_date=None,
        lejepa_first_breach=None, rv20_first_breach=None, vix_first_breach=None,
        lejepa_lead_days=None, rv20_lead_days=None, vix_lead_days=None,
        lejepa_episodes_2021=0, rv20_episodes_2021=0, vix_episodes_2021=0,
        lejepa_alert_days_2021=0, rv20_alert_days_2021=0, vix_alert_days_2021=0,
        criterion_1=False, criterion_2=False, criterion_3=True, criterion_4=True,
        partial_signal=False, all_pass=False,
    )
    OutputWriter(tmp_path).write(_make_scores_df(), verdict, _make_config())
    assert (tmp_path / "soxx_lejepa_chart.png").exists()


# ---------------------------------------------------------------------------
# Console summary
# ---------------------------------------------------------------------------

def test_console_summary_prints_all_breach_dates(tmp_path, capsys):
    verdict = _make_verdict()
    OutputWriter(tmp_path).write(_make_scores_df(), verdict, _make_config())
    out = capsys.readouterr().out
    assert str(verdict.lejepa_first_breach.date()) in out
    assert str(verdict.rv20_first_breach.date()) in out
    assert str(verdict.vix_first_breach.date()) in out


def test_console_summary_prints_na_when_no_breach(tmp_path, capsys):
    verdict = Verdict(
        crash_onset_date=None,
        lejepa_first_breach=None, rv20_first_breach=None, vix_first_breach=None,
        lejepa_lead_days=None, rv20_lead_days=None, vix_lead_days=None,
        lejepa_episodes_2021=0, rv20_episodes_2021=0, vix_episodes_2021=0,
        lejepa_alert_days_2021=0, rv20_alert_days_2021=0, vix_alert_days_2021=0,
        criterion_1=False, criterion_2=False, criterion_3=True, criterion_4=True,
        partial_signal=False, all_pass=False,
    )
    OutputWriter(tmp_path).write(_make_scores_df(), verdict, _make_config())
    out = capsys.readouterr().out
    assert "N/A" in out


def test_console_summary_prints_lead_times(tmp_path, capsys):
    verdict = _make_verdict()
    OutputWriter(tmp_path).write(_make_scores_df(), verdict, _make_config())
    out = capsys.readouterr().out
    assert str(verdict.lejepa_lead_days) in out
    assert str(verdict.rv20_lead_days) in out
    assert str(verdict.vix_lead_days) in out


def test_console_summary_prints_2021_episode_counts(tmp_path, capsys):
    verdict = _make_verdict()
    OutputWriter(tmp_path).write(_make_scores_df(), verdict, _make_config())
    out = capsys.readouterr().out
    assert "1 episodes" in out or "1 episode" in out


def test_console_summary_disclaimer_verbatim(tmp_path, capsys):
    OutputWriter(tmp_path).write(_make_scores_df(), _make_verdict(), _make_config())
    assert _DISCLAIMER in capsys.readouterr().out


def test_console_summary_final_verdict_pass(tmp_path, capsys):
    OutputWriter(tmp_path).write(_make_scores_df(), _make_verdict(all_pass=True), _make_config())
    out = capsys.readouterr().out
    assert "Final verdict: PASS" in out


def test_console_summary_final_verdict_fail(tmp_path, capsys):
    verdict = Verdict(
        crash_onset_date=pd.bdate_range("2021-01-04", periods=30)[10],
        lejepa_first_breach=pd.bdate_range("2021-01-04", periods=30)[8],
        rv20_first_breach=pd.bdate_range("2021-01-04", periods=30)[3],
        vix_first_breach=pd.bdate_range("2021-01-04", periods=30)[4],
        lejepa_lead_days=2,
        rv20_lead_days=7,
        vix_lead_days=6,
        lejepa_episodes_2021=3,
        rv20_episodes_2021=1,
        vix_episodes_2021=1,
        lejepa_alert_days_2021=10,
        rv20_alert_days_2021=5,
        vix_alert_days_2021=4,
        criterion_1=True,
        criterion_2=False,
        criterion_3=False,
        criterion_4=True,
        partial_signal=False,
        all_pass=False,
    )
    OutputWriter(tmp_path).write(_make_scores_df(), verdict, _make_config())
    out = capsys.readouterr().out
    assert "Final verdict: FAIL" in out


def test_console_summary_final_verdict_partial_signal(tmp_path, capsys):
    verdict = Verdict(
        crash_onset_date=pd.bdate_range("2021-01-04", periods=30)[10],
        lejepa_first_breach=pd.bdate_range("2021-01-04", periods=30)[5],
        rv20_first_breach=pd.bdate_range("2021-01-04", periods=30)[7],
        vix_first_breach=pd.bdate_range("2021-01-04", periods=30)[3],
        lejepa_lead_days=5,
        rv20_lead_days=3,
        vix_lead_days=7,
        lejepa_episodes_2021=1,
        rv20_episodes_2021=2,
        vix_episodes_2021=2,
        lejepa_alert_days_2021=3,
        rv20_alert_days_2021=5,
        vix_alert_days_2021=4,
        criterion_1=True,
        criterion_2=False,
        criterion_3=True,
        criterion_4=True,
        partial_signal=True,
        all_pass=False,
    )
    OutputWriter(tmp_path).write(_make_scores_df(), verdict, _make_config())
    out = capsys.readouterr().out
    assert "Final verdict: PARTIAL SIGNAL" in out
