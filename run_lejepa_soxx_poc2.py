"""
LeJEPA SOXX Market Stress POC-2

Identical pipeline to POC-1 with one change: the calibration window excludes
days where the RV20 z-score (normalised on 2010-2016 training stats) exceeds
the 95th percentile of training-period RV20 z-scores.

NOTE: The test period (2021-2022) has been observed in POC-1. Results must be
interpreted as calibration-window sensitivity, not a fresh blind test.

Usage:
    python run_lejepa_soxx_poc2.py

Outputs (never overwrites POC-1 files):
    outputs/scores_poc2.csv
    outputs/soxx_lejepa_poc2_chart.png
    outputs/run_config_poc2.json

Caches (shared with diagnostics.py or POC-2 specific):
    outputs/diag_model.pt            — trained model weights (reused if present)
    outputs/poc2_calib_latents.npz   — all 3 latent arrays + dates
"""
from __future__ import annotations

import dataclasses
import json
import math
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch

from run_lejepa_soxx_poc import (
    _build_scores_df,
    _collect_calib_latents,
    _compute_forward_drawdown,
    _crash_onset_date_from_close,
    _fix_seeds,
    _get_git_commit,
    _split_df,
)
from src.baselines import BaselineScorer
from src.calibration import MahalanobisCalibrator, regime_filter_mask
from src.data_pipeline import DataPipeline
from src.evaluation import Evaluator
from src.model import LeJEPAModel
from src.output import _CSV_COLUMNS, _EVAL_ONLY_COMMENT, _json_default, _sanitize_floats, _verdict_summary
from src.scoring import Scorer
from src.training import Trainer
from src.types import Config

OUTPUT_DIR = Path("outputs")
MODEL_CKPT = OUTPUT_DIR / "diag_model.pt"
CALIB_CACHE = OUTPUT_DIR / "poc2_calib_latents.npz"

REGIME_PCT: int = 95


# ---------------------------------------------------------------------------
# Testable helpers
# ---------------------------------------------------------------------------

def _apply_regime_filter(
    z_target: np.ndarray,
    z_pred: np.ndarray,
    z_context: np.ndarray,
    calib_window_dates: pd.DatetimeIndex,
    calib_df: pd.DataFrame,
    rv20_series: pd.Series,
    rv20_mean: float,
    rv20_std: float,
    regime_threshold: float,
    min_rows: int = 50,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, pd.DataFrame]:
    """Apply regime filter to latent arrays and calib_df; assert ≥ min_rows remain."""
    n_before_windows = len(z_target)
    n_before_days = len(calib_df)

    window_mask = regime_filter_mask(
        calib_window_dates, rv20_series, rv20_mean, rv20_std, regime_threshold
    )
    daily_mask = regime_filter_mask(
        calib_df.index, rv20_series, rv20_mean, rv20_std, regime_threshold
    )

    n_after_windows = int(window_mask.sum())
    n_after_days = int(daily_mask.sum())

    print(f"[poc2] Regime filter: calib windows {n_before_windows} -> {n_after_windows}")
    print(f"[poc2] Regime filter: calib_df rows {n_before_days} -> {n_after_days}")

    assert n_after_windows >= min_rows, (
        f"Regime filter left {n_after_windows} calibration windows, need ≥ {min_rows}. "
        "Lower regime_pct or widen the calibration window."
    )

    return z_target[window_mask], z_pred[window_mask], z_context[window_mask], calib_df[daily_mask]


# ---------------------------------------------------------------------------
# Cache helpers
# ---------------------------------------------------------------------------

def _load_or_train_poc2(config: Config, dataset) -> LeJEPAModel:
    model = LeJEPAModel(config)
    if MODEL_CKPT.exists():
        print(f"[poc2] Loading model checkpoint: {MODEL_CKPT}")
        model.load_state_dict(torch.load(MODEL_CKPT, map_location="cpu", weights_only=True))
    else:
        print("[poc2] Training LeJEPA model (will cache at outputs/diag_model.pt)...")
        model = Trainer(config).fit(model, dataset.train, dataset.val)
        torch.save(model.state_dict(), MODEL_CKPT)
        print(f"[poc2] Checkpoint saved: {MODEL_CKPT}")
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model


def _load_or_collect_calib_poc2(
    model: LeJEPAModel, dataset
) -> tuple[np.ndarray, np.ndarray, np.ndarray, pd.DatetimeIndex]:
    """Return (z_target, z_pred, z_context, calib_dates) — all three latent arrays."""
    if CALIB_CACHE.exists():
        print(f"[poc2] Loading calib latents: {CALIB_CACHE}")
        d = np.load(CALIB_CACHE)
        dates = pd.DatetimeIndex(pd.to_datetime(d["dates_ns"], unit="ns"))
        return d["z_target"], d["z_pred"], d["z_context"], dates
    print("[poc2] Collecting calibration latents (all 3 arrays)...")
    zt, zp, zc = _collect_calib_latents(model, dataset.calib)
    dates_ns = np.array([w.date.value for w in dataset.calib], dtype=np.int64)
    np.savez(CALIB_CACHE, z_target=zt, z_pred=zp, z_context=zc, dates_ns=dates_ns)
    print(f"[poc2] Saved: {CALIB_CACHE}")
    return zt, zp, zc, pd.DatetimeIndex([w.date for w in dataset.calib])


# ---------------------------------------------------------------------------
# POC-2 output helpers
# ---------------------------------------------------------------------------

def _write_poc2_scores_csv(scores_df: pd.DataFrame, config: Config) -> None:
    df = scores_df.copy()
    df["lejepa_threshold"] = config.lejepa_threshold
    df_out = df[_CSV_COLUMNS]
    path = OUTPUT_DIR / "scores_poc2.csv"
    with open(path, "w", newline="") as f:
        f.write(_EVAL_ONLY_COMMENT)
        df_out.to_csv(f, index_label="date")
    print(f"[poc2] Written: {path}")


def _write_poc2_run_config(verdict, config: Config, regime_params: dict) -> None:
    cfg_dict = dataclasses.asdict(config)
    _sanitize_floats(cfg_dict)
    payload = {
        "config": cfg_dict,
        "regime_filter": regime_params,
        "verdict": _verdict_summary(verdict),
    }
    path = OUTPUT_DIR / "run_config_poc2.json"
    with open(path, "w") as f:
        json.dump(payload, f, indent=2, default=_json_default)
    print(f"[poc2] Written: {path}")


def _write_poc2_chart(scores_df: pd.DataFrame, verdict, config: Config) -> None:
    import matplotlib.pyplot as plt
    plt.switch_backend("Agg")

    fig, (ax_top, ax_bot) = plt.subplots(2, 1, figsize=(14, 8), sharex=True)

    ax_top.plot(
        scores_df.index, scores_df["soxx_close"],
        color="black", lw=1.2, label="SOXX Close",
    )
    if verdict.crash_onset_date is not None:
        ax_top.axvline(
            verdict.crash_onset_date, color="red", ls="--", lw=1.2, label="Crash Onset",
        )
    ax_top.set_ylabel("Close")
    ax_top.set_title("LeJEPA SOXX POC-2 - Regime-Filtered Calibration", fontsize=11)
    ax_top.legend(fontsize=8)

    ax_bot.plot(
        scores_df.index, scores_df["lejepa_score"],
        color="steelblue", lw=1.0, label="LeJEPA score (POC-2)",
    )
    if not math.isnan(config.lejepa_threshold):
        ax_bot.axhline(
            config.lejepa_threshold, color="steelblue", ls=":", lw=0.8, label="LeJEPA thr",
        )

    if "rv20_zscore" in scores_df.columns:
        ax_bot.plot(
            scores_df.index, scores_df["rv20_zscore"],
            color="orange", lw=1.0, label="RV20 z-score",
        )
        if "rv20_threshold" in scores_df.columns:
            ax_bot.axhline(
                scores_df["rv20_threshold"].iloc[0],
                color="orange", ls=":", lw=0.8, label="RV20 thr",
            )

    if "vix_zscore" in scores_df.columns:
        ax_bot.plot(
            scores_df.index, scores_df["vix_zscore"],
            color="green", lw=1.0, label="VIX z-score",
        )
        if "vix_threshold" in scores_df.columns:
            ax_bot.axhline(
                scores_df["vix_threshold"].iloc[0],
                color="green", ls=":", lw=0.8, label="VIX thr",
            )

    for breach_date, color in [
        (verdict.lejepa_first_breach, "steelblue"),
        (verdict.rv20_first_breach, "orange"),
        (verdict.vix_first_breach, "green"),
    ]:
        if breach_date is not None and breach_date in scores_df.index:
            ax_bot.axvline(breach_date, color=color, ls="-.", lw=0.8, alpha=0.8)

    ax_bot.set_ylabel("Score / Z-score")
    ax_bot.legend(fontsize=8)

    fig.tight_layout()
    path = OUTPUT_DIR / "soxx_lejepa_poc2_chart.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"[poc2] Written: {path}")


def _print_poc2_summary(verdict, regime_params: dict) -> None:
    sep = "=" * 60
    lines = [
        sep,
        "LeJEPA SOXX POC-2 - Regime-Filtered Calibration Summary",
        sep,
        "NOTE: The test period (2021-2022) has been observed in POC-1.",
        "      Results represent calibration-window sensitivity,",
        "      not a fresh blind test.",
        "",
        "Regime filter parameters:",
        f"  regime_pct       : {regime_params['regime_pct']}th percentile",
        f"  rv20_mean        : {regime_params['rv20_mean']:.4f}",
        f"  rv20_std         : {regime_params['rv20_std']:.4f}",
        f"  regime_threshold : {regime_params['regime_threshold']:.4f}",
        f"  n_calib_before   : {regime_params['n_calib_before_filter']}",
        f"  n_calib_after    : {regime_params['n_calib_after_filter']}",
        "",
        f"Crash onset date : {_fmt_date(verdict.crash_onset_date)}",
        "",
        "First breach dates and lead times:",
        f"  LeJEPA : {_fmt_date(verdict.lejepa_first_breach)}"
        f"  (lead {verdict.lejepa_lead_days} days)",
        f"  RV20   : {_fmt_date(verdict.rv20_first_breach)}"
        f"  (lead {verdict.rv20_lead_days} days)",
        f"  VIX    : {_fmt_date(verdict.vix_first_breach)}"
        f"  (lead {verdict.vix_lead_days} days)",
        "",
        "2021 episode counts and alert days:",
        f"  LeJEPA : {verdict.lejepa_episodes_2021} episodes, "
        f"{verdict.lejepa_alert_days_2021} alert days",
        f"  RV20   : {verdict.rv20_episodes_2021} episodes, "
        f"{verdict.rv20_alert_days_2021} alert days",
        f"  VIX    : {verdict.vix_episodes_2021} episodes, "
        f"{verdict.vix_alert_days_2021} alert days",
        "",
        "Pass/fail per criterion:",
        f"  Criterion 1 (breach before onset)     : {verdict.criterion_1}",
        f"  Criterion 2 (beat both baselines)     : {verdict.criterion_2}",
        f"  Criterion 3 (2021 episode efficiency) : {verdict.criterion_3}",
        f"  Criterion 4 (split boundaries)        : {verdict.criterion_4}",
        "",
    ]

    if verdict.all_pass:
        final = "PASS"
    elif verdict.partial_signal:
        final = "PARTIAL SIGNAL"
    else:
        final = "FAIL"

    lines += [f"Final verdict: {final}", sep]
    print("\n".join(lines))


def _fmt_date(ts: Optional[pd.Timestamp]) -> str:
    if ts is None or (isinstance(ts, float) and math.isnan(ts)):
        return "N/A"
    try:
        if pd.isna(ts):
            return "N/A"
    except (TypeError, ValueError):
        pass
    return str(ts.date())


# ---------------------------------------------------------------------------
# Main orchestration
# ---------------------------------------------------------------------------

def main() -> None:
    # 1. Seeds + config
    _fix_seeds(42)
    config = Config()
    config.git_commit = _get_git_commit()
    print(f"[poc2] git_commit = {config.git_commit!r}")

    # 2. Data pipeline
    print("[poc2] Building data pipeline...")
    pipeline = DataPipeline(config)
    dataset = pipeline.build()

    def _make_baseline_input(start: str, end: str) -> pd.DataFrame:
        raw = _split_df(pipeline.raw_df, start, end)
        feat = _split_df(pipeline.features_df, start, end)
        return pd.DataFrame({"rv20": feat["rv20"], "vix": raw["vix"]}).dropna()

    train_df = _make_baseline_input(config.train_start, config.train_end)
    calib_df = _make_baseline_input(config.calib_start, config.calib_end)
    test_df  = _make_baseline_input(config.test_start,  config.test_end)

    # 3. Train or load model
    model = _load_or_train_poc2(config, dataset)

    # 4. Collect or load calibration latents
    z_target_calib, z_pred_calib, z_context_calib, calib_dates = (
        _load_or_collect_calib_poc2(model, dataset)
    )

    # 5. Regime filter — computed entirely from training data before calib data is touched
    print("[poc2] Computing regime filter threshold from training data...")
    bs_train = BaselineScorer(train_df, config)
    train_rv20_z = (train_df["rv20"] - bs_train.rv20_mean) / bs_train.rv20_std
    regime_threshold = float(np.percentile(train_rv20_z, REGIME_PCT))
    rv20_all = pipeline.features_df["rv20"]

    print(
        f"[poc2] Regime threshold (p{REGIME_PCT} of training RV20 z-scores): "
        f"{regime_threshold:.4f}"
    )

    n_calib_before = len(z_target_calib)
    (
        z_target_calib, z_pred_calib, z_context_calib, calib_df
    ) = _apply_regime_filter(
        z_target_calib, z_pred_calib, z_context_calib,
        calib_dates, calib_df,
        rv20_all, bs_train.rv20_mean, bs_train.rv20_std,
        regime_threshold,
    )
    n_calib_after = len(z_target_calib)

    # 6. Calibrate on filtered data
    print("[poc2] Calibrating Mahalanobis distributions on filtered window...")
    target_cal = MahalanobisCalibrator()
    target_cal.fit(z_target_calib)
    target_cal.set_threshold(z_pred_calib, percentile=config.threshold_percentile)

    context_cal = MahalanobisCalibrator()
    context_cal.fit(z_context_calib)
    context_cal.set_threshold(z_context_calib, percentile=config.threshold_percentile)

    config.lejepa_threshold = target_cal.threshold
    print(f"[poc2] lejepa_threshold = {config.lejepa_threshold:.4f}")

    # 7. Score test period
    print("[poc2] Scoring test period with LeJEPA...")
    lejepa_df = Scorer(
        model, target_cal, context_cal, dataset.test_dates
    ).score(dataset.test)

    # 8. Baseline scoring (fit_calibration on filtered calib_df)
    print("[poc2] Scoring baselines...")
    baseline_df = (
        BaselineScorer(train_df, config)
        .fit_calibration(calib_df)
        .score(test_df)
    )

    # 9. Evaluation-only columns
    test_close = (
        _split_df(pipeline.raw_df[["close"]], config.test_start, config.test_end)["close"]
        .reindex(dataset.test_dates)
    )
    forward_dd = _compute_forward_drawdown(test_close, window=config.drawdown_window_days)
    crash_onset = _crash_onset_date_from_close(
        test_close,
        window=config.drawdown_window_days,
        threshold=config.drawdown_threshold,
    )
    crash_flag = pd.Series(False, index=dataset.test_dates)
    if crash_onset is not None and crash_onset in crash_flag.index:
        crash_flag.loc[crash_onset] = True

    scores_df = _build_scores_df(lejepa_df, baseline_df, test_close, forward_dd, crash_flag)

    # 10. Evaluate
    print("[poc2] Evaluating results...")
    verdict = Evaluator(scores_df, test_close).evaluate()

    # 11. Write POC-2 outputs
    print("[poc2] Writing outputs...")
    regime_params = {
        "regime_pct": REGIME_PCT,
        "rv20_mean": bs_train.rv20_mean,
        "rv20_std": bs_train.rv20_std,
        "regime_threshold": regime_threshold,
        "n_calib_before_filter": n_calib_before,
        "n_calib_after_filter": n_calib_after,
    }
    _write_poc2_scores_csv(scores_df, config)
    _write_poc2_run_config(verdict, config, regime_params)
    _write_poc2_chart(scores_df, verdict, config)
    _print_poc2_summary(verdict, regime_params)


if __name__ == "__main__":
    main()
