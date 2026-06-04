"""
LeJEPA SOXX Market Stress POC-3 — Extended Baselines

Runs all five anomaly signals (LeJEPA, RV20, VIX, PCA, Isolation Forest)
against the 2021-2022 test period using a single model (seed=42, latent_dim=16).
Each signal is evaluated independently against the crash oracle using the same
four pass/fail criteria as POC-1 (signal treated as the "LeJEPA" signal,
compared against RV20 and VIX as reference baselines).

Outputs written to outputs/poc3/ — no other files are written.

Usage:
    python run_poc3_baselines.py

Outputs:
    outputs/poc3/baselines_results.csv
    outputs/poc3/baselines_chart.png
    outputs/poc3/poc3_summary.md
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.baselines import BaselineScorer
from src.calibration import MahalanobisCalibrator
from src.data_pipeline import DataPipeline
from src.evaluation import Evaluator
from src.model import LeJEPAModel
from src.poc3_baselines import IsolationForestScorer, PCAScorer
from src.scoring import Scorer
from src.training import Trainer
from src.types import Config, Verdict

from run_lejepa_soxx_poc import (
    _collect_calib_latents,
    _compute_forward_drawdown,
    _crash_onset_date_from_close,
    _fix_seeds,
    _split_df,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _append_poc3_summary(outdir: Path, rows: list[dict]) -> None:
    n_pass = sum(1 for r in rows if r["all_pass"])
    n_total = len(rows)
    signal_parts = ", ".join(
        f"{r['signal_name']}={'PASS' if r['all_pass'] else 'FAIL'}" for r in rows
    )
    line = f"baselines: {n_pass}/{n_total} signals pass ({signal_parts})\n"
    with open(outdir / "poc3_summary.md", "a") as fh:
        fh.write(line)


def _print_console_summary(rows: list[dict]) -> None:
    print("\n=== Extended Baselines Summary ===")
    print(f"{'signal':>8}  {'first_breach':>14}  {'lead_days':>10}  {'verdict':>7}")
    print("-" * 46)
    for r in rows:
        fb = r["first_breach"]
        breach_str = fb.strftime("%Y-%m-%d") if fb is not None else "never"
        lead_str = str(r["lead_days"]) if r["lead_days"] is not None else "N/A"
        verdict_str = "PASS" if r["all_pass"] else "FAIL"
        print(f"{r['signal_name']:>8}  {breach_str:>14}  {lead_str:>10}  {verdict_str:>7}")


def _build_signal_row(name: str, verdict: Verdict) -> dict:
    return {
        "signal_name": name,
        "first_breach": verdict.lejepa_first_breach,
        "lead_days": verdict.lejepa_lead_days,
        "episodes_2021": verdict.lejepa_episodes_2021,
        "alert_days_2021": verdict.lejepa_alert_days_2021,
        "all_pass": verdict.all_pass,
    }


def _build_signal_scores_df(
    sig_breach: pd.Series,
    sig_episode: pd.Series,
    baseline_df: pd.DataFrame,
    test_close: pd.Series,
    forward_dd: pd.Series,
    crash_flag: pd.Series,
) -> pd.DataFrame:
    df = pd.DataFrame(
        {
            "lejepa_breach": sig_breach,
            "lejepa_episode_id": sig_episode,
            "rv20_breach": baseline_df["rv20_breach"].reindex(sig_breach.index),
            "rv20_episode_id": baseline_df["rv20_episode_id"].reindex(sig_breach.index),
            "vix_breach": baseline_df["vix_breach"].reindex(sig_breach.index),
            "vix_episode_id": baseline_df["vix_episode_id"].reindex(sig_breach.index),
        },
        index=sig_breach.index,
    )
    df["soxx_close"] = test_close.reindex(df.index)
    df["forward_drawdown_60d"] = forward_dd.reindex(df.index)
    df["crash_onset_flag"] = crash_flag.reindex(df.index)
    return df


def _write_baselines_chart(
    outdir: Path,
    test_close: pd.Series,
    lejepa_df: pd.DataFrame,
    lejepa_threshold: float,
    baseline_df: pd.DataFrame,
    pca_df: pd.DataFrame,
    if_df: pd.DataFrame,
    crash_onset,
) -> None:
    import math

    import matplotlib.pyplot as plt

    plt.switch_backend("Agg")

    fig, (ax_top, ax_bot) = plt.subplots(2, 1, figsize=(14, 8), sharex=True)

    ax_top.plot(test_close.index, test_close, color="black", lw=1.2, label="SOXX Close")
    if crash_onset is not None:
        ax_top.axvline(crash_onset, color="red", ls="--", lw=1.2, label="Crash Onset")
    ax_top.set_ylabel("Close")
    ax_top.legend(fontsize=8)

    ax_bot.plot(lejepa_df.index, lejepa_df["lejepa_score"], color="steelblue", lw=1.0, label="LeJEPA")
    if not math.isnan(lejepa_threshold):
        ax_bot.axhline(lejepa_threshold, color="steelblue", ls=":", lw=0.8)

    ax_bot.plot(baseline_df.index, baseline_df["rv20_zscore"], color="orange", lw=1.0, label="RV20")
    ax_bot.axhline(float(baseline_df["rv20_threshold"].iloc[0]), color="orange", ls=":", lw=0.8)

    ax_bot.plot(baseline_df.index, baseline_df["vix_zscore"], color="green", lw=1.0, label="VIX")
    ax_bot.axhline(float(baseline_df["vix_threshold"].iloc[0]), color="green", ls=":", lw=0.8)

    ax_bot.plot(pca_df.index, pca_df["pca_score"], color="purple", lw=1.0, label="PCA")
    ax_bot.axhline(float(pca_df["pca_threshold"].iloc[0]), color="purple", ls=":", lw=0.8)

    ax_bot.plot(if_df.index, if_df["if_score"], color="brown", lw=1.0, label="IF")
    ax_bot.axhline(float(if_df["if_threshold"].iloc[0]), color="brown", ls=":", lw=0.8)

    if crash_onset is not None:
        ax_bot.axvline(crash_onset, color="red", ls="--", lw=1.2)

    ax_bot.set_ylabel("Score")
    ax_bot.legend(fontsize=8)

    fig.tight_layout()
    fig.savefig(outdir / "baselines_chart.png", dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main runner
# ---------------------------------------------------------------------------

def run_baselines(outdir: Path = Path("outputs/poc3")) -> pd.DataFrame:
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    _fix_seeds(42)
    config = Config()

    print("[baselines] Building data pipeline...")
    pipeline = DataPipeline(config)
    dataset = pipeline.build()

    def _baseline_input(start: str, end: str) -> pd.DataFrame:
        raw = _split_df(pipeline.raw_df, start, end)
        feat = _split_df(pipeline.features_df, start, end)
        return pd.DataFrame({"rv20": feat["rv20"], "vix": raw["vix"]}).dropna()

    train_df_bl = _baseline_input(config.train_start, config.train_end)
    calib_df_bl = _baseline_input(config.calib_start, config.calib_end)
    test_df_bl = _baseline_input(config.test_start, config.test_end)

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

    print("[baselines] Training LeJEPA model (seed=42, latent_dim=16)...")
    model = LeJEPAModel(config)
    model = Trainer(config).fit(model, dataset.train, dataset.val)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)

    z_target, z_pred, z_context = _collect_calib_latents(model, dataset.calib)

    target_cal = MahalanobisCalibrator()
    target_cal.fit(z_target)
    target_cal.set_threshold(z_pred, percentile=config.threshold_percentile)

    context_cal = MahalanobisCalibrator()
    context_cal.fit(z_context)
    context_cal.set_threshold(z_context, percentile=config.threshold_percentile)

    config.lejepa_threshold = target_cal.threshold

    print("[baselines] Scoring all five signals...")
    lejepa_df = Scorer(model, target_cal, context_cal, dataset.test_dates).score(dataset.test)
    baseline_df = BaselineScorer(train_df_bl, config).fit_calibration(calib_df_bl).score(test_df_bl)
    pca_df = PCAScorer(dataset.train, config).fit_calibration(dataset.calib).score(dataset.test)
    if_df = IsolationForestScorer(dataset.train, config).fit_calibration(dataset.calib).score(dataset.test)

    signals = [
        ("lejepa", lejepa_df["lejepa_breach"], lejepa_df["lejepa_episode_id"]),
        ("rv20",   baseline_df["rv20_breach"],  baseline_df["rv20_episode_id"]),
        ("vix",    baseline_df["vix_breach"],   baseline_df["vix_episode_id"]),
        ("pca",    pca_df["pca_breach"],         pca_df["pca_episode_id"]),
        ("if",     if_df["if_breach"],           if_df["if_episode_id"]),
    ]

    rows: list[dict] = []
    for name, sig_breach, sig_episode in signals:
        scores_df = _build_signal_scores_df(
            sig_breach, sig_episode, baseline_df, test_close, forward_dd, crash_flag
        )
        verdict = Evaluator(scores_df, test_close).evaluate()
        rows.append(_build_signal_row(name, verdict))

    result_df = pd.DataFrame(rows)
    csv_path = outdir / "baselines_results.csv"
    result_df.to_csv(csv_path, index=False)
    print(f"[baselines] Results written to {csv_path}")

    _write_baselines_chart(outdir, test_close, lejepa_df, config.lejepa_threshold,
                           baseline_df, pca_df, if_df, crash_onset)
    print(f"[baselines] Chart written to {outdir / 'baselines_chart.png'}")

    _print_console_summary(rows)
    _append_poc3_summary(outdir, rows)

    return result_df


def main() -> None:
    run_baselines(outdir=Path("outputs/poc3"))


if __name__ == "__main__":
    main()
