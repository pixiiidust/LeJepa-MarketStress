"""
LeJEPA SOXX Market Stress POC-3 — Seed Stability Sweep

Runs the full POC-1 pipeline (train → calibrate → score → evaluate) with 10
fixed seeds (0–9). Data loading and scaling happen once; each seed gets its
own independently fitted model, calibration, and threshold.

Results are written to outputs/poc3/seed_sweep_results.csv.
A one-line summary is appended to outputs/poc3/poc3_summary.md.
No files outside outputs/poc3/ are written.

Usage:
    python run_poc3_seed_sweep.py

Outputs:
    outputs/poc3/seed_sweep_results.csv
    outputs/poc3/poc3_summary.md
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from src.baselines import BaselineScorer
from src.calibration import MahalanobisCalibrator
from src.data_pipeline import DataPipeline
from src.evaluation import Evaluator
from src.model import LeJEPAModel
from src.scoring import Scorer
from src.training import CollapseError, Trainer
from src.types import Config, Verdict

from run_lejepa_soxx_poc import (
    _build_scores_df,
    _collect_calib_latents,
    _compute_forward_drawdown,
    _crash_onset_date_from_close,
    _fix_seeds,
    _split_df,
)

SEEDS = list(range(10))
_COLLAPSE_THRESHOLD = 0.05


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _compute_latent_std_min(z: np.ndarray) -> float:
    """Minimum per-dimension std across N latent samples (shape N × latent_dim)."""
    return float(z.std(axis=0).min())


def _build_seed_row(
    seed: int,
    config: Config,
    verdict: Verdict,
    latent_std_min: float,
) -> dict:
    return {
        "seed": seed,
        "lejepa_threshold": config.lejepa_threshold,
        "lejepa_first_breach": verdict.lejepa_first_breach,
        "crash_onset_date": verdict.crash_onset_date,
        "lejepa_lead_days": verdict.lejepa_lead_days,
        "all_pass": verdict.all_pass,
        "criterion_1": verdict.criterion_1,
        "criterion_2": verdict.criterion_2,
        "criterion_3": verdict.criterion_3,
        "criterion_4": verdict.criterion_4,
        "latent_std_min": latent_std_min,
    }


def _print_console_summary(rows: list[dict]) -> None:
    n_pass = sum(1 for r in rows if r["all_pass"])
    n_total = len(rows)
    lead_days = [r["lejepa_lead_days"] for r in rows if r.get("lejepa_lead_days") is not None]
    print(f"\n=== Seed Sweep Summary ===")
    print(f"Pass rate: {n_pass}/{n_total} seeds where all_pass is True")
    if lead_days:
        mean_lead = float(np.mean(lead_days))
        std_lead = float(np.std(lead_days)) if len(lead_days) > 1 else float("nan")
        print(f"Lead days (seeds where LeJEPA fired): mean={mean_lead:.1f} ± std={std_lead:.1f}")
    else:
        print("Lead days: LeJEPA never fired in any seed")


def _append_poc3_summary(outdir: Path, rows: list[dict]) -> None:
    n_pass = sum(1 for r in rows if r["all_pass"])
    n_total = len(rows)
    lead_days = [r["lejepa_lead_days"] for r in rows if r.get("lejepa_lead_days") is not None]
    if lead_days:
        mean_lead = float(np.mean(lead_days))
        std_lead = float(np.std(lead_days)) if len(lead_days) > 1 else float("nan")
    else:
        mean_lead = float("nan")
        std_lead = float("nan")
    line = (
        f"seed_sweep: {n_pass}/{n_total} seeds pass, "
        f"mean lead={mean_lead:.1f}±{std_lead:.1f} days\n"
    )
    with open(outdir / "poc3_summary.md", "a") as fh:
        fh.write(line)


# ---------------------------------------------------------------------------
# Main sweep
# ---------------------------------------------------------------------------

def run_seed_sweep(outdir: Path = Path("outputs/poc3")) -> pd.DataFrame:
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    config_base = Config()
    print("[seed_sweep] Building data pipeline...")
    pipeline = DataPipeline(config_base)
    dataset = pipeline.build()

    def _baseline_input(start: str, end: str) -> pd.DataFrame:
        raw = _split_df(pipeline.raw_df, start, end)
        feat = _split_df(pipeline.features_df, start, end)
        return pd.DataFrame({"rv20": feat["rv20"], "vix": raw["vix"]}).dropna()

    train_df = _baseline_input(config_base.train_start, config_base.train_end)
    calib_df = _baseline_input(config_base.calib_start, config_base.calib_end)
    test_df = _baseline_input(config_base.test_start, config_base.test_end)

    test_close = (
        _split_df(pipeline.raw_df[["close"]], config_base.test_start, config_base.test_end)[
            "close"
        ].reindex(dataset.test_dates)
    )
    forward_dd = _compute_forward_drawdown(test_close, window=config_base.drawdown_window_days)

    rows: list[dict] = []

    for s in SEEDS:
        print(f"\n[seed_sweep] Running seed {s}...")
        _fix_seeds(s)
        config = Config(seed=s)

        model = LeJEPAModel(config)
        try:
            model = Trainer(config).fit(model, dataset.train, dataset.val)
        except CollapseError as exc:
            print(f"[WARNING] seed={s}: CollapseError — {exc}")
            rows.append(
                {
                    "seed": s,
                    "lejepa_threshold": float("nan"),
                    "lejepa_first_breach": None,
                    "crash_onset_date": None,
                    "lejepa_lead_days": None,
                    "all_pass": False,
                    "criterion_1": False,
                    "criterion_2": False,
                    "criterion_3": False,
                    "criterion_4": False,
                    "latent_std_min": 0.0,
                }
            )
            continue

        model.eval()
        for p in model.parameters():
            p.requires_grad_(False)

        z_target, z_pred, z_context = _collect_calib_latents(model, dataset.calib)
        latent_std_min = _compute_latent_std_min(z_pred)
        if latent_std_min < _COLLAPSE_THRESHOLD:
            print(
                f"[WARNING] seed={s}: latent_std_min={latent_std_min:.4f} < "
                f"{_COLLAPSE_THRESHOLD} (calibration latent collapse)"
            )

        target_cal = MahalanobisCalibrator()
        target_cal.fit(z_target)
        target_cal.set_threshold(z_pred, percentile=config.threshold_percentile)

        context_cal = MahalanobisCalibrator()
        context_cal.fit(z_context)
        context_cal.set_threshold(z_context, percentile=config.threshold_percentile)

        config.lejepa_threshold = target_cal.threshold

        lejepa_df = Scorer(model, target_cal, context_cal, dataset.test_dates).score(dataset.test)
        baseline_df = BaselineScorer(train_df, config).fit_calibration(calib_df).score(test_df)

        crash_onset = _crash_onset_date_from_close(
            test_close,
            window=config_base.drawdown_window_days,
            threshold=config_base.drawdown_threshold,
        )
        crash_flag = pd.Series(False, index=dataset.test_dates)
        if crash_onset is not None and crash_onset in crash_flag.index:
            crash_flag.loc[crash_onset] = True

        scores_df = _build_scores_df(lejepa_df, baseline_df, test_close, forward_dd, crash_flag)
        verdict = Evaluator(scores_df, test_close).evaluate()

        rows.append(_build_seed_row(s, config, verdict, latent_std_min))

    result_df = pd.DataFrame(rows)
    csv_path = outdir / "seed_sweep_results.csv"
    result_df.to_csv(csv_path, index=False)
    print(f"\n[seed_sweep] Results written to {csv_path}")

    _print_console_summary(rows)
    _append_poc3_summary(outdir, rows)

    return result_df


def main() -> None:
    run_seed_sweep(outdir=Path("outputs/poc3"))


if __name__ == "__main__":
    main()
