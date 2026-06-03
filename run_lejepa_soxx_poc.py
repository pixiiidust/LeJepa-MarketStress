"""
LeJEPA SOXX Market Stress POC-1

Single entry point for the full experiment:
    1. Set seeds (42)
    2. Build data pipeline (fetch, align, feature-engineer, scale, split)
    3. Train LeJEPA model (2010-2016, early-stop on 2017-2018)
    4. Calibrate Mahalanobis distributions (2019-2020)
    5. Score blind test period (2021-2022)
    6. Score baselines (RV20, VIX level)
    7. Evaluate (crash onset, lead times, episode counts, pass/fail)
    8. Write outputs (scores.csv, chart, run_config.json, console summary)

Usage:
    python run_lejepa_soxx_poc.py

Outputs:
    outputs/scores.csv
    outputs/soxx_lejepa_chart.png
    outputs/run_config.json
"""
from __future__ import annotations

import random
import subprocess
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch

from src.baselines import BaselineScorer
from src.calibration import MahalanobisCalibrator
from src.data_pipeline import DataPipeline
from src.evaluation import Evaluator
from src.model import LeJEPAModel
from src.output import OutputWriter
from src.scoring import Scorer
from src.training import Trainer
from src.types import Config, Window


# ---------------------------------------------------------------------------
# Testable helper functions
# ---------------------------------------------------------------------------

def _fix_seeds(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _get_git_commit() -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        capture_output=True, text=True, check=False,
    )
    return result.stdout.strip() if result.returncode == 0 else ""


def _split_df(df: pd.DataFrame, start: str, end: str) -> pd.DataFrame:
    mask = (df.index >= pd.Timestamp(start)) & (df.index <= pd.Timestamp(end))
    return df.loc[mask]


def _collect_calib_latents(
    model: LeJEPAModel,
    calib_windows: list[Window],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (z_target, z_pred, z_context) arrays from frozen model on calib windows."""
    model.eval()
    z_targets: list[np.ndarray] = []
    z_preds: list[np.ndarray] = []
    z_contexts: list[np.ndarray] = []
    with torch.no_grad():
        for w in calib_windows:
            ctx_flat = torch.tensor(w.context_array.flatten(), dtype=torch.float32)
            z_context = model.encode(ctx_flat)
            z_pred = model.predict(z_context)
            future_ctx = np.concatenate([w.context_array[-20:], w.target_array], axis=0)
            z_target = model.encode(
                torch.tensor(future_ctx.flatten(), dtype=torch.float32)
            )
            z_targets.append(z_target.numpy())
            z_preds.append(z_pred.numpy())
            z_contexts.append(z_context.numpy())
    return np.stack(z_targets), np.stack(z_preds), np.stack(z_contexts)


def _compute_forward_drawdown(close: pd.Series, window: int = 60) -> pd.Series:
    """Per-day forward drawdown; NaN if fewer than window days remain."""
    n = len(close)
    result = np.full(n, np.nan)
    for i in range(n):
        if i + window > n:
            break
        fwd_min = close.iloc[i : i + window].min()
        result[i] = fwd_min / close.iloc[i] - 1
    return pd.Series(result, index=close.index)


def _crash_onset_date_from_close(
    close: pd.Series,
    window: int = 60,
    threshold: float = -0.20,
) -> Optional[pd.Timestamp]:
    """First test day with a full window whose forward drawdown ≤ threshold."""
    n = len(close)
    for i in range(n):
        if i + window > n:
            break
        fwd_min = close.iloc[i : i + window].min()
        if fwd_min / close.iloc[i] - 1 <= threshold:
            return close.index[i]
    return None


def _build_scores_df(
    lejepa_df: pd.DataFrame,
    baseline_df: pd.DataFrame,
    soxx_close: pd.Series,
    forward_drawdown: pd.Series,
    crash_onset_flag: pd.Series,
) -> pd.DataFrame:
    """Merge lejepa and baseline frames; attach evaluation-only columns."""
    scores_df = pd.concat([lejepa_df, baseline_df], axis=1)
    scores_df["soxx_close"] = soxx_close.reindex(scores_df.index)
    scores_df["forward_drawdown_60d"] = forward_drawdown.reindex(scores_df.index)
    scores_df["crash_onset_flag"] = crash_onset_flag.reindex(scores_df.index)
    return scores_df


# ---------------------------------------------------------------------------
# Main orchestration
# ---------------------------------------------------------------------------

def main() -> None:
    # 1. Seeds
    _fix_seeds(42)

    # 2. Config + git commit
    config = Config()
    config.git_commit = _get_git_commit()
    print(f"[main] git_commit = {config.git_commit!r}")

    # 3. Data pipeline
    print("[main] Building data pipeline...")
    pipeline = DataPipeline(config)
    dataset = pipeline.build()

    def _make_baseline_input(start: str, end: str) -> pd.DataFrame:
        raw = _split_df(pipeline.raw_df, start, end)
        feat = _split_df(pipeline.features_df, start, end)
        return pd.DataFrame({"rv20": feat["rv20"], "vix": raw["vix"]}).dropna()

    train_df = _make_baseline_input(config.train_start, config.train_end)
    calib_df = _make_baseline_input(config.calib_start, config.calib_end)
    test_df  = _make_baseline_input(config.test_start,  config.test_end)

    # 4. Train model + freeze
    print("[main] Training LeJEPA model...")
    model = LeJEPAModel(config)
    model = Trainer(config).fit(model, dataset.train, dataset.val)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)

    # 5. Calibration: fit + set threshold BEFORE any test data touches the model
    print("[main] Calibrating Mahalanobis distributions...")
    z_target_calib, z_pred_calib, z_context_calib = _collect_calib_latents(
        model, dataset.calib
    )

    target_cal = MahalanobisCalibrator()
    target_cal.fit(z_target_calib)
    target_cal.set_threshold(z_pred_calib, percentile=config.threshold_percentile)

    context_cal = MahalanobisCalibrator()
    context_cal.fit(z_context_calib)
    context_cal.set_threshold(z_context_calib, percentile=config.threshold_percentile)

    # Freeze threshold into config before Scorer runs (acceptance criterion 5)
    config.lejepa_threshold = target_cal.threshold
    print(f"[main] lejepa_threshold = {config.lejepa_threshold:.4f}")

    # 6. Score test period
    print("[main] Scoring test period with LeJEPA...")
    lejepa_df = Scorer(
        model, target_cal, context_cal, dataset.test_dates
    ).score(dataset.test)

    # 7. Baseline scoring
    print("[main] Scoring baselines...")
    baseline_df = (
        BaselineScorer(train_df, config)
        .fit_calibration(calib_df)
        .score(test_df)
    )

    # 8. Merge + evaluation-only columns
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

    # 9. Evaluate
    print("[main] Evaluating results...")
    verdict = Evaluator(scores_df, test_close).evaluate()

    # 10. Write outputs
    print("[main] Writing outputs...")
    OutputWriter(Path("outputs")).write(scores_df, verdict, config)
    print("[main] Done.")


if __name__ == "__main__":
    main()
