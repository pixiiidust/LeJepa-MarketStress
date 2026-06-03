"""
OutputWriter: writes scores.csv, soxx_lejepa_chart.png, run_config.json, console summary.

Chart layout:
    Top panel:    SOXX close, crash onset vertical line, shaded drawdown window
    Bottom panel: LeJEPA score + threshold, RV20 z-score + threshold, VIX z-score + threshold,
                  first breach markers for each signal

Public interface:
    OutputWriter(output_dir).write(scores_df, verdict, config)
"""
from __future__ import annotations

import dataclasses
import json
import math
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd

from src.types import Config, Verdict

_DISCLAIMER = (
    "This POC tests whether LeJEPA latent drift can function as an unsupervised "
    "early warning signal for market regime stress, and whether it beats simple "
    "volatility and VIX baselines in a leakage-controlled historical backtest."
)

_EVAL_ONLY_COMMENT = (
    "# forward_drawdown_60d and crash_onset_flag are evaluation-only "
    "(future-looking) columns\n"
)

_CSV_COLUMNS = [
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


class OutputWriter:
    def __init__(self, output_dir: Path) -> None:
        self._dir = Path(output_dir)
        self._dir.mkdir(parents=True, exist_ok=True)

    def write(
        self,
        scores_df: pd.DataFrame,
        verdict: Verdict,
        config: Config,
    ) -> None:
        self._write_scores_csv(scores_df, config)
        self._write_run_config(verdict, config)
        self._write_chart(scores_df, verdict, config)
        self._print_summary(verdict)

    # ------------------------------------------------------------------

    def _write_scores_csv(self, scores_df: pd.DataFrame, config: Config) -> None:
        df = scores_df.copy()
        df["lejepa_threshold"] = config.lejepa_threshold
        df_out = df[_CSV_COLUMNS]
        with open(self._dir / "scores.csv", "w", newline="") as f:
            f.write(_EVAL_ONLY_COMMENT)
            df_out.to_csv(f, index_label="date")

    def _write_run_config(self, verdict: Verdict, config: Config) -> None:
        cfg_dict = dataclasses.asdict(config)
        _sanitize_floats(cfg_dict)
        payload = {
            "config": cfg_dict,
            "verdict": _verdict_summary(verdict),
        }
        with open(self._dir / "run_config.json", "w") as f:
            json.dump(payload, f, indent=2, default=_json_default)

    def _write_chart(
        self,
        scores_df: pd.DataFrame,
        verdict: Verdict,
        config: Config,
    ) -> None:
        import matplotlib.pyplot as plt
        plt.switch_backend("Agg")

        fig, (ax_top, ax_bot) = plt.subplots(2, 1, figsize=(14, 8), sharex=True)

        # Top panel — SOXX close + crash onset
        ax_top.plot(
            scores_df.index, scores_df["soxx_close"],
            color="black", lw=1.2, label="SOXX Close",
        )
        if verdict.crash_onset_date is not None:
            ax_top.axvline(
                verdict.crash_onset_date, color="red", ls="--", lw=1.2, label="Crash Onset",
            )
        ax_top.set_ylabel("Close")
        ax_top.legend(fontsize=8)

        # Bottom panel — scores + thresholds + first-breach markers
        ax_bot.plot(
            scores_df.index, scores_df["lejepa_score"],
            color="steelblue", lw=1.0, label="LeJEPA score",
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
        fig.savefig(self._dir / "soxx_lejepa_chart.png", dpi=150)
        plt.close(fig)

    def _print_summary(self, verdict: Verdict) -> None:
        sep = "=" * 60
        lines = [
            sep,
            "LeJEPA SOXX POC-1 — Experiment Summary",
            sep,
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

        lines += [
            f"Final verdict: {final}",
            "",
            _DISCLAIMER,
            sep,
        ]
        print("\n".join(lines))


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

def _fmt_date(ts: Optional[pd.Timestamp]) -> str:
    if ts is None or (isinstance(ts, float) and math.isnan(ts)):
        return "N/A"
    try:
        if pd.isna(ts):
            return "N/A"
    except (TypeError, ValueError):
        pass
    return str(ts.date())


def _sanitize_floats(d: dict) -> None:
    """Replace float NaN with None in-place so json.dump produces valid JSON."""
    for k, v in d.items():
        if isinstance(v, float) and math.isnan(v):
            d[k] = None
        elif isinstance(v, dict):
            _sanitize_floats(v)


def _json_default(obj: Any) -> Any:
    if isinstance(obj, pd.Timestamp):
        try:
            if pd.isna(obj):
                return None
        except (TypeError, ValueError):
            pass
        return str(obj.date())
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        f = float(obj)
        return None if math.isnan(f) else f
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, tuple):
        return list(obj)
    raise TypeError(f"Not JSON serializable: {type(obj)}")


def _verdict_summary(verdict: Verdict) -> dict:
    def _ts(ts: Optional[pd.Timestamp]) -> Optional[str]:
        if ts is None:
            return None
        try:
            if pd.isna(ts):
                return None
        except (TypeError, ValueError):
            pass
        return str(ts.date())

    return {
        "crash_onset_date": _ts(verdict.crash_onset_date),
        "lejepa_first_breach": _ts(verdict.lejepa_first_breach),
        "rv20_first_breach": _ts(verdict.rv20_first_breach),
        "vix_first_breach": _ts(verdict.vix_first_breach),
        "lejepa_lead_days": verdict.lejepa_lead_days,
        "rv20_lead_days": verdict.rv20_lead_days,
        "vix_lead_days": verdict.vix_lead_days,
        "lejepa_episodes_2021": verdict.lejepa_episodes_2021,
        "rv20_episodes_2021": verdict.rv20_episodes_2021,
        "vix_episodes_2021": verdict.vix_episodes_2021,
        "lejepa_alert_days_2021": verdict.lejepa_alert_days_2021,
        "rv20_alert_days_2021": verdict.rv20_alert_days_2021,
        "vix_alert_days_2021": verdict.vix_alert_days_2021,
        "criterion_1": verdict.criterion_1,
        "criterion_2": verdict.criterion_2,
        "criterion_3": verdict.criterion_3,
        "criterion_4": verdict.criterion_4,
        "partial_signal": verdict.partial_signal,
        "all_pass": verdict.all_pass,
    }
