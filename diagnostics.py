"""
diagnostics.py — post-hoc sensitivity analysis for LeJEPA SOXX POC-1

POST-HOC ONLY. Not part of the preregistered protocol.

Scenarios
---------
  original          — 2019-2020 calibration, p99  (reference, matches run_config.json)
  pre-COVID p99     — 2019-01-01 to 2020-01-31, p99
  excl Feb-Apr 2020 — 2019-2020 minus 2020-02-01:2020-04-30, p99
  p97.5             — 2019-2020 calibration, p97.5
  p99.5             — 2019-2020 calibration, p99.5

Outputs
-------
  console                       — comparison table
  outputs/diagnostics.png       — score-vs-threshold visualisation

Caches (delete to force refresh)
---------------------------------
  outputs/diag_model.pt         — trained model weights
  outputs/diag_calib_latents.npz — calibration-period latent vectors + dates

Usage
-----
    python diagnostics.py
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch

from run_lejepa_soxx_poc import (
    _collect_calib_latents,
    _crash_onset_date_from_close,
    _fix_seeds,
)
from src.baselines import BaselineScorer
from src.calibration import MahalanobisCalibrator, regime_filter_mask
from src.data_pipeline import DataPipeline
from src.model import LeJEPAModel
from src.training import Trainer
from src.types import Config

OUTPUT_DIR = Path("outputs")
MODEL_CKPT = OUTPUT_DIR / "diag_model.pt"
CALIB_CACHE = OUTPUT_DIR / "diag_calib_latents.npz"

SCENARIOS = [
    {"label": "original",          "cutoff": None,         "exclude": None,                         "pct": 99.0, "regime_pct": None},
    {"label": "pre-COVID p99",     "cutoff": "2020-01-31", "exclude": None,                         "pct": 99.0, "regime_pct": None},
    {"label": "excl Feb-Apr 2020", "cutoff": None,         "exclude": ("2020-02-01", "2020-04-30"), "pct": 99.0, "regime_pct": None},
    {"label": "p97.5",             "cutoff": None,         "exclude": None,                         "pct": 97.5, "regime_pct": None},
    {"label": "p99.5",             "cutoff": None,         "exclude": None,                         "pct": 99.5, "regime_pct": None},
    # POC-2 regime-filter variants — post-hoc only
    {"label": "regime-p95",        "cutoff": None,         "exclude": None,                         "pct": 99.0, "regime_pct": 95},
    {"label": "regime-p90",        "cutoff": None,         "exclude": None,                         "pct": 99.0, "regime_pct": 90},
    {"label": "regime-p99",        "cutoff": None,         "exclude": None,                         "pct": 99.0, "regime_pct": 99},
]

_COLORS = ["steelblue", "darkorange", "tomato", "mediumpurple", "dimgray",
           "forestgreen", "saddlebrown", "teal"]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _filter_latents(
    z_target: np.ndarray,
    z_pred: np.ndarray,
    dates: pd.DatetimeIndex,
    cutoff: Optional[str],
    exclude: Optional[tuple],
    regime_mask: Optional[np.ndarray] = None,
) -> tuple[np.ndarray, np.ndarray]:
    mask = np.ones(len(dates), dtype=bool)
    if cutoff:
        mask &= dates <= pd.Timestamp(cutoff)
    if exclude:
        e0, e1 = pd.Timestamp(exclude[0]), pd.Timestamp(exclude[1])
        mask &= ~((dates >= e0) & (dates <= e1))
    if regime_mask is not None:
        mask &= regime_mask
    return z_target[mask], z_pred[mask]


def _filter_df(
    df: pd.DataFrame,
    cutoff: Optional[str],
    exclude: Optional[tuple],
    regime_mask: Optional[np.ndarray] = None,
) -> pd.DataFrame:
    mask = np.ones(len(df), dtype=bool)
    if cutoff:
        mask &= df.index <= pd.Timestamp(cutoff)
    if exclude:
        e0, e1 = pd.Timestamp(exclude[0]), pd.Timestamp(exclude[1])
        mask &= ~((df.index >= e0) & (df.index <= e1))
    if regime_mask is not None:
        mask &= regime_mask
    return df[mask]


def _first_breach(
    scores: np.ndarray,
    threshold: float,
    dates: pd.DatetimeIndex,
) -> Optional[pd.Timestamp]:
    hits = dates[scores > threshold]
    return hits[0] if len(hits) else None


def _lead_days(
    breach: Optional[pd.Timestamp],
    onset: Optional[pd.Timestamp],
    dates: pd.DatetimeIndex,
) -> Optional[int]:
    if breach is None or onset is None:
        return None
    try:
        return int(dates.get_loc(onset) - dates.get_loc(breach))
    except KeyError:
        return None


# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------

def _load_or_train(config: Config, dataset) -> LeJEPAModel:
    model = LeJEPAModel(config)
    if MODEL_CKPT.exists():
        print(f"[diag] Loading checkpoint: {MODEL_CKPT}")
        model.load_state_dict(torch.load(MODEL_CKPT, map_location="cpu", weights_only=True))
    else:
        print("[diag] No checkpoint — training (will be cached after first run)...")
        model = Trainer(config).fit(model, dataset.train, dataset.val)
        torch.save(model.state_dict(), MODEL_CKPT)
        print(f"[diag] Checkpoint saved: {MODEL_CKPT}")
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model


def _load_or_collect_calib(
    model: LeJEPAModel, dataset
) -> tuple[np.ndarray, np.ndarray, pd.DatetimeIndex]:
    if CALIB_CACHE.exists():
        print(f"[diag] Loading calib latents: {CALIB_CACHE}")
        d = np.load(CALIB_CACHE)
        dates = pd.DatetimeIndex(pd.to_datetime(d["dates_ns"], unit="ns"))
        return d["z_target"], d["z_pred"], dates
    print("[diag] Collecting calibration latents...")
    zt, zp, _ = _collect_calib_latents(model, dataset.calib)
    dates_ns = np.array([w.date.value for w in dataset.calib], dtype=np.int64)
    np.savez(CALIB_CACHE, z_target=zt, z_pred=zp, dates_ns=dates_ns)
    print(f"[diag] Saved: {CALIB_CACHE}")
    return zt, zp, pd.DatetimeIndex([w.date for w in dataset.calib])


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    _fix_seeds(42)
    config = Config()

    print("[diag] Building data pipeline...")
    pipeline = DataPipeline(config)
    dataset = pipeline.build()

    def _bl_df(start: str, end: str) -> pd.DataFrame:
        raw = pipeline.raw_df
        feat = pipeline.features_df
        r = raw.loc[(raw.index >= pd.Timestamp(start)) & (raw.index <= pd.Timestamp(end))]
        f = feat.loc[(feat.index >= pd.Timestamp(start)) & (feat.index <= pd.Timestamp(end))]
        return pd.DataFrame({"rv20": f["rv20"], "vix": r["vix"]}).dropna()

    train_df = _bl_df(config.train_start, config.train_end)
    calib_df_full = _bl_df(config.calib_start, config.calib_end)
    test_df = _bl_df(config.test_start, config.test_end)

    model = _load_or_train(config, dataset)
    z_target_full, z_pred_full, calib_dates = _load_or_collect_calib(model, dataset)

    # Load test z_pred from scores.csv — avoids re-running model on test period.
    # These are the raw 16-dim latent prediction vectors; Mahalanobis distances
    # are recomputed below against each scenario's calibration distribution.
    scores_csv = pd.read_csv(
        OUTPUT_DIR / "scores.csv", skiprows=1, parse_dates=["date"], index_col="date"
    )
    test_zpred = scores_csv[[f"z_pred_{j}" for j in range(16)]].values
    test_dates = scores_csv.index
    test_close = scores_csv["soxx_close"]

    crash_onset = _crash_onset_date_from_close(
        test_close,
        window=config.drawdown_window_days,
        threshold=config.drawdown_threshold,
    )
    print(f"[diag] Crash onset: {crash_onset.date() if crash_onset else 'N/A'}")

    # Pre-compute regime filter masks for all regime_pct variants.
    # Instantiate a temporary BaselineScorer to access training-period rv20 stats.
    # The threshold is derived from training data only — no calibration data is touched here.
    bs_train = BaselineScorer(train_df, config)
    train_rv20_z = (train_df["rv20"] - bs_train.rv20_mean) / bs_train.rv20_std
    regime_pct_values = {sc["regime_pct"] for sc in SCENARIOS if sc.get("regime_pct") is not None}
    regime_thresholds: dict[int, float] = {
        pct: float(np.percentile(train_rv20_z, pct)) for pct in regime_pct_values
    }
    rv20_all = pipeline.features_df["rv20"]

    # Regime masks: one per (window dates, daily index) pair for each pct.
    # window mask → applied to latent arrays (one row per window, indexed by w.date)
    # daily mask  → applied to calib_df_full (one row per trading day)
    regime_window_masks: dict[int, np.ndarray] = {
        pct: regime_filter_mask(calib_dates, rv20_all, bs_train.rv20_mean, bs_train.rv20_std, thr)
        for pct, thr in regime_thresholds.items()
    }
    regime_daily_masks: dict[int, np.ndarray] = {
        pct: regime_filter_mask(calib_df_full.index, rv20_all, bs_train.rv20_mean, bs_train.rv20_std, thr)
        for pct, thr in regime_thresholds.items()
    }

    if regime_thresholds:
        print("[diag] Regime filter thresholds (training-period RV20 z-score percentiles):")
        for pct in sorted(regime_thresholds):
            n_kept_windows = int(regime_window_masks[pct].sum())
            n_kept_days = int(regime_daily_masks[pct].sum())
            print(f"  p{pct}: z <= {regime_thresholds[pct]:.3f}  "
                  f"({n_kept_windows}/{len(calib_dates)} windows, "
                  f"{n_kept_days}/{len(calib_df_full)} days retained)")
    print()

    results: list[dict] = []
    scenario_scores: dict[str, tuple[np.ndarray, float]] = {}

    for sc in SCENARIOS:
        lbl = sc["label"]
        cutoff, exclude, pct = sc["cutoff"], sc["exclude"], sc["pct"]
        regime_pct = sc.get("regime_pct")

        lj_regime_mask = regime_window_masks.get(regime_pct) if regime_pct is not None else None
        df_regime_mask = regime_daily_masks.get(regime_pct) if regime_pct is not None else None

        zt_sub, zp_sub = _filter_latents(
            z_target_full, z_pred_full, calib_dates, cutoff, exclude, lj_regime_mask
        )
        calib_df_sub = _filter_df(calib_df_full, cutoff, exclude, df_regime_mask)

        cal = MahalanobisCalibrator()
        cal.fit(zt_sub)
        cal.set_threshold(zp_sub, percentile=pct)

        test_scores = np.array([cal.score(test_zpred[i]) for i in range(len(test_zpred))])

        bs = BaselineScorer(train_df, config)
        bs.fit_calibration(calib_df_sub)
        bl = bs.score(test_df)

        lj_fb = _first_breach(test_scores, cal.threshold, test_dates)

        rv20_hits = np.where(bl["rv20_breach"].values)[0]
        vix_hits = np.where(bl["vix_breach"].values)[0]
        rv20_fb = test_df.index[rv20_hits[0]] if len(rv20_hits) else None
        vix_fb = test_df.index[vix_hits[0]] if len(vix_hits) else None

        results.append({
            "label": lbl,
            "n_calib": len(zp_sub),
            "pct": pct,
            "lj_thr": cal.threshold,
            "rv20_thr": bs.rv20_threshold,
            "vix_thr": bs.vix_threshold,
            "lj_fb": lj_fb,
            "rv20_fb": rv20_fb,
            "vix_fb": vix_fb,
            "lj_lead": _lead_days(lj_fb, crash_onset, test_dates),
            "rv20_lead": _lead_days(rv20_fb, crash_onset, test_df.index),
            "vix_lead": _lead_days(vix_fb, crash_onset, test_df.index),
            "max_pct_thr": float(np.nanmax(test_scores) / cal.threshold * 100),
        })
        scenario_scores[lbl] = (test_scores, cal.threshold)

    _print_table(results, crash_onset)
    _save_chart(scores_csv, scenario_scores, crash_onset, results)
    print(f"[diag] Chart saved: {OUTPUT_DIR / 'diagnostics.png'}")


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------

def _print_table(results: list[dict], crash_onset: Optional[pd.Timestamp]) -> None:
    sep = "=" * 138
    print(sep)
    print("POST-HOC SENSITIVITY ANALYSIS — LeJEPA SOXX POC-1")
    print(f"Crash onset: {crash_onset.date() if crash_onset else 'N/A'}")
    print(sep)
    print(
        f"{'Scenario':<22} {'N_cal':>5} {'Pct':>5}  "
        f"{'LJ_thr':>7} {'RV20_thr':>9} {'VIX_thr':>8}  "
        f"{'LJ_breach':<12} {'LJ_lead':>8}  "
        f"{'RV20_breach':<12} {'RV20_lead':>10}  "
        f"{'VIX_breach':<12} {'VIX_lead':>9}  "
        f"{'max%thr':>7}  {'C1':>4}  {'C2':>7}"
    )
    print("-" * 138)
    for r in results:
        lj_b   = r["lj_fb"].strftime("%Y-%m-%d")   if r["lj_fb"]   else "N/A"
        rv20_b = r["rv20_fb"].strftime("%Y-%m-%d")  if r["rv20_fb"] else "N/A"
        vix_b  = r["vix_fb"].strftime("%Y-%m-%d")   if r["vix_fb"]  else "N/A"
        lj_l   = str(r["lj_lead"])   if r["lj_lead"]   is not None else "N/A"
        rv20_l = str(r["rv20_lead"]) if r["rv20_lead"] is not None else "N/A"
        vix_l  = str(r["vix_lead"])  if r["vix_lead"]  is not None else "N/A"
        c1 = "PASS" if r["lj_lead"] is not None and r["lj_lead"] >= 0 else "FAIL"
        beats_rv20 = r["lj_fb"] is not None and r["rv20_fb"] is not None and r["lj_fb"] < r["rv20_fb"]
        beats_vix  = r["lj_fb"] is not None and r["vix_fb"]  is not None and r["lj_fb"] < r["vix_fb"]
        c2 = "PASS" if (beats_rv20 and beats_vix) else ("PARTIAL" if (beats_rv20 ^ beats_vix) else "FAIL")
        print(
            f"{r['label']:<22} {r['n_calib']:>5} {r['pct']:>5.1f}  "
            f"{r['lj_thr']:>7.2f} {r['rv20_thr']:>9.2f} {r['vix_thr']:>8.2f}  "
            f"{lj_b:<12} {lj_l:>8}  "
            f"{rv20_b:<12} {rv20_l:>10}  "
            f"{vix_b:<12} {vix_l:>9}  "
            f"{r['max_pct_thr']:>7.1f}  {c1:>4}  {c2:>7}"
        )
    print(sep)
    print("LJ_lead = trading-day lead (positive = before crash onset)")
    print()


def _save_chart(
    scores_csv: pd.DataFrame,
    scenario_scores: dict[str, tuple[np.ndarray, float]],
    crash_onset: Optional[pd.Timestamp],
    results: list[dict],
) -> None:
    import matplotlib.pyplot as plt
    plt.switch_backend("Agg")

    fig, (ax_top, ax_mid, ax_bot) = plt.subplots(3, 1, figsize=(14, 10), sharex=True)

    # --- Panel 1: SOXX close ---
    ax_top.plot(scores_csv.index, scores_csv["soxx_close"], color="black", lw=1.2, label="SOXX")
    if crash_onset is not None:
        ax_top.axvline(crash_onset, color="red", ls="--", lw=1.2, label="Crash onset")
    ax_top.set_ylabel("SOXX Close")
    ax_top.legend(fontsize=8)
    ax_top.set_title("LeJEPA SOXX POC-1 — Post-hoc Sensitivity Analysis", fontsize=11)

    # --- Panel 2: LeJEPA scores ---
    # Show three score series: original, pre-COVID, and regime-p95
    for sc_label, color, alpha in [
        ("original", "steelblue", 0.55),
        ("pre-COVID p99", "darkorange", 0.55),
        ("regime-p95", "forestgreen", 0.55),
    ]:
        if sc_label in scenario_scores:
            sc_scores, _ = scenario_scores[sc_label]
            ax_mid.plot(
                scores_csv.index, sc_scores,
                color=color, lw=0.85, alpha=alpha, label=f"score ({sc_label})",
            )

    # Threshold lines for all five scenarios
    ls_cycle = ["-", "--", "-.", ":", (0, (3, 1, 1, 1)), "--", "-.", ":"]
    for i, sc in enumerate(SCENARIOS):
        lbl = sc["label"]
        if lbl in scenario_scores:
            _, thr = scenario_scores[lbl]
            ax_mid.axhline(
                thr, color=_COLORS[i], ls=ls_cycle[i], lw=1.3,
                label=f"{lbl}  thr={thr:.2f}",
            )

    # Vertical breach markers for scenarios that fired
    for i, r in enumerate(results):
        if r["lj_fb"] is not None:
            ax_mid.axvline(
                r["lj_fb"], color=_COLORS[i], ls="-.", lw=0.7, alpha=0.55,
            )

    if crash_onset is not None:
        ax_mid.axvline(crash_onset, color="red", ls="--", lw=1.0, alpha=0.6)

    ax_mid.set_ylabel("Mahalanobis score")
    ax_mid.legend(fontsize=7, loc="upper left", ncol=2)

    # --- Panel 3: RV20 and VIX z-scores ---
    ax_bot.plot(
        scores_csv.index, scores_csv["rv20_zscore"],
        color="orange", lw=0.9, alpha=0.8, label="RV20 z-score",
    )
    ax_bot.plot(
        scores_csv.index, scores_csv["vix_zscore"],
        color="green", lw=0.9, alpha=0.8, label="VIX z-score",
    )

    orig = results[0]
    pre_covid = next((r for r in results if "pre-COVID" in r["label"]), None)

    ax_bot.axhline(
        orig["rv20_thr"], color="orange", ls="-", lw=1.2,
        label=f"RV20 orig={orig['rv20_thr']:.1f}",
    )
    ax_bot.axhline(
        orig["vix_thr"], color="green", ls="-", lw=1.2,
        label=f"VIX orig={orig['vix_thr']:.1f}",
    )
    if pre_covid is not None:
        ax_bot.axhline(
            pre_covid["rv20_thr"], color="orange", ls="--", lw=1.2,
            label=f"RV20 pre-COVID={pre_covid['rv20_thr']:.1f}",
        )
        ax_bot.axhline(
            pre_covid["vix_thr"], color="green", ls="--", lw=1.2,
            label=f"VIX pre-COVID={pre_covid['vix_thr']:.1f}",
        )

    if crash_onset is not None:
        ax_bot.axvline(crash_onset, color="red", ls="--", lw=1.0, alpha=0.6)

    ax_bot.set_ylabel("Z-score")
    ax_bot.legend(fontsize=7, loc="upper left", ncol=2)

    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "diagnostics.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    main()
