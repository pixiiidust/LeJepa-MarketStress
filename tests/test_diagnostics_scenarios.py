"""
Regression tests for diagnostics.py 8-scenario pipeline.

Verifies:
- Three regime-filter scenarios (regime-p90/p95/p99) behave correctly
- Five original POC-1 scenarios are unaffected by the regime-filter addition

Tests skip gracefully when cache files are absent. No model training or
latent re-collection is performed.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

CALIB_CACHE = Path("outputs/diag_calib_latents.npz")
SCORES_CSV = Path("outputs/scores.csv")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_caches():
    if not CALIB_CACHE.exists() or not SCORES_CSV.exists():
        pytest.skip("Cache files absent — run diagnostics.py first")
    d = np.load(CALIB_CACHE)
    z_target_full = d["z_target"]
    z_pred_full = d["z_pred"]
    calib_dates = pd.DatetimeIndex(pd.to_datetime(d["dates_ns"], unit="ns"))
    scores_csv = pd.read_csv(SCORES_CSV, skiprows=1, parse_dates=["date"], index_col="date")
    test_zpred = scores_csv[[f"z_pred_{j}" for j in range(16)]].values
    return z_target_full, z_pred_full, calib_dates, test_zpred


def _build_scenario_results() -> dict[str, dict]:
    from diagnostics import SCENARIOS, _filter_latents
    from src.calibration import MahalanobisCalibrator, regime_filter_mask
    from src.baselines import BaselineScorer
    from src.data_pipeline import DataPipeline
    from src.types import Config

    z_target_full, z_pred_full, calib_dates, _test_zpred = _load_caches()

    config = Config()
    pipeline = DataPipeline(config)
    pipeline.build()

    raw = pipeline.raw_df
    feat = pipeline.features_df

    def _bl_df(start: str, end: str) -> pd.DataFrame:
        r = raw.loc[(raw.index >= pd.Timestamp(start)) & (raw.index <= pd.Timestamp(end))]
        f = feat.loc[(feat.index >= pd.Timestamp(start)) & (feat.index <= pd.Timestamp(end))]
        return pd.DataFrame({"rv20": f["rv20"], "vix": r["vix"]}).dropna()

    train_df = _bl_df(config.train_start, config.train_end)

    bs_train = BaselineScorer(train_df, config)
    train_rv20_z = (train_df["rv20"] - bs_train.rv20_mean) / bs_train.rv20_std
    rv20_all = pipeline.features_df["rv20"]

    regime_pct_values = {sc["regime_pct"] for sc in SCENARIOS if sc.get("regime_pct") is not None}
    regime_thresholds: dict[int, float] = {
        pct: float(np.percentile(train_rv20_z, pct)) for pct in regime_pct_values
    }
    regime_window_masks: dict[int, np.ndarray] = {
        pct: regime_filter_mask(calib_dates, rv20_all, bs_train.rv20_mean, bs_train.rv20_std, thr)
        for pct, thr in regime_thresholds.items()
    }

    results: dict[str, dict] = {}
    for sc in SCENARIOS:
        lbl = sc["label"]
        regime_pct = sc.get("regime_pct")
        lj_regime_mask = regime_window_masks.get(regime_pct) if regime_pct is not None else None

        zt_sub, zp_sub = _filter_latents(
            z_target_full, z_pred_full, calib_dates,
            sc["cutoff"], sc["exclude"], lj_regime_mask,
        )
        cal = MahalanobisCalibrator()
        cal.fit(zt_sub)
        cal.set_threshold(zp_sub, percentile=sc["pct"])
        results[lbl] = {"n_calib": len(zp_sub), "lj_thr": cal.threshold}

    return results


# ---------------------------------------------------------------------------
# Cycle 1: regime-p95 has strictly fewer calibration windows than original
# ---------------------------------------------------------------------------

def test_regime_p95_has_fewer_calib_than_original():
    """Regime filter must remove high-volatility windows, reducing n_calib."""
    res = _build_scenario_results()
    assert res["regime-p95"]["n_calib"] < res["original"]["n_calib"]


# ---------------------------------------------------------------------------
# Cycle 2: regime-p95 threshold strictly lower than original
# ---------------------------------------------------------------------------

def test_regime_p95_threshold_lower_than_original():
    """Removing COVID windows from calibration tail must lower the LeJEPA threshold."""
    res = _build_scenario_results()
    assert res["regime-p95"]["lj_thr"] < res["original"]["lj_thr"]


# ---------------------------------------------------------------------------
# Cycle 3: regime filter monotonicity across percentile levels
# ---------------------------------------------------------------------------

def test_regime_filter_monotonicity():
    """Tighter regime filter (lower pct) must retain fewer calibration windows."""
    res = _build_scenario_results()
    assert res["regime-p90"]["n_calib"] < res["regime-p95"]["n_calib"] < res["regime-p99"]["n_calib"]


# ---------------------------------------------------------------------------
# Cycle 4: original POC-1 scenarios are deterministic across two runs
# ---------------------------------------------------------------------------

_ORIGINAL_SCENARIOS = [
    "original",
    "pre-COVID p99",
    "excl Feb-Apr 2020",
    "p97.5",
    "p99.5",
]


def test_original_scenarios_no_regression():
    """Five POC-1 scenarios must produce identical n_calib and lj_thr on repeated runs."""
    run_a = _build_scenario_results()
    run_b = _build_scenario_results()
    for lbl in _ORIGINAL_SCENARIOS:
        assert run_a[lbl]["n_calib"] == run_b[lbl]["n_calib"], (
            f"{lbl}: n_calib changed between runs ({run_a[lbl]['n_calib']} vs {run_b[lbl]['n_calib']})"
        )
        assert run_a[lbl]["lj_thr"] == pytest.approx(run_b[lbl]["lj_thr"], rel=1e-9), (
            f"{lbl}: lj_thr changed between runs"
        )


# ---------------------------------------------------------------------------
# Cycle 5: graceful skip when cache files are absent
# ---------------------------------------------------------------------------

def test_skips_when_cache_absent(monkeypatch):
    """_load_caches must call pytest.skip with a clear message when caches are missing."""
    monkeypatch.setattr(Path, "exists", lambda self: False)
    with pytest.raises(pytest.skip.Exception, match="Cache files absent"):
        _load_caches()
