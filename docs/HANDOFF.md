# Handoff: LeJEPA SOXX — POC-2 Issue #17 (diagnostics regime-filter regression tests)

## State

Issue #16 complete and pushed (`2dddbb4` on `main`). Next session targets **issue #17** only.

**Do not re-run `run_lejepa_soxx_poc.py`.** The locked POC-1 blind result is in `outputs/run_config.json`.

---

## What Was Done This Session

| Change | Detail |
|--------|--------|
| `run_lejepa_soxx_poc2.py` | Full POC-2 entry point — regime-filtered calibration (466 → 398 windows, threshold 6.33 → 4.29). FAIL verdict (breach 18 days after onset). |
| `tests/test_poc2_entry.py` | 4 unit tests for `_apply_regime_filter` — window mask, daily mask, min-rows assertion, before/after logging |
| `outputs/poc2_calib_latents.npz` | New cache (z_target, z_pred, z_context, dates_ns — all 3 arrays; untracked) |
| `outputs/scores_poc2.csv` | POC-2 test scores (untracked) |
| `outputs/run_config_poc2.json` | POC-2 config with regime fields (untracked) |
| `outputs/soxx_lejepa_poc2_chart.png` | POC-2 chart (untracked) |
| GitHub | Issue #16 closed |
| git | Committed and pushed — commit `2dddbb4` on `main` |

111/111 tests passing.

### POC-2 regime filter result (for reference)
- `regime_pct = 95`, `regime_threshold = 1.9689` (p95 of training-period RV20 z-scores)
- `n_calib_before = 466`, `n_calib_after = 398`
- `lejepa_threshold = 4.2945` (down from POC-1 6.33 — COVID days removed from calibration tail)
- Verdict: **FAIL** (LeJEPA first breach 2022-01-06, crash onset 2021-12-10, lead = -18 days)

---

## Next Session Target: Issue #17

**URL:** https://github.com/pixiiidust/LeJepa-MarketStress/issues/17  
**Title:** Regression tests for diagnostics.py regime-filter scenarios  
**Skill to invoke:** `/tdd`

### What to build

A new test file (`tests/test_diagnostics_scenarios.py`) that verifies the three regime-filter scenarios added to `diagnostics.py` behave correctly, and that the five original POC-1 scenarios are unaffected.

Tests **must skip gracefully** (pytest.skip) when cache files are absent. They must **not** re-train the model.

### How the tests should work

`diagnostics.py` already has the full 8-scenario pipeline implemented. The test can import `SCENARIOS`, `regime_filter_mask`, `BaselineScorer`, `MahalanobisCalibrator`, and the `_filter_latents` / `_filter_df` helpers directly from `diagnostics.py`, load the caches, and run the calibration logic against them.

**Required caches at test time** (both must exist or tests skip):
- `outputs/diag_calib_latents.npz` — keys: `z_target` (466, 16), `z_pred` (466, 16), `dates_ns` (int64 ns timestamps). No `z_context` — irrelevant for these tests (only z_target and z_pred are needed for Mahalanobis calibration in the diagnostics scenarios).
- `outputs/scores.csv` — 503 rows (2021-2022 test period), first row is a comment; `skiprows=1`. Columns include `z_pred_0..z_pred_15` (raw latent vectors) used to re-score test period.

### Acceptance criteria (from issue)

1. `regime-p95` scenario → strictly fewer `n_calib` than `original`
2. `regime-p95` scenario → strictly lower `lj_thr` than `original` (COVID removed from tail)
3. Monotonicity: `regime-p90.n_calib < regime-p95.n_calib < regime-p99.n_calib`
4. Five original POC-1 scenarios (`original`, `pre-COVID p99`, `excl Feb-Apr 2020`, `p97.5`, `p99.5`) produce identical `n_calib` and `lj_thr` when run via the updated code path — i.e., no regression. (Run them twice and compare, or compare to values derived from a single reference run.)
5. Tests skip with a clear message if cache absent
6. Tests do not re-train or re-collect latents

### Key diagnostic internals to reuse

From `diagnostics.py` (all importable):
- `SCENARIOS` — list of 8 dicts with keys `label`, `cutoff`, `exclude`, `pct`, `regime_pct`
- `_filter_latents(z_target, z_pred, dates, cutoff, exclude, regime_mask=None)` → `(zt_sub, zp_sub)`
- `_filter_df(df, cutoff, exclude, regime_mask=None)` → `pd.DataFrame`
- The regime mask computation: `regime_filter_mask(calib_dates, rv20_all, rv20_mean, rv20_std, threshold)`
- `regime_thresholds` and `regime_window_masks` are computed inside `main()` — the test must replicate this logic inline (a few lines: instantiate `BaselineScorer(train_df, config)`, compute percentiles of training RV20 z-scores, call `regime_filter_mask`)

### Data sources for tests

```python
import numpy as np, pandas as pd
from src.calibration import MahalanobisCalibrator, regime_filter_mask
from src.baselines import BaselineScorer
from src.data_pipeline import DataPipeline
from src.types import Config
from diagnostics import SCENARIOS, _filter_latents, _filter_df

config = Config()
pipeline = DataPipeline(config)  # reads from local cache — fast, no network
dataset = pipeline.build()

d = np.load("outputs/diag_calib_latents.npz")
z_target_full = d["z_target"]   # (466, 16)
z_pred_full   = d["z_pred"]     # (466, 16)
calib_dates   = pd.DatetimeIndex(pd.to_datetime(d["dates_ns"], unit="ns"))

scores_csv = pd.read_csv("outputs/scores.csv", skiprows=1, parse_dates=["date"], index_col="date")
test_zpred = scores_csv[[f"z_pred_{j}" for j in range(16)]].values  # (503, 16)

# Training baseline (for regime threshold + n_calib reference)
train_df = ...  # built from pipeline.raw_df / pipeline.features_df for 2010-2016
calib_df_full = ...  # 2019-2020 daily rv20+vix
rv20_all = pipeline.features_df["rv20"]
bs_train = BaselineScorer(train_df, config)
```

Then per scenario: apply `_filter_latents` / `_filter_df`, fit `MahalanobisCalibrator`, call `set_threshold`. Compare `n_calib` (= len of filtered z_pred) and `lj_thr` (= `cal.threshold`) across scenarios.

---

## Session Protocol

Each session targets **one issue only**. When issue is complete:
1. Push changes to `origin/main`
2. Run `/handoff` with args: `save to docs/HANDOFF.md, target next session for issue #18 or wrap-up if no more open issues`

Issue order: ~~#15~~ → ~~#16~~ → **#17**

---

## Issue Queue

| Issue | Title | Blocked by |
|-------|-------|-----------|
| ~~[#15](https://github.com/pixiiidust/LeJepa-MarketStress/issues/15)~~ | ~~Unit tests for `regime_filter_mask()`~~ | Done |
| ~~[#16](https://github.com/pixiiidust/LeJepa-MarketStress/issues/16)~~ | ~~`run_lejepa_soxx_poc2.py` entry point~~ | Done |
| [#17](https://github.com/pixiiidust/LeJepa-MarketStress/issues/17) | Diagnostics regime-filter regression tests | None (cache warm) |

---

## Key Files

| Path | Purpose |
|------|---------|
| `diagnostics.py` | 8 scenarios including 3 new regime-filter ones; `_filter_latents`, `_filter_df` helpers importable |
| `src/calibration.py` | `regime_filter_mask()` at line 30 |
| `src/baselines.py` | `BaselineScorer` — `rv20_mean`, `rv20_std` are public attrs |
| `outputs/diag_calib_latents.npz` | Cached latents — z_target, z_pred, dates_ns (untracked, must exist) |
| `outputs/scores.csv` | Locked POC-1 test scores — z_pred_0..15 columns usable for re-scoring |
| `outputs/run_config.json` | Locked POC-1 result — do not modify |
| `run_lejepa_soxx_poc2.py` | POC-2 entry point (just built) — `_apply_regime_filter` is the core new helper |
| `tests/test_poc2_entry.py` | 4 unit tests for `_apply_regime_filter` — good reference for test style |
| `tests/test_calibration.py` | 10 tests for `MahalanobisCalibrator` + `regime_filter_mask` — good style reference |
| `docs/PRD-POC2.md` | Full POC-2 spec — testing decisions section has additional guidance |

---

## Suggested Skills

- `/tdd` — primary skill for #17; write one scenario test at a time (tracer bullet per acceptance criterion)
- `/handoff` — run when #17 is done: `save to docs/HANDOFF.md, target next session for issue #18 or wrap-up if no more open issues`
