# Handoff: LeJEPA SOXX — POC-3 Sensitivity Experiments

## State

POC-1 and POC-2 complete (both FAIL). 162/162 tests passing on `main`. PRD written at `docs/PRD-POC3.md`. Issue #20 closed this session; two issues remain open.

**Next session target: issue #21** — PCA and Isolation Forest anomaly scorers (unblocked, independent of any other open issue).

---

## Open Issues (POC-3)

| # | Title | Blocked by | Status |
|---|-------|------------|--------|
| [#21](https://github.com/pixiiidust/LeJepa-MarketStress/issues/21) | PCA and Isolation Forest anomaly scorers | — | open |
| [#22](https://github.com/pixiiidust/LeJepa-MarketStress/issues/22) | Extended baselines entry point (all five signals) | #21 | open |

---

## What Changed This Session (issue #20)

**Commit:** `6ae38f5` — `feat(#20): latent dim sweep runner for dims 8, 16, 32 (seed=42)`

- `run_poc3_latent_sweep.py`: full POC-1 pipeline per latent dim variant (8, 16, 32). Data loaded once via `DataPipeline`; each dim independently trains a fresh `LeJEPAModel` with `Config(latent_dim=d, encoder_dims=(180,64,d), predictor_dims=(d,2*d,d))`, seed fixed at 42. Calibration, scoring, and evaluation follow the same pattern as `run_poc3_seed_sweep.py`. Warns if `latent_std_min < 0.05`. Writes `outputs/poc3/latent_dim_results.csv`; appends one-line summary to `outputs/poc3/poc3_summary.md`. No writes outside `outputs/poc3/`.
- `tests/test_poc3_latent_sweep.py`: 20 tests across 5 cycles — `_compute_latent_std_min`, `_build_dim_row`, `_print_console_summary`, `_append_poc3_summary`, and `run_latent_sweep` integration (all heavy deps monkeypatched).
- Test count: 142 → 162, all passing.

---

## Experiment Results (Historical)

### POC-1 — Full 2019–2020 Calibration (Locked)

```
Threshold:            lejepa_threshold = 6.33  (p99 of calibration Mahalanobis scores)
Crash onset:          2021-12-10
LeJEPA first breach:  2022-11-04  (lead = -227 trading days)
RV20 first breach:    never
VIX first breach:     never
Verdict:              FAIL — all four criteria failed
```

Result locked in `outputs/run_config.json`. **Do not re-run `run_lejepa_soxx_poc.py`.**

### POC-2 — Regime-Filtered Calibration (Post-Hoc)

```
regime_pct = 95,  regime_threshold = 1.9689
lejepa_threshold = 4.2945  (down from 6.33)
LeJEPA first breach: 2022-01-06  (lead = -18 trading days after crash onset 2021-12-10)
Verdict:             FAIL — breach after crash onset
```

---

## Repository Layout

```
run_lejepa_soxx_poc.py       POC-1 entry point (LOCKED — do not re-run)
run_lejepa_soxx_poc2.py      POC-2 entry point
run_poc3_seed_sweep.py       POC-3 seed sweep (issue #19, done)
run_poc3_latent_sweep.py     POC-3 latent dim sweep (issue #20, done)
diagnostics.py               8-scenario post-hoc sensitivity analysis

src/
  types.py                   Config, Window, WindowDataset, Verdict
  data_pipeline.py           DataPipeline — ingestion, scaling, windowing
  model.py                   LeJEPAModel (builds from config.encoder_dims/predictor_dims)
  sigreg.py                  SIGReg collapse-prevention loss
  training.py                Trainer — AdamW, early stopping; CollapseError
  calibration.py             MahalanobisCalibrator, regime_filter_mask
  baselines.py               BaselineScorer (RV20, VIX z-scores)
  scoring.py                 Scorer — Mahalanobis distance, episodes (latent cols config-driven)
  evaluation.py              Evaluator — crash oracle, 4-criterion verdict
  output.py                  OutputWriter — CSV, chart, JSON
  poc3_baselines.py          (to be created in #21) PCAScorer, IsolationForestScorer

tests/                       162 tests total, all passing
  test_types.py
  test_utils.py
  test_data_pipeline.py
  test_model.py              dim-8 and dim-32 shape tests added (#18)
  test_calibration.py
  test_baselines.py
  test_scoring.py            latent column count tests added (#18)
  test_evaluation.py
  test_output.py
  test_main.py
  test_poc2_entry.py
  test_diagnostics_scenarios.py
  test_poc3_seed_sweep.py    20 tests for seed sweep runner (issue #19)
  test_poc3_latent_sweep.py  20 tests for latent dim sweep runner (issue #20)
  test_poc3_baselines.py     (to be created in #21)
  test_poc3_entry.py         (to be created in #22)

docs/
  PRD.md                     POC-1 preregistered spec + results
  PRD-POC2.md                POC-2 spec and testing decisions
  PRD-POC3.md                POC-3 sensitivity experiment spec
  HANDOFF.md                 This file

outputs/                     (mostly untracked)
  run_config.json            LOCKED POC-1 blind result
  scores.csv                 LOCKED POC-1 test scores
  soxx_lejepa_chart.png      POC-1 chart
  diag_calib_latents.npz     Cached calibration latents for diagnostics.py
  diag_model.pt              Cached model weights for diagnostics.py
  poc3/                      all POC-3 outputs go here
```

---

## Issue #21 Sketch (PCA and Isolation Forest Anomaly Scorers)

Create `src/poc3_baselines.py` with two classes: `PCAScorer` and `IsolationForestScorer`. Both follow the same fit/calibrate/score interface as `src/baselines.py:BaselineScorer`, but accept `list[Window]` (not DataFrames) and operate on flattened 180-dim context arrays.

**Input preparation** (shared by both scorers):
- Flatten each window's `context_array` (shape 30×6) to 180-dim vector.
- Standardise using a `RobustScaler` fitted on the training windows (same leakage discipline as the data pipeline).

**PCAScorer**:
- Constructor: `PCAScorer(train_windows: list[Window], config: Config)`
- Fits `sklearn.decomposition.PCA` on training vectors; selects smallest k where cumulative explained variance ≥ 95%.
- `fit_calibration(calib_windows) → PCAScorer`: sets `pca_threshold` = p99 of calibration reconstruction L2 errors.
- `score(windows) → pd.DataFrame`: columns `pca_score`, `pca_threshold`, `pca_breach`, `pca_episode_id`. Episode detection uses `config.episode_cooldown_days` (same as all signals).

**IsolationForestScorer**:
- Constructor: `IsolationForestScorer(train_windows: list[Window], config: Config)`
- Fits `IsolationForest(n_estimators=200, random_state=42)` on training vectors.
- Anomaly score = `−score_samples(X)` (negated so higher = more anomalous, directionally consistent).
- `fit_calibration(calib_windows) → IsolationForestScorer`: sets `if_threshold` = p99 of calibration scores.
- `score(windows) → pd.DataFrame`: columns `if_score`, `if_threshold`, `if_breach`, `if_episode_id`.

**Test file**: `tests/test_poc3_baselines.py`. Key tests to target:
- `PCAScorer.score()` returns columns `pca_score`, `pca_threshold`, `pca_breach`, `pca_episode_id`.
- `pca_breach` is True when `pca_score > pca_threshold`.
- Reconstruction error near-zero for in-distribution windows, non-trivially positive for out-of-distribution windows.
- `fit_calibration()` sets threshold = p99 of calibration errors.
- `IsolationForestScorer.score()` returns columns `if_score`, `if_threshold`, `if_breach`, `if_episode_id`.
- Out-of-distribution window scores above threshold from clean calibration set.

Pattern to follow: `src/baselines.py` for scorer structure; `tests/test_baselines.py` for test style; `tests/test_poc3_seed_sweep.py` for how to use synthetic `list[Window]` data.

---

## POC-3 Workflow Per Issue

For each issue: run `/tdd` to implement with red-green-refactor, then push, close the issue, then run `/handoff` with args: save to `docs/HANDOFF.md`, target next session for the next open issue with no open blockers.

---

## Key Constraints (Do Not Violate)

1. **Never re-run `run_lejepa_soxx_poc.py`** — `outputs/run_config.json` is the locked blind result
2. **Split boundaries are hard** — train 2010–2016, calibration 2019–2020, test 2021+; no data from later splits may influence earlier-split fitting
3. **No tuning against test period** — POC-3 is post-hoc sensitivity analysis; test period was observed in POC-1
4. **All POC-3 output goes to `outputs/poc3/`** — no other files under `outputs/` may be written by any POC-3 entry point
5. **Do NOT run any entry point scripts** (run_poc3_*.py, run_lejepa_soxx_poc*.py) — the researcher runs these manually

---

## Suggested Skills

- `/tdd` — implement the next issue with red-green-refactor (start with red tests, then green, then refactor)
- `/handoff` with args: `save to docs/HANDOFF.md, target next session for the next open issue with no open blockers`
