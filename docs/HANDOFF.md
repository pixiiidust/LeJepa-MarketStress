# Handoff: LeJEPA SOXX — POC-3 Sensitivity Experiments

## State

POC-1 and POC-2 complete (both FAIL). 185/185 tests passing on `main`. PRD written at `docs/PRD-POC3.md`. Issue #21 closed this session; one issue remains open.

**Next session target: issue #22** — Extended baselines entry point (all five signals). Now unblocked: #21 (`PCAScorer` / `IsolationForestScorer`) is complete.

---

## Open Issues (POC-3)

| # | Title | Blocked by | Status |
|---|-------|------------|--------|
| [#22](https://github.com/pixiiidust/LeJepa-MarketStress/issues/22) | Extended baselines entry point (all five signals) | — | open |

---

## What Changed This Session (issue #21)

**Commit:** `32a3eb8` — `feat(#21): PCA and Isolation Forest anomaly scorers`

- `src/poc3_baselines.py`: `PCAScorer` and `IsolationForestScorer`. Both flatten each window's `context_array` (30×6 → 180-dim), standardise with `RobustScaler` fitted on train windows. `PCAScorer` selects k components at cumulative explained variance ≥ 95%; anomaly score = L2 reconstruction residual. `IsolationForestScorer` uses `IsolationForest(n_estimators=200, random_state=42)`; anomaly score = `−score_samples(X)`. Both set threshold = p99 of calibration scores via `fit_calibration()`, and produce episode IDs via `detect_episodes`. Shared helpers `_flatten()` and `_dates()` keep the two classes symmetric.
- `tests/test_poc3_baselines.py`: 23 tests across 10 cycles — output columns, breach logic, OOD detection, p99 threshold, `fit_calibration` returns self, episode ID identity, idempotence — for both scorers.
- Test count: 162 → 185, all passing.

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
run_poc3_baselines.py        (to be created in #22) extended baselines entry point
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
  poc3_baselines.py          PCAScorer, IsolationForestScorer (issue #21, done)

tests/                       185 tests total, all passing
  test_types.py
  test_utils.py
  test_data_pipeline.py
  test_model.py
  test_calibration.py
  test_baselines.py
  test_scoring.py
  test_evaluation.py
  test_output.py
  test_main.py
  test_poc2_entry.py
  test_diagnostics_scenarios.py
  test_poc3_seed_sweep.py    20 tests for seed sweep runner (issue #19)
  test_poc3_latent_sweep.py  20 tests for latent dim sweep runner (issue #20)
  test_poc3_baselines.py     23 tests for PCAScorer / IsolationForestScorer (issue #21)
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

## Issue #22 Sketch (Extended Baselines Entry Point)

Create `run_poc3_baselines.py` that trains one model (seed=42, latent_dim=16) and evaluates all five signals — LeJEPA, RV20 z-score, VIX z-score, PCA reconstruction error, Isolation Forest — against the 2021–2022 test period using the same four pass/fail criteria as POC-1.

**Pipeline** (follow `run_poc3_seed_sweep.py` as structural template):
1. Load data via `DataPipeline`; build `WindowDataset`.
2. Train `LeJEPAModel` with `Trainer` (seed=42).
3. Calibrate `MahalanobisCalibrator` → LeJEPA scores via `Scorer`.
4. Fit and calibrate `BaselineScorer` → RV20 + VIX scores.
5. Fit and calibrate `PCAScorer` → PCA scores.
6. Fit and calibrate `IsolationForestScorer` → IF scores.
7. Evaluate each signal independently against crash oracle via `Evaluator`.
8. Write outputs to `outputs/poc3/`.

**Outputs** (all inside `outputs/poc3/`):
- `baselines_results.csv` — 5 rows, one per signal: `signal_name`, `first_breach`, `lead_days`, `episodes_2021`, `alert_days_2021`, `all_pass`
- `baselines_chart.png` — two-panel: SOXX close (top), all five score series with thresholds and crash onset line (bottom)
- One-line result appended to `poc3_summary.md`

**Test file**: `tests/test_poc3_entry.py`. Key tests to target:
- `_build_signal_row(name, verdict)` returns dict with required keys
- `_print_console_summary(rows)` prints a readable five-signal table
- `_append_poc3_summary(outdir, rows)` appends a line containing `"baselines"` to `poc3_summary.md`
- `run_baselines(outdir)` integration test (all heavy deps monkeypatched): returns DataFrame with 5 rows, writes `baselines_results.csv`, writes `baselines_chart.png`, appends to `poc3_summary.md`

Pattern to follow: `run_poc3_seed_sweep.py` for structure; `tests/test_poc3_seed_sweep.py` for monkeypatching style; `src/poc3_baselines.py` for how to call `PCAScorer` / `IsolationForestScorer`.

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

- `/tdd` — implement issue #22 with red-green-refactor (start with red tests, then green, then refactor)
- `/handoff` with args: `save to docs/HANDOFF.md, target next session for the next open issue with no open blockers`
