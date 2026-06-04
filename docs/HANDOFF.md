# Handoff: LeJEPA SOXX — POC-3 Complete

## State

POC-1, POC-2, and POC-3 all complete. **205/205 tests passing on `main`.** No open GitHub issues remain.

**POC-3 is fully implemented.** All three sensitivity entry points exist and are tested; the researcher has not yet run them against real data.

---

## What Changed This Session (issue #22)

**Commit:** `b3166f7` — `feat(#22): extended baselines entry point — all five signals`

- `run_poc3_baselines.py`: trains one model (seed=42, latent_dim=16), scores all five signals — LeJEPA, RV20, VIX, PCAScorer, IsolationForestScorer — evaluates each independently against the crash oracle (each signal treated as "LeJEPA", compared against RV20/VIX). Writes `baselines_results.csv` (5 rows), `baselines_chart.png` (two-panel), and appends to `poc3_summary.md`.
- `tests/test_poc3_entry.py`: 20 tests across 4 cycles — `_build_signal_row` keys, `_print_console_summary` output, `_append_poc3_summary` append/content, `run_baselines` integration (all heavy deps monkeypatched, Evaluator called exactly 5 times).
- Test count: 185 → 205, all passing.

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

### POC-3 — Sensitivity Experiments (Entry Points Ready, Not Yet Run)

All three entry points are implemented and tested. The researcher runs them manually:

| Entry point | Output | Status |
|-------------|--------|--------|
| `run_poc3_seed_sweep.py` | `outputs/poc3/seed_sweep_results.csv` | Ready |
| `run_poc3_latent_sweep.py` | `outputs/poc3/latent_dim_results.csv` | Ready |
| `run_poc3_baselines.py` | `outputs/poc3/baselines_results.csv` + `baselines_chart.png` | Ready |

---

## Next Session

No open GitHub issues exist. The next session depends on what the researcher decides after reviewing POC-3 results.

**Likely paths:**
1. **Results interpretation**: after the researcher runs the three POC-3 entry points, review `outputs/poc3/poc3_summary.md` and the three CSVs, and write a summary finding (likely to `docs/PRD-POC3.md` or a new `docs/FINDINGS.md`).
2. **New issue creation**: if the researcher identifies further experiments (POC-4: rolling calibration, additional assets, attention encoder), create new GitHub issues and continue the `/tdd` cycle.
3. **Wrap-up**: if POC-3 results confirm the POC-1 FAIL verdict is structural, write a final conclusion.

**Out-of-scope items from `docs/PRD-POC3.md`** that could become next issues:
- Rolling / walk-forward calibration
- Additional assets (QQQ, SPY, SMH)
- Recurrent or attention-based encoder architectures
- Composite regime filter

---

## Repository Layout

```
run_lejepa_soxx_poc.py       POC-1 entry point (LOCKED — do not re-run)
run_lejepa_soxx_poc2.py      POC-2 entry point
run_poc3_seed_sweep.py       POC-3 seed sweep (issue #19, done)
run_poc3_latent_sweep.py     POC-3 latent dim sweep (issue #20, done)
run_poc3_baselines.py        POC-3 extended baselines (issue #22, done)
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

tests/                       205 tests total, all passing
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
  test_poc3_seed_sweep.py    20 tests (issue #19)
  test_poc3_latent_sweep.py  20 tests (issue #20)
  test_poc3_baselines.py     23 tests for PCAScorer / IsolationForestScorer (issue #21)
  test_poc3_entry.py         20 tests for run_poc3_baselines (issue #22)

docs/
  PRD.md                     POC-1 preregistered spec + results
  PRD-POC2.md                POC-2 spec and testing decisions
  PRD-POC3.md                POC-3 sensitivity experiment spec (all user stories + implementation decisions)
  HANDOFF.md                 This file

outputs/                     (mostly untracked)
  run_config.json            LOCKED POC-1 blind result
  scores.csv                 LOCKED POC-1 test scores
  soxx_lejepa_chart.png      POC-1 chart
  diag_calib_latents.npz     Cached calibration latents for diagnostics.py
  diag_model.pt              Cached model weights for diagnostics.py
  poc3/                      all POC-3 outputs go here (written when researcher runs entry points)
```

---

## Key Constraints (Do Not Violate)

1. **Never re-run `run_lejepa_soxx_poc.py`** — `outputs/run_config.json` is the locked blind result
2. **Split boundaries are hard** — train 2010–2016, calibration 2019–2020, test 2021+; no data from later splits may influence earlier-split fitting
3. **No tuning against test period** — POC-3 is post-hoc sensitivity analysis; test period was observed in POC-1
4. **All POC-3 output goes to `outputs/poc3/`** — no other files under `outputs/` may be written by any POC-3 entry point
5. **Do NOT run any entry point scripts** (`run_poc3_*.py`, `run_lejepa_soxx_poc*.py`) — the researcher runs these manually

---

## Suggested Skills

- `/tdd` — if new issues are created for POC-4 or further analysis
- `/handoff` with args: `save to docs/HANDOFF.md, target next session for the next open issue with no open blockers`
