# Handoff: LeJEPA SOXX — Project Complete (All Issues Closed)

## State

All 17 issues closed. No open work items. 116/116 tests passing on `main` (commit `d6dc406`).

**This is a wrap-up document.** There is no next session target. If work continues it will be a new initiative (POC-3, publication, or a new experiment).

---

## What Was Built (Full Arc)

The project implemented a self-supervised market stress detector for the SOXX semiconductor ETF using a JEPA-style latent prediction architecture, then ran two preregistered experiments against it.

### Phase 1 — Scaffold and Core Modules (Issues #1–#8)

| Issue | Module | Purpose |
|-------|--------|---------|
| #1 | `src/types.py` | Shared types: `Config`, `Window`, `WindowDataset`, `Verdict` |
| #2–#6 | `src/utils.py` | `detect_episodes()`, `make_windows()`, stub fixes |
| #7 | `src/data_pipeline.py` | SOXX/VIX/DGS10 ingestion, RobustScaler, sliding windows |
| #8 | `src/model.py`, `src/sigreg.py`, `src/training.py` | LeJEPA encoder/predictor MLP, SIGReg loss, AdamW trainer |

### Phase 2 — Calibration, Scoring, Evaluation (Issues #9–#13)

| Issue | Module | Purpose |
|-------|--------|---------|
| #9 | `src/calibration.py` | `MahalanobisCalibrator` (Ledoit-Wolf); `regime_filter_mask()` added later |
| #10 | `src/baselines.py` | `BaselineScorer` — RV20 z-score and VIX level z-score |
| #11 | `src/evaluation.py` | `Evaluator` — crash onset detection, 4-criterion pass/fail |
| #12 | `src/scoring.py` | `Scorer` — Mahalanobis distances, episode detection |
| #13 | `src/output.py` | `OutputWriter` — scores.csv, chart PNG, run_config.json |

### Phase 3 — Entry Points and Blind Results (Issue #14)

- `run_lejepa_soxx_poc.py` — single entry point, leakage-controlled end-to-end run
- Locked POC-1 blind result in `outputs/run_config.json`

### Phase 4 — POC-2 Regime Filter (Issues #15–#17)

| Issue | Deliverable |
|-------|-------------|
| #15 | Unit tests for `regime_filter_mask()` in `src/calibration.py` |
| #16 | `run_lejepa_soxx_poc2.py` — regime-filtered calibration entry point |
| #17 | `tests/test_diagnostics_scenarios.py` — regression tests for all 8 `diagnostics.py` scenarios |

Between #14 and #15: `diagnostics.py` was extended with 3 regime-filter scenarios (regime-p90/p95/p99) on top of the original 5 POC-1 scenarios; `src/calibration.py` gained `regime_filter_mask()`; `docs/PRD-POC2.md` documented the full POC-2 spec.

---

## Experiment Results

### POC-1 — Full 2019–2020 Calibration (Locked)

```
Threshold:          lejepa_threshold = 6.33  (p99 of calibration Mahalanobis scores)
Crash onset:        2021-12-10
LeJEPA first breach: 2022-11-04  (lead = -227 trading days)
RV20 first breach:  never
VIX first breach:   never
Verdict:            FAIL — all four criteria failed
```

Result locked in `outputs/run_config.json`. **Do not re-run `run_lejepa_soxx_poc.py`.**

### POC-2 — Regime-Filtered Calibration (Post-Hoc)

```
regime_pct = 95,  regime_threshold = 1.9689 (p95 of training-period RV20 z-scores)
n_calib_before = 466,  n_calib_after = 398
lejepa_threshold = 4.2945  (down from 6.33 — COVID windows removed from calibration tail)
LeJEPA first breach: 2022-01-06  (lead = -18 trading days after crash onset 2021-12-10)
Verdict:            FAIL — breach after crash onset
```

Both experiments produced FAIL verdicts. The regime filter lowered the threshold substantially (6.33 → 4.29) but was not sufficient to produce a leading signal.

---

## Repository Layout

```
run_lejepa_soxx_poc.py      POC-1 entry point (locked — do not re-run)
run_lejepa_soxx_poc2.py     POC-2 entry point
diagnostics.py              8-scenario post-hoc sensitivity analysis

src/
  types.py                  Config, Window, WindowDataset, Verdict
  data_pipeline.py          DataPipeline — ingestion, scaling, windowing
  model.py                  LeJEPAModel (encoder + predictor MLPs)
  sigreg.py                 SIGReg collapse-prevention loss
  training.py               Trainer — AdamW, early stopping
  calibration.py            MahalanobisCalibrator, regime_filter_mask
  baselines.py              BaselineScorer (RV20, VIX z-scores)
  scoring.py                Scorer — Mahalanobis distance, episodes
  evaluation.py             Evaluator — crash oracle, 4-criterion verdict
  output.py                 OutputWriter — CSV, chart, JSON

tests/                      116 tests total, all passing
  test_types.py
  test_utils.py
  test_data_pipeline.py
  test_model.py
  test_calibration.py       includes regime_filter_mask tests (#15)
  test_baselines.py
  test_scoring.py
  test_evaluation.py
  test_output.py
  test_main.py
  test_poc2_entry.py        _apply_regime_filter unit tests (#16)
  test_diagnostics_scenarios.py  8-scenario regression tests (#17)

docs/
  PRD.md                    Full POC-1 preregistered design spec
  PRD-POC2.md               POC-2 spec and testing decisions
  HANDOFF.md                This file

outputs/                    (mostly untracked)
  run_config.json           LOCKED POC-1 blind result
  scores.csv                LOCKED POC-1 test scores (z_pred_0..15 + soxx_close)
  soxx_lejepa_chart.png     POC-1 chart
  diag_calib_latents.npz    Cached calibration latents for diagnostics.py
  diag_model.pt             Cached model weights for diagnostics.py
```

---

## If Work Continues

No open issues. Potential directions (none preregistered):

- **POC-3** — latent dim sensitivity (8/16/32), seed stability sweep, or additional baselines (PCA, Isolation Forest, correlated ETFs) as documented in `docs/PRD-POC2.md` under "POC-2 scope (deferred)"
- **Publication / write-up** — the two FAIL results are a legitimate scientific finding; the regime filter reduced threshold by 32% but did not invert the verdict
- **Feature engineering** — the FAIL with both POC-1 and POC-2 suggests the latent structure learned from 2010–2016 may not generalise to the 2022 drawdown regime; sector rotation or cross-ETF context could be explored

If starting a new initiative, run `/grill-me` or `/to-prd` to preregister decisions before touching the locked outputs.

---

## Key Constraints (Do Not Violate)

1. **Never re-run `run_lejepa_soxx_poc.py`** — `outputs/run_config.json` is the locked blind result
2. **Split boundaries are hard** — no data from 2017+ may influence training; no 2021+ data may influence calibration
3. **No tuning against test period** — any threshold or feature changes must be preregistered before looking at test scores
