# Handoff: LeJEPA SOXX — POC-3 Sensitivity Experiments

## State

POC-1 and POC-2 complete (both FAIL). 116/116 tests passing on `main`. PRD written at `docs/PRD-POC3.md`. Five issues open and ready for agent work.

**Next session target: issue #18** — the foundation issue that unblocks both sweep runners.

---

## Open Issues (POC-3)

| # | Title | Blocked by | Status |
|---|-------|------------|--------|
| [#18](https://github.com/pixiiidust/LeJepa-MarketStress/issues/18) | Parameterise `LeJEPAModel` from Config + fix `Scorer` latent column count | — | open |
| [#19](https://github.com/pixiiidust/LeJepa-MarketStress/issues/19) | Seed stability sweep runner | #18 | open |
| [#20](https://github.com/pixiiidust/LeJepa-MarketStress/issues/20) | Latent dim sweep runner | #18 | open |
| [#21](https://github.com/pixiiidust/LeJepa-MarketStress/issues/21) | PCA and Isolation Forest anomaly scorers | — | open |
| [#22](https://github.com/pixiiidust/LeJepa-MarketStress/issues/22) | Extended baselines entry point (all five signals) | #21 | open |

Parallel starts: #18 and #21 have no blockers and can run simultaneously.

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
diagnostics.py               8-scenario post-hoc sensitivity analysis

src/
  types.py                   Config, Window, WindowDataset, Verdict
  data_pipeline.py           DataPipeline — ingestion, scaling, windowing
  model.py                   LeJEPAModel (encoder + predictor MLPs) ← #18 modifies this
  sigreg.py                  SIGReg collapse-prevention loss
  training.py                Trainer — AdamW, early stopping
  calibration.py             MahalanobisCalibrator, regime_filter_mask
  baselines.py               BaselineScorer (RV20, VIX z-scores)
  scoring.py                 Scorer — Mahalanobis distance, episodes ← #18 modifies this
  evaluation.py              Evaluator — crash oracle, 4-criterion verdict
  output.py                  OutputWriter — CSV, chart, JSON
  poc3_baselines.py          (to be created in #21) PCAScorer, IsolationForestScorer

tests/                       116 tests total, all passing
  test_types.py
  test_utils.py
  test_data_pipeline.py
  test_model.py              ← #18 adds dim-8 and dim-32 shape tests
  test_calibration.py
  test_baselines.py
  test_scoring.py            ← #18 adds latent column count tests
  test_evaluation.py
  test_output.py
  test_main.py
  test_poc2_entry.py
  test_diagnostics_scenarios.py
  test_poc3_*.py             (to be created in #19, #20, #21, #22)

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
  poc3/                      (to be created) all POC-3 outputs go here
```

---

## POC-3 Workflow Per Issue

For each issue: run `/tdd` to implement with red-green-refactor, then push, close the issue, then run `/handoff` with args: save to `docs/HANDOFF.md`, target next session for the next open issue.

---

## Key Constraints (Do Not Violate)

1. **Never re-run `run_lejepa_soxx_poc.py`** — `outputs/run_config.json` is the locked blind result
2. **Split boundaries are hard** — train 2010–2016, calibration 2019–2020, test 2021+; no data from later splits may influence earlier-split fitting
3. **No tuning against test period** — POC-3 is post-hoc sensitivity analysis; test period was observed in POC-1
4. **All POC-3 output goes to `outputs/poc3/`** — no other files under `outputs/` may be written by any POC-3 entry point
