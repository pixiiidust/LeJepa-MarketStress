# PRD: LeJEPA SOXX Market Stress POC-2

## Problem Statement

POC-1 returned FAIL because the 2019–2020 calibration window included the COVID-19 volatility regime. The 99th-percentile thresholds were set at levels only reachable during a COVID-scale shock: LeJEPA reached 91.7% of its threshold but did not breach until 227 trading days after crash onset. Neither RV20 nor VIX fired at all.

The root cause is not the model architecture, the latent dimension, or the threshold percentile. It is the calibration window design. A calibration window that includes any COVID-era regime inflation will produce thresholds unreachable in a normal-stress environment. Excluding only the acute crash (Feb–Apr 2020) does not fix this: the LeJEPA threshold barely moves (6.33 → 6.43), showing that COVID-adjacent volatility outside the acute crash dominated the calibration tail.

A researcher needs a principled, leakage-clean way to exclude extreme-regime days from the calibration window without resorting to hand-selected date ranges.

---

## Solution

A second entry point (`run_lejepa_soxx_poc2.py`) that runs the identical pipeline as POC-1 but applies a **regime-filtered calibration window** before fitting the Mahalanobis distribution and baseline thresholds.

The filter excludes any calibration day where the RV20 z-score — normalised using 2010–2016 training statistics — exceeds the 95th percentile of training-period RV20 z-scores. The exclusion threshold is derived entirely from training data; no calibration or test data informs the filter criterion.

A pure helper function `regime_filter_mask()` encapsulates the filter logic and is shared between the entry point and `diagnostics.py` sensitivity scenarios. The `diagnostics.py` SCENARIOS system is extended to support a `regime_pct` key so that the 90th-, 95th-, and 99th-percentile filter variants can be compared in the same post-hoc table that was used for POC-1 diagnostics.

All other design decisions — model architecture, training protocol, scoring hierarchy, pass/fail criteria, test period — are inherited unchanged from POC-1.

---

## User Stories

1. As a researcher, I want the regime filter to use only 2010–2016 training statistics, so that no calibration or test data informs the exclusion criterion.
2. As a researcher, I want the regime-filter threshold to be computed as the 95th percentile of training-period RV20 z-scores, so that the cutoff is grounded in the normal-stress distribution rather than a hand-picked date.
3. As a researcher, I want the RV20 z-score used for filtering to be normalised with the same training-period mean and std as the BaselineScorer, so that the filter and the baseline signal share identical leakage discipline.
4. As a researcher, I want the regime filter to be implemented as a standalone pure function that accepts dates, raw RV20 values, training-period normalisation constants, and a percentile threshold, so that the filter logic can be unit-tested in isolation and reused across the entry point and diagnostics.
5. As a researcher, I want `regime_filter_mask()` to return a boolean array where `True` means "keep this day", so that the calling code can apply the mask to any aligned array or DataFrame without additional logic.
6. As a researcher, I want the filter function to accept any DatetimeIndex — both window dates and raw trading-day indices — so that the same function can filter both the latent arrays (466 rows, one per window) and the baseline calibration DataFrame (505 rows, one per trading day) without duplicating logic.
7. As a researcher, I want the regime-filter percentile threshold computed once from training data before any calibration data is touched, so that the threshold cannot be influenced by calibration-period volatility levels.
8. As a researcher, I want the filtered latent arrays and filtered baseline DataFrame to use the same filter criterion applied to their respective date indices, so that the Mahalanobis distribution and the baseline thresholds are calibrated against the same regime.
9. As a researcher, I want the number of calibration samples retained after filtering logged to the console, so that I can verify the filter is removing the expected COVID-era days.
10. As a researcher, I want the POC-2 entry point to reuse the trained model from POC-1 if a cached checkpoint exists, so that I do not need to retrain when the only change is the calibration window.
11. As a researcher, I want the POC-2 entry point to reuse the cached calibration latents from POC-1 if available, so that the expensive forward-pass collection step is not repeated.
12. As a researcher, I want the regime filter applied to calibration latent arrays after collection, not during collection, so that the full cache remains reusable for diagnostics at any filter level.
13. As a researcher, I want the filtered z_target, z_pred, and z_context latent arrays to be passed to `MahalanobisCalibrator.fit()` and `set_threshold()`, so that the Mahalanobis distribution and threshold are calibrated on the filtered sample only.
14. As a researcher, I want the filtered baseline DataFrame to be passed to `BaselineScorer.fit_calibration()`, so that the RV20 and VIX thresholds are also calibrated on the regime-filtered window.
15. As a researcher, I want all test-period scoring, episode detection, and pass/fail evaluation to proceed identically to POC-1 after calibration, so that the only experimental variable is the calibration window design.
16. As a researcher, I want the POC-2 entry point to assert that the filtered calibration sample has at least 50 rows, so that the Mahalanobis estimator receives enough samples to produce a well-conditioned covariance.
17. As a researcher, I want the POC-2 regime-filter threshold value stored in `run_config_poc2.json`, so that any run is fully reproducible and the exact filter threshold is preserved alongside all other POC-1 hyperparameters.
18. As a researcher, I want `run_config_poc2.json` to record the regime_pct parameter (95), the training-period rv20_mean and rv20_std, the computed regime_threshold, and the number of calibration samples before and after filtering, so that the calibration window design is fully auditable from the config alone.
19. As a researcher, I want `run_config_poc2.json` to record the same fields as `run_config.json` for all inherited parameters (seed, architecture, splits, scoring, pass/fail criteria), so that a reviewer does not need to cross-reference two documents.
20. As a researcher, I want the POC-2 output scores written to `scores_poc2.csv` with the same column contract as `scores.csv`, so that the same downstream analysis tools can be applied without modification.
21. As a researcher, I want the POC-2 chart written to `soxx_lejepa_poc2_chart.png` with the same two-panel layout as POC-1, so that timing relationships are visually inspectable in the same format.
22. As a researcher, I want the console summary to print the regime-filter parameters alongside the threshold and lead-time results, so that the calibration window design is immediately visible in the run output.
23. As a researcher, I want the pass/fail verdict in the POC-2 console summary to state explicitly that the test period has been observed in POC-1, so that the pre-registration status of the result is transparent.
24. As a researcher, I want the four pass/fail criteria applied to POC-2 results to be identical to POC-1, so that the comparison is apples-to-apples.
25. As a researcher, I want the crash onset date computed identically to POC-1 (first test day T with a full 60-day forward window where the drawdown exceeds 20%), so that the target event definition is not adjusted between experiments.
26. As a researcher, I want the diagnostics script extended to support a `regime_pct` key in SCENARIOS, so that the 90th-, 95th-, and 99th-percentile filter variants appear in the same post-hoc comparison table.
27. As a researcher, I want the diagnostics script to compute the regime-filter masks for all three variants before the scenario loop, so that training-period normalisation statistics are instantiated once and reused.
28. As a researcher, I want the diagnostics scenario table to show the number of calibration samples retained for each regime-filter variant alongside the existing N_cal column, so that the tradeoff between filter strictness and sample size is immediately visible.
29. As a researcher, I want the `_filter_latents` and `_filter_df` helpers to accept a pre-computed boolean mask as an optional parameter, so that regime-pct scenarios can pass the mask in without threading RV20 stats through the filter functions.
30. As a researcher, I want the diagnostics SCENARIOS list to include `regime-p95`, `regime-p90`, and `regime-p99` entries with the same pct=99.0 threshold percentile as the original, so that the only variable is the calibration window design.
31. As a researcher, I want the diagnostics chart to include a regime-p95 score series alongside the original, so that the two calibration designs are visually comparable.
32. As a researcher, I want a composite-filter variant documented in the PRD as a future sensitivity check (exclude days where max(RV20 z-score, VIX z-score) > 95th pct of training scores), so that the design space is captured even if it is not implemented in POC-2.

---

## Implementation Decisions

### New Function: `regime_filter_mask()`

Add a module-level function in the calibration module alongside `MahalanobisCalibrator`. It accepts a DatetimeIndex, a raw RV20 series, training-period normalisation constants, and a z-score percentile threshold. It returns a boolean numpy array (`True` = keep). The function reindexes the RV20 series to the provided date index before computing z-scores, so it handles both window-date indices (466 rows) and trading-day indices (505 rows) transparently.

The function has no side effects and no dependency on model state. It is a pure transformation from inputs to a boolean mask, making it independently testable.

### Entry Point: `run_lejepa_soxx_poc2.py`

The POC-2 entry point is structurally identical to `run_lejepa_soxx_poc.py`. The regime-filter logic is inserted in one place: after calibration latents are collected and before `MahalanobisCalibrator.fit()` is called. The sequence is:

1. Instantiate a temporary `BaselineScorer` on training data to access `rv20_mean` and `rv20_std` (these are already public attributes — no changes to `BaselineScorer` required).
2. Compute training-period RV20 z-scores and derive the 95th-percentile threshold.
3. Apply `regime_filter_mask()` twice — once to the window date index (for latent arrays), once to the raw trading-day index (for the baseline calibration DataFrame).
4. Subset all three latent arrays (`z_target`, `z_pred`, `z_context`) using the window mask.
5. Subset `calib_df` using the trading-day mask.
6. Assert that the filtered sample has ≥ 50 rows.
7. Proceed with `MahalanobisCalibrator.fit()`, `set_threshold()`, `BaselineScorer.fit_calibration()`, and all downstream steps identically to POC-1.

No changes are required to `MahalanobisCalibrator`, `BaselineScorer`, `DataPipeline`, `Evaluator`, or `Scorer`. They already accept filtered inputs.

### Outputs

POC-2 produces separate output files to preserve the locked POC-1 result:

| File | Purpose |
|------|---------|
| `outputs/scores_poc2.csv` | Same column contract as `scores.csv` |
| `outputs/soxx_lejepa_poc2_chart.png` | Same two-panel layout as POC-1 chart |
| `outputs/run_config_poc2.json` | Same fields as `run_config.json` plus regime-filter parameters |

The locked POC-1 files (`scores.csv`, `soxx_lejepa_chart.png`, `run_config.json`) are never overwritten.

### Diagnostics Extension

The diagnostics SCENARIOS list is extended with three new entries, each using `regime_pct` as the filter key and `pct=99.0` as the Mahalanobis threshold percentile:

```
{"label": "regime-p95",  "cutoff": None, "exclude": None, "pct": 99.0, "regime_pct": 95}
{"label": "regime-p90",  "cutoff": None, "exclude": None, "pct": 99.0, "regime_pct": 90}
{"label": "regime-p99",  "cutoff": None, "exclude": None, "pct": 99.0, "regime_pct": 99}
```

(Interface sketch from prefactoring analysis — encodes the decision precisely.)

Before the scenario loop, the diagnostics `main()` function instantiates a temporary `BaselineScorer` on training data, computes training-period RV20 z-scores, and derives the 90th-, 95th-, and 99th-percentile filter thresholds. The `_filter_latents` and `_filter_df` helpers are extended with an optional `regime_mask` parameter. When a scenario has `regime_pct`, the pre-computed mask is passed in; when it does not, the parameter is `None` and the existing cutoff/exclude logic applies unchanged.

### Hyperparameter Registry (POC-2 additions)

```
regime_filter_feature    = rv20
regime_filter_pct        = 95           # primary; 90 and 99 are sensitivity variants
regime_filter_normaliser = rv20_mean, rv20_std from 2010-2016 (same as BaselineScorer)
min_filtered_calib_rows  = 50
```

All POC-1 hyperparameters are inherited unchanged.

### Key Invariant

The regime-filter threshold is computed from training-period statistics only. The calibration-period RV20 series is never used to set or inform the filter threshold. This is enforced by the structure of the code: the training-period percentile is computed before `calib_df` is touched, and `regime_filter_mask()` receives only training-period normalisation constants, not calibration-period statistics.

---

## Testing Decisions

A good test exercises external behavior through the module's public interface and asserts on observable output — not on internal implementation details. Tests should be deterministic (synthetic data with known properties), small enough to reason about by hand, and fail clearly when the invariant is violated.

### `regime_filter_mask()` (new)

- On a synthetic RV20 series where five dates have z-scores above the 95th percentile of a known training distribution, assert that exactly those five dates are marked `False` in the returned mask.
- Assert that the function returns `True` for all dates when the threshold is set above the maximum z-score in the series.
- Assert that the function returns `False` for all dates when the threshold is set below the minimum z-score.
- Assert that the returned array length equals the length of the input DatetimeIndex, even when the RV20 series has gaps (reindex fills NaN, which should be treated as exceeding the threshold — i.e., `NaN <= threshold` is `False`).
- Assert that the mask applied to latent arrays and to a daily DataFrame both correctly exclude the same regime dates (end-to-end consistency check on a synthetic two-row-per-window vs one-row-per-day setup).

### `diagnostics.py` regime scenarios

- Assert that a regime-p95 scenario produces fewer calibration samples than the original scenario on the real data.
- Assert that the regime-p95 threshold is lower than the original threshold (filtering COVID reduces the calibration tail).
- Assert that the existing POC-1 scenarios (original, pre-COVID, excl Feb-Apr, p97.5, p99.5) are unaffected by the new code path (no regression).

### Prior art

`tests/test_calibration.py` contains the existing `MahalanobisCalibrator` tests. The `regime_filter_mask()` tests should live in the same file, following the same pattern: synthetic data, deterministic assertions, no reliance on real market data.

---

## Out of Scope for POC-2

- Composite filter (exclude days where max(RV20 z-score, VIX z-score) > 95th pct): documented as a future sensitivity check; not implemented as a primary scenario.
- Seed stability sweep (8–10 seeds): deferred to POC-3.
- Latent dim sensitivity (8, 16, 32): deferred to POC-3.
- Additional baselines (PCA, Isolation Forest, correlated ETFs): deferred to POC-3.
- Rolling or walk-forward calibration: deferred to POC-3.
- Additional assets (QQQ, SPY, SMH): deferred to POC-3.
- Recurrent or attention-based encoder architectures: deferred to POC-3.
- Any change to the model, training protocol, scoring hierarchy, or pass/fail criteria.
- Re-running POC-1 or modifying its locked output files.

---

## Pre-registration Note

The regime-filter definition (RV20 z-score > 95th percentile of training-period scores) is locked before running the experiment. The test period (2021–2022) has been observed in POC-1, so results must be interpreted as calibration-window sensitivity rather than a fully fresh blind test. The protection against post-hoc tuning is strict preregistration of the filter definition and public documentation of the constraint.

The three sensitivity variants (90th, 99th percentile) are explicitly post-hoc and do not affect the primary verdict.

---

## Further Notes

- **The filter targets the root cause directly.** The POC-1 diagnostics showed that excluding Feb–Apr 2020 barely moves the LeJEPA threshold (6.33 → 6.43). The COVID regime contamination extends beyond the acute crash. A percentile-based filter on RV20 z-score is the principled way to exclude the entire extreme-regime cluster without manually specifying dates.

- **Leakage discipline is identical to POC-1.** The regime-filter threshold is derived from the same training split (2010–2016) as the scaler, baseline normalisation constants, and everything else. The calibration-period RV20 values are used only to compute z-scores (using training-period constants) — the calibration distribution never informs the filter threshold.

- **The test period is not blind.** POC-2 is calibration-window sensitivity analysis, not a fresh experiment. The 2021–2022 crash onset date was observed in POC-1. The experimental claim is accordingly narrow: whether a principled, leakage-clean calibration window design produces thresholds that allow the signal to fire.

- **Caches are safe to reuse.** The trained model and calibration latent cache from POC-1 diagnostics (`diag_model.pt`, `diag_calib_latents.npz`) contain no calibration window information — they cache the model weights and the full unfiltered latent array. The regime filter is applied after cache load, so both files are valid inputs to POC-2.

- **Minimum sample guard.** Ledoit-Wolf shrinkage is well-conditioned at 16 latent dimensions with as few as ~50 samples, but the assertion at ≥ 50 filtered rows is a safety net to catch catastrophic over-filtering (e.g. if the filter threshold is set below 1.0 and removes nearly all calibration data).

---

## Key Files

| Path | Purpose |
|------|---------|
| `docs/PRD.md` | POC-1 preregistered spec + results |
| `run_lejepa_soxx_poc.py` | POC-1 entry point (do not re-run) |
| `run_lejepa_soxx_poc2.py` | POC-2 entry point (to be created) |
| `diagnostics.py` | Post-hoc sensitivity (extend for regime-pct scenarios) |
| `src/calibration.py` | Add `regime_filter_mask()` here |
| `src/baselines.py` | `rv20_mean`, `rv20_std` are public attrs — no changes needed |
| `outputs/run_config.json` | Locked POC-1 blind result — do not overwrite |
| `outputs/run_config_poc2.json` | POC-2 config (to be created) |
