# PRD: LeJEPA SOXX Market Stress POC-3 — Sensitivity Experiments

## Problem Statement

POC-1 and POC-2 both returned FAIL verdicts. Before drawing conclusions about the viability of the LeJEPA architecture for market stress detection, a researcher needs to understand how sensitive the FAIL results are to implementation choices that were fixed at arbitrary or convenient values in POC-1: the random seed, the latent dimensionality, and the choice of anomaly detection baseline.

Three open questions remain unanswered by the existing experiments:

1. **Seed stability**: Is the FAIL verdict stable across random seeds, or does it change depending on which local minimum the encoder finds? A single seed cannot distinguish a structural failure from a lucky (or unlucky) random initialisation.
2. **Latent dim sensitivity**: The encoder was fixed at 16 latent dimensions. If 8 dims collapse the representation and 32 dims over-parameterise it, the POC-1 failure could be a latent space sizing artefact rather than an architectural finding.
3. **Baseline quality**: RV20 and VIX are simple linear signals. If neither they nor LeJEPA ever fire, the FAIL verdict may reflect a hard detection problem rather than LeJEPA underperforming competitive alternatives. PCA reconstruction error and Isolation Forest are standard anomaly detection methods that can be compared against LeJEPA on the same evaluation criteria.

All three questions are post-hoc sensitivity analysis — the test period (2021–2022) has already been observed in POC-1. Results from these experiments cannot overturn or modify the locked POC-1 verdict; they contextualise it.

---

## Solution

Three new sweep entry points, each producing aggregate result tables in `outputs/poc3/`:

1. **Seed sweep** (`run_poc3_seed_sweep.py`): Run the full POC-1 pipeline (train → calibrate → score → evaluate) with 10 different seeds (0–9). Report per-seed verdict, first-breach date, and lead time. Compute mean/std of lead time across seeds.
2. **Latent dim sweep** (`run_poc3_latent_sweep.py`): Run the full POC-1 pipeline with `latent_dim` in {8, 16, 32}. The dim-16 run reuses the POC-1 result to avoid retraining. Report per-dim verdict, threshold, first-breach date, and lead time.
3. **Extended baselines** (`run_poc3_baselines.py`): Add PCA reconstruction-error and Isolation Forest anomaly scores alongside the existing RV20 and VIX baselines. Apply the same calibration-window threshold logic (99th percentile of calibration-period scores). Evaluate all five signals against the same pass/fail criteria on the same test period.

No locked POC-1 output files (`outputs/run_config.json`, `outputs/scores.csv`, `outputs/soxx_lejepa_chart.png`) are touched by any POC-3 entry point. All POC-3 results are written under `outputs/poc3/`.

---

## User Stories

### Seed Sweep

1. As a researcher, I want to run the full POC-1 pipeline (train, calibrate, score, evaluate) with 10 different random seeds, so that I can assess whether the FAIL verdict is stable or seed-dependent.
2. As a researcher, I want the 10 seeds to be a fixed sequence (0 through 9), so that the sweep is reproducible without additional configuration.
3. As a researcher, I want each seed run to be fully independent (separate model weights, separate calibration, separate threshold), so that seeds do not share any fitted state.
4. As a researcher, I want each seed run to reuse the same data pipeline output (downloaded and scaled data), so that the sweep does not re-download market data 10 times.
5. As a researcher, I want the per-seed results table to include: seed, `lejepa_threshold`, `lejepa_first_breach`, `crash_onset_date`, `lejepa_lead_days`, `all_pass`, and each of the four criterion flags, so that I can inspect which criteria vary across seeds.
6. As a researcher, I want the aggregate summary to report mean and standard deviation of `lejepa_lead_days` across seeds (excluding seeds where LeJEPA never fires), so that I can quantify timing uncertainty.
7. As a researcher, I want the fraction of seeds where `all_pass` is True printed in the console summary, so that verdict stability is immediately visible.
8. As a researcher, I want the per-seed results written to `outputs/poc3/seed_sweep_results.csv`, so that I can load and inspect them in downstream analysis.
9. As a researcher, I want the seed sweep to complete without modifying any file under `outputs/` that does not begin with `outputs/poc3/`, so that locked POC-1 outputs are protected by design.
10. As a researcher, I want a warning printed if any seed produces latent_std_min below 0.05 (collapse threshold), so that collapsed seeds are flagged rather than silently included in the aggregate.

### Latent Dim Sweep

11. As a researcher, I want to run the full POC-1 pipeline with `latent_dim` in {8, 16, 32}, so that I can test whether the FAIL is robust to latent space dimensionality.
12. As a researcher, I want the dim-16 variant to produce a fresh independent run rather than copying the locked POC-1 result, so that all three variants share the same training code path and are directly comparable.
13. As a researcher, I want `LeJEPAModel` to read encoder and predictor dimensions from `Config` rather than hardcoding them, so that all three variants can be run with the same code.
14. As a researcher, I want the encoder architecture for dim-8 to be (180→64→8) and for dim-32 to be (180→64→32), so that only the output projection width changes and intermediate capacity is held constant.
15. As a researcher, I want the predictor architecture for dim-8 to be (8→16→8) and for dim-32 to be (32→64→32), maintaining the same 2× hidden-layer ratio as the POC-1 predictor, so that capacity scales proportionally.
16. As a researcher, I want the number of latent columns written to the per-run scores CSV to match the actual `latent_dim` for each run, so that downstream column access does not assume 16 columns.
17. As a researcher, I want each dim variant to use the same seed (42) as POC-1, so that latent dimensionality is the only experimental variable.
18. As a researcher, I want the per-dim results table to include: `latent_dim`, `lejepa_threshold`, `lejepa_first_breach`, `lejepa_lead_days`, `all_pass`, and `latent_std_min`, so that I can see quality and verdict together.
19. As a researcher, I want the aggregate table written to `outputs/poc3/latent_dim_results.csv`, so that all three variants are inspectable in one place.
20. As a researcher, I want the console summary to print each variant's threshold, first-breach date, and verdict side by side, so that the comparison is immediately visible without loading the CSV.

### Extended Baselines

21. As a researcher, I want to add a PCA reconstruction-error baseline fitted on 2010–2016 training windows, so that I can compare LeJEPA against a classical dimensionality-reduction anomaly detector.
22. As a researcher, I want the PCA baseline to use the same flattened 30-day context window (180-dimensional) as the LeJEPA encoder input, so that both methods operate on identical inputs.
23. As a researcher, I want the number of PCA components selected as the smallest k such that cumulative explained variance exceeds 95% of training variance, so that the component count is data-driven and not hand-tuned.
24. As a researcher, I want the PCA reconstruction error on each day computed as the L2 norm of the residual (original − reconstructed input), so that the anomaly score is interpretable and unit-consistent.
25. As a researcher, I want the PCA threshold set at the 99th percentile of calibration-period (2019–2020) reconstruction errors, so that the false-alert budget matches LeJEPA and the existing baselines.
26. As a researcher, I want to add an Isolation Forest baseline fitted on 2010–2016 training windows, so that I can compare LeJEPA against a tree-based unsupervised anomaly detector.
27. As a researcher, I want the Isolation Forest baseline to use the same 180-dimensional flattened context window as the PCA baseline, so that all baselines share an identical input contract.
28. As a researcher, I want the Isolation Forest anomaly score to be the negative of `score_samples` output (so that higher = more anomalous, consistent with the other baselines' direction), so that all five signals are directionally consistent.
29. As a researcher, I want the Isolation Forest threshold set at the 99th percentile of calibration-period anomaly scores, so that the false-alert budget matches LeJEPA.
30. As a researcher, I want Isolation Forest fitted with a fixed random seed (42) and `n_estimators=200`, so that the result is reproducible and the ensemble is large enough to be stable.
31. As a researcher, I want both new baselines to use RobustScaler statistics fitted on 2010–2016 training data (identical to the data pipeline), so that leakage discipline is enforced at the same boundary as every other signal.
32. As a researcher, I want the extended baselines entry point to evaluate all five signals — LeJEPA, RV20, VIX, PCA, Isolation Forest — against the same four pass/fail criteria on the same test period, so that the comparison is apples-to-apples.
33. As a researcher, I want episode detection applied to PCA and Isolation Forest using the same 10-day cooldown as the existing signals, so that episode counts are comparable.
34. As a researcher, I want the per-signal results table to include: signal_name, `first_breach`, `lead_days`, `episodes_2021`, `alert_days_2021`, `all_pass`, so that all five signals can be ranked by timing performance.
35. As a researcher, I want the extended baselines results written to `outputs/poc3/baselines_results.csv`, so that all five signals are inspectable in one place.
36. As a researcher, I want the extended baselines chart to show all five signals and their thresholds in the score panel alongside SOXX close in the price panel, so that timing relationships are visually inspectable without loading the CSV.
37. As a researcher, I want the chart written to `outputs/poc3/baselines_chart.png`, so that it is separate from the locked POC-1 chart.
38. As a researcher, I want the console summary to print first-breach date and lead time for all five signals in a single table, so that the full competitive comparison is immediately visible.

### Shared Constraints

39. As a researcher, I want all three entry points to assert that no file under `outputs/` outside of `outputs/poc3/` is written, so that locked POC-1 outputs are structurally protected.
40. As a researcher, I want all three entry points to enforce the same split boundaries (train 2010–2016, calibration 2019–2020, test 2021+) as POC-1, so that the experimental timeline is not adjusted between POC-1 and POC-3.
41. As a researcher, I want a shared `outputs/poc3/poc3_summary.md` written at the end of each run that appends a one-line result to a human-readable log, so that all POC-3 experiments are auditable without reading individual CSVs.

---

## Implementation Decisions

### `LeJEPAModel` Parameterisation (Required Change to Existing Module)

`LeJEPAModel` currently hardcodes layer sizes (180→64→16, 16→32→16) and ignores `Config.encoder_dims` and `Config.predictor_dims`. For the latent dim sweep, the model must be built from config. The change is localised to the constructor: read `encoder_dims` and `predictor_dims` from `config` and build `nn.Sequential` layers programmatically. The public interface (`encode`, `predict`, `forward`) is unchanged.

`Config.encoder_dims` and `Config.predictor_dims` already carry the right shape. For dim-8: `encoder_dims=(180, 64, 8)`, `predictor_dims=(8, 16, 8)`. For dim-32: `encoder_dims=(180, 64, 32)`, `predictor_dims=(32, 64, 32)`.

### `Scorer` Latent Column Count (Required Change to Existing Module)

`Scorer.score()` currently writes `z_pred_{j}` and `z_context_{j}` for `j in range(16)`. This must be driven by the model's actual latent dim (readable from config or inferred from the model's output shape) so that dim-8 and dim-32 runs write the correct number of columns.

### Seed Sweep Runner

A new entry point that runs the full pipeline once per seed. It instantiates `Config(seed=s)` for each seed s in `range(10)`, then calls the same train → calibrate → score → evaluate sequence as `run_lejepa_soxx_poc.py`. Data loading and scaling happen once before the loop; the loop begins at model instantiation.

Each seed's `Verdict` is serialised to a row in the aggregate table. The runner does not call `OutputWriter` per seed (no per-seed charts or JSON); it writes one aggregate CSV at the end.

### Latent Dim Sweep Runner

A new entry point that runs the full pipeline for each dim in `(8, 16, 32)`. It constructs `Config(latent_dim=d, encoder_dims=(180, 64, d), predictor_dims=(d, 2*d, d))` for each d. The seed is fixed at 42 for all three. Data loading happens once; model construction and all downstream steps are inside the loop.

No per-dim charts or JSON; one aggregate CSV at the end.

### PCA Anomaly Scorer

A new module (or class in a new module) with the same fit/calibrate/score pattern as `BaselineScorer`. It accepts `train_windows: list[Window]` and `config: Config` on construction. `fit_calibration(calib_windows)` sets the threshold. `score(windows) -> pd.DataFrame` returns `pca_score`, `pca_threshold`, `pca_breach`, `pca_episode_id` columns.

The 180-dimensional input vector (flattened context window) is standardised with training-period statistics before PCA, so the component count is not skewed by feature scales.

### Isolation Forest Scorer

Same pattern as `PCAScorer`. Accepts `train_windows: list[Window]`, fits `IsolationForest(n_estimators=200, random_state=42)` on training feature vectors. Anomaly score = `−score_samples(X)`. Threshold = 99th percentile of calibration-period scores.

### Extended Baselines Entry Point

The extended baselines entry point trains the model once (seed=42, latent_dim=16), runs standard scoring to get LeJEPA and baseline scores, then runs `PCAScorer` and `IsolationForestScorer` separately. All five score series are merged into a single DataFrame and passed to a generalised evaluator that applies the same four criteria per signal. The merged results are written to `outputs/poc3/baselines_results.csv` and `outputs/poc3/baselines_chart.png`.

### Output Layout

```
outputs/poc3/
  seed_sweep_results.csv       — one row per seed
  latent_dim_results.csv       — one row per latent_dim
  baselines_results.csv        — one row per signal (5 rows)
  baselines_chart.png          — 5-signal comparison chart
  poc3_summary.md              — human-readable one-line-per-run log
```

No files outside `outputs/poc3/` are written by any POC-3 entry point.

---

## Testing Decisions

A good test exercises the module's public interface with synthetic data that has known properties, asserts on observable output, and fails clearly when an invariant is broken. Tests must not depend on real market data or trained model weights.

### `LeJEPAModel` Parameterisation Tests

- Assert that a model constructed with `Config(latent_dim=8, encoder_dims=(180,64,8), predictor_dims=(8,16,8))` produces `encode()` output with shape `(8,)` and `predict()` output with shape `(8,)`.
- Assert that a model constructed with `Config(latent_dim=32, ...)` produces outputs with shape `(32,)`.
- Assert that the dim-16 model (default config) is backward-compatible with existing test expectations.

Prior art: `tests/test_model.py`.

### `Scorer` Latent Column Tests

- Assert that `Scorer.score()` on a model with latent_dim=8 writes columns `z_pred_0`..`z_pred_7` and `z_context_0`..`z_context_7` (8 each, not 16).
- Assert that no `z_pred_8`..`z_pred_15` columns appear in the output for a dim-8 model.

Prior art: `tests/test_scoring.py`.

### `PCAScorer` Tests

- On a synthetic 20-window training set with known principal structure, assert that reconstruction error is near zero for windows drawn from the same distribution and non-trivially positive for out-of-distribution windows.
- Assert that `fit_calibration()` sets a threshold equal to the 99th percentile of calibration reconstruction errors on a known synthetic set.
- Assert that `score()` returns a DataFrame with columns `pca_score`, `pca_threshold`, `pca_breach`, `pca_episode_id`.
- Assert that `pca_breach` is `True` when `pca_score > pca_threshold` and `False` otherwise.

### `IsolationForestScorer` Tests

- Assert that `score()` returns a DataFrame with the expected columns.
- Assert that anomaly score direction is positive for anomalous inputs: inject an out-of-distribution window and assert its score exceeds the threshold set on a clean calibration set.
- Assert that `fit_calibration()` sets a threshold equal to the 99th percentile of calibration-period scores.

### Seed Sweep Runner Tests

- Assert that the output CSV has exactly 10 rows (one per seed) with a `seed` column containing 0–9.
- Assert that the `lejepa_threshold` values differ across seeds (model is not deterministic across seeds).

### Latent Dim Sweep Runner Tests

- Assert that the output CSV has exactly 3 rows with `latent_dim` values {8, 16, 32}.
- Assert that the latent dim sweep does not write any file outside `outputs/poc3/`.

All new tests live in `tests/` following the existing naming convention (`test_poc3_*.py`). Prior art for entry-point tests: `tests/test_main.py` and `tests/test_poc2_entry.py`.

---

## Out of Scope

- Rolling or walk-forward calibration: deferred beyond POC-3.
- Additional assets (QQQ, SPY, SMH, sector rotation features): deferred beyond POC-3.
- Recurrent or attention-based encoder architectures: deferred beyond POC-3.
- Composite regime filter (max of RV20 and VIX z-scores): documented in POC-2 PRD as future work; not implemented here.
- Hyperparameter search (learning rate, weight decay, SIGReg lambda): no tuning against test period is permitted.
- Any modification to `run_lejepa_soxx_poc.py` or the locked outputs under `outputs/` (excluding `outputs/poc3/`).
- Re-running POC-2 or modifying `run_config_poc2.json`.
- New feature engineering (sector rotation, cross-ETF context).
- Publication write-up: separate initiative.

---

## Further Notes

- **Post-hoc status.** All three experiments observe a test period (2021–2022) that was already seen in POC-1. Results are sensitivity analysis, not fresh experiments. No POC-3 result can override the locked POC-1 or POC-2 verdicts.

- **`LeJEPAModel` parameterisation is a correctness fix.** `Config` already carries `encoder_dims` and `predictor_dims` but the model ignores them. The dim-16 model produced by the fixed code is architecturally identical to the POC-1 model (same layer sizes), so this change does not alter the locked POC-1 result.

- **Seed sweep interpretation.** If all 10 seeds produce FAIL, the FAIL verdict is seed-stable. If 1–3 seeds produce PASS, the result is sensitive to initialisation — a meaningful finding about the optimisation landscape. If >5 seeds produce PASS, the seed selection in POC-1 was unusually unlucky and the architecture warrants further investigation.

- **PCA component count.** Selecting k via 95% explained variance on 2010–2016 training data is the standard approach. It avoids hand-tuning while being reproducible from the training data alone. The k value should be logged alongside the results so a reviewer can see how many components were retained.

- **Isolation Forest output direction.** `sklearn.ensemble.IsolationForest.score_samples` returns negative scores where more negative = more anomalous. Negating this gives a positive anomaly score aligned with the other signals. The threshold is set on the negated scores, so the breach condition (`score > threshold`) is directionally consistent.

- **No per-run charts for seed or latent dim sweeps.** Generating 10 or 3 charts adds visual noise without proportional insight. The aggregate CSVs are the primary output. The extended baselines run produces a chart because five-signal timing comparison is difficult to inspect in tabular form.
