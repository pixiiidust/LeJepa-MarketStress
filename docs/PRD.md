# PRD: LeJEPA SOXX Market Stress POC-1

## Problem Statement

A researcher wants to test whether a small Linear Joint-Embedding Predictive Architecture (LeJEPA) can produce an unsupervised early warning signal before the 2022 semiconductor sector drawdown — and whether that signal leads simple volatility and fear-index baselines in a leakage-controlled historical backtest.

This is not a crash prediction system. The question is whether LeJEPA latent drift gives an earlier and cleaner warning than realized volatility (RV20) and VIX level under the same false-alert budget and the same data discipline. Without a structured, leakage-controlled experiment, any apparent outperformance could be an artifact of lookahead bias, overfitting to the test period, or circular threshold calibration. The researcher needs a single reproducible script that runs the full experiment end-to-end, enforces all split boundaries in code, and produces a binary pass/fail verdict that a skeptical reviewer could replicate from the `run_config.json` alone.

---

## Solution

A single Python script (`run_lejepa_soxx_poc.py`) that:

1. Downloads and aligns SOXX, VIX, and DGS10 data; computes six causal features; fits a RobustScaler on training data only.
2. Trains a small LeJEPA model (MLP encoder + predictor, latent dim 16) on 2010–2016 with early stopping on 2017–2018 validation loss.
3. Freezes the model; fits a Ledoit-Wolf Mahalanobis distribution on calibration-period (2019–2020) target latents; sets the alert threshold at the 99th percentile of calibration-period predicted latent scores.
4. Scores each day in the blind test period (2021–2022) using only data available up to that day; computes the same scores for RV20 and VIX baselines under identical split discipline.
5. Determines the mechanical crash onset date, computes lead times, episode counts, and alert days; applies four pass/fail criteria; writes `scores.csv`, `soxx_lejepa_chart.png`, and `run_config.json`.

The final claim is narrow and defensible: whether LeJEPA latent drift functions as an unsupervised early warning signal for market regime stress, and whether it beats simple baselines in a leakage-controlled backtest.

---

## User Stories

1. As a researcher, I want to run a single script that downloads all required data, so that the full dataset is reproducibly assembled without manual steps.
2. As a researcher, I want SOXX and VIX fetched from yfinance and DGS10 fetched from FRED, so that each series uses its most reliable available source.
3. As a researcher, I want all series aligned to SOXX trading days with DGS10 forward-filled at most 3 days, so that feature computation is consistent and alignment gaps do not silently corrupt features.
4. As a researcher, I want missing-row counts logged to the console and run_config.json, so that data quality issues are visible without inspecting raw downloads.
5. As a researcher, I want an assertion that no NaN values remain after alignment, so that downstream feature computation cannot silently operate on missing data.
6. As a researcher, I want six features computed from data available as of each trading day, so that no feature introduces lookahead bias into the encoder.
7. As a researcher, I want log-volume ratio computed as log(volume_t / mean(volume[t-20:t-1])), so that volume is stationary and the rolling mean excludes the current day.
8. As a researcher, I want rolling drawdown computed from the 252-trading-day rolling high (expanding for the first 252 days), so that the feature is stationary and regime-comparable across the full 2010–2022 period.
9. As a researcher, I want the RobustScaler fitted only on 2010–2016 training data and applied to all other splits, so that calibration and test distributions are never used to normalize features.
10. As a researcher, I want split boundaries enforced as hard assertions before any model training, so that rolling window overlap leakage cannot occur silently.
11. As a researcher, I want the training split (2010–2016), validation split (2017–2018), calibration split (2019–2020), and blind test split (2021–2022) to be fixed and non-overlapping, so that the experiment is reproducible and the timeline is unambiguous.
12. As a researcher, I want the LeJEPA encoder and predictor to be small MLPs, so that the model cannot memorize training-period idiosyncrasies with only ~1,760 training windows.
13. As a researcher, I want the encoder to map a 30-day context window (180-dimensional flattened input) to a 16-dimensional latent vector, so that the latent space is small enough for reliable Mahalanobis estimation on the calibration sample.
14. As a researcher, I want the predictor to map the context latent to a predicted target latent, so that the model learns to anticipate future latent structure rather than merely compressing present state.
15. As a researcher, I want the target window to be the 10 trading days immediately following the context window, so that the prediction task is forward-looking and the diagnostic score is available within two calendar weeks.
16. As a researcher, I want training loss to combine MSE prediction error and SIGReg regularization (λ=0.02), so that the encoder learns predictive structure without collapsing all latents to a constant vector.
17. As a researcher, I want SIGReg applied to z_context, z_target, and z_pred within each batch, so that isotropy is enforced across all three embedding populations.
18. As a researcher, I want AdamW with learning rate 3e-4 and weight decay 1e-4, so that regularization is decoupled from the adaptive gradient update.
19. As a researcher, I want early stopping on validation prediction loss (patience 30, min_delta 1e-5), so that training halts when the model has converged without overfitting to the validation split.
20. As a researcher, I want all random seeds fixed to 42 (random, numpy, torch, cuda), so that the result is exactly reproducible across runs on the same hardware.
21. As a researcher, I want collapse checks logged at each epoch (latent_std_min, latent_rank, covariance_condition_number), so that I can detect latent collapse before it invalidates the scoring.
22. As a researcher, I want the script to halt with a clear error if latent_std_min falls below 0.05 at the end of training, so that a collapsed model cannot silently produce meaningless anomaly scores.
23. As a researcher, I want the model frozen after training and never updated during calibration or scoring, so that calibration-period data cannot influence the encoder weights.
24. As a researcher, I want the Mahalanobis distribution fitted on z_target vectors from 2019–2020 using Ledoit-Wolf shrinkage, so that the covariance estimate is well-conditioned for 464 calibration samples and 16 latent dimensions.
25. As a researcher, I want the alert threshold computed as the 99th percentile of Mahalanobis scores on calibration-period z_pred vectors, so that the threshold is calibrated in the same space as the live score.
26. As a researcher, I want a separate Mahalanobis distribution fitted on calibration-period z_context vectors, so that the secondary score is independently calibrated.
27. As a researcher, I want the primary live anomaly score on each test day to be Mahalanobis(z_pred, calib_target_dist), so that the alert uses only data available up to that day.
28. As a researcher, I want the diagnostic prediction error (‖z_pred − z_target‖₂) logged with a valid_date of T+10 trading days, so that the true JEPA breakdown signal is preserved without being confused for a live signal.
29. As a researcher, I want the diagnostic score set to NaN for the last 10 days of the test period, so that incomplete target windows are not silently filled.
30. As a researcher, I want RV20 and VIX level z-scores normalized using training-period mean and std, so that baselines operate under the same leakage discipline as LeJEPA.
31. As a researcher, I want each baseline threshold set at the 99th percentile of its calibration-period z-scores, so that the false-alert budget is symmetric across all three signals.
32. As a researcher, I want alert episodes defined identically for LeJEPA and both baselines: starting on the first breach and ending after ≥10 consecutive non-breach trading days, so that episode-count comparisons are apples-to-apples.
33. As a researcher, I want alert episode IDs incremented per episode and stored in the CSV, so that individual episodes can be inspected in downstream analysis.
34. As a researcher, I want the crash onset date defined mechanically as the first test day with a full forward 60-trading-day drawdown of at least 20% (using same-day close as reference), so that the oracle label is fixed before model results are reviewed.
35. As a researcher, I want days without a full 60-trading-day forward window excluded from crash onset eligibility, so that incomplete windows do not produce spurious onset dates near the end of 2022.
36. As a researcher, I want a "no qualifying onset" result documented if no test day meets the crash onset criterion, so that a null result is reported rather than hidden.
37. As a researcher, I want pass/fail criterion 1 to require LeJEPA's first breach to occur on or before the crash onset date, so that the warning precedes the event it is supposed to signal.
38. As a researcher, I want pass/fail criterion 2 to require LeJEPA's first breach to precede both RV20 and VIX first breaches, so that the timing advantage is conservative and cannot be satisfied by beating only the noisier baseline.
39. As a researcher, I want a partial-signal classification when LeJEPA beats exactly one baseline on timing, so that borderline results are documented transparently rather than forced into pass or fail.
40. As a researcher, I want pass/fail criterion 3 to require LeJEPA's 2021 episode count to be at most max(rv20_episodes_2021, vix_episodes_2021), with alert days as a tiebreaker, so that the false-alert floor is the noisier baseline rather than the cleaner one.
41. As a researcher, I want max episode duration logged as a diagnostic column in the CSV but excluded from pass/fail, so that sticky false alerts are visible without adding a fourth criterion that could be gamed.
42. As a researcher, I want all first-breach dates and lead times printed in the console summary regardless of pass/fail outcome, so that the verdict is immediately auditable without reading the CSV.
43. As a researcher, I want z_pred and z_context latent vectors stored as 16 separate columns in scores.csv, so that the latent space can be explored in downstream analysis without parsing serialized arrays.
44. As a researcher, I want run_config.json to record seed, all hyperparameters, split dates, threshold value, data sources, and git commit hash, so that any run is fully reproducible from a single file.
45. As a researcher, I want the two-panel chart to show SOXX close with crash onset line in the top panel and LeJEPA score, baseline scores, and thresholds in the bottom panel, so that timing relationships are visually inspectable without loading the CSV.

---

## Implementation Decisions

### Module Architecture

Nine modules, each with a narrow public interface:

**DataPipeline**
Fetches SOXX and VIX from yfinance; DGS10 from FRED via pandas_datareader. Aligns all series to the SOXX trading day index. Forward-fills DGS10 at most 3 days. Asserts no NaN after alignment. Computes all six features. Fits RobustScaler on the 2010–2016 slice and applies it to all splits. Enforces split boundary assertions before returning window datasets. Returns train, validation, calibration, and test window sets as separate objects.

**LeJEPAModel**
Wraps Encoder (Linear 180→64→ReLU→Linear 64→16) and Predictor (Linear 16→32→ReLU→Linear 32→16). Exposes `encode(x) → z` and `predict(z) → z_pred`. Latent dim fixed at 16. No statefulness beyond weights.

**SIGRegLoss**
Adapts the SIGReg (Sketched Isotropic Gaussian Regularization) implementation from galilai-group/lejepa for time-series batch shapes (batch × 16). Applied to z_context, z_target, and z_pred within each training step. λ fixed at 0.02.

**Trainer**
Runs the AdamW training loop on the train split, evaluates on the validation split after each epoch, applies early stopping (patience=30, min_delta=1e-5 on validation prediction loss), and logs collapse diagnostics (latent_std_min, latent_rank, covariance_condition_number) per epoch. Raises an error if latent_std_min < 0.05 at training completion. Returns the best checkpoint by validation loss.

**MahalanobisCalibrator**
Accepts a set of z_target vectors from the calibration period, fits a Ledoit-Wolf covariance estimator (sklearn.covariance.LedoitWolf), and exposes `score(z) → float` for any input vector. Computes the alert threshold as the 99th percentile of calibration-period z_pred scores. Instantiated twice: once for the target-latent distribution, once for the context-latent distribution.

**Scorer**
Iterates over the blind test period in causal order. For each day T: encodes the 30-day context window, computes z_pred, scores against the calibration target distribution (primary) and calibration context distribution (secondary), logs z_pred and z_context latent vectors, and computes the diagnostic prediction error using the target window (logging valid_date = T+10). Applies alert episode detection using ≥10 consecutive non-breach days as the episode-end rule.

**BaselineScorer**
Fits RV20 and VIX level z-score normalizers on 2010–2016 training statistics. Scores calibration and test periods. Computes per-baseline thresholds from calibration 99th percentiles. Applies identical alert episode detection logic as Scorer.

**Evaluator**
Computes crash onset: iterates test days with full 60-day forward windows, finds the first day where min(close[T:T+60])/close[T]−1 ≤ −0.20. Computes lead times for LeJEPA and each baseline. Counts 2021 episodes and alert days for each signal. Applies all four pass/fail criteria and emits a structured verdict including partial-signal classification.

**OutputWriter**
Assembles scores.csv from Scorer and BaselineScorer output. Renders the two-panel matplotlib chart. Writes run_config.json. Prints the console summary.

---

### Key Invariants

- **No rolling window crosses a split boundary.** The last training window's target end date must be ≤ 2016-12-31. The first calibration window's context start must be ≥ 2019-01-01. Enforced as assertions in DataPipeline before any model construction.
- **2019–2020 data never influences training, early stopping, architecture, or hyperparameter choices.** Calibration data is passed to MahalanobisCalibrator only after the model is fully frozen.
- **The crash onset date is computed before pass/fail evaluation begins.** The Evaluator computes and logs the onset date in one step and the verdict in a subsequent step, so there is no opportunity to adjust the onset definition based on model output.
- **The threshold value is fixed before any test-period data is scored.** Threshold is stored in run_config.json before Scorer iterates the test period.

---

### Feature Definitions

```
soxx_log_return     = log(close_t / close_{t-1})
log_volume_ratio    = log(volume_t / mean(volume[t-20 : t-1]))
rv20                = std(log_return[t-19 : t]) * sqrt(252)
rolling_drawdown    = close_t / max(close[t-252 : t]) - 1  [expanding for first 252 days]
vix_daily_change    = vix_t - vix_{t-1}
dgs10_daily_change  = dgs10_t - dgs10_{t-1}
```

All features use only data available as of day T.

---

### Anomaly Score Hierarchy

| Score | Formula | Timing | Use |
|---|---|---|---|
| Primary live | Mahalanobis(z_pred, calib_target_dist) | Available day T | Alert trigger, threshold breach |
| Diagnostic | ‖z_pred − z_target‖₂ | Valid at T+10 | Model-breakdown evidence |
| Secondary | Mahalanobis(z_context, calib_context_dist) | Available day T | Static embedding baseline |

---

### Pass/Fail Criteria

| # | Criterion | Threshold |
|---|---|---|
| 1 | LeJEPA first breach ≤ crash onset date | Hard |
| 2 | LeJEPA first breach < RV20 first breach **AND** < VIX first breach | Hard; partial signal if one baseline beaten |
| 3 | LeJEPA 2021 episode count ≤ max(rv20_episodes, vix_episodes); tiebreaker: alert days | Hard |
| 4 | All split boundary assertions passed | Hard |

---

### scores.csv Column Contract

```
date, soxx_close,
lejepa_score, lejepa_threshold, lejepa_breach, lejepa_episode_id,
pred_error, pred_error_valid_date,
lejepa_context_score,
z_pred_0 … z_pred_15,
z_context_0 … z_context_15,
rv20_zscore, rv20_threshold, rv20_breach, rv20_episode_id,
vix_zscore, vix_threshold, vix_breach, vix_episode_id,
forward_drawdown_60d, crash_onset_flag
```

`forward_drawdown_60d` and `crash_onset_flag` are labeled as evaluation-only (future-looking) in column documentation.

---

### Hyperparameter Registry

```
seed                     = 42
encoder                  = [180, 64, 16], ReLU
predictor                = [16, 32, 16], ReLU
latent_dim               = 16
context_window           = 30 trading days
target_window            = 10 trading days
optimizer                = AdamW
learning_rate            = 3e-4
weight_decay             = 1e-4
batch_size               = 64
max_epochs               = 500
early_stopping_patience  = 30
early_stopping_min_delta = 1e-5
sigreg_lambda            = 0.02
threshold_percentile     = 99
mahalanobis_estimator    = LedoitWolf
scaler                   = RobustScaler
episode_cooldown_days    = 10
drawdown_threshold       = -0.20
drawdown_window_days     = 60
drawdown_lookback_days   = 252
```

---

## Testing Decisions

A good test exercises external behavior through the module's public interface and asserts on observable output — not on internal implementation details like weight values, intermediate tensors, or private methods. Tests should be deterministic (seed-fixed), use synthetic data small enough to reason about by hand, and fail clearly when the invariant is violated.

**DataPipeline**
- Assert that the scaler fitted on a training slice does not have access to calibration or test statistics (check fitted mean and scale against training-only ground truth on synthetic data).
- Assert that alignment produces no NaN on a synthetic series with known gaps.
- Assert that split boundary enforcement raises an assertion error when a window spans two splits.
- Assert that log_volume_ratio uses `t-20:t-1` (not `t-20:t`) by checking the value on a day where today's volume differs sharply from yesterday's.

**MahalanobisCalibrator**
- Assert that `threshold` equals the 99th percentile of calibration scores on synthetic normally-distributed latent vectors (exact equality, not tolerance).
- Assert that `score(z)` increases monotonically as z moves further from the calibration distribution mean.
- Assert that Ledoit-Wolf is used: perturb calibration to be low-sample and confirm covariance is regularized (condition number bounded).

**Scorer (episode detection)**
- On a synthetic breach sequence with a 9-day gap, assert that two episodes are detected (gap is below the 10-day cooldown).
- On a synthetic breach sequence with a 10-day gap, assert that two episodes are detected (gap equals the cooldown, so it resets).
- Assert that `pred_error_valid_date` equals T+10 trading days for mid-period rows and NaN for the last 10 rows.
- Assert that `lejepa_episode_id` increments correctly across a multi-episode sequence.

**Evaluator (crash onset + pass/fail)**
- On a synthetic close series with a known 20% forward drop at a specific day, assert that crash_onset equals that day.
- On a synthetic series where the last qualifying day is within 60 days of period end, assert that the onset is not set to a day with an incomplete window.
- Assert that criterion 2 returns "partial signal" when LeJEPA beats RV20 but not VIX.
- Assert that criterion 3 tiebreaker fires correctly when episode counts are tied but alert days differ.

---

## Out of Scope

- Additional assets (QQQ, SPY, SMH, ARKK, XLF): deferred to post-POC-1 extension.
- Additional unsupervised baselines (PCA anomaly score, Isolation Forest): deferred.
- Sensitivity sweep on threshold percentile (99.5th, 99.7th): deferred to POC-2.
- Sensitivity sweep on latent dim (8, 32): deferred to POC-2.
- Sensitivity sweep on random seed: deferred to POC-2.
- Live or forward-looking inference beyond 2022: out of scope for this POC.
- Hyperparameter search of any kind: all hyperparameters are preregistered; no tuning is permitted.
- Recurrent or attention-based encoder architectures: out of scope for POC-1.
- GPU training: the model is small enough to train on CPU in minutes.
- Claiming crash prediction: the final claim is limited to whether LeJEPA latent drift functions as an unsupervised early warning signal in a leakage-controlled historical backtest.

---

## Further Notes

- **Leakage is the primary risk.** The four hard assertions — split boundaries, scaler fitting scope, calibration-only threshold setting, and frozen model during scoring — are the experiment's integrity layer. Any reviewer attacking the result will attack these four points first. They should be enforced in code, not convention.

- **The SIGReg implementation requires adaptation.** The galilai-group/lejepa repository implements SIGReg expecting embeddings in a specific shape and training context. The adapter must verify that SIGReg receives (batch × 16) tensors for z_context, z_target, and z_pred and that the loss scale is compatible with the MSE prediction loss at λ=0.02. A collapse check after the first few training epochs is the fastest way to confirm the adaptation is correct.

- **If 2021 triggers the crash onset first, that is a finding.** The 60-day / 20% rule is intentionally mechanical and not anchored to the 2022 main event. If it fires in 2021 (possible during any sharp drawdown that year), the correct response is to document it as "rule too loose or stress began earlier than expected" — not to adjust the rule after seeing the result.

- **POC-2 scope.** After the blind result is reviewed: seed stability sweep (8–10 seeds), threshold sensitivity (99th vs 99.5th), latent dim sensitivity (8, 16, 32), and additional baseline signals (PCA, Isolation Forest, correlated ETFs).

- **Final claim standard.** The script must not claim LeJEPA predicts crashes. The console summary pass/fail verdict should read: "This POC tests whether LeJEPA latent drift can function as an unsupervised early warning signal for market regime stress, and whether it beats simple volatility and VIX baselines in a leakage-controlled historical backtest."
