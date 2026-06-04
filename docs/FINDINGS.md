# Findings: LeJEPA SOXX Market Stress Experiment

## Hypothesis

A small Linear Joint-Embedding Predictive Architecture (LeJEPA), trained on 2010–2016 SOXX data and calibrated on a 2019–2020 low-volatility anchor, can produce an unsupervised early warning signal before the 2022 semiconductor sector drawdown — and that signal leads simple volatility baselines (RV20, VIX) under identical leakage-controlled conditions.

---

## Verdict: FAIL — Structural, Not Architectural

The hypothesis is falsified. LeJEPA did not fire before the crash onset under any configuration tested. More importantly, **no signal of any kind fired before the crash onset** — including VIX and RV20. The failure is not specific to LeJEPA. It is specific to the event.

---

## Experiment Summary

### POC-1 — Preregistered blind test

| Parameter | Value |
|---|---|
| Calibration threshold | lejepa_threshold = 6.33 (p99 of calibration Mahalanobis scores) |
| Crash onset | 2021-12-10 |
| LeJEPA first breach | 2022-11-04 |
| Lead days | −227 trading days (breach 227 days *after* onset) |
| RV20 first breach | never |
| VIX first breach | never |
| **Verdict** | **FAIL — all four criteria failed** |

Result locked in `outputs/run_config.json`. This was the primary test.

### POC-2 — Regime-filtered calibration (post-hoc)

Applied a regime filter (regime_pct=95) to exclude high-vol days from the calibration distribution, lowering the threshold from 6.33 to 4.29.

| Parameter | Value |
|---|---|
| LeJEPA first breach | 2022-01-06 |
| Lead days | −18 trading days (still after onset) |
| **Verdict** | **FAIL** |

The softer threshold brought the breach date forward by ~10 months but could not cross the crash onset boundary.

### POC-3 — Sensitivity analysis

Three sweeps to test whether the POC-1 failure was an artefact of a specific hyperparameter choice.

**Seed sweep (10 seeds, latent_dim=16)**

| Seed | First breach | Lead days |
|---|---|---|
| 0 | 2022-11-04 | −227 |
| 1 | 2022-11-04 | −227 |
| 2 | 2022-05-17 | −108 |
| 3 | 2022-08-31 | −181 |
| 4 | 2022-05-11 | −104 |
| 5 | 2022-05-31 | −117 |
| 6 | 2022-05-12 | −105 |
| 7 | 2022-05-18 | −109 |
| 8 | 2022-05-16 | −107 |
| 9 | 2022-03-25 | −72 |

Pass rate: **0/10**. Mean lead: −135.7 ± 52.3 days. The best seed (9) fires 72 trading days after the crash onset. Variance across seeds is substantial but the distribution is entirely in the wrong direction.

**Latent dim sweep (dims 8, 16, 32; seed=42)**

All three converge on the same breach date (2022-11-04, −227 days). Model capacity has no effect on the result.

**Extended baselines (seed=42, latent_dim=16)**

| Signal | First breach | Episodes in 2021 | Alert days in 2021 |
|---|---|---|---|
| LeJEPA | never | 0 | 0 |
| RV20 | never | 0 | 0 |
| VIX | never | 0 | 0 |
| PCA scorer | never | 0 | 0 |
| Isolation Forest | never | 0 | 0 |

**0/5 signals fired at any point in 2021.** No signal — traditional or learned — detected any anomaly in the lead-up to or at the crash onset.

---

## Structural Interpretation

The 2022 SOXX drawdown was a slow-burn macro correction: rising rate expectations, post-COVID supply chain normalization, and semiconductor multiple compression. The decline unfolded over roughly 12 months, without a sharp volatility spike at onset.

Every signal in this experiment is fundamentally a volatility-regime detector. LeJEPA is trained to predict latent structure from context windows; when that structure deviates from the 2019–2020 calibration anchor, it fires. RV20 and VIX measure realized and implied volatility directly. PCA and Isolation Forest detect statistical outliers in the feature space. All of these approaches share the same implicit assumption: that a crash is preceded by a distinguishable anomaly in vol or feature structure relative to a normal-regime anchor.

That assumption does not hold for this event. The 2022 SOXX drawdown was not anomalous in volatility space before it happened — it was anomalous in price space only in hindsight. There was nothing in the daily feature distribution to separate late 2021 from the calibration period, which is precisely why VIX itself shows zero 2021 alerts.

**The failure is not a model architecture problem. It is an event-type mismatch.**

---

## What This Does and Does Not Say

**This experiment does not show that LeJEPA is ineffective as a stress detector.** It shows that LeJEPA (and all volatility-based anomaly detection) cannot predict a macro-driven, slow-burn sector correction that does not manifest as a volatility regime shift.

**The approach may perform differently on vol-spike crashes** — events like March 2020 (COVID), August 2015 (China devaluation), or February 2018 (Volmageddon), where a sharp volatility regime change preceded or coincided with the drawdown. The LeJEPA signal is designed to detect exactly that kind of regime departure.

Every major vol-spike crash before 2021 (including COVID March 2020) already falls inside an existing split, so testing on a different event would require redesigning the splits from scratch — it is not a simple continuation of this experiment.

---

## Data Discipline Notes

All split boundaries were enforced in code throughout:

- Train: 2010–2016
- Validation: 2017–2018
- Calibration: 2019–2020
- Test: 2021–2022

The RobustScaler was fitted on training data only. The Mahalanobis distribution was fitted on calibration data only. No test-period information influenced any threshold, filter, or model weight. All POC-3 sensitivity experiments used the same split boundaries and wrote only to `outputs/poc3/`.
