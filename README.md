# LeJEPA Market Stress

**Experiment concluded.** See `docs/FINDINGS.md` for the full write-up.

Tests whether a small Linear Joint-Embedding Predictive Architecture (LeJEPA) produces an unsupervised early warning signal before the 2022 semiconductor drawdown, and whether it leads simple volatility baselines in a leakage-controlled backtest.

## Verdict: FAIL — structural

No signal fired before the crash onset — not LeJEPA, not RV20, not VIX, not PCA, not Isolation Forest. The 2022 SOXX drawdown was a slow-burn macro correction with no volatility spike at onset. All signals in this experiment are vol-regime detectors; the event type was wrong for the approach.

| POC | Verdict | Note |
|-----|---------|------|
| POC-1 | FAIL | LeJEPA breach 227 trading days after crash onset; no baseline fired |
| POC-2 | FAIL | Regime-filtered calibration; breach still 18 days after onset |
| POC-3 | FAIL | 0/10 seeds, 0/3 latent dims, 0/5 signals pass; failure is consistent |

Note: every vol-spike crash before 2021 (including COVID March 2020) falls inside an existing split — testing the approach on a different event would require redesigning the splits.

## Splits

| Split | Period |
|---|---|
| Train | 2010–2016 |
| Validation | 2017–2018 |
| Calibration | 2019–2020 |
| Blind test | 2021–2022 |

## Model

```
Encoder:   [180 → 64 → 16]  ReLU
Predictor: [16  → 32 → 16]  ReLU
Loss:      MSE(z_pred, z_target) + 0.02 × SIGReg
Optimizer: AdamW  lr=3e-4  wd=1e-4
```

Context window: 30 trading days → 10-day future target. Anomaly score: Mahalanobis distance of predicted latent from calibration distribution.

## Structure

```
src/                    module implementations
tests/                  205 tests, all passing
docs/FINDINGS.md        experiment conclusions
docs/PRD.md             POC-1 preregistered spec
docs/PRD-POC2.md        POC-2 spec
docs/PRD-POC3.md        POC-3 sensitivity spec
outputs/run_config.json locked POC-1 blind result
outputs/poc3/           POC-3 results
```
