# LeJEPA Market Stress

Tests whether a small [Linear Joint-Embedding Predictive Architecture](https://github.com/galilai-group/lejepa) produces an unsupervised early warning signal before the 2022 semiconductor drawdown, and whether it leads simple volatility and VIX baselines in a leakage-controlled backtest.

**This is not a crash prediction system.** The claim is narrow: does LeJEPA latent drift beat RV20 and VIX level as an early warning signal under the same false-alert budget?

## Results

| POC | Verdict | Root cause |
|-----|---------|------------|
| POC-1 | **FAIL** | COVID-era calibration inflated thresholds to unreachable levels |
| POC-2 | In progress | Regime-filtered calibration (exclude RV20 z-score > 95th pct of training) |

POC-1 blind result: LeJEPA reached 91.7% of threshold, breached 227 days late. Neither baseline fired. See `docs/PRD.md` for full diagnostics.

## Experiment Design

| | |
|---|---|
| Asset | SOXX |
| Train | 2010–2016 |
| Validation | 2017–2018 |
| Calibration | 2019–2020 |
| Blind test | 2021–2022 |
| Main event | 2022 semiconductor drawdown |

**Anomaly score:** Mahalanobis distance of the predicted latent (`z_pred`) from the calibration target latent distribution — no future data used.

**Baselines:** 20-day realized volatility z-score, VIX level z-score. Both normalised on training data only.

## Quickstart

```bash
pip install -r requirements.txt
python run_lejepa_soxx_poc.py   # POC-1 (result locked)
python run_lejepa_soxx_poc2.py  # POC-2 (regime-filtered calibration)
```

## Model

```
Encoder:   [180 → 64 → 16]  ReLU
Predictor: [16  → 32 → 16]  ReLU
Loss:      MSE(z_pred, z_target) + 0.02 × SIGReg
Optimizer: AdamW  lr=3e-4  wd=1e-4
Seed:      42
```

Context window: 30 trading days → 10-day future target.

## Structure

```
src/                    module implementations
tests/                  unit tests
docs/PRD.md             POC-1 spec + results
docs/PRD-POC2.md        POC-2 spec
diagnostics.py          post-hoc sensitivity analysis
outputs/                generated artifacts (gitignored except .gitkeep)
```
