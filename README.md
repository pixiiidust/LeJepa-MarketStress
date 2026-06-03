# LeJEPA Market Stress — POC-1

Tests whether a small [Linear Joint-Embedding Predictive Architecture](https://github.com/galilai-group/lejepa) produces an unsupervised early warning signal before the 2022 semiconductor drawdown, and whether it leads simple volatility and VIX baselines in a leakage-controlled backtest.

**This is not a crash prediction system.** The claim is narrow: does LeJEPA latent drift beat RV20 and VIX level as an early warning signal under the same false-alert budget?

## Experiment Design

| | |
|---|---|
| Asset | SOXX |
| Train | 2010–2016 |
| Early-stop validation | 2017–2018 |
| Calibration | 2019–2020 |
| Blind test | 2021–2022 |
| Main event | 2022 semiconductor drawdown |

**Anomaly score:** Mahalanobis distance of the predicted latent (`z_pred`) from the calibration target latent distribution — no future data used.

**Baselines:** 20-day realized volatility z-score, VIX level z-score. Both normalized on training data only.

**Pass criteria:**
1. LeJEPA first breach ≤ crash onset date
2. LeJEPA first breach earlier than **both** RV20 and VIX
3. LeJEPA 2021 false-alert episodes ≤ noisier baseline's count

## Quickstart

```bash
pip install -r requirements.txt
python run_lejepa_soxx_poc.py
```

Outputs written to `outputs/`:
- `scores.csv` — daily scores, breach flags, latent vectors, baselines
- `soxx_lejepa_chart.png` — two-panel chart
- `run_config.json` — full run config for reproducibility (seed, splits, hyperparams, threshold)

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
src/              module implementations
tests/            unit tests (DataPipeline, Calibrator, Scorer, Evaluator)
docs/PRD.md       full product requirements and design decisions
outputs/          generated artifacts (gitignored except .gitkeep)
run_lejepa_soxx_poc.py   single entry point
```
