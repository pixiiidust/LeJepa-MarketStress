# LeJEPA Market Stress

Tests whether a small [Linear Joint-Embedding Predictive Architecture](https://github.com/galilai-group/lejepa) produces an unsupervised early warning signal before the 2022 semiconductor drawdown, and whether it leads simple volatility and VIX baselines in a leakage-controlled backtest.

**This is not a crash prediction system.** The claim is narrow: does LeJEPA latent drift beat RV20 and VIX level as an early warning signal under the same false-alert budget?

## Results

| POC | Verdict | Finding |
|-----|---------|---------|
| POC-1 | **FAIL** | COVID-era calibration inflated threshold to 6.33; LeJEPA breached 227 days late; neither baseline fired |
| POC-2 | **FAIL** | Regime-filtered calibration lowered threshold to 4.29 (−32%) but breach still 18 days after crash onset |
| POC-3 | In progress | Sensitivity sweep: seed stability (10 seeds), latent dim {8/16/32}, PCA + Isolation Forest baselines |

Both FAIL verdicts are locked. POC-3 contextualises them — see `docs/PRD-POC3.md`.

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
python run_lejepa_soxx_poc.py    # POC-1 (result locked — do not re-run)
python run_lejepa_soxx_poc2.py   # POC-2 (regime-filtered calibration)
python run_poc3_seed_sweep.py    # POC-3 seed stability sweep (10 seeds)
python run_poc3_latent_sweep.py  # POC-3 latent dim sweep {8, 16, 32}
python run_poc3_baselines.py     # POC-3 extended baselines (PCA, Isolation Forest)
```

POC-3 entry points write only to `outputs/poc3/`.

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
tests/                  unit tests (116 passing)
docs/PRD.md             POC-1 preregistered spec + results
docs/PRD-POC2.md        POC-2 spec + results
docs/PRD-POC3.md        POC-3 sensitivity experiment spec
docs/HANDOFF.md         session handoff state
diagnostics.py          post-hoc sensitivity analysis (8 scenarios)
outputs/                generated artifacts (gitignored except .gitkeep)
outputs/poc3/           POC-3 results (seed sweep, latent dim, baselines)
```
