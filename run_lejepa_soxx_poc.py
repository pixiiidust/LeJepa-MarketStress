"""
LeJEPA SOXX Market Stress POC-1

Single entry point for the full experiment:
    1. Set seeds (42)
    2. Build data pipeline (fetch, align, feature-engineer, scale, split)
    3. Train LeJEPA model (2010-2016, early-stop on 2017-2018)
    4. Calibrate Mahalanobis distributions (2019-2020)
    5. Score blind test period (2021-2022)
    6. Score baselines (RV20, VIX level)
    7. Evaluate (crash onset, lead times, episode counts, pass/fail)
    8. Write outputs (scores.csv, chart, run_config.json, console summary)

Usage:
    python run_lejepa_soxx_poc.py

Outputs:
    outputs/scores.csv
    outputs/soxx_lejepa_chart.png
    outputs/run_config.json
"""
