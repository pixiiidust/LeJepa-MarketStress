"""
Scorer: per-day anomaly scoring and alert episode detection for the blind test period.

Iterates 2021-2022 in causal order. For each day T:
  - primary score:    Mahalanobis(z_pred, calib_target_dist)
  - secondary score:  Mahalanobis(z_context, calib_context_dist)
  - diagnostic score: ||z_pred - z_target||_2, valid at T+10 (NaN for last 10 days)

Episode detection: episode starts on breach, ends after >=10 consecutive non-breach days.

Public interface:
    Scorer(model, target_cal, context_cal).score(test_windows, test_close) -> pd.DataFrame
"""
