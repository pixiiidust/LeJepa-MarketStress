"""
Scorer: per-day anomaly scoring and alert episode detection for the blind test period.

Iterates 2021-2022 in causal order. For each day T:
  - primary score:    Mahalanobis(z_pred, calib_target_dist)
  - secondary score:  Mahalanobis(z_context, calib_context_dist)
  - diagnostic score: ||z_pred - z_target||_2, valid at T+10 trading days (NaN for last 10)

pred_error_valid_date uses the trading calendar (WindowDataset.test_dates), not
calendar days. "Last 10 rows" means the last 10 entries of test_dates, not the
last 10 calendar days. Pass test_dates explicitly so the T+10 lookup is unambiguous
and independently testable at test-period boundaries.

Episode detection delegates to src.utils.detect_episodes.

Public interface:
    Scorer(
        model,
        target_cal: MahalanobisCalibrator,
        context_cal: MahalanobisCalibrator,
        trading_days: pd.DatetimeIndex,   # = WindowDataset.test_dates
    ).score(test_windows: list[Window]) -> pd.DataFrame
"""
