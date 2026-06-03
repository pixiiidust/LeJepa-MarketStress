"""
BaselineScorer: RV20 z-score and VIX level z-score baselines.

Normalization statistics fitted on 2010-2016 training data only.
Threshold = 99th percentile of calibration-period z-scores (2019-2020).
Identical alert episode logic as Scorer (>=10 consecutive non-breach days to end).

Public interface:
    BaselineScorer().fit(train_df).score_calibration(calib_df).score_test(test_df) -> pd.DataFrame
"""
