"""
Tests for DataPipeline.

Cover:
- Scaler fitted on train slice only (stats match training-period ground truth)
- Alignment produces no NaN after forward-fill on synthetic series with gaps
- Split boundary assertion fires when a window spans two splits
- log_volume_ratio excludes today (t-20:t-1, not t-20:t)
"""
