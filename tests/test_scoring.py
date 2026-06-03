"""
Tests for Scorer (episode detection).

Cover:
- 9-day gap in breaches: two episodes detected (gap < 10-day cooldown)
- 10-day gap in breaches: two episodes detected (gap >= cooldown resets)
- pred_error_valid_date = T+10 trading days for mid-period rows
- pred_error = NaN for last 10 rows of test period
- lejepa_episode_id increments correctly across multi-episode sequences
"""
