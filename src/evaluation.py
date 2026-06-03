"""
Evaluator: crash onset detection, lead time computation, and pass/fail verdict.

Crash onset: first test day T with full 60-day forward window where
    min(close[T:T+60]) / close[T] - 1 <= -0.20

Pass/fail criteria:
    1. lejepa_first_breach <= crash_onset_date
    2. lejepa_first_breach < rv20_first_breach AND < vix_first_breach
       (partial signal if one beaten, not both)
    3. lejepa_episodes_2021 <= max(rv20_episodes_2021, vix_episodes_2021)
       tiebreaker: alert_days_2021
    4. split boundary assertions passed (evaluated by DataPipeline)

Public interface:
    Evaluator(scores_df, close_series).evaluate() -> Verdict
"""
