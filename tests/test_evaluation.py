"""
Tests for Evaluator (crash onset and pass/fail logic).

Cover:
- Crash onset = correct day on synthetic close series with known 20% forward drop
- No onset set for days with incomplete 60-day forward window near period end
- Criterion 2 = partial signal when LeJEPA beats RV20 but not VIX
- Criterion 3 tiebreaker fires correctly when episode counts tie but alert days differ
"""
