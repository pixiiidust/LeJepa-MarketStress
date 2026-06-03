"""
Tests for Evaluator (crash onset and pass/fail logic).

Cover:
- Crash onset = correct day on synthetic close series with known 20% forward drop
- No onset set when series has no complete 60-day forward window
- Criterion 2 = partial signal when LeJEPA beats exactly one baseline on timing
- Criterion 3 tiebreaker fires correctly when episode counts tie but alert days differ
"""
import random

import numpy as np
import pandas as pd
import pytest

from src.evaluation import Evaluator
from src.utils import detect_episodes

random.seed(42)
np.random.seed(42)


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------

def _make_close(n: int, drop_at: int | None = None, drop_value: float = 75.0) -> pd.Series:
    """n business-day close series starting 2021-01-04, all 100.0 except optional drop."""
    dates = pd.bdate_range("2021-01-04", periods=n)
    close = pd.Series(100.0, index=dates)
    if drop_at is not None:
        close.iloc[drop_at] = drop_value
    return close


def _make_scores_df(
    dates: pd.DatetimeIndex,
    lejepa_breach_days: list[int] | None = None,
    rv20_breach_days: list[int] | None = None,
    vix_breach_days: list[int] | None = None,
) -> pd.DataFrame:
    lejepa_breach = pd.Series(False, index=dates)
    rv20_breach = pd.Series(False, index=dates)
    vix_breach = pd.Series(False, index=dates)

    for col, days in [
        (lejepa_breach, lejepa_breach_days),
        (rv20_breach, rv20_breach_days),
        (vix_breach, vix_breach_days),
    ]:
        if days:
            for d in days:
                col.iloc[d] = True

    lejepa_ep = detect_episodes(lejepa_breach, cooldown=10)
    rv20_ep = detect_episodes(rv20_breach, cooldown=10)
    vix_ep = detect_episodes(vix_breach, cooldown=10)

    return pd.DataFrame(
        {
            "lejepa_breach": lejepa_breach,
            "lejepa_episode_id": lejepa_ep,
            "rv20_breach": rv20_breach,
            "rv20_episode_id": rv20_ep,
            "vix_breach": vix_breach,
            "vix_episode_id": vix_ep,
        },
        index=dates,
    )


# ---------------------------------------------------------------------------
# Crash onset tests
# ---------------------------------------------------------------------------

def test_crash_onset_detected_at_correct_day():
    # drop at iloc[69] = 75 (from 100).
    # Window [T:T+60] first contains iloc[69] when T=10 (positions 10..69).
    # 10+60=70 <= 120 → full window. Expected onset = dates[10].
    close = _make_close(120, drop_at=69, drop_value=75.0)
    scores_df = _make_scores_df(close.index)
    v = Evaluator(scores_df, close).evaluate()
    assert v.crash_onset_date == close.index[10]


def test_crash_onset_none_when_no_complete_forward_window():
    # n=59: first window would need i+60 <= 59, impossible for any i >= 0.
    # Even with a 25% drop present, no full window exists → onset must be None.
    close = _make_close(59, drop_at=10, drop_value=75.0)
    scores_df = _make_scores_df(close.index)
    v = Evaluator(scores_df, close).evaluate()
    assert v.crash_onset_date is None


# ---------------------------------------------------------------------------
# Criterion 2 — partial signal
# ---------------------------------------------------------------------------

def test_criterion_2_partial_signal_when_exactly_one_baseline_beaten():
    # lejepa breaches day 5; rv20 day 8 (beaten); vix day 3 (not beaten).
    # criterion_2 must be False; partial_signal must be True.
    close = _make_close(120)
    scores_df = _make_scores_df(
        close.index,
        lejepa_breach_days=[5],
        rv20_breach_days=[8],
        vix_breach_days=[3],
    )
    v = Evaluator(scores_df, close).evaluate()
    assert v.criterion_2 is False
    assert v.partial_signal is True


def test_criterion_2_true_and_no_partial_when_both_baselines_beaten():
    close = _make_close(120)
    scores_df = _make_scores_df(
        close.index,
        lejepa_breach_days=[5],
        rv20_breach_days=[8],
        vix_breach_days=[9],
    )
    v = Evaluator(scores_df, close).evaluate()
    assert v.criterion_2 is True
    assert v.partial_signal is False


def test_criterion_2_false_and_no_partial_when_no_baseline_beaten():
    # LeJEPA is last to breach — beats neither baseline.
    close = _make_close(120)
    scores_df = _make_scores_df(
        close.index,
        lejepa_breach_days=[9],
        rv20_breach_days=[3],
        vix_breach_days=[5],
    )
    v = Evaluator(scores_df, close).evaluate()
    assert v.criterion_2 is False
    assert v.partial_signal is False


# ---------------------------------------------------------------------------
# Criterion 3 — episode count + tiebreaker
# ---------------------------------------------------------------------------

def test_criterion_3_tiebreaker_passes_when_lejepa_alert_days_within_max():
    # 2021: lejepa 1 ep / 5 alert days; rv20 1 ep / 10 days; vix 1 ep / 3 days.
    # Counts tie → tiebreaker fires: 5 <= max(10, 3) = 10 → criterion_3 = True.
    dates = pd.bdate_range("2021-01-04", periods=504)
    close = pd.Series(100.0, index=dates)
    scores_df = _make_scores_df(
        dates,
        lejepa_breach_days=list(range(2, 7)),    # 5 days → 1 episode
        rv20_breach_days=list(range(30, 40)),     # 10 days → 1 episode
        vix_breach_days=list(range(50, 53)),      # 3 days → 1 episode
    )
    v = Evaluator(scores_df, close).evaluate()

    assert v.lejepa_episodes_2021 == 1
    assert v.rv20_episodes_2021 == 1
    assert v.vix_episodes_2021 == 1
    assert v.lejepa_alert_days_2021 == 5
    assert v.rv20_alert_days_2021 == 10
    assert v.criterion_3 is True


def test_criterion_3_tiebreaker_fails_when_lejepa_alert_days_exceed_max():
    # Same structure but lejepa has 15 alert days vs max baseline 10 → fails.
    dates = pd.bdate_range("2021-01-04", periods=504)
    close = pd.Series(100.0, index=dates)
    scores_df = _make_scores_df(
        dates,
        lejepa_breach_days=list(range(2, 17)),   # 15 days → 1 episode
        rv20_breach_days=list(range(30, 40)),     # 10 days → 1 episode
        vix_breach_days=list(range(50, 53)),      # 3 days → 1 episode
    )
    v = Evaluator(scores_df, close).evaluate()

    assert v.lejepa_episodes_2021 == 1
    assert v.lejepa_alert_days_2021 == 15
    assert v.criterion_3 is False


def test_criterion_3_tiebreaker_does_not_fire_when_episode_counts_differ():
    # lejepa has fewer episodes → tiebreaker must NOT fire; criterion_3 passes directly.
    dates = pd.bdate_range("2021-01-04", periods=504)
    close = pd.Series(100.0, index=dates)
    # Give lejepa 1 ep, baselines 2 eps each (two separated bursts, cooldown=10).
    scores_df = _make_scores_df(
        dates,
        lejepa_breach_days=list(range(2, 7)),          # 1 episode
        rv20_breach_days=list(range(2, 7)) + list(range(25, 30)),   # 2 episodes
        vix_breach_days=list(range(2, 7)) + list(range(25, 30)),    # 2 episodes
    )
    v = Evaluator(scores_df, close).evaluate()

    assert v.lejepa_episodes_2021 == 1
    assert v.rv20_episodes_2021 == 2
    # criterion_3: 1 < max(2,2)=2 → True without tiebreaker
    assert v.criterion_3 is True


# ---------------------------------------------------------------------------
# all_pass and Verdict consistency
# ---------------------------------------------------------------------------

def test_all_pass_true_when_all_four_criteria_pass():
    # Crash onset at dates[10]; lejepa first at day 5, rv20 day 7, vix day 8.
    # criterion_1 ✓, criterion_2 ✓, criterion_3 ✓ (tiebreaker: 1<=1), criterion_4 ✓.
    dates = pd.bdate_range("2021-01-04", periods=504)
    close = pd.Series(100.0, index=dates)
    close.iloc[69] = 75.0  # onset at dates[10]
    scores_df = _make_scores_df(
        dates,
        lejepa_breach_days=[5],
        rv20_breach_days=[7],
        vix_breach_days=[8],
    )
    v = Evaluator(scores_df, close, boundary_assertions_passed=True).evaluate()

    assert v.criterion_1 is True
    assert v.criterion_2 is True
    assert v.criterion_3 is True
    assert v.criterion_4 is True
    assert v.all_pass is True


def test_all_pass_false_when_criterion_4_fails():
    dates = pd.bdate_range("2021-01-04", periods=504)
    close = pd.Series(100.0, index=dates)
    close.iloc[69] = 75.0
    scores_df = _make_scores_df(
        dates,
        lejepa_breach_days=[5],
        rv20_breach_days=[7],
        vix_breach_days=[8],
    )
    v = Evaluator(scores_df, close, boundary_assertions_passed=False).evaluate()

    assert v.criterion_4 is False
    assert v.all_pass is False


def test_all_pass_equals_conjunction_of_four_criteria():
    # Property test: all_pass must equal criterion_1 & 2 & 3 & 4 for any Verdict.
    dates = pd.bdate_range("2021-01-04", periods=504)
    close = pd.Series(100.0, index=dates)
    close.iloc[69] = 75.0
    scores_df = _make_scores_df(
        dates,
        lejepa_breach_days=[5],
        rv20_breach_days=[3],  # lejepa doesn't beat vix here → criterion_2 False
        vix_breach_days=[8],
    )
    v = Evaluator(scores_df, close).evaluate()

    assert v.all_pass == (v.criterion_1 and v.criterion_2 and v.criterion_3 and v.criterion_4)
