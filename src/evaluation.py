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
from __future__ import annotations

from typing import Optional

import pandas as pd

from src.types import Verdict


class Evaluator:
    def __init__(
        self,
        scores_df: pd.DataFrame,
        close_series: pd.Series,
        boundary_assertions_passed: bool = True,
    ) -> None:
        self._scores = scores_df
        self._close = close_series
        self._boundary_ok = boundary_assertions_passed

    def evaluate(self) -> Verdict:
        # Step 1 — crash onset computed and frozen before any criterion is evaluated.
        crash_onset_date = self._compute_crash_onset()

        # Step 2 — first breach dates.
        lejepa_fb = self._first_breach("lejepa_breach")
        rv20_fb = self._first_breach("rv20_breach")
        vix_fb = self._first_breach("vix_breach")

        # Step 3 — lead times (trading days; positive = early warning).
        lejepa_lead = self._lead_days(lejepa_fb, crash_onset_date)
        rv20_lead = self._lead_days(rv20_fb, crash_onset_date)
        vix_lead = self._lead_days(vix_fb, crash_onset_date)

        # Step 4 — 2021 episode / alert-day counts.
        lejepa_ep, lejepa_ad = self._count_2021("lejepa")
        rv20_ep, rv20_ad = self._count_2021("rv20")
        vix_ep, vix_ad = self._count_2021("vix")

        # Step 5 — evaluate criteria.
        criterion_1 = (
            lejepa_fb is not None
            and crash_onset_date is not None
            and lejepa_fb <= crash_onset_date
        )

        beats_rv20 = lejepa_fb is not None and rv20_fb is not None and lejepa_fb < rv20_fb
        beats_vix = lejepa_fb is not None and vix_fb is not None and lejepa_fb < vix_fb
        criterion_2 = beats_rv20 and beats_vix
        partial_signal = beats_rv20 ^ beats_vix  # exactly one

        max_ep = max(rv20_ep, vix_ep)
        if lejepa_ep < max_ep:
            criterion_3 = True
        elif lejepa_ep == max_ep:
            criterion_3 = lejepa_ad <= max(rv20_ad, vix_ad)
        else:
            criterion_3 = False

        criterion_4 = self._boundary_ok

        all_pass = criterion_1 and criterion_2 and criterion_3 and criterion_4

        return Verdict(
            crash_onset_date=crash_onset_date,
            lejepa_first_breach=lejepa_fb,
            rv20_first_breach=rv20_fb,
            vix_first_breach=vix_fb,
            lejepa_lead_days=lejepa_lead,
            rv20_lead_days=rv20_lead,
            vix_lead_days=vix_lead,
            lejepa_episodes_2021=lejepa_ep,
            rv20_episodes_2021=rv20_ep,
            vix_episodes_2021=vix_ep,
            lejepa_alert_days_2021=lejepa_ad,
            rv20_alert_days_2021=rv20_ad,
            vix_alert_days_2021=vix_ad,
            criterion_1=criterion_1,
            criterion_2=criterion_2,
            criterion_3=criterion_3,
            criterion_4=criterion_4,
            partial_signal=partial_signal,
            all_pass=all_pass,
        )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _compute_crash_onset(self) -> Optional[pd.Timestamp]:
        close = self._close
        n = len(close)
        for i in range(n):
            if i + 60 > n:
                break
            fwd_min = close.iloc[i : i + 60].min()
            if fwd_min / close.iloc[i] - 1 <= -0.20:
                return close.index[i]
        return None

    def _first_breach(self, col: str) -> Optional[pd.Timestamp]:
        breach = self._scores[col]
        hits = breach[breach.astype(bool)]
        if hits.empty:
            return None
        return hits.index[0]

    def _lead_days(
        self,
        breach_date: Optional[pd.Timestamp],
        crash_date: Optional[pd.Timestamp],
    ) -> Optional[int]:
        if breach_date is None or crash_date is None:
            return None
        dates = self._scores.index
        try:
            return int(dates.get_loc(crash_date) - dates.get_loc(breach_date))
        except KeyError:
            return None

    def _count_2021(self, prefix: str) -> tuple[int, int]:
        df = self._scores[self._scores.index.year == 2021]
        ep_col = f"{prefix}_episode_id"
        episodes = int(df[ep_col][df[ep_col] > 0].nunique())
        alert_days = int((df[ep_col] > 0).sum())
        return episodes, alert_days
