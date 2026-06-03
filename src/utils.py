"""
Shared utilities: detect_episodes, make_windows.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.types import Window


def detect_episodes(breach: pd.Series, cooldown: int = 10) -> pd.Series:
    """
    Assign episode IDs to a boolean breach series.

    An episode starts on the first breach day. It remains active until
    `cooldown` consecutive non-breach days have elapsed, at which point it
    ends. A subsequent breach starts a new episode.

    Returns a pd.Series of int aligned to breach.index:
        0  = no active episode
        N  = Nth episode (1-based)
    """
    ids = pd.Series(0, index=breach.index, dtype=int)
    episode = 0
    quiet_streak = cooldown  # pre-seeded so first breach immediately opens ep 1

    for loc, is_breach in enumerate(breach):
        if is_breach:
            if quiet_streak >= cooldown:
                episode += 1
            quiet_streak = 0
            ids.iloc[loc] = episode
        else:
            if episode > 0 and quiet_streak < cooldown:
                quiet_streak += 1

    return ids


def make_windows(
    features: np.ndarray,
    dates: pd.DatetimeIndex,
    context: int,
    target: int,
) -> list[Window]:
    """
    Slice (features, dates) into (context, target) Window pairs.

    Each Window's date is the last day of its context slice (day T).
    Trailing incomplete windows are dropped.

    Args:
        features: (N, n_features) float array, rows ordered by trading day.
        dates:    DatetimeIndex of length N aligned to features rows.
        context:  Number of trading days in the context window.
        target:   Number of trading days in the target window.

    Returns:
        List of Window, length = N - context - target + 1.
    """
    n = len(features)
    windows = []
    for start in range(n - context - target + 1):
        ctx_end = start + context
        tgt_end = ctx_end + target
        windows.append(Window(
            context_array=features[start:ctx_end],
            target_array=features[ctx_end:tgt_end],
            date=dates[ctx_end - 1],
        ))
    return windows
