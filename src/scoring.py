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
from __future__ import annotations

import numpy as np
import pandas as pd
import torch

from src.calibration import MahalanobisCalibrator
from src.model import LeJEPAModel
from src.types import Window
from src.utils import detect_episodes

_EPISODE_COOLDOWN = 10


class Scorer:
    def __init__(
        self,
        model: LeJEPAModel,
        target_cal: MahalanobisCalibrator,
        context_cal: MahalanobisCalibrator,
        trading_days: pd.DatetimeIndex,
    ) -> None:
        self._model = model
        self._target_cal = target_cal
        self._context_cal = context_cal
        self._trading_days = trading_days

    def score(self, test_windows: list[Window]) -> pd.DataFrame:
        self._model.eval()
        n_days = len(self._trading_days)
        rows: list[dict] = []

        with torch.no_grad():
            for i, window in enumerate(test_windows):
                # Encode context window → z_context (16-dim)
                ctx_flat = torch.tensor(
                    window.context_array.flatten(), dtype=torch.float32
                )
                z_context = self._model.encode(ctx_flat)
                z_pred = self._model.predict(z_context)

                # Encode future-context window → z_target (last 20 context + 10 target days)
                future_ctx = np.concatenate(
                    [window.context_array[-20:], window.target_array], axis=0
                )
                z_target = self._model.encode(
                    torch.tensor(future_ctx.flatten(), dtype=torch.float32)
                )

                z_pred_np: np.ndarray = z_pred.numpy()
                z_ctx_np: np.ndarray = z_context.numpy()

                lejepa_score = float(self._target_cal.score(z_pred_np))
                lejepa_context_score = float(self._context_cal.score(z_ctx_np))

                # Diagnostic: pred_error valid only when T+10 is within the trading calendar
                if i + 10 < n_days:
                    pred_error: float = float(
                        np.linalg.norm(z_pred_np - z_target.numpy())
                    )
                    pred_error_valid_date = self._trading_days[i + 10]
                else:
                    pred_error = float("nan")
                    pred_error_valid_date = pd.NaT

                lejepa_breach = lejepa_score > self._target_cal.threshold

                row: dict = {
                    "lejepa_score": lejepa_score,
                    "lejepa_context_score": lejepa_context_score,
                    "pred_error": pred_error,
                    "pred_error_valid_date": pred_error_valid_date,
                    "lejepa_breach": lejepa_breach,
                }
                for j in range(16):
                    row[f"z_pred_{j}"] = float(z_pred_np[j])
                    row[f"z_context_{j}"] = float(z_ctx_np[j])

                rows.append(row)

        df = pd.DataFrame(rows, index=self._trading_days[: len(rows)])
        df["lejepa_episode_id"] = detect_episodes(
            df["lejepa_breach"], cooldown=_EPISODE_COOLDOWN
        )
        return df
