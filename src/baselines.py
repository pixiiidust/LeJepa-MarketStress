"""
BaselineScorer: RV20 z-score and VIX level z-score baselines.

Normalisation statistics (mean, std) are computed from train_df in __init__.
Call fit_calibration(calib_df) once to set rv20_threshold and vix_threshold
(99th percentile of calibration-period z-scores). Then call score(df) freely
for any split — the thresholds and normalisation stats are frozen.

Episode detection delegates to src.utils.detect_episodes so the logic is
identical to Scorer (>=10 consecutive non-breach days to end an episode).

Public interface:
    bs = BaselineScorer(train_df: pd.DataFrame, config: Config)
    bs.fit_calibration(calib_df: pd.DataFrame) -> BaselineScorer
    bs.score(df: pd.DataFrame) -> pd.DataFrame
        columns: rv20_zscore, rv20_threshold, rv20_breach, rv20_episode_id,
                 vix_zscore, vix_threshold, vix_breach, vix_episode_id
    bs.rv20_threshold -> float
    bs.vix_threshold  -> float
"""
import numpy as np
import pandas as pd

from src.types import Config
from src.utils import detect_episodes


class BaselineScorer:
    def __init__(self, train_df: pd.DataFrame, config: Config) -> None:
        self._config = config
        self.rv20_mean: float = float(train_df["rv20"].mean())
        self.rv20_std: float = float(train_df["rv20"].std())
        self.vix_mean: float = float(train_df["vix"].mean())
        self.vix_std: float = float(train_df["vix"].std())
        self.rv20_threshold: float = float("nan")
        self.vix_threshold: float = float("nan")

    def fit_calibration(self, calib_df: pd.DataFrame) -> "BaselineScorer":
        rv20_z = (calib_df["rv20"] - self.rv20_mean) / self.rv20_std
        vix_z = (calib_df["vix"] - self.vix_mean) / self.vix_std
        self.rv20_threshold = float(np.percentile(rv20_z, 99))
        self.vix_threshold = float(np.percentile(vix_z, 99))
        return self

    def score(self, df: pd.DataFrame) -> pd.DataFrame:
        rv20_z = (df["rv20"] - self.rv20_mean) / self.rv20_std
        vix_z = (df["vix"] - self.vix_mean) / self.vix_std

        rv20_breach = rv20_z > self.rv20_threshold
        vix_breach = vix_z > self.vix_threshold

        cooldown = self._config.episode_cooldown_days
        rv20_episode_id = detect_episodes(rv20_breach, cooldown=cooldown)
        vix_episode_id = detect_episodes(vix_breach, cooldown=cooldown)

        return pd.DataFrame(
            {
                "rv20_zscore": rv20_z.values,
                "rv20_threshold": self.rv20_threshold,
                "rv20_breach": rv20_breach.values,
                "rv20_episode_id": rv20_episode_id.values,
                "vix_zscore": vix_z.values,
                "vix_threshold": self.vix_threshold,
                "vix_breach": vix_breach.values,
                "vix_episode_id": vix_episode_id.values,
            },
            index=df.index,
        )
