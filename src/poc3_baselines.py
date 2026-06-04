"""
PCAScorer and IsolationForestScorer: classical anomaly baselines for POC-3.

Both operate on 180-dim flattened context windows (identical input to the LeJEPA
encoder). Training statistics are frozen from the training split; thresholds are
the 99th percentile of calibration-period scores.

Public interface (mirrors BaselineScorer):
    PCAScorer(train_windows, config).fit_calibration(calib_windows).score(windows)
        columns: pca_score, pca_threshold, pca_breach, pca_episode_id

    IsolationForestScorer(train_windows, config).fit_calibration(calib_windows).score(windows)
        columns: if_score, if_threshold, if_breach, if_episode_id
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import RobustScaler

from src.types import Config, Window
from src.utils import detect_episodes


def _flatten(windows: list[Window]) -> np.ndarray:
    return np.array([w.context_array.flatten() for w in windows], dtype=np.float64)


def _dates(windows: list[Window]) -> pd.DatetimeIndex:
    return pd.DatetimeIndex([w.date for w in windows])


class PCAScorer:
    def __init__(self, train_windows: list[Window], config: Config) -> None:
        self._config = config
        self._scaler = RobustScaler()
        X_train = self._scaler.fit_transform(_flatten(train_windows))

        full_pca = PCA().fit(X_train)
        cumvar = np.cumsum(full_pca.explained_variance_ratio_)
        k = int(np.searchsorted(cumvar, 0.95)) + 1
        self._pca = PCA(n_components=k).fit(X_train)

        self.pca_threshold: float = float("nan")

    def fit_calibration(self, calib_windows: list[Window]) -> "PCAScorer":
        calib_scores = self._raw_scores(calib_windows)
        self.pca_threshold = float(np.percentile(calib_scores, 99))
        return self

    def _raw_scores(self, windows: list[Window]) -> np.ndarray:
        X = self._scaler.transform(_flatten(windows))
        X_reconstructed = self._pca.inverse_transform(self._pca.transform(X))
        residuals = X - X_reconstructed
        return np.linalg.norm(residuals, axis=1)

    def score(self, windows: list[Window]) -> pd.DataFrame:
        scores = self._raw_scores(windows)
        breach = pd.Series(scores > self.pca_threshold, dtype=bool)
        episode_ids = detect_episodes(breach, cooldown=self._config.episode_cooldown_days)
        return pd.DataFrame(
            {
                "pca_score": scores,
                "pca_threshold": self.pca_threshold,
                "pca_breach": breach.values,
                "pca_episode_id": episode_ids.values,
            },
            index=_dates(windows),
        )


class IsolationForestScorer:
    def __init__(self, train_windows: list[Window], config: Config) -> None:
        self._config = config
        self._scaler = RobustScaler()
        X_train = self._scaler.fit_transform(_flatten(train_windows))
        self._iforest = IsolationForest(n_estimators=200, random_state=42)
        self._iforest.fit(X_train)

        self.if_threshold: float = float("nan")

    def fit_calibration(self, calib_windows: list[Window]) -> "IsolationForestScorer":
        calib_scores = self._raw_scores(calib_windows)
        self.if_threshold = float(np.percentile(calib_scores, 99))
        return self

    def _raw_scores(self, windows: list[Window]) -> np.ndarray:
        X = self._scaler.transform(_flatten(windows))
        return -self._iforest.score_samples(X)

    def score(self, windows: list[Window]) -> pd.DataFrame:
        scores = self._raw_scores(windows)
        breach = pd.Series(scores > self.if_threshold, dtype=bool)
        episode_ids = detect_episodes(breach, cooldown=self._config.episode_cooldown_days)
        return pd.DataFrame(
            {
                "if_score": scores,
                "if_threshold": self.if_threshold,
                "if_breach": breach.values,
                "if_episode_id": episode_ids.values,
            },
            index=_dates(windows),
        )
