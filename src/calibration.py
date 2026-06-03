"""
MahalanobisCalibrator: Ledoit-Wolf Mahalanobis distribution fitting and scoring.

Instantiated twice — once per latent population:

    Primary (target-latent):
        cal.fit(z_target_calib)          # fit distribution on z_target vectors
        cal.set_threshold(z_pred_calib)  # threshold = 99th pct of z_pred scores
        cal.score(z_pred_t)              # live score on day T

    Secondary (context-latent):
        cal.fit(z_context_calib)         # fit distribution on z_context vectors
        cal.set_threshold(z_context_calib)  # threshold = 99th pct of z_context scores
        cal.score(z_context_t)           # secondary score on day T

Public interface:
    cal = MahalanobisCalibrator()
    cal.fit(z_fit: ndarray[N, 16])
    cal.set_threshold(z_calib: ndarray[N, 16], percentile=99)
    cal.score(z: ndarray[16]) -> float
    cal.threshold -> float

    regime_filter_mask(dates, rv20_series, rv20_mean, rv20_std, pct_threshold) -> ndarray[bool]
"""
import numpy as np
import pandas as pd
from sklearn.covariance import LedoitWolf


def regime_filter_mask(
    dates: pd.DatetimeIndex,
    rv20_series: pd.Series,
    rv20_mean: float,
    rv20_std: float,
    pct_threshold: float,
) -> np.ndarray:
    """Return a boolean mask (True = keep) for days where RV20 z-score <= pct_threshold.

    Reindexes rv20_series to dates before computing z-scores, so gaps become NaN.
    NaN z-scores are treated as exceeding the threshold (excluded).
    Safe to apply to either window-date indices or raw trading-day indices.
    """
    zscores = (rv20_series.reindex(dates) - rv20_mean) / rv20_std
    return (zscores <= pct_threshold).values


class MahalanobisCalibrator:
    def __init__(self) -> None:
        self.mean_: np.ndarray | None = None
        self.precision_matrix_: np.ndarray | None = None
        self.threshold: float | None = None

    def fit(self, z_fit: np.ndarray) -> None:
        lw = LedoitWolf()
        lw.fit(z_fit)
        self.mean_ = lw.location_
        self.precision_matrix_ = lw.get_precision()

    def score(self, z: np.ndarray) -> float:
        if self.mean_ is None:
            raise RuntimeError("Call fit() before score()")
        diff = z - self.mean_
        return float(np.sqrt(diff @ self.precision_matrix_ @ diff))

    def set_threshold(self, z_calib: np.ndarray, percentile: int = 99) -> None:
        if self.mean_ is None:
            raise RuntimeError("Call fit() before set_threshold()")
        scores = np.array([self.score(z_calib[i]) for i in range(len(z_calib))])
        self.threshold = float(np.percentile(scores, percentile))
