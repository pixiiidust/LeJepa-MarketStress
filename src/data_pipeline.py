"""
DataPipeline: fetch, align, feature-engineer, scale, and split market data.

Public interface:
    DataPipeline(config).build() -> WindowDataset

Enforces split boundary assertions before returning any window dataset.
RobustScaler is fitted on 2010-2016 only and applied to all splits.
"""
from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler

from src.types import Config, WindowDataset
from src.utils import make_windows

logger = logging.getLogger(__name__)

FEATURE_COLS = [
    "soxx_log_return",
    "log_volume_ratio",
    "rv20",
    "rolling_drawdown",
    "vix_daily_change",
    "dgs10_daily_change",
]


def compute_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute six causal features from an aligned raw DataFrame.

    Input columns: close, volume, vix, dgs10
    Output columns: soxx_log_return, log_volume_ratio, rv20, rolling_drawdown,
                    vix_daily_change, dgs10_daily_change
    """
    out = pd.DataFrame(index=df.index)

    log_return = np.log(df["close"] / df["close"].shift(1))
    out["soxx_log_return"] = log_return

    # mean(volume[t-20:t-1]): shift(1) brings t-1 to position t; rolling(20) takes 20 prev days
    vol_mean_prev20 = df["volume"].shift(1).rolling(20).mean()
    out["log_volume_ratio"] = np.log(df["volume"] / vol_mean_prev20)

    out["rv20"] = log_return.rolling(20).std() * np.sqrt(252)

    # expanding for first 252 days, then rolling 252-day high
    out["rolling_drawdown"] = df["close"] / df["close"].rolling(252, min_periods=1).max() - 1

    out["vix_daily_change"] = df["vix"] - df["vix"].shift(1)
    out["dgs10_daily_change"] = df["dgs10"] - df["dgs10"].shift(1)

    return out


def forward_fill_dgs10(dgs10: pd.Series, max_fill: int = 3) -> pd.Series:
    """Forward-fill DGS10 up to max_fill consecutive missing values."""
    filled = dgs10.ffill(limit=max_fill)
    remaining = int(filled.isna().sum())
    if remaining > 0:
        logger.warning(
            "DGS10: %d NaN values remain after forward-fill (max_fill=%d). "
            "Gaps longer than %d trading days were not filled.",
            remaining, max_fill, max_fill,
        )
    return filled


def fit_and_apply_scaler(
    features_df: pd.DataFrame,
    train_end: str,
) -> tuple[pd.DataFrame, RobustScaler]:
    """
    Fit RobustScaler on rows where index <= train_end; apply to all rows.

    NaN positions in features_df are preserved in the output.
    Returns (scaled_df, fitted_scaler).
    """
    train_mask = features_df.index <= pd.Timestamp(train_end)
    scaler = RobustScaler()
    scaler.fit(features_df.loc[train_mask].dropna())

    nan_mask = features_df.isna()
    scaled_values = scaler.transform(features_df.fillna(0.0))
    scaled_df = pd.DataFrame(
        scaled_values,
        index=features_df.index,
        columns=features_df.columns,
    )
    scaled_df[nan_mask] = np.nan
    return scaled_df, scaler


def assert_split_boundaries(
    train_dates: pd.DatetimeIndex,
    calib_dates: pd.DatetimeIndex,
    config: Config,
) -> None:
    """
    Assert that feature date ranges respect the hard split boundaries.

    Raises AssertionError if train_dates[-1] > train_end or
    calib_dates[0] < calib_start.
    """
    train_end = pd.Timestamp(config.train_end)
    calib_start = pd.Timestamp(config.calib_start)

    assert train_dates[-1] <= train_end, (
        f"Last training date {train_dates[-1].date()} > train_end {config.train_end}. "
        "Windows would cross the train/validation split boundary."
    )
    assert calib_dates[0] >= calib_start, (
        f"First calibration date {calib_dates[0].date()} < calib_start {config.calib_start}. "
        "Windows would cross the validation/calibration split boundary."
    )


class DataPipeline:
    """Fetch, align, feature-engineer, scale, and split market data."""

    def __init__(self, config: Config):
        self.config = config
        self.scaler: Optional[RobustScaler] = None

    def build(self) -> WindowDataset:
        """Full pipeline: fetch → align → features → scale → assert → windows."""
        raw_df = self._fetch()
        return self._build_from_raw(raw_df)

    # ------------------------------------------------------------------
    # Internal helpers (separated so tests can exercise processing logic
    # without making network calls)
    # ------------------------------------------------------------------

    def _fetch(self) -> pd.DataFrame:
        """Download SOXX, VIX, DGS10 and return aligned raw DataFrame."""
        import yfinance as yf
        import pandas_datareader as pdr

        cfg = self.config
        start, end = cfg.train_start, cfg.test_end

        soxx = yf.download(cfg.equity_ticker, start=start, end=end, progress=False)
        soxx.columns = soxx.columns.get_level_values(0).str.lower()
        soxx = soxx[["close", "volume"]].dropna()

        vix_raw = yf.download(cfg.vix_ticker, start=start, end=end, progress=False)
        vix_raw.columns = vix_raw.columns.get_level_values(0).str.lower()
        vix = vix_raw["close"].reindex(soxx.index)

        dgs10_raw = pdr.get_data_fred(cfg.bond_series, start=start, end=end)
        dgs10 = dgs10_raw[cfg.bond_series].reindex(soxx.index)
        dgs10 = forward_fill_dgs10(dgs10, max_fill=3)

        missing_vix = int(vix.isna().sum())
        missing_dgs10 = int(dgs10.isna().sum())
        print(f"[DataPipeline] Missing VIX rows after alignment: {missing_vix}")
        print(f"[DataPipeline] Missing DGS10 rows after alignment: {missing_dgs10}")

        raw_df = pd.DataFrame(
            {"close": soxx["close"], "volume": soxx["volume"], "vix": vix, "dgs10": dgs10},
            index=soxx.index,
        )
        assert raw_df.isna().sum().sum() == 0, (
            f"NaN values remain after alignment:\n{raw_df.isna().sum()}"
        )
        return raw_df

    def _build_from_raw(self, raw_df: pd.DataFrame) -> WindowDataset:
        """Process an aligned raw DataFrame into a WindowDataset."""
        cfg = self.config

        features = compute_features(raw_df)
        self.raw_df = raw_df
        self.features_df = features
        scaled, self.scaler = fit_and_apply_scaler(features, cfg.train_end)
        scaled = scaled.dropna()

        def _slice(s: str, e: str):
            mask = (scaled.index >= pd.Timestamp(s)) & (scaled.index <= pd.Timestamp(e))
            sl = scaled.loc[mask]
            return sl, sl.index

        train_f, train_dates = _slice(cfg.train_start, cfg.train_end)
        val_f,   val_dates   = _slice(cfg.val_start,   cfg.val_end)
        calib_f, calib_dates = _slice(cfg.calib_start, cfg.calib_end)
        test_f,  test_dates  = _slice(cfg.test_start,  cfg.test_end)

        assert_split_boundaries(train_dates, calib_dates, cfg)

        for name, dates in [
            ("train", train_dates), ("val", val_dates),
            ("calib", calib_dates), ("test", test_dates),
        ]:
            print(f"[DataPipeline] {name}: {len(dates)} trading days")

        ctx, tgt = cfg.context_window, cfg.target_window
        return WindowDataset(
            train=make_windows(train_f.values, train_dates, ctx, tgt),
            val=make_windows(val_f.values,   val_dates,   ctx, tgt),
            calib=make_windows(calib_f.values, calib_dates, ctx, tgt),
            test=make_windows(test_f.values,  test_dates,  ctx, tgt),
            train_dates=train_dates,
            val_dates=val_dates,
            calib_dates=calib_dates,
            test_dates=test_dates,
        )
