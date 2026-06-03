"""
DataPipeline: fetch, align, feature-engineer, scale, and split market data.

Public interface:
    DataPipeline().build() -> WindowDataset (train, val, calib, test)

Enforces split boundary assertions before returning any window dataset.
RobustScaler is fitted on 2010-2016 only and applied to all splits.
"""
