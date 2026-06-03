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
