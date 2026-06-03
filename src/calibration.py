"""
MahalanobisCalibrator: Ledoit-Wolf Mahalanobis distribution fitting and scoring.

Fitted on z_target vectors from the calibration period (2019-2020).
Threshold = 99th percentile of calibration-period z_pred scores.
Instantiated separately for target-latent and context-latent distributions.

Public interface:
    cal = MahalanobisCalibrator()
    cal.fit(z_target: ndarray[N, 16])
    cal.set_threshold(z_pred_calib: ndarray[N, 16], percentile=99)
    cal.score(z: ndarray[16]) -> float
    cal.threshold -> float
"""
