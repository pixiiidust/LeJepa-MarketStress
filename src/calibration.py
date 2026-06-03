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
"""
