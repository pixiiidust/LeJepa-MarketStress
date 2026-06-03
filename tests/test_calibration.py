"""
Tests for MahalanobisCalibrator.

Cover:
- threshold equals 99th percentile of calibration scores on synthetic normal vectors
- score(z) increases monotonically as z moves away from the distribution mean
- Ledoit-Wolf shrinkage produces bounded condition number on low-sample synthetic data
"""
