"""
Unit tests for the Data Drift statistical monitoring module.
"""

import numpy as np
import pandas as pd
from src.monitoring.drift import DataDriftDetector, calculate_psi


def test_calculate_psi_no_drift():
    """Verify PSI is near zero for samples from the same distribution."""
    rng = np.random.RandomState(42)
    expected = rng.normal(10, 2, 1000)
    actual = rng.normal(10, 2, 1000)

    psi = calculate_psi(expected, actual)
    assert psi < 0.10


def test_data_drift_detector_detects_shift():
    """Verify drift detector flags deliberate feature distribution shift."""
    rng = np.random.RandomState(42)
    ref_df = pd.DataFrame({
        "alcohol": rng.normal(10.5, 1.2, 500),
        "pH": rng.normal(3.2, 0.15, 500),
    })

    # Shifted data: alcohol is heavily shifted
    current_df = pd.DataFrame({
        "alcohol": rng.normal(14.0, 1.2, 200),
        "pH": rng.normal(3.2, 0.15, 200),
    })

    detector = DataDriftDetector(ref_df)
    report = detector.evaluate_drift(current_df)

    assert "alcohol" in report["drifted_features"]
    assert report["drifted_features_count"] >= 1
    assert report["feature_details"]["alcohol"]["p_value"] < 0.05
