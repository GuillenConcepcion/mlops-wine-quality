"""
Data Drift Detection Engine.
Calculates statistical drift between baseline training data and production inference batches
using the two-sample Kolmogorov-Smirnov Test (KS-Test) and Population Stability Index (PSI).
"""

from typing import Dict, Any, List, Optional
import numpy as np
import pandas as pd
from scipy.stats import ks_2samp


def calculate_psi(
    expected: np.ndarray,
    actual: np.ndarray,
    num_buckets: int = 10,
    eps: float = 1e-4,
) -> float:
    """
    Computes Population Stability Index (PSI) between two numeric distributions.
    Rule of Thumb:
      - PSI < 0.1: No significant change / stable
      - 0.1 <= PSI < 0.2: Moderate drift / monitoring advised
      - PSI >= 0.2: Significant drift / retrain recommended
    """
    expected = expected[~np.isnan(expected)]
    actual = actual[~np.isnan(actual)]

    if len(expected) == 0 or len(actual) == 0:
        return 0.0

    # Determine quantile bins based on expected baseline
    quantiles = np.linspace(0, 100, num_buckets + 1)
    bins = np.percentile(expected, quantiles)
    bins[0] = -np.inf
    bins[-1] = np.inf

    expected_counts, _ = np.histogram(expected, bins=bins)
    actual_counts, _ = np.histogram(actual, bins=bins)

    expected_pct = expected_counts / len(expected) + eps
    actual_pct = actual_counts / len(actual) + eps

    psi_val = np.sum((actual_pct - expected_pct) * np.log(actual_pct / expected_pct))
    return float(psi_val)


class DataDriftDetector:
    """
    Evaluates covariate data drift for production batches compared to reference baseline.
    """

    def __init__(
        self,
        reference_data: pd.DataFrame,
        alpha_ks: float = 0.05,
        psi_threshold: float = 0.2,
    ):
        self.reference_data = reference_data
        self.alpha_ks = alpha_ks
        self.psi_threshold = psi_threshold

    def evaluate_drift(
        self,
        current_data: pd.DataFrame,
        features: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """
        Runs KS-test and PSI for all evaluated features.
        Returns detailed per-feature diagnostics and a global system health verdict.
        """
        if features is None:
            features = [col for col in self.reference_data.columns if col in current_data.columns]

        drifted_features = []
        feature_reports = {}

        for feat in features:
            ref_series = self.reference_data[feat].dropna().values
            curr_series = current_data[feat].dropna().values

            if len(curr_series) < 10:
                continue

            # Two-sample KS-Test
            ks_stat, p_value = ks_2samp(ref_series, curr_series)
            psi = calculate_psi(ref_series, curr_series)

            is_ks_drift = bool(p_value < self.alpha_ks)
            is_psi_drift = bool(psi >= self.psi_threshold)
            drift_detected = is_ks_drift or is_psi_drift

            if drift_detected:
                drifted_features.append(feat)

            feature_reports[feat] = {
                "ks_statistic": round(float(ks_stat), 4),
                "p_value": round(float(p_value), 6),
                "psi": round(float(psi), 4),
                "ks_drift": is_ks_drift,
                "psi_drift": is_psi_drift,
                "drift_detected": drift_detected,
            }

        drift_ratio = len(drifted_features) / len(features) if features else 0.0
        retrain_recommended = drift_ratio >= 0.30 or any(
            feature_reports[f]["psi"] >= 0.25 for f in drifted_features
        )

        return {
            "n_reference_samples": len(self.reference_data),
            "n_current_samples": len(current_data),
            "total_features_evaluated": len(features),
            "drifted_features_count": len(drifted_features),
            "drifted_features": drifted_features,
            "drift_ratio": round(drift_ratio, 4),
            "retrain_recommended": retrain_recommended,
            "feature_details": feature_reports,
        }
