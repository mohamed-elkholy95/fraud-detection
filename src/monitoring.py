"""Monitoring and drift detection."""
import logging
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from scipy import stats

from src.config import KS_THRESHOLD, PSI_THRESHOLD, DRIFT_WINDOW_SIZE

logger = logging.getLogger(__name__)


class DriftDetector:
    """Monitor feature drift and performance degradation.

    Uses KS test and PSI to detect distribution shifts between
    reference (training) and current (production) data.
    """

    def __init__(
        self,
        reference_data: Optional[pd.DataFrame] = None,
        window_size: int = DRIFT_WINDOW_SIZE,
        ks_threshold: float = KS_THRESHOLD,
        psi_threshold: float = PSI_THRESHOLD,
    ) -> None:
        self._reference = reference_data
        self._window: List[Dict] = []
        self._window_size = window_size
        self._ks_threshold = ks_threshold
        self._psi_threshold = psi_threshold
        self._reference_stats: Dict[str, Dict] = {}

        if reference_data is not None:
            self._compute_reference_stats(reference_data)

    def _compute_reference_stats(self, df: pd.DataFrame) -> None:
        """Pre-compute reference distribution statistics."""
        for col in df.select_dtypes(include=[np.number]).columns:
            self._reference_stats[col] = {
                "mean": df[col].mean(),
                "std": df[col].std(),
                "min": df[col].min(),
                "max": df[col].max(),
                "q25": df[col].quantile(0.25),
                "q50": df[col].quantile(0.50),
                "q75": df[col].quantile(0.75),
            }

    def check_feature_drift(
        self,
        current: pd.DataFrame,
        features: Optional[List[str]] = None,
    ) -> Dict[str, Dict]:
        """Run KS test and PSI for each feature.

        Args:
            current: Current production data.
            features: Features to check. If None, checks all numeric.

        Returns:
            Dict with drift metrics per feature.
        """
        if self._reference is None:
            raise RuntimeError("Reference data not set")

        if features is None:
            features = [c for c in self._reference.select_dtypes(include=[np.number]).columns
                        if c in current.columns]

        results = {}
        for feat in features:
            ref_vals = self._reference[feat].dropna().values
            cur_vals = current[feat].dropna().values

            if len(ref_vals) == 0 or len(cur_vals) == 0:
                continue

            # KS Test
            ks_stat, ks_pvalue = stats.ks_2samp(ref_vals, cur_vals)
            ks_drift = ks_pvalue < self._ks_threshold

            # PSI
            psi = self._compute_psi(ref_vals, cur_vals)
            psi_drift = psi > self._psi_threshold

            results[feat] = {
                "ks_statistic": round(float(ks_stat), 4),
                "ks_pvalue": round(float(ks_pvalue), 6),
                "ks_drift_detected": ks_drift,
                "psi": round(psi, 4),
                "psi_drift_detected": psi_drift,
                "drift_detected": ks_drift or psi_drift,
            }

        return results

    @staticmethod
    def _compute_psi(expected: np.ndarray, actual: np.ndarray, bins: int = 10) -> float:
        """Compute Population Stability Index.

        Args:
            expected: Reference distribution values.
            actual: Current distribution values.
            bins: Number of bins.

        Returns:
            PSI value.
        """
        breakpoints = np.percentile(expected, np.linspace(0, 100, bins + 1))
        breakpoints[0] = -np.inf
        breakpoints[-1] = np.inf

        expected_counts = np.histogram(expected, bins=breakpoints)[0]
        actual_counts = np.histogram(actual, bins=breakpoints)[0]

        # Avoid division by zero
        expected_pct = expected_counts / len(expected) + 1e-9
        actual_pct = actual_counts / len(actual) + 1e-9

        psi = np.sum((actual_pct - expected_pct) * np.log(actual_pct / expected_pct))
        return float(psi)

    def check_performance_drift(
        self,
        reference_metrics: Dict[str, float],
        current_metrics: Dict[str, float],
        threshold: float = 0.05,
    ) -> Dict[str, Dict]:
        """Compare current vs reference metrics.

        Args:
            reference_metrics: Baseline metric values.
            current_metrics: Current metric values.
            threshold: Relative degradation threshold.

        Returns:
            Drift assessment per metric.
        """
        results = {}
        for metric, ref_val in reference_metrics.items():
            cur_val = current_metrics.get(metric, 0.0)
            degradation = ref_val - cur_val
            relative = degradation / ref_val if ref_val != 0 else 0.0
            results[metric] = {
                "reference": round(ref_val, 4),
                "current": round(cur_val, 4),
                "degradation": round(degradation, 4),
                "relative_degradation": round(relative, 4),
                "drift_detected": relative > threshold,
            }
        return results

    def log_prediction(self, prediction: Dict, actual: Optional[int] = None) -> None:
        """Store prediction for monitoring.

        Args:
            prediction: Prediction dict.
            actual: Ground truth label if available.
        """
        if actual is not None:
            prediction["actual"] = actual
        self._window.append(prediction)
        if len(self._window) > self._window_size:
            self._window = self._window[-self._window_size:]
