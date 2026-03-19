"""Tests for monitoring."""
import pytest
import numpy as np
import pandas as pd

from src.monitoring import DriftDetector


@pytest.fixture
def reference_df():
    rng = np.random.default_rng(42)
    return pd.DataFrame({f"V{i}": rng.normal(0, 1, 500) for i in range(1, 10)})


@pytest.fixture
def current_df_stable(reference_df):
    """Same distribution as reference."""
    rng = np.random.default_rng(42)
    return pd.DataFrame({f"V{i}": rng.normal(0, 1, 200) for i in range(1, 10)})


@pytest.fixture
def current_df_drifted():
    """Different distribution."""
    rng = np.random.default_rng(42)
    return pd.DataFrame({f"V{i}": rng.normal(3.0, 2.0, 200) for i in range(1, 10)})


class TestDriftDetector:
    def test_init_with_reference(self, reference_df):
        detector = DriftDetector(reference_data=reference_df)
        assert detector._reference is not None
        assert len(detector._reference_stats) > 0

    def test_check_feature_drift_stable(self, reference_df, current_df_stable):
        detector = DriftDetector(reference_data=reference_df)
        results = detector.check_feature_drift(current_df_stable)
        assert isinstance(results, dict)
        assert len(results) > 0
        for feat, metrics in results.items():
            assert "ks_statistic" in metrics
            assert "psi" in metrics
            assert "drift_detected" in metrics

    def test_check_feature_drift_detected(self, reference_df, current_df_drifted):
        detector = DriftDetector(reference_data=reference_df, ks_threshold=0.001)
        results = detector.check_feature_drift(current_df_drifted)
        any_drift = any(m["drift_detected"] for m in results.values())
        assert any_drift, "Drift should be detected with significantly shifted data"

    def test_check_performance_drift(self):
        detector = DriftDetector()
        ref = {"roc_auc": 0.997, "pr_auc": 0.89}
        cur = {"roc_auc": 0.980, "pr_auc": 0.85}
        results = detector.check_performance_drift(ref, cur, threshold=0.01)
        assert "roc_auc" in results
        assert results["roc_auc"]["degradation"] > 0

    def test_log_prediction(self):
        detector = DriftDetector()
        detector.log_prediction({"score": 0.8}, actual=1)
        assert len(detector._window) == 1
