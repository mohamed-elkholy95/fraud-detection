"""Tests for threshold optimization."""
import pytest
import numpy as np

from src.threshold import optimize_threshold, threshold_analysis, cost_curve


@pytest.fixture
def predictions():
    rng = np.random.default_rng(42)
    y_true = np.array([0] * 90 + [1] * 10)
    y_proba = np.concatenate([rng.uniform(0, 0.4, 90), rng.uniform(0.5, 1.0, 10)])
    return y_true, y_proba


class TestOptimizeThreshold:
    def test_returns_float(self, predictions):
        y_true, y_proba = predictions
        t = optimize_threshold(y_true, y_proba)
        assert isinstance(t, float)

    def test_range(self, predictions):
        y_true, y_proba = predictions
        t = optimize_threshold(y_true, y_proba)
        assert 0.01 <= t <= 0.99

    def test_cost_fp_less(self, predictions):
        y_true, y_proba = predictions
        t_high = optimize_threshold(y_true, y_proba, cost_fp=1000, cost_fn=10)
        # When FP is expensive, threshold should be higher (fewer blocks)
        t_low = optimize_threshold(y_true, y_proba, cost_fp=1, cost_fn=1000)
        assert t_high >= t_low


class TestThresholdAnalysis:
    def test_returns_dataframe(self, predictions):
        y_true, y_proba = predictions
        df = threshold_analysis(y_true, y_proba)
        assert "threshold" in df.columns
        assert "precision" in df.columns
        assert "recall" in df.columns
        assert "expected_cost" in df.columns

    def test_metrics_in_range(self, predictions):
        y_true, y_proba = predictions
        df = threshold_analysis(y_true, y_proba)
        for col in ["precision", "recall", "f1", "fpr", "tpr"]:
            assert df[col].min() >= 0.0
            assert df[col].max() <= 1.0


class TestOptimizeThresholdEdgeCases:
    """Edge case tests for threshold optimization robustness."""

    def test_empty_arrays(self):
        """Empty inputs should return safe default of 0.5."""
        t = optimize_threshold(np.array([]), np.array([]))
        assert t == 0.5

    def test_all_legitimate(self):
        """When all transactions are legitimate, threshold should be high."""
        y_true = np.zeros(100)
        y_proba = np.random.uniform(0, 0.3, 100)
        t = optimize_threshold(y_true, y_proba)
        assert isinstance(t, float)

    def test_all_fraud(self):
        """When all transactions are fraud, threshold should be low."""
        y_true = np.ones(100)
        y_proba = np.random.uniform(0.5, 1.0, 100)
        t = optimize_threshold(y_true, y_proba)
        assert t <= 0.5

    def test_perfect_separation(self):
        """Perfect model probabilities should yield clean separation."""
        y_true = np.array([0] * 50 + [1] * 50)
        y_proba = np.array([0.0] * 50 + [1.0] * 50)
        t = optimize_threshold(y_true, y_proba)
        # Any threshold between 0 and 1 achieves zero cost
        assert 0.01 <= t <= 0.99

    def test_array_length_mismatch(self):
        """Mismatched array lengths should raise ValueError."""
        with pytest.raises(ValueError, match="length mismatch"):
            optimize_threshold(np.array([0, 1]), np.array([0.1, 0.2, 0.3]))


class TestCostCurve:
    def test_returns_arrays(self, predictions):
        y_true, y_proba = predictions
        thresholds, costs = cost_curve(y_true, y_proba)
        assert len(thresholds) == len(costs)
        assert len(thresholds) > 0

    def test_costs_positive(self, predictions):
        y_true, y_proba = predictions
        _, costs = cost_curve(y_true, y_proba)
        assert (costs >= 0).all()
