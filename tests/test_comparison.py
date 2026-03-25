"""Tests for model comparison utilities."""
import numpy as np
import pytest

from src.models.comparison import (
    compare_models,
    format_comparison_table,
    select_best_model,
)


class MockModel:
    """Simple mock model for testing comparison logic."""

    def __init__(self, fraud_score: float = 0.3):
        self._fraud_score = fraud_score

    def predict_proba(self, X):
        n = X.shape[0]
        fraud_proba = np.full(n, self._fraud_score)
        return np.column_stack([1 - fraud_proba, fraud_proba])


@pytest.fixture
def test_data():
    rng = np.random.default_rng(42)
    n = 200
    X = rng.normal(0, 1, (n, 10))
    y = rng.choice([0, 1], n, p=[0.95, 0.05])
    return X, y


@pytest.fixture
def mock_models():
    return {
        "high_score": MockModel(fraud_score=0.6),
        "low_score": MockModel(fraud_score=0.1),
        "medium_score": MockModel(fraud_score=0.3),
    }


class TestCompareModels:
    def test_returns_results_for_all_models(self, mock_models, test_data):
        X, y = test_data
        results = compare_models(mock_models, X, y)
        assert len(results) == 3
        names = {r["name"] for r in results}
        assert names == {"high_score", "low_score", "medium_score"}

    def test_results_sorted_by_pr_auc(self, mock_models, test_data):
        X, y = test_data
        results = compare_models(mock_models, X, y)
        pr_aucs = [r["pr_auc"] for r in results if "error" not in r]
        assert pr_aucs == sorted(pr_aucs, reverse=True)

    def test_results_contain_expected_metrics(self, mock_models, test_data):
        X, y = test_data
        results = compare_models(mock_models, X, y)
        expected_keys = {
            "name", "pr_auc", "roc_auc", "f1", "precision", "recall",
            "total_cost", "false_positives", "false_negatives",
            "latency_ms", "latency_per_sample_ms",
        }
        for r in results:
            if "error" not in r:
                assert expected_keys.issubset(set(r.keys()))

    def test_cost_calculation(self, test_data):
        X, y = test_data
        model = MockModel(fraud_score=0.6)
        results = compare_models({"test": model}, X, y, cost_fp=10.0, cost_fn=500.0)
        assert results[0]["total_cost"] >= 0

    def test_handles_broken_model(self, test_data):
        X, y = test_data

        class BrokenModel:
            def predict_proba(self, X):
                raise RuntimeError("model failed")

        results = compare_models({"broken": BrokenModel()}, X, y)
        assert len(results) == 1
        assert "error" in results[0]


class TestFormatComparisonTable:
    def test_produces_markdown(self, mock_models, test_data):
        X, y = test_data
        results = compare_models(mock_models, X, y)
        table = format_comparison_table(results)
        assert "| Model |" in table
        assert "PR-AUC" in table

    def test_empty_results(self):
        table = format_comparison_table([])
        assert "No model comparison" in table


class TestSelectBestModel:
    def test_selects_highest_pr_auc(self):
        results = [
            {"name": "a", "pr_auc": 0.7, "latency_ms": 10, "total_cost": 100},
            {"name": "b", "pr_auc": 0.9, "latency_ms": 50, "total_cost": 200},
            {"name": "c", "pr_auc": 0.8, "latency_ms": 20, "total_cost": 150},
        ]
        assert select_best_model(results) == "b"

    def test_latency_constraint(self):
        results = [
            {"name": "fast", "pr_auc": 0.7, "latency_ms": 5, "total_cost": 100},
            {"name": "slow", "pr_auc": 0.9, "latency_ms": 500, "total_cost": 50},
        ]
        best = select_best_model(results, max_latency_ms=100)
        assert best == "fast"

    def test_cost_constraint(self):
        results = [
            {"name": "cheap", "pr_auc": 0.7, "latency_ms": 10, "total_cost": 50},
            {"name": "expensive", "pr_auc": 0.9, "latency_ms": 10, "total_cost": 5000},
        ]
        best = select_best_model(results, max_cost=100)
        assert best == "cheap"

    def test_no_eligible_models(self):
        results = [
            {"name": "a", "pr_auc": 0.9, "latency_ms": 500, "total_cost": 5000},
        ]
        best = select_best_model(results, max_latency_ms=10, max_cost=10)
        assert best is None

    def test_skips_error_results(self):
        results = [
            {"name": "broken", "error": "failed"},
            {"name": "ok", "pr_auc": 0.7, "latency_ms": 10, "total_cost": 100},
        ]
        assert select_best_model(results) == "ok"
