"""Tests for cross_validate_models() in src/evaluation.py."""
import numpy as np
import pytest
from unittest.mock import MagicMock

from src.evaluation import cross_validate_models


@pytest.fixture
def binary_data():
    """Balanced 300-sample dataset for fast CV tests."""
    rng = np.random.default_rng(0)
    n = 300
    X = rng.normal(0, 1, (n, 10))
    y = rng.choice([0, 1], n, p=[0.8, 0.2])
    return X, y


def _dummy_train_func(X_train, y_train, **kwargs):
    """Minimal mock model: always predicts class priors."""
    from sklearn.dummy import DummyClassifier
    model = DummyClassifier(strategy="stratified", random_state=42)
    model.fit(X_train, y_train)
    return model


def _lr_train_func(X_train, y_train, **kwargs):
    """LogisticRegression-based train func for realistic CV."""
    from sklearn.linear_model import LogisticRegression
    model = LogisticRegression(max_iter=200, random_state=42)
    model.fit(X_train, y_train)
    return model


class TestCrossValidate:
    def test_returns_metrics(self, binary_data):
        """Result dict contains aggregated metric keys."""
        X, y = binary_data
        result = cross_validate_models(X, y, _dummy_train_func, cv=3)
        assert isinstance(result, dict)
        assert "roc_auc_mean" in result
        assert "pr_auc_mean" in result

    def test_metric_keys(self, binary_data):
        """All expected mean and std keys are present."""
        X, y = binary_data
        result = cross_validate_models(X, y, _lr_train_func, cv=3)
        expected_metrics = ["roc_auc", "pr_auc", "f1", "precision", "recall"]
        for m in expected_metrics:
            assert f"{m}_mean" in result, f"Missing {m}_mean"
            assert f"{m}_std" in result, f"Missing {m}_std"

    def test_cv_folds(self, binary_data):
        """n_folds in result matches the requested cv count."""
        X, y = binary_data
        for cv in [3, 4, 5]:
            result = cross_validate_models(X, y, _lr_train_func, cv=cv)
            assert result["n_folds"] == cv, f"Expected {cv} folds, got {result['n_folds']}"

    def test_with_mock_model(self, binary_data):
        """Works with a mock train func that returns a mock model."""
        X, y = binary_data

        def mock_train(X_train, y_train, **kwargs):
            rng = np.random.default_rng(1)

            class _DynamicMock:
                """Mock that returns correctly-sized probabilities for any X."""
                def predict_proba(self, X):  # noqa: N805
                    n = X.shape[0]
                    p = rng.uniform(0, 1, (n, 2))
                    p = p / p.sum(axis=1, keepdims=True)
                    return p

            return _DynamicMock()

        result = cross_validate_models(X, y, mock_train, cv=3)
        assert "roc_auc_mean" in result
        assert 0.0 <= result["roc_auc_mean"] <= 1.0

    def test_fold_results_length(self, binary_data):
        """fold_results list has one entry per completed fold."""
        X, y = binary_data
        result = cross_validate_models(X, y, _lr_train_func, cv=4)
        assert len(result["fold_results"]) == 4

    def test_metric_values_in_range(self, binary_data):
        """All mean metric values are in [0, 1]."""
        X, y = binary_data
        result = cross_validate_models(X, y, _lr_train_func, cv=3)
        for m in ["roc_auc", "pr_auc", "f1", "precision", "recall"]:
            val = result[f"{m}_mean"]
            assert 0.0 <= val <= 1.0, f"{m}_mean={val} out of [0,1]"

    def test_kwargs_forwarded(self, binary_data):
        """Extra kwargs are forwarded to the train function."""
        X, y = binary_data
        received_kwargs = {}

        def capturing_train(X_train, y_train, **kwargs):
            received_kwargs.update(kwargs)
            return _lr_train_func(X_train, y_train)

        cross_validate_models(X, y, capturing_train, cv=2, my_param=42)
        assert received_kwargs.get("my_param") == 42
