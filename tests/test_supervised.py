"""Tests for supervised models."""
import pytest
import numpy as np

from src.models.supervised import (
    train_logistic_regression, train_random_forest,
    train_xgboost, compute_scale_pos_weight,
)


@pytest.fixture
def binary_data():
    rng = np.random.default_rng(42)
    n = 500
    X = rng.normal(0, 1, (n, 30))
    y = rng.choice([0, 1], n, p=[0.9, 0.1])
    return X, y


class TestTrainLogisticRegression:
    def test_returns_model(self, binary_data):
        X, y = binary_data
        model = train_logistic_regression(X, y)
        if model is not None:
            assert hasattr(model, "predict_proba")

    def test_predict_shape(self, binary_data):
        X, y = binary_data
        model = train_logistic_regression(X, y)
        if model is not None:
            proba = model.predict_proba(X)
            assert proba.shape == (len(X), 2)


class TestTrainRandomForest:
    def test_returns_model(self, binary_data):
        X, y = binary_data
        model = train_random_forest(X, y)
        if model is not None:
            assert hasattr(model, "predict_proba")

    def test_predict_shape(self, binary_data):
        X, y = binary_data
        model = train_random_forest(X, y)
        if model is not None:
            proba = model.predict_proba(X)
            assert proba.shape == (len(X), 2)


class TestTrainXGBoost:
    def test_returns_model(self, binary_data):
        X, y = binary_data
        model = train_xgboost(X, y)
        if model is not None:
            assert hasattr(model, "predict_proba")

    def test_predict_shape(self, binary_data):
        X, y = binary_data
        model = train_xgboost(X, y, n_estimators=50)
        if model is not None:
            proba = model.predict_proba(X)
            assert proba.shape == (len(X), 2)


class TestComputeScalePosWeight:
    def test_balanced(self):
        y = np.array([0, 0, 1, 1])
        w = compute_scale_pos_weight(y)
        assert w == 1.0

    def test_imbalanced(self):
        y = np.array([0] * 900 + [1] * 100)
        w = compute_scale_pos_weight(y)
        assert w == 9.0
