"""Tests for ensemble model."""
import pytest
import numpy as np
from unittest.mock import MagicMock

from src.models.ensemble import FraudEnsemble


class MockModel:
    """Mock classifier with predict_proba."""
    def __init__(self, proba_value=0.1):
        self.proba_value = proba_value

    def predict_proba(self, X):
        n = X.shape[0]
        return np.column_stack([
            np.full(n, 1 - self.proba_value),
            np.full(n, self.proba_value),
        ])

    def decision_function(self, X):
        return np.full(X.shape[0], -1.0)


@pytest.fixture
def ensemble():
    return FraudEnsemble()


@pytest.fixture
def sample_data():
    return np.random.default_rng(42).normal(0, 1, (100, 30))


class TestFraudEnsemble:
    def test_empty_raises(self, ensemble, sample_data):
        with pytest.raises(RuntimeError, match="No models"):
            ensemble.predict_proba(sample_data)

    def test_add_model(self, ensemble):
        ensemble.add_model("lr", MockModel(0.1), weight=0.5, model_type="supervised")
        assert "lr" in ensemble.model_names

    def test_predict_proba_shape(self, ensemble, sample_data):
        ensemble.add_model("lr", MockModel(0.1), weight=0.5, model_type="supervised")
        proba = ensemble.predict_proba(sample_data)
        assert proba.shape == (100, 2)

    def test_predict_proba_range(self, ensemble, sample_data):
        ensemble.add_model("lr", MockModel(0.3), weight=1.0, model_type="supervised")
        proba = ensemble.predict_proba(sample_data)
        assert proba.min() >= 0.0
        assert proba.max() <= 1.0

    def test_predict_shape(self, ensemble, sample_data):
        ensemble.add_model("lr", MockModel(0.1), weight=1.0, model_type="supervised")
        preds = ensemble.predict(sample_data)
        assert preds.shape == (100,)
        assert set(preds).issubset({0, 1})

    def test_multiple_models(self, ensemble, sample_data):
        ensemble.add_model("lr", MockModel(0.1), weight=0.5, model_type="supervised")
        ensemble.add_model("rf", MockModel(0.2), weight=0.5, model_type="supervised")
        proba = ensemble.predict_proba(sample_data)
        assert proba.shape == (100, 2)

    def test_explain_prediction(self, ensemble, sample_data):
        ensemble.add_model("lr", MockModel(0.3), weight=0.5, model_type="supervised")
        ensemble.add_model("iso", MockModel(0.1), weight=0.5, model_type="unsupervised")
        result = ensemble.explain_prediction(sample_data[:1])
        assert "model_contributions" in result
        assert "final_score" in result
        assert "prediction" in result
        assert 0.0 <= result["final_score"] <= 1.0
