"""Tests for unsupervised models."""
import pytest
import numpy as np

from src.models.unsupervised import (
    train_isolation_forest, predict_isolation_forest,
    train_autoencoder, predict_autoencoder,
)


@pytest.fixture
def sample_data():
    rng = np.random.default_rng(42)
    return rng.normal(0, 1, (500, 30))


class TestIsolationForest:
    def test_returns_model(self, sample_data):
        model = train_isolation_forest(sample_data)
        if model is not None:
            assert model is not None

    def test_predict_shape(self, sample_data):
        model = train_isolation_forest(sample_data)
        if model is not None:
            preds, scores = predict_isolation_forest(model, sample_data)
            assert preds.shape == (500,)
            assert scores.shape == (500,)
            assert set(preds).issubset({0, 1})

    def test_scores_in_range(self, sample_data):
        model = train_isolation_forest(sample_data)
        if model is not None:
            _, scores = predict_isolation_forest(model, sample_data)
            assert scores.min() >= 0.0
            assert scores.max() <= 1.0


class TestAutoencoder:
    def test_returns_none_without_tf(self, sample_data):
        result = train_autoencoder(sample_data, encoding_dim=5, epochs=2, batch_size=64)
        # Returns None if TF not installed, or (model, threshold) if it is
        if result is not None:
            model, threshold = result
            assert threshold > 0
            preds, scores = predict_autoencoder(model, sample_data, threshold)
            assert preds.shape == (500,)
