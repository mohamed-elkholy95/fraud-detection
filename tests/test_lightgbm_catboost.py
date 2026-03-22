"""Tests for LightGBM and CatBoost model training functions."""
import numpy as np
import pytest

# Check availability at import time
try:
    import lightgbm  # noqa: F401
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False

try:
    import catboost  # noqa: F401
    CATBOOST_AVAILABLE = True
except ImportError:
    CATBOOST_AVAILABLE = False

from src.models.supervised import train_lightgbm, train_catboost


@pytest.fixture
def binary_data():
    """500-sample imbalanced binary classification dataset."""
    rng = np.random.default_rng(42)
    n = 500
    X = rng.normal(0, 1, (n, 20))
    y = rng.choice([0, 1], n, p=[0.9, 0.1])
    return X, y


@pytest.fixture
def split_data(binary_data):
    """Train / validation split for early stopping tests."""
    X, y = binary_data
    split = int(len(X) * 0.8)
    return X[:split], y[:split], X[split:], y[split:]


# ─── LightGBM ─────────────────────────────────────────────────────────────────

@pytest.mark.skipif(not LIGHTGBM_AVAILABLE, reason="lightgbm not installed")
class TestLightGBM:
    def test_returns_model(self, binary_data):
        """train_lightgbm returns a fitted model."""
        X, y = binary_data
        model = train_lightgbm(X, y, n_estimators=50)
        assert model is not None
        assert hasattr(model, "predict_proba")

    def test_predict_shape(self, binary_data):
        """predict_proba output shape is (n_samples, 2)."""
        X, y = binary_data
        model = train_lightgbm(X, y, n_estimators=50)
        assert model is not None
        proba = model.predict_proba(X)
        assert proba.shape == (len(X), 2)

    def test_with_validation_data(self, split_data):
        """Early stopping with validation data doesn't crash."""
        X_tr, y_tr, X_val, y_val = split_data
        model = train_lightgbm(
            X_tr, y_tr, X_val=X_val, y_val=y_val,
            n_estimators=100, early_stopping_rounds=10,
        )
        assert model is not None
        proba = model.predict_proba(X_val)
        assert proba.shape == (len(X_val), 2)

    def test_class_balance(self, binary_data):
        """Model uses is_unbalance so it handles imbalanced data."""
        X, y = binary_data
        model = train_lightgbm(X, y, n_estimators=50)
        assert model is not None
        # With is_unbalance=True, predictions should include positive class
        proba = model.predict_proba(X)[:, 1]
        assert proba.max() > 0.0


class TestLightGBMNotInstalled:
    """Behaviour when lightgbm is absent — always runs."""

    def test_returns_none_when_unavailable(self, binary_data, monkeypatch):
        """train_lightgbm returns None when lightgbm is not importable."""
        import src.models.supervised as sup_module
        original = sup_module.LIGHTGBM_AVAILABLE
        sup_module.LIGHTGBM_AVAILABLE = False
        try:
            X, y = binary_data
            result = train_lightgbm(X, y, n_estimators=10)
            assert result is None
        finally:
            sup_module.LIGHTGBM_AVAILABLE = original


# ─── CatBoost ─────────────────────────────────────────────────────────────────

@pytest.mark.skipif(not CATBOOST_AVAILABLE, reason="catboost not installed")
class TestCatBoost:
    def test_returns_model(self, binary_data):
        """train_catboost returns a fitted model."""
        X, y = binary_data
        model = train_catboost(X, y, n_estimators=50)
        assert model is not None
        assert hasattr(model, "predict_proba")

    def test_predict_shape(self, binary_data):
        """predict_proba output shape is (n_samples, 2)."""
        X, y = binary_data
        model = train_catboost(X, y, n_estimators=50)
        assert model is not None
        proba = model.predict_proba(X)
        assert proba.shape == (len(X), 2)

    def test_with_validation_data(self, split_data):
        """Early stopping with validation data doesn't crash."""
        X_tr, y_tr, X_val, y_val = split_data
        model = train_catboost(
            X_tr, y_tr, X_val=X_val, y_val=y_val,
            n_estimators=100, early_stopping_rounds=10,
        )
        assert model is not None
        proba = model.predict_proba(X_val)
        assert proba.shape == (len(X_val), 2)

    def test_class_balance(self, binary_data):
        """auto_class_weights='Balanced' handles imbalanced data."""
        X, y = binary_data
        model = train_catboost(X, y, n_estimators=50)
        assert model is not None
        proba = model.predict_proba(X)[:, 1]
        assert proba.max() > 0.0


class TestCatBoostNotInstalled:
    """Behaviour when catboost is absent — always runs."""

    def test_returns_none_when_unavailable(self, binary_data, monkeypatch):
        """train_catboost returns None when catboost is not importable."""
        import src.models.supervised as sup_module
        original = sup_module.CATBOOST_AVAILABLE
        sup_module.CATBOOST_AVAILABLE = False
        try:
            X, y = binary_data
            result = train_catboost(X, y, n_estimators=10)
            assert result is None
        finally:
            sup_module.CATBOOST_AVAILABLE = original
