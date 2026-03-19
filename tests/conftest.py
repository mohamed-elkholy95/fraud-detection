"""Shared test fixtures."""
import os, sys
import pytest
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


@pytest.fixture
def synthetic_data():
    """Generate synthetic fraud detection dataset."""
    rng = np.random.default_rng(42)
    n_normal = 900
    n_fraud = 100
    n = n_normal + n_fraud

    v_cols = {f"V{i}": rng.normal(0, 1, n) for i in range(1, 29)}
    df = pd.DataFrame(v_cols)
    df["Time"] = rng.uniform(0, 172800, n)
    df["Amount"] = rng.lognormal(2.5, 1.5, n)
    # First n_normal rows are normal, last n_fraud are fraud
    y = np.array([0] * n_normal + [1] * n_fraud)
    # Make fraud slightly different
    for i in range(28):
        col = f"V{i+1}"
        df.loc[n_normal:, col] += rng.normal(1.0, 0.5, n_fraud)
    df["Class"] = y
    return df, y


@pytest.fixture
def synthetic_features(synthetic_data):
    """Return X, y numpy arrays."""
    df, y = synthetic_data
    v_cols = [f"V{i}" for i in range(1, 29)]
    X = df[v_cols + ["Amount"]].values
    return X, y


@pytest.fixture
def single_transaction():
    """Single transaction feature array."""
    rng = np.random.default_rng(42)
    return rng.normal(0, 1, 30).reshape(1, -1)
