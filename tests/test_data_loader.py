"""Tests for data loader."""
import pytest
import pandas as pd
import numpy as np
from pathlib import Path

from src.data.loader import (
    load_fraud_data, split_temporal, generate_synthetic_data,
    get_dataset_stats, EXPECTED_COLUMNS,
)


class TestGenerateSyntheticData:
    def test_shape(self):
        df = generate_synthetic_data(n_samples=1000)
        assert df.shape[0] == 1000
        assert df.shape[1] >= 30

    def test_columns(self):
        df = generate_synthetic_data(n_samples=100)
        for col in EXPECTED_COLUMNS:
            assert col in df.columns

    def test_class_values(self):
        df = generate_synthetic_data(n_samples=1000)
        assert set(df["Class"].unique()).issubset({0, 1})

    def test_fraud_present(self):
        df = generate_synthetic_data(n_samples=10000, fraud_ratio=0.01)
        assert df["Class"].sum() > 0


class TestSplitTemporal:
    def test_split_sizes(self):
        df = generate_synthetic_data(n_samples=1000)
        train, test = split_temporal(df, test_ratio=0.2)
        assert len(train) == 800
        assert len(test) == 200

    def test_no_overlap(self):
        df = generate_synthetic_data(n_samples=1000)
        train, test = split_temporal(df, test_ratio=0.2)
        assert len(train) + len(test) == len(df)

    def test_temporal_ordering(self):
        df = generate_synthetic_data(n_samples=1000)
        train, test = split_temporal(df, test_ratio=0.2)
        assert train["Time"].max() <= test["Time"].min()


class TestGetDatasetStats:
    def test_returns_dict(self):
        df = generate_synthetic_data(n_samples=1000)
        stats = get_dataset_stats(df)
        assert isinstance(stats, dict)

    def test_required_keys(self):
        df = generate_synthetic_data(n_samples=1000)
        stats = get_dataset_stats(df)
        for key in ["n_rows", "n_fraud", "fraud_rate", "amount_mean"]:
            assert key in stats


class TestLoadFraudData:
    def test_load_generates(self, tmp_path):
        path = str(tmp_path / "test_fraud.csv")
        df = load_fraud_data(path=path, download=True)
        assert isinstance(df, pd.DataFrame)
        assert len(df) > 0

    def test_load_existing(self, tmp_path):
        path = str(tmp_path / "test_fraud.csv")
        df_orig = generate_synthetic_data(n_samples=500)
        df_orig.to_csv(path, index=False)
        df = load_fraud_data(path=path, download=True)
        assert len(df) == 500
