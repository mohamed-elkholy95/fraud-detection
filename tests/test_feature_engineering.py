"""Tests for feature engineering."""
import pytest
import numpy as np

from src.data.feature_engineering import TransactionFeatureEngineer


@pytest.fixture
def engineer():
    return TransactionFeatureEngineer(seed=42)


@pytest.fixture
def sample_df():
    rng = np.random.default_rng(42)
    n = 100
    data = {f"V{i}": rng.normal(0, 1, n) for i in range(1, 29)}
    data["Time"] = rng.uniform(0, 172800, n)
    data["Amount"] = rng.lognormal(2.5, 1.5, n)
    return __import__("pandas").DataFrame(data)


class TestTransactionFeatureEngineer:
    def test_fit(self, engineer, sample_df):
        result = engineer.fit(sample_df)
        assert result is engineer
        assert result._is_fitted

    def test_transform_raises_unfitted(self, engineer, sample_df):
        with pytest.raises(RuntimeError, match="fit"):
            engineer.transform(sample_df)

    def test_transform_shape(self, engineer, sample_df):
        engineer.fit(sample_df)
        result = engineer.transform(sample_df)
        assert result.shape[0] == sample_df.shape[0]
        assert result.shape[1] > sample_df.shape[1]

    def test_fit_transform(self, sample_df):
        engineer = TransactionFeatureEngineer()
        result = engineer.fit_transform(sample_df)
        assert result.shape[0] == sample_df.shape[0]

    def test_new_features_present(self, engineer, sample_df):
        engineer.fit(sample_df)
        result = engineer.transform(sample_df)
        expected_new = [
            "hour_of_day", "amount_log1p", "amount_zscore",
            "v_features_mean", "v_features_std", "v1_v2_interaction",
        ]
        for feat in expected_new:
            assert feat in result.columns, f"Missing feature: {feat}"

    def test_hour_of_day_range(self, engineer, sample_df):
        engineer.fit(sample_df)
        result = engineer.transform(sample_df)
        assert result["hour_of_day"].min() >= 0
        assert result["hour_of_day"].max() <= 23

    def test_get_feature_names(self, engineer):
        original = ["V1", "Time", "Amount"]
        names = engineer.get_feature_names(original)
        assert len(names) > len(original)
