"""Tests for data validation utilities."""
import numpy as np
import pandas as pd
import pytest

from src.data.validation import (
    ValidationReport,
    validate_training_data,
    validate_prediction_input,
)


@pytest.fixture
def valid_fraud_df():
    """Create a valid fraud detection DataFrame."""
    rng = np.random.default_rng(42)
    n = 1000
    data = {f"V{i}": rng.normal(0, 1, n) for i in range(1, 29)}
    data["Time"] = rng.uniform(0, 172800, n)
    data["Amount"] = rng.lognormal(2.5, 1.5, n)
    data["Class"] = rng.choice([0, 1], n, p=[0.98, 0.02])
    return pd.DataFrame(data)


class TestValidationReport:
    def test_default_is_valid(self):
        report = ValidationReport()
        assert report.is_valid is True
        assert len(report.errors) == 0
        assert len(report.warnings) == 0

    def test_str_representation(self):
        report = ValidationReport(
            is_valid=False,
            errors=["missing column"],
            warnings=["low fraud rate"],
        )
        output = str(report)
        assert "FAILED" in output
        assert "missing column" in output
        assert "low fraud rate" in output

    def test_str_passed(self):
        report = ValidationReport()
        assert "PASSED" in str(report)


class TestValidateTrainingData:
    def test_valid_data_passes(self, valid_fraud_df):
        report = validate_training_data(valid_fraud_df)
        assert report.is_valid is True
        assert report.stats["n_rows"] == 1000

    def test_missing_target_column(self, valid_fraud_df):
        df = valid_fraud_df.drop(columns=["Class"])
        report = validate_training_data(df)
        assert report.is_valid is False
        assert any("Target column" in e for e in report.errors)

    def test_insufficient_rows(self):
        df = pd.DataFrame({"Class": [0, 1], "V1": [1.0, 2.0]})
        report = validate_training_data(df)
        assert report.is_valid is False
        assert any("Insufficient" in e for e in report.errors)

    def test_low_fraud_rate_warning(self, valid_fraud_df):
        valid_fraud_df["Class"] = 0
        valid_fraud_df.iloc[0, valid_fraud_df.columns.get_loc("Class")] = 1
        report = validate_training_data(valid_fraud_df, min_fraud_rate=0.01)
        assert any("below minimum" in w for w in report.warnings)

    def test_high_fraud_rate_warning(self, valid_fraud_df):
        valid_fraud_df["Class"] = 1
        report = validate_training_data(valid_fraud_df, max_fraud_rate=0.10)
        assert any("above maximum" in w for w in report.warnings)

    def test_missing_values_error(self, valid_fraud_df):
        valid_fraud_df.loc[:100, "V1"] = np.nan
        report = validate_training_data(valid_fraud_df, max_missing_pct=0.05)
        assert report.is_valid is False
        assert any("missing values" in e for e in report.errors)

    def test_infinite_values_error(self, valid_fraud_df):
        valid_fraud_df.loc[0, "V1"] = np.inf
        report = validate_training_data(valid_fraud_df)
        assert report.is_valid is False
        assert any("infinite" in e for e in report.errors)

    def test_duplicate_rows_warning(self, valid_fraud_df):
        # Add many duplicate rows
        dupes = pd.concat([valid_fraud_df.iloc[[0]]] * 50, ignore_index=True)
        df = pd.concat([valid_fraud_df, dupes], ignore_index=True)
        report = validate_training_data(df, max_duplicate_pct=0.01)
        assert any("duplicate" in w for w in report.warnings)

    def test_non_binary_target_error(self, valid_fraud_df):
        valid_fraud_df["Class"] = np.random.choice([0, 1, 2], len(valid_fraud_df))
        report = validate_training_data(valid_fraud_df)
        assert report.is_valid is False
        assert any("non-binary" in e for e in report.errors)


class TestValidatePredictionInput:
    def test_valid_input(self):
        features = np.random.randn(1, 30)
        is_valid, error = validate_prediction_input(features, expected_dim=30)
        assert is_valid is True
        assert error is None

    def test_1d_input(self):
        features = np.random.randn(30)
        is_valid, error = validate_prediction_input(features, expected_dim=30)
        assert is_valid is True

    def test_wrong_dimensions(self):
        features = np.random.randn(1, 25)
        is_valid, error = validate_prediction_input(features, expected_dim=30)
        assert is_valid is False
        assert "Expected 30" in error

    def test_nan_input(self):
        features = np.array([[1.0, np.nan, 3.0]])
        is_valid, error = validate_prediction_input(features, expected_dim=3)
        assert is_valid is False
        assert "NaN" in error

    def test_inf_input(self):
        features = np.array([[1.0, np.inf, 3.0]])
        is_valid, error = validate_prediction_input(features, expected_dim=3)
        assert is_valid is False
        assert "infinite" in error
