"""Data validation utilities for fraud detection pipeline.

Validates incoming data quality before model training or inference.
Catches issues early — before they silently degrade model performance.

Common data quality issues in production fraud systems:
- Missing values in PCA components (upstream pipeline failures)
- Infinite values from division-by-zero in feature engineering
- Duplicate transactions from retry logic in payment systems
- Label corruption where fraud labels are systematically missing
"""
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class ValidationReport:
    """Results of a data validation check.

    Attributes:
        is_valid: Whether the data passed all critical checks.
        warnings: Non-critical issues that should be investigated.
        errors: Critical issues that must be fixed before proceeding.
        stats: Summary statistics about the validated data.
    """

    is_valid: bool = True
    warnings: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)
    stats: Dict[str, float] = field(default_factory=dict)

    def __str__(self) -> str:
        status = "PASSED" if self.is_valid else "FAILED"
        parts = [f"Validation {status}"]
        if self.errors:
            parts.append(f"  Errors ({len(self.errors)}):")
            parts.extend(f"    - {e}" for e in self.errors)
        if self.warnings:
            parts.append(f"  Warnings ({len(self.warnings)}):")
            parts.extend(f"    - {w}" for w in self.warnings)
        return "\n".join(parts)


def validate_training_data(
    df: pd.DataFrame,
    target_col: str = "Class",
    min_fraud_rate: float = 0.0001,
    max_fraud_rate: float = 0.10,
    max_missing_pct: float = 0.05,
    max_duplicate_pct: float = 0.01,
) -> ValidationReport:
    """Validate a DataFrame before model training.

    Checks for schema compliance, missing values, duplicates, label
    distribution, and numerical validity. These checks prevent common
    silent failures in ML pipelines.

    Args:
        df: Training DataFrame.
        target_col: Name of the target column.
        min_fraud_rate: Minimum expected fraud rate (flags label issues).
        max_fraud_rate: Maximum expected fraud rate (flags label contamination).
        max_missing_pct: Maximum allowed missing value percentage per column.
        max_duplicate_pct: Maximum allowed duplicate row percentage.

    Returns:
        ValidationReport with findings.
    """
    report = ValidationReport()

    # Check target column exists
    if target_col not in df.columns:
        report.errors.append(f"Target column '{target_col}' not found")
        report.is_valid = False
        return report

    # Check minimum row count
    if len(df) < 100:
        report.errors.append(f"Insufficient data: {len(df)} rows (minimum 100)")
        report.is_valid = False

    # Check fraud rate
    fraud_rate = df[target_col].mean()
    report.stats["fraud_rate"] = round(float(fraud_rate), 6)
    report.stats["n_rows"] = len(df)
    report.stats["n_fraud"] = int(df[target_col].sum())

    if fraud_rate < min_fraud_rate:
        report.warnings.append(
            f"Fraud rate {fraud_rate:.6f} below minimum {min_fraud_rate:.6f} — "
            f"possible label issue or insufficient fraud samples"
        )
    if fraud_rate > max_fraud_rate:
        report.warnings.append(
            f"Fraud rate {fraud_rate:.4f} above maximum {max_fraud_rate:.4f} — "
            f"possible label contamination or sampling bias"
        )

    # Check for missing values
    missing_pct = df.isnull().mean()
    cols_with_missing = missing_pct[missing_pct > 0]
    if len(cols_with_missing) > 0:
        worst_col = cols_with_missing.idxmax()
        worst_pct = cols_with_missing.max()
        report.stats["missing_columns"] = len(cols_with_missing)
        if worst_pct > max_missing_pct:
            report.errors.append(
                f"Column '{worst_col}' has {worst_pct:.1%} missing values "
                f"(threshold: {max_missing_pct:.1%})"
            )
            report.is_valid = False
        else:
            report.warnings.append(
                f"{len(cols_with_missing)} columns have missing values "
                f"(worst: {worst_col} at {worst_pct:.2%})"
            )

    # Check for infinite values in numeric columns
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    inf_counts = np.isinf(df[numeric_cols]).sum()
    cols_with_inf = inf_counts[inf_counts > 0]
    if len(cols_with_inf) > 0:
        report.errors.append(
            f"{len(cols_with_inf)} columns contain infinite values: "
            f"{list(cols_with_inf.index[:5])}"
        )
        report.is_valid = False

    # Check for duplicates
    n_duplicates = df.duplicated().sum()
    dup_pct = n_duplicates / len(df)
    report.stats["duplicate_rows"] = int(n_duplicates)
    if dup_pct > max_duplicate_pct:
        report.warnings.append(
            f"{n_duplicates} duplicate rows ({dup_pct:.2%}) — "
            f"possible data pipeline issue"
        )

    # Check target is binary
    unique_labels = df[target_col].dropna().unique()
    if not set(unique_labels).issubset({0, 1, 0.0, 1.0}):
        report.errors.append(
            f"Target column has non-binary values: {sorted(unique_labels)[:10]}"
        )
        report.is_valid = False

    logger.info("Data validation: %s", report)
    return report


def validate_prediction_input(
    features: np.ndarray,
    expected_dim: int,
) -> Tuple[bool, Optional[str]]:
    """Validate feature array before model inference.

    Quick sanity checks for production prediction requests.

    Args:
        features: Input feature array.
        expected_dim: Expected number of features.

    Returns:
        (is_valid, error_message) tuple.
    """
    if features.ndim == 1:
        features = features.reshape(1, -1)

    if features.shape[1] != expected_dim:
        return False, (
            f"Expected {expected_dim} features, got {features.shape[1]}"
        )

    if np.any(np.isnan(features)):
        return False, "Input contains NaN values"

    if np.any(np.isinf(features)):
        return False, "Input contains infinite values"

    return True, None
