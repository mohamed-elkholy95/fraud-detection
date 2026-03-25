"""Feature engineering for fraud transaction data.

This module implements a stateful feature transformer following the sklearn
fit/transform pattern. Key engineering strategies:

1. **Temporal features**: Extract hour-of-day and day segments to capture
   time-based fraud patterns (fraud often clusters at unusual hours).

2. **Amount transformations**: Log transform reduces right-skew; z-score
   normalization enables comparison across different spending profiles.

3. **V-feature aggregates**: Since V1-V28 are PCA components, their aggregate
   statistics (mean, std, range) capture overall transaction "shape" that
   individual components might miss.

4. **Interaction terms**: Product of specific PCA components captures non-linear
   relationships that tree-based models might split inefficiently on.

5. **Cross-feature terms**: Combining V14/V17 with Amount creates features
   that measure "how anomalous is this amount given the PCA profile" — a
   strong fraud signal.

Note: The fit() step stores training statistics (amount mean/std) to prevent
data leakage during transform() on test/production data.
"""
import logging
from typing import Optional

import numpy as np
import pandas as pd

from src.config import RANDOM_SEED, AMOUNT_BINS, AMOUNT_BIN_LABELS

logger = logging.getLogger(__name__)


class TransactionFeatureEngineer:
    """Engineer features from raw credit card transaction data.

    Creates temporal, amount-based, and V-feature aggregate features
    to improve fraud detection model performance.
    """

    def __init__(self, seed: int = RANDOM_SEED) -> None:
        self._is_fitted: bool = False
        self._amount_mean: float = 0.0
        self._amount_std: float = 1.0
        self._seed = seed

    def fit(self, df: pd.DataFrame) -> "TransactionFeatureEngineer":
        """Compute statistics for normalization from training data.

        Args:
            df: Training DataFrame with 'Amount' column.

        Returns:
            Self.
        """
        self._amount_mean = df["Amount"].mean()
        self._amount_std = df["Amount"].std()
        if self._amount_std == 0:
            self._amount_std = 1.0
        self._is_fitted = True
        logger.info("FeatureEngineer fitted: amount_mean=%.2f, amount_std=%.2f",
                    self._amount_mean, self._amount_std)
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply all feature transformations.

        Args:
            df: DataFrame with V1-V28, Time, Amount columns.

        Returns:
            DataFrame with original + engineered features.
        """
        if not self._is_fitted:
            raise RuntimeError("Call fit() before transform()")

        result = df.copy()

        # Temporal features
        result["hour_of_day"] = (result["Time"] % 86400) / 3600
        result["hour_of_day"] = result["hour_of_day"].astype(int) % 24
        result["day_segment"] = pd.cut(
            result["hour_of_day"],
            bins=[0, 6, 12, 18, 24],
            labels=["night", "morning", "afternoon", "evening"],
        ).astype(str)

        # Amount features
        result["amount_log1p"] = np.log1p(result["Amount"])
        result["amount_zscore"] = (result["Amount"] - self._amount_mean) / self._amount_std
        result["amount_bins"] = pd.cut(
            result["Amount"], bins=AMOUNT_BINS, labels=AMOUNT_BIN_LABELS,
        ).astype(str)

        # V-feature aggregates
        v_cols = [f"V{i}" for i in range(1, 29)]
        result["v_features_mean"] = result[v_cols].mean(axis=1)
        result["v_features_std"] = result[v_cols].std(axis=1)
        result["v_features_min"] = result[v_cols].min(axis=1)
        result["v_features_max"] = result[v_cols].max(axis=1)
        result["v_features_range"] = result["v_features_max"] - result["v_features_min"]
        result["v_features_abs_mean"] = result[v_cols].abs().mean(axis=1)

        # Known interaction terms
        result["v1_v2_interaction"] = result["V1"] * result["V2"]
        result["v3_v7_interaction"] = result["V3"] * result["V7"]
        result["v4_v11_interaction"] = result["V4"] * result["V11"]

        # Distance features
        result["v14_amount"] = result["V14"] * result["Amount"]
        result["v17_amount"] = result["V17"] * result["Amount"]

        logger.info("Engineered %d features", result.shape[1] - df.shape[1])
        return result

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Fit and transform in one step.

        Args:
            df: Training DataFrame.

        Returns:
            Transformed DataFrame.
        """
        return self.fit(df).transform(df)

    def get_feature_names(self, original_cols: list) -> list:
        """Get list of all feature names after transformation.

        Args:
            original_cols: Original column names.

        Returns:
            List of all column names after transformation.
        """
        new_features = [
            "hour_of_day", "day_segment", "amount_log1p", "amount_zscore",
            "amount_bins", "v_features_mean", "v_features_std",
            "v_features_min", "v_features_max", "v_features_range",
            "v_features_abs_mean", "v1_v2_interaction", "v3_v7_interaction",
            "v4_v11_interaction", "v14_amount", "v17_amount",
        ]
        return original_cols + new_features
