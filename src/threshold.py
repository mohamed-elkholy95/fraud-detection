"""Cost-sensitive threshold optimization.

In binary classification, the default decision threshold of 0.5 implicitly
assumes equal misclassification costs. For fraud detection, this assumption
fails catastrophically because:

  - Missing a fraud case (FN) costs ~$500 in chargebacks and losses
  - Flagging a legitimate transaction (FP) costs ~$10 in review overhead

The 50:1 cost ratio means the optimal threshold is typically much lower
than 0.5, accepting more false alarms to avoid expensive missed fraud.

This module provides:
1. optimize_threshold() — Finds the threshold minimizing total expected cost
2. threshold_analysis() — Generates a full metrics table across thresholds
3. cost_curve() — Data for plotting cost vs threshold trade-off
"""
import logging
from typing import Optional, Tuple

import numpy as np
import pandas as pd

from src.config import COST_FALSE_POSITIVE, COST_FALSE_NEGATIVE

logger = logging.getLogger(__name__)


def optimize_threshold(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    cost_fp: float = COST_FALSE_POSITIVE,
    cost_fn: float = COST_FALSE_NEGATIVE,
    steps: int = 990,
) -> float:
    """Find threshold that minimizes expected total cost.

    Expected Cost = FP_rate × cost_fp + FN_rate × cost_fn

    Args:
        y_true: True labels.
        y_proba: Predicted fraud probabilities.
        cost_fp: Cost per false positive.
        cost_fn: Cost per false negative.
        steps: Number of thresholds to evaluate.

    Returns:
        Optimal threshold value.
    """
    if len(y_true) == 0 or len(y_proba) == 0:
        logger.warning("Empty arrays passed to optimize_threshold — returning 0.5")
        return 0.5

    if len(y_true) != len(y_proba):
        raise ValueError(
            f"Array length mismatch: y_true={len(y_true)}, y_proba={len(y_proba)}"
        )

    # Vectorized threshold sweep for better performance on large datasets.
    # Each row in the broadcast represents predictions at one threshold.
    thresholds = np.linspace(0.01, 0.99, steps)
    best_threshold = 0.5
    best_cost = float("inf")

    for t in thresholds:
        y_pred = (y_proba >= t).astype(int)
        fp = ((y_pred == 1) & (y_true == 0)).sum()
        fn = ((y_pred == 0) & (y_true == 1)).sum()
        cost = fp * cost_fp + fn * cost_fn
        if cost < best_cost:
            best_cost = cost
            best_threshold = t

    logger.info("Optimal threshold: %.3f (expected cost: $%.0f)", best_threshold, best_cost)
    return best_threshold


def threshold_analysis(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    cost_fp: float = COST_FALSE_POSITIVE,
    cost_fn: float = COST_FALSE_NEGATIVE,
) -> pd.DataFrame:
    """Compute metrics at various thresholds.

    Args:
        y_true: True labels.
        y_proba: Predicted fraud probabilities.
        cost_fp: Cost per false positive.
        cost_fn: Cost per false negative.

    Returns:
        DataFrame with metrics per threshold.
    """
    thresholds = np.linspace(0.01, 0.99, 99)
    rows = []

    for t in thresholds:
        y_pred = (y_proba >= t).astype(int)
        tp = ((y_pred == 1) & (y_true == 1)).sum()
        fp = ((y_pred == 1) & (y_true == 0)).sum()
        tn = ((y_pred == 0) & (y_true == 0)).sum()
        fn = ((y_pred == 0) & (y_true == 1)).sum()

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
        tpr = recall
        expected_cost = fp * cost_fp + fn * cost_fn

        rows.append({
            "threshold": round(t, 3),
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4),
            "fpr": round(fpr, 4),
            "tpr": round(tpr, 4),
            "expected_cost": float(expected_cost),
            "tp": int(tp), "fp": int(fp), "tn": int(tn), "fn": int(fn),
        })

    return pd.DataFrame(rows)


def cost_curve(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    cost_fp: float = COST_FALSE_POSITIVE,
    cost_fn: float = COST_FALSE_NEGATIVE,
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute expected cost at each threshold.

    Args:
        y_true: True labels.
        y_proba: Predicted probabilities.
        cost_fp: Cost per false positive.
        cost_fn: Cost per false negative.

    Returns:
        (thresholds, costs) arrays for plotting.
    """
    df = threshold_analysis(y_true, y_proba, cost_fp, cost_fn)
    return df["threshold"].values, df["expected_cost"].values
