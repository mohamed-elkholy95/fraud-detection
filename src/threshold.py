"""Cost-sensitive threshold optimization."""
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
