"""Evaluation metrics and reporting."""
import logging
from typing import Any, Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)

try:
    from sklearn.metrics import (
        accuracy_score, precision_score, recall_score, f1_score,
        roc_auc_score, average_precision_score, confusion_matrix,
        classification_report, precision_recall_curve, roc_curve,
    )
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    logger.warning("scikit-learn not installed")


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: Optional[np.ndarray] = None,
) -> Dict[str, float]:
    """Compute comprehensive classification metrics.

    Args:
        y_true: True labels.
        y_pred: Predicted labels.
        y_proba: Predicted fraud probabilities.

    Returns:
        Dict with accuracy, precision, recall, f1, roc_auc, pr_auc.
    """
    if not SKLEARN_AVAILABLE:
        return {"accuracy": 0.0, "precision": 0.0, "recall": 0.0, "f1": 0.0}

    metrics: Dict[str, float] = {
        "accuracy": round(float(accuracy_score(y_true, y_pred)), 4),
        "precision": round(float(precision_score(y_true, y_pred, zero_division=0)), 4),
        "recall": round(float(recall_score(y_true, y_pred, zero_division=0)), 4),
        "f1": round(float(f1_score(y_true, y_pred, zero_division=0)), 4),
    }

    if y_proba is not None:
        try:
            metrics["roc_auc"] = round(float(roc_auc_score(y_true, y_proba)), 4)
        except ValueError:
            metrics["roc_auc"] = 0.0
        try:
            metrics["pr_auc"] = round(float(average_precision_score(y_true, y_proba)), 4)
        except ValueError:
            metrics["pr_auc"] = 0.0

    return metrics


def compute_confusion_matrix(
    y_true: np.ndarray, y_pred: np.ndarray,
) -> Dict[str, int]:
    """Compute confusion matrix values.

    Args:
        y_true: True labels.
        y_pred: Predicted labels.

    Returns:
        Dict with tp, fp, tn, fn counts.
    """
    if not SKLEARN_AVAILABLE:
        return {"tp": 0, "fp": 0, "tn": 0, "fn": 0}

    cm = confusion_matrix(y_true, y_pred)
    return {
        "tn": int(cm[0, 0]),
        "fp": int(cm[0, 1]),
        "fn": int(cm[1, 0]),
        "tp": int(cm[1, 1]),
    }


def compute_cost_analysis(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    cost_fp: float = 10.0,
    cost_fn: float = 500.0,
) -> Dict[str, float]:
    """Compute business cost analysis.

    Args:
        y_true: True labels.
        y_pred: Predicted labels.
        cost_fp: Cost per false positive.
        cost_fn: Cost per false negative.

    Returns:
        Cost analysis dict.
    """
    cm = compute_confusion_matrix(y_true, y_pred)
    total_fp_cost = cm["fp"] * cost_fp
    total_fn_cost = cm["fn"] * cost_fn
    return {
        "total_cost": round(total_fp_cost + total_fn_cost, 2),
        "fp_cost": round(total_fp_cost, 2),
        "fn_cost": round(total_fn_cost, 2),
        "n_false_positives": cm["fp"],
        "n_false_negatives": cm["fn"],
        "cost_per_transaction": round((total_fp_cost + total_fn_cost) / len(y_true), 4),
    }


def get_pr_curve_data(
    y_true: np.ndarray, y_proba: np.ndarray,
) -> Dict[str, np.ndarray]:
    """Get precision-recall curve data for plotting.

    Args:
        y_true: True labels.
        y_proba: Predicted probabilities.

    Returns:
        Dict with precision, recall, thresholds arrays.
    """
    if not SKLEARN_AVAILABLE or y_proba is None:
        return {"precision": np.array([]), "recall": np.array([]), "thresholds": np.array([])}

    precision, recall, thresholds = precision_recall_curve(y_true, y_proba)
    return {"precision": precision, "recall": recall, "thresholds": thresholds}


def get_roc_curve_data(
    y_true: np.ndarray, y_proba: np.ndarray,
) -> Dict[str, np.ndarray]:
    """Get ROC curve data for plotting.

    Args:
        y_true: True labels.
        y_proba: Predicted probabilities.

    Returns:
        Dict with fpr, tpr, thresholds arrays.
    """
    if not SKLEARN_AVAILABLE or y_proba is None:
        return {"fpr": np.array([]), "tpr": np.array([]), "thresholds": np.array([])}

    fpr, tpr, thresholds = roc_curve(y_true, y_proba)
    return {"fpr": fpr, "tpr": tpr, "thresholds": thresholds}


def generate_evaluation_report(metrics: Dict[str, float]) -> str:
    """Generate markdown evaluation report.

    Args:
        metrics: Metrics dict.

    Returns:
        Markdown formatted report string.
    """
    lines = [
        "# Fraud Detection — Evaluation Report",
        "",
        "## Performance Metrics",
        "",
        "| Metric | Value |",
        "|--------|-------|",
    ]
    for k, v in metrics.items():
        if isinstance(v, float):
            lines.append(f"| {k} | {v:.4f} |")
        else:
            lines.append(f"| {k} | {v} |")

    return "\n".join(lines)
