"""Model comparison and selection utilities.

Provides structured comparison of multiple fraud detection models across
key metrics, helping select the best model for deployment.

In fraud detection, model selection is nuanced because:
1. Different metrics tell different stories (precision vs recall trade-off)
2. Business cost matters more than statistical accuracy
3. Latency requirements may disqualify high-performing but slow models
4. Interpretability needs may favor simpler models for regulatory compliance
"""
import logging
import time
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

try:
    from sklearn.metrics import (
        average_precision_score,
        roc_auc_score,
        f1_score,
        precision_score,
        recall_score,
    )

    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False


def compare_models(
    models: Dict[str, Any],
    X_test: np.ndarray,
    y_test: np.ndarray,
    cost_fp: float = 10.0,
    cost_fn: float = 500.0,
) -> List[Dict[str, Any]]:
    """Compare multiple trained models on the same test set.

    Evaluates each model on standard classification metrics plus
    business-specific cost analysis and inference latency. Results
    are sorted by PR-AUC (the primary metric for imbalanced fraud data).

    Args:
        models: Dict mapping model name to fitted model object.
            Each model must implement predict_proba(X) -> (n, 2).
        X_test: Test feature matrix.
        y_test: Test labels.
        cost_fp: Dollar cost per false positive.
        cost_fn: Dollar cost per false negative.

    Returns:
        List of dicts with metrics per model, sorted by PR-AUC descending.

    Example:
        >>> results = compare_models(
        ...     {"rf": rf_model, "xgb": xgb_model},
        ...     X_test, y_test,
        ... )
        >>> print(results[0]["name"], results[0]["pr_auc"])
        xgb 0.8234
    """
    if not SKLEARN_AVAILABLE:
        logger.warning("scikit-learn not available for model comparison")
        return []

    results: List[Dict[str, Any]] = []

    for name, model in models.items():
        try:
            # Measure inference latency
            start_time = time.perf_counter()
            y_proba = model.predict_proba(X_test)
            latency_ms = (time.perf_counter() - start_time) * 1000

            if y_proba.ndim == 2 and y_proba.shape[1] == 2:
                y_proba = y_proba[:, 1]

            y_pred = (y_proba >= 0.5).astype(int)

            # Compute confusion matrix components for cost analysis
            fp = int(((y_pred == 1) & (y_test == 0)).sum())
            fn = int(((y_pred == 0) & (y_test == 1)).sum())
            total_cost = fp * cost_fp + fn * cost_fn

            result = {
                "name": name,
                "pr_auc": round(float(average_precision_score(y_test, y_proba)), 4),
                "roc_auc": round(float(roc_auc_score(y_test, y_proba)), 4),
                "f1": round(float(f1_score(y_test, y_pred, zero_division=0)), 4),
                "precision": round(float(precision_score(y_test, y_pred, zero_division=0)), 4),
                "recall": round(float(recall_score(y_test, y_pred, zero_division=0)), 4),
                "total_cost": round(total_cost, 2),
                "false_positives": fp,
                "false_negatives": fn,
                "latency_ms": round(latency_ms, 2),
                "latency_per_sample_ms": round(latency_ms / len(X_test), 4),
            }
            results.append(result)
            logger.info(
                "Model '%s': PR-AUC=%.4f, cost=$%.0f, latency=%.1fms",
                name, result["pr_auc"], total_cost, latency_ms,
            )

        except Exception as exc:
            logger.warning("Failed to evaluate model '%s': %s", name, exc)
            results.append({"name": name, "error": str(exc)})

    # Sort by PR-AUC descending (best first)
    results.sort(key=lambda r: r.get("pr_auc", 0), reverse=True)
    return results


def format_comparison_table(results: List[Dict[str, Any]]) -> str:
    """Format model comparison results as a markdown table.

    Args:
        results: Output from compare_models().

    Returns:
        Markdown-formatted comparison table.
    """
    if not results:
        return "No model comparison results available."

    lines = [
        "| Model | PR-AUC | ROC-AUC | F1 | Precision | Recall | Cost ($) | Latency (ms) |",
        "|-------|--------|---------|----|-----------|---------|-----------|----|",
    ]

    for r in results:
        if "error" in r:
            lines.append(f"| {r['name']} | ERROR: {r['error']} | | | | | | |")
        else:
            lines.append(
                f"| {r['name']} "
                f"| {r['pr_auc']:.4f} "
                f"| {r['roc_auc']:.4f} "
                f"| {r['f1']:.4f} "
                f"| {r['precision']:.4f} "
                f"| {r['recall']:.4f} "
                f"| {r['total_cost']:,.0f} "
                f"| {r['latency_ms']:.1f} |"
            )

    return "\n".join(lines)


def select_best_model(
    results: List[Dict[str, Any]],
    primary_metric: str = "pr_auc",
    max_latency_ms: Optional[float] = None,
    max_cost: Optional[float] = None,
) -> Optional[str]:
    """Select the best model given constraints.

    Applies optional latency and cost constraints, then selects the model
    with the highest value for the primary metric among eligible candidates.

    Args:
        results: Output from compare_models().
        primary_metric: Metric to optimize (default: pr_auc).
        max_latency_ms: Maximum allowed inference latency.
        max_cost: Maximum allowed total cost.

    Returns:
        Name of the best model, or None if no model meets constraints.
    """
    candidates = [r for r in results if "error" not in r]

    if max_latency_ms is not None:
        candidates = [r for r in candidates if r.get("latency_ms", float("inf")) <= max_latency_ms]

    if max_cost is not None:
        candidates = [r for r in candidates if r.get("total_cost", float("inf")) <= max_cost]

    if not candidates:
        logger.warning("No models meet the specified constraints")
        return None

    best = max(candidates, key=lambda r: r.get(primary_metric, 0))
    logger.info(
        "Selected model '%s' with %s=%.4f",
        best["name"], primary_metric, best.get(primary_metric, 0),
    )
    return best["name"]
