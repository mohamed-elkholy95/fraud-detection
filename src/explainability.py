"""Explainability wrappers for fraud predictions."""
import logging
from typing import Any, Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)

try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    logger.warning("SHAP not installed — explainability limited")

try:
    import lime
    import lime.lime_tabular
    LIME_AVAILABLE = True
except ImportError:
    LIME_AVAILABLE = False
    logger.warning("LIME not installed")


def explain_with_shap(
    model: Any,
    X: np.ndarray,
    feature_names: Optional[List[str]] = None,
    max_display: int = 10,
) -> Dict[str, Any]:
    """Generate SHAP explanation for predictions.

    Args:
        model: Fitted model (tree-based for TreeExplainer, any for KernelExplainer).
        X: Feature matrix (n_samples).
        feature_names: Feature name list.
        max_display: Max features to show.

    Returns:
        Dict with feature contributions and summary.
    """
    if not SHAP_AVAILABLE:
        return _mock_explanation(feature_names, max_display)

    try:
        if hasattr(model, "get_booster") or "xgboost" in str(type(model)).lower():
            explainer = shap.TreeExplainer(model)
        elif hasattr(model, "estimators_"):
            explainer = shap.TreeExplainer(model)
        else:
            background = shap.sample(X, min(100, len(X)))
            explainer = shap.KernelExplainer(model.predict_proba, background)

        shap_values = explainer.shap_values(X)

        if isinstance(shap_values, list):
            shap_vals = shap_values[1] if len(shap_values) > 1 else shap_values[0]
        else:
            shap_vals = shap_values

        # Global importance
        mean_abs_shap = np.abs(shap_vals).mean(axis=0)
        if feature_names is None:
            feature_names = [f"feature_{i}" for i in range(X.shape[1])]

        top_indices = np.argsort(mean_abs_shap)[-max_display:][::-1]
        top_features = [
            {"feature": feature_names[i], "importance": round(float(mean_abs_shap[i]), 4)}
            for i in top_indices
        ]

        return {
            "method": "shap",
            "top_features": top_features,
            "shap_values": shap_vals.tolist(),
            "base_value": round(float(explainer.expected_value), 4) if not isinstance(explainer.expected_value, np.ndarray) else round(float(explainer.expected_value[1]), 4),
        }
    except Exception as exc:
        logger.warning("SHAP explanation failed: %s", exc)
        return _mock_explanation(feature_names, max_display)


def explain_with_lime(
    model: Any,
    X: np.ndarray,
    feature_names: Optional[List[str]] = None,
    num_features: int = 10,
) -> Dict[str, Any]:
    """Generate LIME explanation.

    Args:
        model: Fitted model with predict_proba.
        X: Single sample feature array.
        feature_names: Feature name list.
        num_features: Max features to explain.

    Returns:
        Dict with LIME feature contributions.
    """
    if not LIME_AVAILABLE:
        return _mock_explanation(feature_names, num_features)

    if X.ndim == 1:
        X = X.reshape(1, -1)

    if feature_names is None:
        feature_names = [f"feature_{i}" for i in range(X.shape[1])]

    try:
        explainer = lime.lime_tabular.LimeTabularExplainer(
            training_data=X,
            feature_names=feature_names,
            mode="classification",
            discretize_continuous=True,
        )
        explanation = explainer.explain_instance(
            X[0], model.predict_proba, num_features=num_features,
        )
        features = [
            {"feature": feat, "weight": round(weight, 4)}
            for feat, weight in explanation.as_list()
        ]
        return {"method": "lime", "top_features": features}
    except Exception as exc:
        logger.warning("LIME explanation failed: %s", exc)
        return _mock_explanation(feature_names, num_features)


def _mock_explanation(
    feature_names: Optional[List[str]], max_display: int,
) -> Dict[str, Any]:
    """Generate mock explanation when real explainability tools unavailable.

    Args:
        feature_names: Feature name list.
        max_display: Max features.

    Returns:
        Mock explanation dict.
    """
    if feature_names is None:
        feature_names = [f"feature_{i}" for i in range(max_display)]

    rng = np.random.default_rng(42)
    n = min(max_display, len(feature_names))
    indices = rng.choice(len(feature_names), size=n, replace=False)
    importances = rng.uniform(0.01, 0.5, size=n)
    importances /= importances.sum()

    features = [
        {"feature": feature_names[i], "importance": round(float(importances[j]), 4)}
        for j, i in enumerate(indices)
    ]
    features.sort(key=lambda x: x["importance"], reverse=True)

    return {"method": "mock", "top_features": features, "base_value": 0.5}
