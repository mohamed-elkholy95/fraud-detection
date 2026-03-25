"""Ensemble model for fraud detection.

The ensemble approach is critical for production fraud systems because:

1. **No single model dominates**: Logistic regression catches linear patterns,
   tree ensembles handle interactions, and isolation forest detects novel anomalies.

2. **Diversity reduces false negatives**: If XGBoost misses a fraud pattern that
   LightGBM catches, the ensemble still flags it.

3. **Graceful degradation**: If one model fails (e.g., missing a dependency),
   the ensemble continues with remaining models rather than failing entirely.

Weight assignment uses PR-AUC on validation data — models that better separate
fraud from legitimate transactions contribute more to the final score.
"""
import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

try:
    from sklearn.metrics import average_precision_score
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False


class FraudEnsemble:
    """Weighted ensemble of multiple fraud detection models.

    Combines supervised (predict_proba) and unsupervised (anomaly scores)
    models via weighted averaging.
    """

    def __init__(self) -> None:
        self._models: Dict[str, Any] = {}
        self._weights: Dict[str, float] = {}
        self._model_types: Dict[str, str] = {}

    @property
    def model_names(self) -> List[str]:
        """List of registered model names."""
        return list(self._models.keys())

    def add_model(
        self,
        name: str,
        model: Any,
        weight: float,
        model_type: str = "supervised",
    ) -> None:
        """Register a model with its weight.

        Args:
            name: Model identifier.
            model: Fitted model object.
            weight: Weight in ensemble (will be normalized).
            model_type: 'supervised' or 'unsupervised'.
        """
        self._models[name] = model
        self._weights[name] = weight
        self._model_types[name] = model_type
        logger.info("Added model '%s' (type=%s, weight=%.2f)", name, model_type, weight)

    def predict(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        """Combined binary prediction using weighted average.

        Args:
            X: Feature matrix.
            threshold: Decision threshold.

        Returns:
            Binary predictions (0/1).
        """
        proba = self.predict_proba(X)[:, 1]
        return (proba >= threshold).astype(int)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Weighted average of fraud probabilities.

        Args:
            X: Feature matrix.

        Returns:
            Probability matrix shape (n_samples, 2) — [P(legit), P(fraud)].
        """
        if not self._models:
            raise RuntimeError("No models registered")

        n_samples = X.shape[0]
        total_weight = sum(self._weights.values())

        fraud_scores = np.zeros(n_samples)

        for name, model in self._models.items():
            weight = self._weights[name] / total_weight
            model_type = self._model_types[name]

            try:
                if model_type == "supervised":
                    # predict_proba returns (n, 2) — take fraud column
                    proba = model.predict_proba(X)
                    if proba.ndim == 2 and proba.shape[1] == 2:
                        score = proba[:, 1]
                    else:
                        score = proba
                else:
                    # Unsupervised: use decision_function or anomaly scores
                    if hasattr(model, "decision_function"):
                        raw = model.decision_function(X)
                        # Normalize to [0, 1]
                        score = 1 - (raw - raw.min()) / (raw.max() - raw.min() + 1e-9)
                    elif hasattr(model, "score_samples"):
                        raw = model.score_samples(X)
                        score = -raw  # Lower score = more anomalous
                        score = (score - score.min()) / (score.max() - score.min() + 1e-9)
                    else:
                        logger.warning("Model '%s' has no scoring method — skipping", name)
                        continue

                fraud_scores += weight * score
            except Exception as exc:
                logger.warning("Model '%s' prediction failed: %s", name, exc)

        # Ensure valid probability range
        fraud_scores = np.clip(fraud_scores, 0.0, 1.0)
        legit_scores = 1.0 - fraud_scores
        return np.column_stack([legit_scores, fraud_scores])

    def train_full_pipeline(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
    ) -> Dict[str, Any]:
        """Train all available models and build an optimized ensemble.

        Automatically detects which model packages are installed, trains each
        model, evaluates PR-AUC on the validation set, and uses those scores
        as ensemble weights.

        Args:
            X_train: Training features.
            y_train: Training labels.
            X_val: Validation features.
            y_val: Validation labels.

        Returns:
            Dict with per-model metrics, ensemble metrics, and weight map.
        """
        # Import train functions lazily to avoid circular imports
        from src.models.supervised import (
            train_logistic_regression,
            train_random_forest,
            train_xgboost,
            train_lightgbm,
            train_catboost,
            LIGHTGBM_AVAILABLE,
            CATBOOST_AVAILABLE,
        )
        from src.models.unsupervised import train_isolation_forest

        # Clear existing models before (re-)training
        self._models.clear()
        self._weights.clear()
        self._model_types.clear()

        candidates: List[Tuple[str, Any, str]] = [
            ("logistic_regression", train_logistic_regression, "supervised"),
            ("random_forest", train_random_forest, "supervised"),
            ("xgboost", train_xgboost, "supervised"),
        ]
        if LIGHTGBM_AVAILABLE:
            candidates.append(("lightgbm", train_lightgbm, "supervised"))
        if CATBOOST_AVAILABLE:
            candidates.append(("catboost", train_catboost, "supervised"))
        candidates.append(("isolation_forest", train_isolation_forest, "unsupervised"))

        per_model_metrics: Dict[str, Dict[str, float]] = {}
        weights: Dict[str, float] = {}

        for name, train_func, model_type in candidates:
            try:
                logger.info("Training %s ...", name)
                if name in {"xgboost", "lightgbm", "catboost"}:
                    model = train_func(
                        X_train, y_train,
                        X_val=X_val, y_val=y_val,
                        n_estimators=100,  # fast for pipeline run
                    )
                elif model_type == "unsupervised":
                    # Unsupervised models only take features, not labels
                    model = train_func(X_train)
                else:
                    model = train_func(X_train, y_train)

                if model is None:
                    logger.warning("Skipping %s — model returned None", name)
                    continue

                # Compute PR-AUC on validation set
                if model_type == "supervised" and hasattr(model, "predict_proba") and SKLEARN_AVAILABLE:
                    val_proba = model.predict_proba(X_val)
                    if val_proba.ndim == 2 and val_proba.shape[1] == 2:
                        val_proba = val_proba[:, 1]
                    pr_auc = float(average_precision_score(y_val, val_proba))
                else:
                    # Unsupervised: assign a small base weight
                    pr_auc = 0.1

                self.add_model(name, model, weight=pr_auc, model_type=model_type)
                weights[name] = round(pr_auc, 4)
                per_model_metrics[name] = {"pr_auc": round(pr_auc, 4)}
                logger.info("%s PR-AUC=%.4f", name, pr_auc)

            except Exception as exc:
                logger.warning("Failed to train %s: %s", name, exc)

        # Compute ensemble metrics
        ensemble_metrics: Dict[str, float] = {}
        if self._models and SKLEARN_AVAILABLE:
            try:
                ens_proba = self.predict_proba(X_val)[:, 1]
                ensemble_metrics["pr_auc"] = round(float(average_precision_score(y_val, ens_proba)), 4)
                logger.info("Ensemble PR-AUC=%.4f", ensemble_metrics["pr_auc"])
            except Exception as exc:
                logger.warning("Ensemble evaluation failed: %s", exc)

        return {
            "per_model": per_model_metrics,
            "ensemble": ensemble_metrics,
            "weights": weights,
            "models_trained": list(per_model_metrics.keys()),
        }

    def explain_prediction(
        self,
        X: np.ndarray,
        feature_names: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """Per-model contribution breakdown.

        Args:
            X: Single sample feature array (1D or 2D with 1 row).
            feature_names: Optional feature name list.

        Returns:
            Dict with per-model scores and final score.
        """
        if X.ndim == 1:
            X = X.reshape(1, -1)

        proba = self.predict_proba(X)
        total_weight = sum(self._weights.values())
        contributions = {}

        for name, model in self._models.items():
            weight = self._weights[name] / total_weight
            model_type = self._model_types[name]

            try:
                if model_type == "supervised":
                    p = model.predict_proba(X)
                    score = float(p[0, 1]) if p.ndim == 2 and p.shape[1] == 2 else float(p[0])
                elif hasattr(model, "decision_function"):
                    raw = model.decision_function(X)
                    score = float(1 - (raw[0] - raw.min()) / (raw.max() - raw.min() + 1e-9))
                else:
                    score = 0.0
                contributions[name] = {"score": round(score, 4), "weight": round(weight, 4)}
            except Exception:
                contributions[name] = {"score": 0.0, "weight": round(weight, 4)}

        return {
            "model_contributions": contributions,
            "final_score": round(float(proba[0, 1]), 4),
            "prediction": int(proba[0, 1] >= 0.5),
        }
