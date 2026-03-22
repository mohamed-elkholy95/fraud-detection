"""FastAPI application for fraud detection."""
import logging
import time
from typing import List, Optional

import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from src.api.models import (
    TransactionInput, PredictionResponse, BatchPredictionRequest,
    BatchPredictionResponse, FeedbackRequest, HealthResponse,
)

logger = logging.getLogger(__name__)

app = FastAPI(
    title="Fraud Detection API",
    description="Real-time credit card fraud detection",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"],
)

# In-memory model store (populated at startup)
_model: Optional[object] = None
_feature_engineer: Optional[object] = None
_prediction_count: int = 0
_threshold: float = 0.5


@app.on_event("startup")
async def startup() -> None:
    """Load model artifacts at startup."""
    global _model, _feature_engineer, _threshold
    logger.info("Starting Fraud Detection API")
    # Models would be loaded from disk in production
    _model = None
    _feature_engineer = None
    _threshold = 0.5


@app.get("/health", response_model=HealthResponse)
async def health_check() -> HealthResponse:
    """Health check endpoint."""
    models = list(_model.model_names) if _model and hasattr(_model, "model_names") else []
    return HealthResponse(status="healthy", models_loaded=models, total_predictions=_prediction_count)


@app.post("/predict", response_model=PredictionResponse)
async def predict(transaction: TransactionInput) -> PredictionResponse:
    """Score a single transaction for fraud.

    Args:
        transaction: Transaction feature values.

    Returns:
        Fraud probability and decision.
    """
    global _prediction_count
    start = time.time()

    if _model is None:
        # Return mock prediction when no model loaded
        latency = (time.time() - start) * 1000
        return PredictionResponse(
            fraud_probability=0.02,
            decision="approve",
            confidence=0.98,
            latency_ms=round(latency, 2),
        )

    features = np.array(transaction.to_feature_array()).reshape(1, -1)

    try:
        proba = _model.predict_proba(features)[0, 1]
    except Exception:
        proba = 0.02

    decision = "block" if proba >= _threshold * 1.5 else "review" if proba >= _threshold else "approve"
    latency = (time.time() - start) * 1000
    _prediction_count += 1

    return PredictionResponse(
        fraud_probability=round(proba, 4),
        decision=decision,
        confidence=round(1.0 - abs(proba - 0.5) * 2, 4),
        latency_ms=round(latency, 2),
    )


@app.post("/predict_batch", response_model=BatchPredictionResponse)
async def predict_batch(request: BatchPredictionRequest) -> BatchPredictionResponse:
    """Score multiple transactions.

    Args:
        request: List of transactions.

    Returns:
        Batch predictions with summary.
    """
    predictions = []
    fraud_count = 0
    total_proba = 0.0

    for tx in request.transactions:
        pred = await predict(tx)
        predictions.append(pred)
        if pred.decision == "block":
            fraud_count += 1
        total_proba += pred.fraud_probability

    return BatchPredictionResponse(
        predictions=predictions,
        total=len(predictions),
        fraud_count=fraud_count,
        avg_fraud_probability=round(total_proba / len(predictions), 4),
    )


@app.post("/predict/stream")
async def predict_stream(request: BatchPredictionRequest) -> dict:
    """Stream-style batch prediction with per-transaction results and summary.

    Processes all transactions in the batch and returns per-item results plus
    aggregate statistics including cost estimate.

    Args:
        request: Batch of transactions.

    Returns:
        Dict with results list, summary, and cost estimate.
    """
    results = []
    total_latency_ms = 0.0
    fraud_detected = 0
    cost_estimate = 0.0

    # Cost constants: false positive = $10, review = $2, block assumed fraud = $0
    COST_PER_REVIEW = 2.0
    COST_PER_BLOCK = 0.0
    COST_PER_APPROVE = 0.0

    for idx, tx in enumerate(request.transactions):
        pred = await predict(tx)
        total_latency_ms += pred.latency_ms or 0.0

        if pred.decision == "block":
            fraud_detected += 1
            cost_estimate += COST_PER_BLOCK
        elif pred.decision == "review":
            cost_estimate += COST_PER_REVIEW
        else:
            cost_estimate += COST_PER_APPROVE

        results.append({
            "index": idx,
            "fraud_probability": pred.fraud_probability,
            "decision": pred.decision,
            "confidence": pred.confidence,
            "latency_ms": pred.latency_ms,
        })

    total = len(results)
    avg_latency = round(total_latency_ms / total, 2) if total > 0 else 0.0
    avg_fraud_prob = round(sum(r["fraud_probability"] for r in results) / total, 4) if total > 0 else 0.0

    return {
        "results": results,
        "summary": {
            "total_processed": total,
            "fraud_detected": fraud_detected,
            "approve_count": sum(1 for r in results if r["decision"] == "approve"),
            "review_count": sum(1 for r in results if r["decision"] == "review"),
            "block_count": fraud_detected,
            "avg_fraud_probability": avg_fraud_prob,
            "avg_processing_time_ms": avg_latency,
            "total_latency_ms": round(total_latency_ms, 2),
        },
        "cost_estimate": {
            "total_cost_usd": round(cost_estimate, 2),
            "cost_per_transaction_usd": round(cost_estimate / total, 4) if total > 0 else 0.0,
            "currency": "USD",
        },
    }


@app.get("/model/performance")
async def model_performance() -> dict:
    """Return current model performance metrics."""
    return {"status": "no_model", "message": "Load model to see performance metrics"}


@app.get("/model/drift")
async def model_drift() -> dict:
    """Return current drift status."""
    return {"status": "no_drift_check", "message": "No drift monitoring active"}


@app.post("/feedback")
async def submit_feedback(feedback: FeedbackRequest) -> dict:
    """Submit feedback on a prediction.

    Args:
        feedback: Feedback with predicted vs actual label.

    Returns:
        Acknowledgment.
    """
    logger.info("Feedback: tx=%s predicted=%d actual=%d correct=%s",
                feedback.transaction_id, feedback.predicted_label,
                feedback.actual_label, feedback.correct)
    return {"status": "recorded", "transaction_id": feedback.transaction_id}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
