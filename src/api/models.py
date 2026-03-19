"""Pydantic models for the fraud detection API."""
from typing import Dict, List, Optional
from pydantic import BaseModel, Field


class TransactionInput(BaseModel):
    """Input for a single transaction prediction."""
    amount: float = Field(..., gt=0, description="Transaction amount")
    time: Optional[float] = Field(None, description="Seconds from first transaction")
    v1: float = 0.0
    v2: float = 0.0
    v3: float = 0.0
    v4: float = 0.0
    v5: float = 0.0
    v6: float = 0.0
    v7: float = 0.0
    v8: float = 0.0
    v9: float = 0.0
    v10: float = 0.0
    v11: float = 0.0
    v12: float = 0.0
    v13: float = 0.0
    v14: float = 0.0
    v15: float = 0.0
    v16: float = 0.0
    v17: float = 0.0
    v18: float = 0.0
    v19: float = 0.0
    v20: float = 0.0
    v21: float = 0.0
    v22: float = 0.0
    v23: float = 0.0
    v24: float = 0.0
    v25: float = 0.0
    v26: float = 0.0
    v27: float = 0.0
    v28: float = 0.0

    def to_feature_array(self) -> List[float]:
        """Convert to ordered feature list."""
        return [
            self.time or 0.0,
            *[getattr(self, f"v{i}") for i in range(1, 29)],
            self.amount,
        ]


class PredictionResponse(BaseModel):
    """Response for a fraud prediction."""
    fraud_probability: float = Field(..., ge=0.0, le=1.0)
    decision: str = Field(..., description="'approve', 'block', or 'review'")
    confidence: float = Field(..., ge=0.0, le=1.0)
    model_contributions: Optional[Dict[str, float]] = None
    latency_ms: Optional[float] = None


class BatchPredictionRequest(BaseModel):
    """Request for batch predictions."""
    transactions: List[TransactionInput] = Field(..., min_length=1, max_length=1000)


class BatchPredictionResponse(BaseModel):
    """Response for batch predictions."""
    predictions: List[PredictionResponse]
    total: int
    fraud_count: int
    avg_fraud_probability: float


class FeedbackRequest(BaseModel):
    """Feedback on a previous prediction."""
    transaction_id: str
    predicted_label: int
    actual_label: int
    correct: bool


class HealthResponse(BaseModel):
    """API health check response."""
    status: str = "healthy"
    models_loaded: List[str]
    total_predictions: int = 0
