# Project Plan: Real-Time Credit Card Fraud Detection

**Project:** 13-fraud-detection  
**Category:** Anomaly Detection / Supervised Classification  
**Difficulty:** Advanced (⭐⭐⭐⭐)  
**Timeline:** 12 days  
**Status:** 🔨 In Progress

---

## Overview

Build a production-grade fraud detection system that combines multiple ML approaches — supervised classification, unsupervised anomaly detection, and deep learning — into a real-time ensemble with explainability and monitoring.

The system must handle extreme class imbalance (0.17% fraud rate), make decisions in under 100ms, and minimize total cost (not just maximize accuracy). Every blocked transaction costs $10 in customer friction; every missed fraud costs $500+.

---

## Phase 1: Problem Definition & Setup (Day 1)

### Business Problem
- Detect fraudulent credit card transactions in real-time
- Two types of errors with asymmetric costs:
  - **False Positive (FP):** Legitimate transaction blocked → customer inconvenience → ~$10 cost
  - **False Negative (FN):** Fraud missed → financial loss + chargeback fees → ~$500 cost
- This asymmetry means threshold selection is critical — not just model accuracy

### Success Metrics
| Metric | Target | Rationale |
|---|---|---|
| ROC-AUC | > 0.99 | Standard fraud detection benchmark |
| PR-AUC | > 0.85 | More informative with imbalanced classes |
| F1 Score | > 0.80 | Balanced precision/recall |
| Inference Latency | < 50ms | Real-time requirement |
| API Response | < 100ms | End-to-end including overhead |

### Constraints
- Class imbalance: 492 fraud / 284,315 legitimate (0.172% positive rate)
- Time-based ordering must be respected (no future leakage)
- Model must be interpretable enough for regulatory compliance (SR 11-7)

### Environment Setup
```bash
python -m venv .venv
source .venv/bin/activate
pip install pandas numpy scikit-learn xgboost shap lime imbalanced-learn \
    fastapi uvicorn redis streamlit plotly tensorflow
```

---

## Phase 2: Data Pipeline (Days 1-2)

### Dataset: Kaggle Credit Card Fraud Detection
- **Size:** 284,807 transactions (2 days in September 2013)
- **Features:** V1-V28 (PCA-transformed for confidentiality), Time (seconds from first transaction), Amount
- **Labels:** Class (0=legitimate, 1=fraud)
- **Imbalance:** 492 fraud / 284,315 normal

### File: `src/data/loader.py`

```python
def load_fraud_data(path: str) -> pd.DataFrame:
    """Load creditcard.csv and validate schema."""
    ...

def split_temporal(
    df: pd.DataFrame,
    test_ratio: float = 0.2
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Time-based split respecting temporal ordering.
    First 80% of transactions → train
    Last 20% → test (simulates deployment)
    """
    ...
```

### File: `src/data/augmentation.py`

```python
def oversample_minority(
    X: np.ndarray,
    y: np.ndarray,
    method: str = "smote"  # smote | adasyn | borderline | random
) -> tuple[np.ndarray, np.ndarray]:
    """Apply SMOTE or ADASYN to training data only."""
    ...

def augment_with_vae(
    X_fraud: np.ndarray,
    n_samples: int = 1000
) -> np.ndarray:
    """
    Train a Variational Autoencoder on fraud samples.
    Generate synthetic fraud transactions.
    Returns: synthetic fraud feature matrix.
    """
    ...
```

### File: `src/data/feature_engineering.py`

```python
class TransactionFeatureEngineer:
    """
    Engineer features from raw transaction data.
    
    New Features:
    - hour_of_day: transaction hour (0-23) derived from Time
    - day_of_week: transaction day derived from Time
    - amount_bins: categorical binning of Amount (0-10, 10-50, 50-200, 200-500, 500+)
    - amount_zscore: standardized Amount within current hour
    - rolling_mean_amount: rolling 100-transaction mean of Amount
    - rolling_std_amount: rolling 100-transaction std of Amount
    - v_features_mean: mean of V1-V28
    - v_features_std: std of V1-V28
    - v_features_min: min of V1-V28
    - v_features_max: max of V1-V28
    - v1_v2_interaction: V1 * V2 (known fraud signal)
    - amount_log1p: log1p transform of Amount
    """
    
    def fit(self, df: pd.DataFrame) -> "TransactionFeatureEngineer":
        """Compute statistics for rolling features from training data."""
        ...
    
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply all feature transformations."""
        ...
    
    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        ...
```

---

## Phase 3: Model Building (Days 3-6)

Build 5 diverse models and one ensemble. Diversity is key — different models catch different fraud patterns.

### Model 1: Logistic Regression (Baseline)
- Interpretable, fast inference
- Class weights balanced via `class_weight='balanced'`
- L2 regularization
- **Purpose:** Establish minimum acceptable performance

### Model 2: Random Forest
- Handles non-linear relationships
- Robust to outliers
- `n_estimators=500`, `class_weight='balanced_subsample'`
- Feature importance via permutation importance

### Model 3: XGBoost
- Typically best performance on tabular data
- `scale_pos_weight = n_negative / n_positive` for imbalance
- `max_depth=6`, `subsample=0.8`, `colsample_bytree=0.8`
- Early stopping on PR-AUC

### Model 4: Isolation Forest (Unsupervised)
- No labels needed — learns "normal" behavior
- Flags transactions that are anomalous
- `contamination=0.00172` (matching true fraud rate)
- **Use case:** Zero-day fraud patterns not in training data

### Model 5: Autoencoder (Deep Learning)
- Trains on normal transactions only
- Reconstruction error is the anomaly score
- High error → transaction doesn't look normal → potential fraud
- Architecture: 30 → 20 → 10 → 5 → 10 → 20 → 30

### Model 6: Ensemble
Combine all models via weighted voting.

### File: `src/models/supervised.py`

```python
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb

def train_logistic_regression(
    X_train: np.ndarray,
    y_train: np.ndarray,
    C: float = 1.0
) -> LogisticRegression:
    """Train logistic regression with balanced class weights."""
    ...

def train_xgboost(
    X_train: np.ndarray,
    y_train: np.ndarray,
    scale_pos_weight: float = None,
    n_estimators: int = 1000,
    learning_rate: float = 0.05,
    max_depth: int = 6
) -> xgb.XGBClassifier:
    """Train XGBoost with early stopping on validation PR-AUC."""
    ...

def train_random_forest(
    X_train: np.ndarray,
    y_train: np.ndarray,
    n_estimators: int = 500
) -> RandomForestClassifier:
    """Train Random Forest with balanced subsample class weights."""
    ...
```

### File: `src/models/unsupervised.py`

```python
from sklearn.ensemble import IsolationForest
import tensorflow as tf

def train_isolation_forest(
    X_train: np.ndarray,
    contamination: float = 0.001
) -> IsolationForest:
    """
    Train Isolation Forest on training data (all classes).
    Returns fitted model. Anomaly scores via decision_function().
    """
    ...

def train_autoencoder(
    X_train: np.ndarray,
    encoding_dim: int = 15,
    epochs: int = 50,
    batch_size: int = 256
) -> tuple[tf.keras.Model, float]:
    """
    Train autoencoder on NORMAL transactions only.
    Returns: (fitted model, reconstruction_error_threshold)
    Threshold: 95th percentile of normal reconstruction error.
    """
    ...
```

### File: `src/models/ensemble.py`

```python
class FraudEnsemble:
    """
    Weighted ensemble of multiple fraud detection models.
    
    Supports both supervised (predict_proba) and unsupervised
    (decision function / reconstruction error) models.
    """
    
    def __init__(self):
        self.models: dict[str, Any] = {}
        self.weights: dict[str, float] = {}
        self.model_types: dict[str, str] = {}  # supervised | unsupervised
    
    def add_model(
        self,
        name: str,
        model: Any,
        weight: float,
        model_type: str = "supervised"
    ) -> None:
        """Register a model with its weight."""
        ...
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Combined prediction using weighted majority vote.
        Returns: binary predictions (0/1)
        """
        ...
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Weighted average of fraud probabilities from all models.
        Unsupervised models: normalize anomaly scores to [0,1].
        Returns: shape (n_samples, 2) — [P(legit), P(fraud)]
        """
        ...
    
    def explain_prediction(
        self,
        X: np.ndarray,
        y_true: np.ndarray = None
    ) -> dict:
        """
        Per-model contributions + aggregated feature importance.
        Returns: {
            "model_scores": {model_name: score},
            "final_score": float,
            "feature_importance": {feature: importance},
            "prediction": 0 or 1
        }
        """
        ...
```

---

## Phase 4: Threshold Optimization (Days 6-7)

Standard 0.5 threshold is wrong for this problem. We optimize for minimum expected cost.

**Expected Cost Formula:**
```
E[Cost] = FP_rate × cost_fp + FN_rate × cost_fn
        = FP_rate × $10 + FN_rate × $500
```

### File: `src/threshold.py`

```python
def optimize_threshold(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    cost_fp: float = 10.0,
    cost_fn: float = 500.0
) -> float:
    """
    Find threshold that minimizes expected total cost.
    Searches thresholds from 0.01 to 0.99 in steps of 0.001.
    Returns: optimal threshold value.
    """
    ...

def threshold_analysis(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    thresholds: np.ndarray = None
) -> pd.DataFrame:
    """
    Compute metrics at each threshold.
    Returns DataFrame with columns:
    [threshold, precision, recall, f1, fpr, tpr, expected_cost, tp, fp, tn, fn]
    """
    ...

def cost_curve(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    cost_fp: float = 10.0,
    cost_fn: float = 500.0
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute expected cost at each threshold.
    Returns: (thresholds, costs) arrays for plotting.
    """
    ...
```

---

## Phase 5: Real-Time API (Days 7-9)

FastAPI with Redis caching for sub-50ms inference.

### Architecture
```
Client → FastAPI → Feature Engineering → Ensemble.predict_proba() → Threshold → Response
                         ↑                       ↑
                   Redis cache              Model artifacts (loaded at startup)
```

### Endpoints

| Method | Path | Description |
|---|---|---|
| POST | `/predict` | Single transaction fraud score |
| POST | `/predict_batch` | Batch scoring (up to 1000) |
| GET | `/model/performance` | Live accuracy, precision, recall |
| GET | `/model/drift` | Current drift status per feature |
| POST | `/feedback` | Report misclassification (labels for retraining) |
| GET | `/health` | Health check + model info |

### File: `src/api/main.py`
FastAPI application with:
- Startup: load model artifacts from `models/` directory
- Middleware: request timing, logging
- CORS configuration
- Swagger/OpenAPI docs at `/docs`

### File: `src/api/predict.py`
```python
async def predict_transaction(transaction: TransactionInput) -> PredictionResponse:
    """
    Single transaction prediction.
    - Check Redis cache (transaction fingerprint)
    - Apply feature engineering
    - Run ensemble predict_proba()
    - Apply cost-optimized threshold
    - Log to monitoring store
    - Return: fraud_score, decision, explanation
    Target: < 50ms
    """
    ...
```

### File: `src/api/monitoring.py`
Endpoints for live metrics retrieval from monitoring store.

### Request/Response Models
```python
class TransactionInput(BaseModel):
    amount: float
    v1: float  # through v28
    ...
    time: Optional[float] = None

class PredictionResponse(BaseModel):
    transaction_id: str
    fraud_probability: float
    decision: Literal["approve", "block", "review"]
    confidence: float
    latency_ms: float
    explanation: Optional[dict] = None
```

---

## Phase 6: Monitoring & Drift Detection (Days 9-10)

Production models degrade. Feature distributions shift. We need to know before accuracy falls.

### Drift Detection Methods
- **KS Test (Kolmogorov-Smirnov):** Compare feature distributions statistically
- **PSI (Population Stability Index):** Binned distribution comparison. PSI > 0.25 = major shift
- **Performance Drift:** Rolling window accuracy/PR-AUC monitoring

### File: `src/monitoring.py`

```python
class DriftDetector:
    """
    Monitor feature drift and performance degradation in production.
    
    Uses sliding window of recent predictions to compare against
    training distribution reference.
    """
    
    def __init__(
        self,
        reference_data: pd.DataFrame,
        window_size: int = 1000,
        ks_threshold: float = 0.05,  # p-value threshold
        psi_threshold: float = 0.25
    ):
        ...
    
    def check_feature_drift(
        self,
        reference: pd.DataFrame,
        current: pd.DataFrame,
        features: list[str] = None
    ) -> dict:
        """
        Run KS test and PSI for each feature.
        Returns: {
            feature: {
                "ks_statistic": float,
                "ks_pvalue": float,
                "psi": float,
                "drift_detected": bool
            }
        }
        """
        ...
    
    def check_performance_drift(
        self,
        reference_metrics: dict,
        current_metrics: dict,
        threshold: float = 0.05
    ) -> dict:
        """
        Compare current vs reference ROC-AUC, PR-AUC.
        Alert if degradation > threshold.
        """
        ...
    
    def log_prediction(
        self,
        prediction: dict,
        actual: int = None
    ) -> None:
        """
        Store prediction for monitoring.
        actual: ground truth label if available (from feedback endpoint)
        """
        ...
```

---

## Phase 7: Streamlit Dashboard (Days 10-11)

Real-time monitoring dashboard for operations team.

### File: `src/dashboard/app.py`

**Sections:**
1. **Header:** Current fraud rate gauge (Plotly indicator)
2. **Transaction Timeline:** Last 24h transactions with fraud markers (scatter plot)
3. **Model Performance:** Rolling ROC-AUC, PR-AUC, F1 over time
4. **Feature Importance:** SHAP bar chart (top 10 features)
5. **Drift Monitor:** Heatmap of feature drift (PSI values)
6. **Alert Feed:** Recent drift/performance alerts with timestamps

**Real-time Updates:** `st.empty()` + `time.sleep(5)` polling for live data.

---

## Phase 8: Explainability (Days 11-12)

Every fraud decision must be explainable — for compliance, debugging, and customer disputes.

### SHAP Explainability
- **TreeExplainer** for tree-based models (fast)
- **DeepExplainer** for autoencoder
- **KernelExplainer** for ensemble (model-agnostic, slower)

### LIME Explainability
- Local approximation for any individual prediction
- More intuitive for business stakeholders

### File: `src/explainability.py`

```python
def explain_transaction(
    model: Any,
    transaction: np.ndarray,
    background_data: np.ndarray,
    method: str = "shap"  # shap | lime
) -> dict:
    """
    Generate explanation for a single transaction prediction.
    
    Returns: {
        "feature_contributions": {feature: value},
        "base_value": float,
        "prediction": float,
        "top_factors": list[dict],  # top 5 contributing features
        "visualization": str  # base64-encoded plot
    }
    """
    ...
```

---

## Evaluation Strategy

### Offline Evaluation
- Time-based train/test split (no data leakage)
- Metrics: ROC-AUC, PR-AUC, F1, Precision, Recall at optimal threshold
- Cost analysis: total expected cost at various thresholds
- Confusion matrix

### Online Evaluation (Production)
- A/B test: old model vs new model on 10% traffic
- Monitor: prediction latency (p50, p95, p99)
- Feedback loop: manual labeling of flagged transactions

---

## Testing Plan

| Test File | Coverage |
|---|---|
| `tests/test_data_loader.py` | load_fraud_data, split_temporal |
| `tests/test_feature_engineering.py` | TransactionFeatureEngineer transformations |
| `tests/test_augmentation.py` | SMOTE shapes, class distribution |
| `tests/test_supervised.py` | Model training, predict_proba shape |
| `tests/test_unsupervised.py` | Isolation Forest, Autoencoder threshold |
| `tests/test_ensemble.py` | FraudEnsemble predict, explain |
| `tests/test_threshold.py` | Cost curve, optimal threshold |
| `tests/test_api.py` | FastAPI endpoints (TestClient) |
| `tests/test_monitoring.py` | DriftDetector KS, PSI |

---

## Dependencies

```txt
pandas>=2.0
numpy>=1.24
scikit-learn>=1.3.0
xgboost>=2.0.0
tensorflow>=2.14.0
shap>=0.43.0
lime>=0.2.0
imbalanced-learn>=0.11.0
fastapi>=0.109.0
uvicorn>=0.27.0
streamlit>=1.30.0
plotly>=5.18.0
redis>=5.0.0
pydantic>=2.0.0
scipy>=1.11.0
pytest>=7.4.0
pytest-asyncio>=0.21.0
httpx>=0.25.0
```

---

## Timeline

| Day | Tasks |
|---|---|
| 1 | Setup, EDA, data pipeline |
| 2 | Feature engineering, augmentation |
| 3-4 | Supervised models (LR, RF, XGB) |
| 5-6 | Unsupervised models (IsoForest, Autoencoder) |
| 6-7 | Ensemble, threshold optimization |
| 7-8 | FastAPI implementation |
| 8-9 | Redis caching, API testing |
| 9-10 | Monitoring, drift detection |
| 10-11 | Streamlit dashboard |
| 11-12 | SHAP/LIME explainability |
| 12 | Integration testing, documentation |

---

## Key Design Decisions

1. **Time-based split over random split** — prevents data leakage; simulates real deployment
2. **Ensemble over single model** — different models catch different fraud types
3. **Cost-sensitive threshold** — optimizes business outcome, not just F1
4. **Autoencoder on normals only** — can detect novel fraud patterns not in training data
5. **SHAP for compliance** — regulators require explainable credit decisions (ECOA, FCRA)
6. **Redis cache** — deduplicates rapid retries, enables sub-50ms latency target
