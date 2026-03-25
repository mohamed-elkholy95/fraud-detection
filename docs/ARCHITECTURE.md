# System Architecture

## High-Level Design

```
┌─────────────────────────────────────────────────────┐
│                   Data Layer                         │
│  ┌──────────┐  ┌─────────────┐  ┌───────────────┐  │
│  │ Raw CSV  │→ │  Validator   │→ │ Feature Eng.  │  │
│  │ (loader) │  │  (schema)    │  │ (transform)   │  │
│  └──────────┘  └─────────────┘  └───────────────┘  │
└──────────────────────┬──────────────────────────────┘
                       │
┌──────────────────────▼──────────────────────────────┐
│                   Model Layer                        │
│  ┌──────────┐  ┌──────────┐  ┌──────────────────┐  │
│  │Supervised│  │Unsuperv. │  │    Ensemble       │  │
│  │ Models   │  │ Models   │→ │ (weighted avg)    │  │
│  │ (5 algos)│  │(IsoForest│  │                   │  │
│  │          │  │ Autoenc) │  │                   │  │
│  └────┬─────┘  └────┬─────┘  └────────┬─────────┘  │
│       └──────────────┴─────────────────┘            │
└──────────────────────┬──────────────────────────────┘
                       │
┌──────────────────────▼──────────────────────────────┐
│               Evaluation & Optimization              │
│  ┌──────────────┐  ┌──────────┐  ┌──────────────┐  │
│  │ Cost-Sensitive│  │Threshold │  │ Cross-       │  │
│  │ Analysis     │  │Optimizer │  │ Validation   │  │
│  └──────────────┘  └──────────┘  └──────────────┘  │
└──────────────────────┬──────────────────────────────┘
                       │
┌──────────────────────▼──────────────────────────────┐
│               Serving & Monitoring                   │
│  ┌──────────┐  ┌──────────────┐  ┌──────────────┐  │
│  │ FastAPI  │  │   Drift      │  │  Streamlit   │  │
│  │ REST API │  │  Detector    │  │  Dashboard   │  │
│  └──────────┘  └──────────────┘  └──────────────┘  │
└─────────────────────────────────────────────────────┘
```

## Module Responsibilities

### `src/data/`
- **`loader.py`** — Data loading, schema validation, synthetic data generation,
  temporal train/test splitting
- **`feature_engineering.py`** — Stateful transformer: temporal features, amount
  transformations, V-feature aggregates, interaction terms
- **`feature_selection.py`** — Hybrid feature ranking (mutual information + model importance)
- **`augmentation.py`** — SMOTE, ADASYN, SMOTE-Tomek oversampling for class balance

### `src/models/`
- **`supervised.py`** — Logistic Regression, Random Forest, XGBoost, LightGBM, CatBoost
  training functions with balanced class handling
- **`unsupervised.py`** — Isolation Forest and Autoencoder anomaly detectors
- **`ensemble.py`** — `FraudEnsemble` class: weighted model combination, full pipeline
  training, prediction explanation
- **`persistence.py`** — Model serialization (save/load with joblib)
- **`network.py`** — Graph-based fraud network analysis

### `src/`
- **`config.py`** — Centralized configuration: paths, costs, thresholds, hyperparameters
- **`evaluation.py`** — Metrics computation, confusion matrix, cost analysis, cross-validation,
  PR/ROC curve data, markdown report generation
- **`threshold.py`** — Cost-sensitive threshold optimization and threshold analysis
- **`monitoring.py`** — `DriftDetector` class: KS-test, PSI computation, performance
  drift tracking, prediction logging
- **`explainability.py`** — SHAP and LIME explanation wrappers with mock fallback
- **`pipeline.py`** — `FraudDetectionPipeline` orchestrator: end-to-end workflow from
  data loading through evaluation and reporting

### `src/api/`
- **`main.py`** — FastAPI application with predict, batch predict, stream, health,
  feedback, and monitoring endpoints
- **`models.py`** — Pydantic request/response schemas

## Design Decisions

### Why Ensemble Over Single Model?
No single algorithm dominates across all fraud patterns. The ensemble combines:
- **Logistic Regression**: Interpretable baseline, good on linear separable patterns
- **Random Forest**: Handles feature interactions, robust to noise
- **XGBoost/LightGBM/CatBoost**: State-of-art gradient boosting for tabular data
- **Isolation Forest**: Catches novel fraud patterns without labeled examples

### Why Temporal Split Over Random Split?
Financial transactions have temporal dependencies. Random splitting would leak
future information into training, producing optimistically biased metrics. Temporal
splitting ensures the model only trains on past data and evaluates on future data,
matching real-world deployment conditions.

### Why PR-AUC as Ensemble Weight?
PR-AUC directly measures a model's ability to rank fraud cases higher than
legitimate ones, respecting the severe class imbalance. Models contributing more
to fraud identification receive higher ensemble weight, maximizing the combined
detector's ability to surface true fraud.
