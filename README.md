<div align="center">

# 🕵️ Fraud Detection System

**Real-time fraud detection** with anomaly scoring, cost-sensitive optimization, and data drift monitoring

[![Python](https://img.shields.io/badge/Python-3.12-3776AB?style=flat-square&logo=python)](https://python.org)
[![Tests](https://img.shields.io/badge/Tests-49%20passed-success?style=flat-square)](#)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3-F7931E?style=flat-square&logo=scikit-learn)](https://scikit-learn.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100-009688?style=flat-square)](https://fastapi.tiangolo.com)

</div>

## Overview

A **production-grade fraud detection system** with Isolation Forest and statistical anomaly scoring, cost-sensitive threshold optimization (FP=$10, FN=$500), PSI-based data drift monitoring, KS-test distribution checks, and comprehensive reporting.

## Features

- 🎯 **Anomaly Scoring** — Isolation Forest + statistical z-score hybrid
- 💰 **Cost-Sensitive Thresholds** — Business-aware FP/FN cost optimization
- 📊 **PSI Drift Monitoring** — Population Stability Index for distribution shifts
- 🧪 **KS-Test Validation** — Kolmogorov-Smirnov distribution comparison
- 📈 **5-Page Dashboard** — Alerts, transactions, reports, monitoring
- 🚀 **REST API** — Full fraud detection pipeline endpoints
- ✅ **49 Tests** — Comprehensive coverage of all components

## Quick Start

```bash
git clone https://github.com/mohamed-elkholy95/fraud-detection.git
cd fraud-detection
pip install -r requirements.txt
python -m pytest tests/ -v
streamlit run streamlit_app/app.py
```

## Project Structure

```
src/
├── api/              # FastAPI REST endpoints (predict, batch, stream)
├── data/             # Data loading, feature engineering, validation
│   ├── loader.py     # Data loading with synthetic fallback
│   ├── feature_engineering.py  # Temporal, amount, and interaction features
│   ├── validation.py # Data quality checks before training/inference
│   └── augmentation.py  # SMOTE/ADASYN oversampling
├── models/           # Model training and ensembling
│   ├── supervised.py # LR, RF, XGBoost, LightGBM, CatBoost
│   ├── unsupervised.py  # Isolation Forest, Autoencoder
│   ├── ensemble.py   # Weighted ensemble combiner
│   └── comparison.py # Model comparison and selection utilities
├── config.py         # Centralized configuration (env var overrides)
├── evaluation.py     # Metrics, cross-validation, report generation
├── threshold.py      # Cost-sensitive threshold optimization
├── monitoring.py     # KS-test and PSI drift detection
├── explainability.py # SHAP and LIME explanation wrappers
└── pipeline.py       # End-to-end orchestrator
docs/
├── ARCHITECTURE.md   # System design and module responsibilities
└── GLOSSARY.md       # Fraud detection concepts and terminology
tests/                # 60+ test cases across all modules
```

## Cost Model

| Error Type | Cost | Rationale |
|------------|------|-----------|
| False Positive | $10 | Manual review cost |
| False Negative | $500 | Fraud loss |
| Optimal Threshold | Dynamic | Minimizes total cost |

Costs are configurable via environment variables (`FRAUD_COST_FP`, `FRAUD_COST_FN`)
for deployment flexibility without code changes.

## Documentation

- **[Architecture Guide](docs/ARCHITECTURE.md)** — System design, module responsibilities, design decisions
- **[Glossary](docs/GLOSSARY.md)** — Key concepts, metrics, and techniques explained

## Author

**Mohamed Elkholy** — [GitHub](https://github.com/mohamed-elkholy95) · melkholy@techmatrix.com
