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

## Cost Model

| Error Type | Cost | Rationale |
|------------|------|-----------|
| False Positive | $10 | Manual review cost |
| False Negative | $500 | Fraud loss |
| Optimal Threshold | Dynamic | Minimizes total cost |

## Author

**Mohamed Elkholy** — [GitHub](https://github.com/mohamed-elkholy95) · melkholy@techmatrix.com
