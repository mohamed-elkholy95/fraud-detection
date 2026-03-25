# Fraud Detection Glossary

A reference guide for key concepts, metrics, and techniques used in this project.

## Classification Metrics

| Term | Definition |
|------|-----------|
| **True Positive (TP)** | Correctly identified fraud transaction |
| **False Positive (FP)** | Legitimate transaction flagged as fraud ("false alarm") |
| **True Negative (TN)** | Correctly identified legitimate transaction |
| **False Negative (FN)** | Missed fraud — the most costly error type |
| **Precision** | TP / (TP + FP) — of all flagged transactions, how many are actually fraud |
| **Recall (Sensitivity)** | TP / (TP + FN) — of all actual fraud, how many did we catch |
| **F1 Score** | Harmonic mean of precision and recall: 2 × (P × R) / (P + R) |
| **ROC-AUC** | Area under the ROC curve; measures discrimination across all thresholds |
| **PR-AUC** | Area under the Precision-Recall curve; preferred metric for imbalanced datasets |

## Why PR-AUC Over ROC-AUC?

In fraud detection, classes are heavily imbalanced (e.g., 0.17% fraud). ROC-AUC can
appear high even with poor fraud detection because TN dominates the FPR denominator.
PR-AUC focuses solely on the positive (fraud) class, making it more informative when
the minority class is the target.

## Cost-Sensitive Optimization

Traditional classification uses a fixed 0.5 threshold. In fraud detection, the
**asymmetric cost structure** changes the optimal operating point:

- **False Positive cost ($10)**: Customer friction, manual review time
- **False Negative cost ($500)**: Undetected fraud, financial loss, chargebacks

The optimal threshold minimizes: `Expected Cost = FP × $10 + FN × $500`

This typically shifts the threshold **below 0.5** to catch more fraud at the expense
of more false alarms — because missing fraud is 50× more expensive.

## Anomaly Detection Approaches

### Isolation Forest
Tree-based method that isolates anomalies by randomly selecting features and split
values. Anomalies require fewer splits to isolate, producing shorter average path
lengths. Does not require labeled fraud examples.

### Autoencoder
Neural network trained to reconstruct normal transactions. Fraud transactions
produce high reconstruction error because the model never learned their patterns.
The anomaly threshold is set at a percentile (e.g., 95th) of training reconstruction errors.

## Data Drift Monitoring

### Kolmogorov-Smirnov (KS) Test
Non-parametric test comparing two distributions. A low p-value (< 0.05) indicates
the feature distribution has shifted between training and production data.

### Population Stability Index (PSI)
Measures how much a variable's distribution has shifted over time:
- PSI < 0.10: No significant shift
- 0.10 ≤ PSI < 0.25: Moderate shift — investigate
- PSI ≥ 0.25: Significant shift — retrain model

Formula: `PSI = Σ (actual% - expected%) × ln(actual% / expected%)`

## Ensemble Methods

This project combines multiple models via weighted averaging, where weights are
proportional to each model's PR-AUC on validation data. This approach:

1. Reduces variance (different models make different errors)
2. Captures both supervised patterns and unsupervised anomalies
3. Provides robustness against any single model's failure mode

## Feature Engineering Concepts

| Feature | Purpose |
|---------|---------|
| **Log-transformed amount** | Reduces skewness of transaction amounts |
| **Z-score normalization** | Centers features for algorithms sensitive to scale |
| **V-feature aggregates** | Captures overall PCA component behavior patterns |
| **Interaction terms** | Models non-linear relationships between PCA components |
| **Temporal features** | Captures time-of-day fraud patterns (fraud peaks at night) |

## Class Imbalance Strategies

| Strategy | How It Works |
|----------|-------------|
| **Class weights** | Penalizes misclassifying the minority class more heavily |
| **SMOTE** | Generates synthetic fraud examples by interpolating between neighbors |
| **ADASYN** | Adaptive SMOTE that focuses on harder-to-learn fraud examples |
| **Undersampling** | Reduces majority class to balance the dataset |
| **Threshold tuning** | Adjusts decision boundary instead of resampling |
