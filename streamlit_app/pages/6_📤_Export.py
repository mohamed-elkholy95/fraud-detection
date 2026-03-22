"""Export & Reporting page — download model reports and data as CSV."""
import io
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Export & Reporting", page_icon="📤", layout="wide")

st.title("📤 Export & Reporting")
st.markdown("Download model comparison reports, prediction results, and drift summaries as CSV.")

# ─── Helpers ──────────────────────────────────────────────────────────────────

def _to_csv_bytes(df: pd.DataFrame) -> bytes:
    """Serialize DataFrame to UTF-8 CSV bytes."""
    buf = io.StringIO()
    df.to_csv(buf, index=False)
    return buf.getvalue().encode("utf-8")


def _now_str() -> str:
    return datetime.utcnow().strftime("%Y%m%d_%H%M%S")


# ─── Generate synthetic data ──────────────────────────────────────────────────

@st.cache_data
def _model_comparison_df() -> pd.DataFrame:
    """Synthetic model comparison report."""
    rng = np.random.default_rng(7)
    models = ["Logistic Regression", "Random Forest", "XGBoost", "LightGBM", "CatBoost", "Ensemble"]
    rows = []
    for m in models:
        base = rng.uniform(0.85, 0.99)
        rows.append({
            "model": m,
            "roc_auc": round(base, 4),
            "pr_auc": round(base - rng.uniform(0.02, 0.08), 4),
            "f1": round(base - rng.uniform(0.04, 0.12), 4),
            "precision": round(base - rng.uniform(0.03, 0.10), 4),
            "recall": round(base - rng.uniform(0.05, 0.15), 4),
            "avg_latency_ms": round(rng.uniform(0.5, 15.0), 2),
            "n_estimators": rng.choice([100, 500, 1000]) if m not in {"Logistic Regression", "Ensemble"} else "N/A",
            "train_time_s": round(rng.uniform(0.2, 60.0), 2),
            "evaluated_at": datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S"),
        })
    return pd.DataFrame(rows)


@st.cache_data
def _prediction_results_df(n: int = 500) -> pd.DataFrame:
    """Synthetic per-transaction prediction results."""
    rng = np.random.default_rng(13)
    is_fraud = rng.choice([0, 1], n, p=[0.97, 0.03])
    proba = np.where(is_fraud, rng.uniform(0.6, 1.0, n), rng.uniform(0.0, 0.15, n))
    decisions = np.where(proba >= 0.75, "block", np.where(proba >= 0.4, "review", "approve"))
    return pd.DataFrame({
        "transaction_id": [f"TXN_{i:06d}" for i in range(n)],
        "timestamp": [(datetime.utcnow() - timedelta(minutes=n - i)).strftime("%Y-%m-%dT%H:%M:%SZ") for i in range(n)],
        "amount_usd": rng.lognormal(2.5, 1.5, n).round(2),
        "fraud_probability": proba.round(4),
        "decision": decisions,
        "true_label": is_fraud,
        "correct": (decisions == np.where(is_fraud, "block", "approve")).astype(int),
        "latency_ms": rng.uniform(0.3, 8.0, n).round(2),
    })


@st.cache_data
def _drift_summary_df() -> pd.DataFrame:
    """Synthetic drift monitoring summary."""
    rng = np.random.default_rng(99)
    n_days = 30
    rows = []
    for day_offset in range(n_days):
        ts = (datetime.utcnow() - timedelta(days=n_days - day_offset)).strftime("%Y-%m-%d")
        drift_score = float(rng.uniform(0.01, 0.25))
        rows.append({
            "date": ts,
            "psi_score": round(drift_score, 4),
            "drift_detected": int(drift_score > 0.15),
            "feature_most_drifted": rng.choice([f"V{i}" for i in range(1, 29)]),
            "mean_fraud_probability": round(float(rng.uniform(0.015, 0.045)), 4),
            "n_transactions": int(rng.integers(800, 1500)),
            "model_accuracy": round(float(rng.uniform(0.90, 0.99)), 4),
            "alert_triggered": int(drift_score > 0.20),
        })
    return pd.DataFrame(rows)


# ─── Summary section ─────────────────────────────────────────────────────────

st.subheader("📊 Key Metrics Summary")
col1, col2, col3, col4 = st.columns(4)

model_df = _model_comparison_df()
best_model = model_df.loc[model_df["roc_auc"].idxmax(), "model"]
best_auc = model_df["roc_auc"].max()

pred_df = _prediction_results_df()
fraud_rate = pred_df["true_label"].mean()
accuracy = pred_df["correct"].mean()

drift_df = _drift_summary_df()
drift_alerts = drift_df["alert_triggered"].sum()

col1.metric("Best Model", best_model)
col2.metric("Best ROC-AUC", f"{best_auc:.4f}")
col3.metric("Fraud Rate", f"{fraud_rate:.2%}")
col4.metric("Drift Alerts (30d)", int(drift_alerts))

st.divider()

# ─── Export: Model Comparison Report ─────────────────────────────────────────

st.subheader("🤖 Model Comparison Report")
st.dataframe(model_df, use_container_width=True)

st.download_button(
    label="⬇️ Download Model Comparison CSV",
    data=_to_csv_bytes(model_df),
    file_name=f"model_comparison_{_now_str()}.csv",
    mime="text/csv",
    help="Download all model performance metrics as CSV",
)

st.divider()

# ─── Export: Prediction Results ───────────────────────────────────────────────

st.subheader("⚡ Individual Prediction Results")

n_rows = st.slider("Number of rows to preview", min_value=10, max_value=100, value=20, step=10)
st.dataframe(pred_df.head(n_rows), use_container_width=True)

st.download_button(
    label="⬇️ Download Prediction Results CSV",
    data=_to_csv_bytes(pred_df),
    file_name=f"prediction_results_{_now_str()}.csv",
    mime="text/csv",
    help="Download per-transaction prediction results as CSV",
)

st.divider()

# ─── Export: Drift Monitoring Summary ────────────────────────────────────────

st.subheader("📈 Drift Monitoring Summary")
st.dataframe(drift_df, use_container_width=True)

st.download_button(
    label="⬇️ Download Drift Summary CSV",
    data=_to_csv_bytes(drift_df),
    file_name=f"drift_summary_{_now_str()}.csv",
    mime="text/csv",
    help="Download 30-day drift monitoring summary as CSV",
)

st.divider()

# ─── Export: Full Summary Report ─────────────────────────────────────────────

st.subheader("📋 Full Summary Report")

summary_data = {
    "metric": [
        "Best Model", "Best ROC-AUC", "Best PR-AUC",
        "Avg Model ROC-AUC", "Avg Model F1",
        "Total Transactions Evaluated", "Fraud Transactions", "Fraud Rate",
        "Overall Accuracy", "Drift Alerts (30 days)",
        "Report Generated At",
    ],
    "value": [
        best_model,
        f"{best_auc:.4f}",
        f"{model_df['pr_auc'].max():.4f}",
        f"{model_df['roc_auc'].mean():.4f}",
        f"{model_df['f1'].mean():.4f}",
        str(len(pred_df)),
        str(int(pred_df['true_label'].sum())),
        f"{fraud_rate:.4%}",
        f"{accuracy:.4%}",
        str(int(drift_alerts)),
        datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC"),
    ],
}
summary_df = pd.DataFrame(summary_data)
st.dataframe(summary_df, use_container_width=True)

st.download_button(
    label="⬇️ Download Full Summary Report CSV",
    data=_to_csv_bytes(summary_df),
    file_name=f"full_summary_report_{_now_str()}.csv",
    mime="text/csv",
    help="Download a compact summary of all key metrics",
)
