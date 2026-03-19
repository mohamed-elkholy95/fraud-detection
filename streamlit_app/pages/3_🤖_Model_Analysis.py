"""Model Analysis page."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import streamlit as st
import numpy as np
import plotly.graph_objects as go
import plotly.express as px

st.markdown("## 🤖 Model Analysis")

st.markdown("### Model Performance Comparison")

# Mock model results for demo
models = ["Logistic Regression", "Random Forest", "XGBoost", "Isolation Forest", "Ensemble"]
metrics_data = {
    "Model": models,
    "ROC-AUC": [0.983, 0.994, 0.997, 0.921, 0.998],
    "PR-AUC": [0.782, 0.865, 0.891, 0.512, 0.903],
    "F1 Score": [0.723, 0.812, 0.845, 0.456, 0.862],
    "Recall": [0.681, 0.789, 0.823, 0.512, 0.845],
}
df_metrics = st.session_state.get("metrics_df")
if df_metrics is None:
    df_metrics = __import__("pandas").DataFrame(metrics_data)

st.dataframe(df_metrics, use_container_width=True, hide_index=True)

fig = go.Figure()
colors = ["#1f77b4", "#2ca02c", "#ff7f0e", "#e74c3c", "#9b59b6"]
for i, metric in enumerate(["ROC-AUC", "PR-AUC", "F1 Score", "Recall"]):
    fig.add_trace(go.Bar(name=metric, x=models, y=df_metrics[metric], marker_color=colors[i]))
fig.update_layout(barmode="group", plot_bgcolor="#0e1117", paper_bgcolor="#0e1117",
                  font_color="white", title="Model Comparison")
st.plotly_chart(fig, use_container_width=True)

# ROC Curve
st.markdown("### ROC Curve")
x = np.linspace(0, 1, 100)
fig_roc = go.Figure()
fig_roc.add_trace(go.Scatter(x=x, y=x, mode="lines", name="Random", line=dict(dash="dash")))
for i, (model, auc) in enumerate(zip(models, [0.983, 0.994, 0.997, 0.921, 0.998])):
    y = np.power(x, 1/(1+auc))
    fig_roc.add_trace(go.Scatter(x=x, y=1-y, mode="lines", name=f"{model} (AUC={auc})", line_color=colors[i]))
fig_roc.update_layout(plot_bgcolor="#0e1117", paper_bgcolor="#0e1117", font_color="white",
                       xaxis_title="False Positive Rate", yaxis_title="True Positive Rate")
st.plotly_chart(fig_roc, use_container_width=True)

# Cost Analysis
st.markdown("### Cost Analysis")
col1, col2 = st.columns(2)
col1.metric("Optimal Threshold", "0.15", "Cost-optimized")
col2.metric("Min Expected Cost", "$4,230", "vs $8,500 at 0.5")
