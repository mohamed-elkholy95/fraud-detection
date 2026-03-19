"""Drift Monitor page."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import streamlit as st
import numpy as np
import plotly.graph_objects as go
import pandas as pd

st.markdown("## 📈 Drift Monitor")
st.markdown("Monitor feature distribution drift and model performance degradation.")

# Simulated drift data
st.markdown("### Feature Drift Heatmap")

rng = np.random.default_rng(42)
features = [f"V{i}" for i in range(1, 15)]
periods = [f"Day {i}" for i in range(1, 8)]

psi_data = rng.uniform(0.0, 0.4, (len(features), len(periods)))
psi_data = np.clip(psi_data, 0, 0.5)

fig = go.Figure(data=go.Heatmap(
    z=psi_data, x=periods, y=features,
    colorscale=[[0, "#2ecc71"], [0.25, "#f39c12"], [1, "#e74c3c"]],
    zmin=0, zmax=0.5,
    colorbar=dict(title="PSI Value"),
))
fig.update_layout(plot_bgcolor="#0e1117", paper_bgcolor="#0e1117", font_color="white",
                  title="PSI Drift Detection (Green=Stable, Red=Drifted)")
st.plotly_chart(fig, use_container_width=True)

# Performance over time
st.markdown("### Model Performance Over Time")
days = list(range(1, 31))
roc_auc = 0.997 - np.cumsum(rng.normal(0, 0.0005, 30))
pr_auc = 0.89 - np.cumsum(rng.normal(0, 0.001, 30))
roc_auc = np.clip(roc_auc, 0.95, 1.0)
pr_auc = np.clip(pr_auc, 0.75, 0.95)

fig2 = go.Figure()
fig2.add_trace(go.Scatter(x=days, y=roc_auc, name="ROC-AUC", line_color="#1f77b4"))
fig2.add_trace(go.Scatter(x=days, y=pr_auc, name="PR-AUC", line_color="#2ca02c"))
fig2.add_hline(y=0.95, line_dash="dash", line_color="#e74c3c",
               annotation_text="ROC-AUC Alert Threshold")
fig2.update_layout(plot_bgcolor="#0e1117", paper_bgcolor="#0e1117", font_color="white",
                   title="Performance Metrics", yaxis_title="Score")
st.plotly_chart(fig2, use_container_width=True)

# Alerts
st.markdown("### Recent Alerts")
alerts = [
    {"Time": "14:22:01", "Type": "PSI Drift", "Feature": "V14", "PSI": 0.28, "Status": "⚠️ Warning"},
    {"Time": "14:18:33", "Type": "Performance", "Metric": "PR-AUC", "Change": "-0.032", "Status": "🔴 Alert"},
    {"Time": "13:55:12", "Type": "PSI Drift", "Feature": "V4", "PSI": 0.15, "Status": "✅ Stable"},
    {"Time": "13:42:07", "Type": "KS Test", "Feature": "V17", "P-value": 0.03, "Status": "⚠️ Warning"},
]
st.dataframe(alerts, use_container_width=True, hide_index=True)
