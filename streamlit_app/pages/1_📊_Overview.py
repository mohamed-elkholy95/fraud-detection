"""Overview page — key metrics and fraud rate."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from pathlib import Path

from src.config import RAW_DATA_PATH, COST_FALSE_POSITIVE, COST_FALSE_NEGATIVE

st.markdown("## 📊 Dashboard Overview")

# Load or generate data
@st.cache_data
def load_data():
    from src.data.loader import load_fraud_data
    return load_fraud_data()

try:
    df = load_data()
except Exception:
    rng = np.random.default_rng(42)
    df = pd.DataFrame({f"V{i}": rng.normal(0, 1, 1000) for i in range(1, 29)})
    df["Amount"] = rng.lognormal(2.5, 1.5, 1000)
    df["Time"] = rng.uniform(0, 172800, 1000)
    df["Class"] = rng.choice([0, 1], 1000, p=[0.998, 0.002])
    df.to_csv(RAW_DATA_PATH, index=False)

stats = {
    "Total Transactions": f"{len(df):,}",
    "Fraud Cases": f"{int(df['Class'].sum()):,}",
    "Fraud Rate": f"{df['Class'].mean()*100:.3f}%",
    "Avg Amount": f"${df['Amount'].mean():,.2f}",
}

col1, col2, col3, col4 = st.columns(4)
for col, (k, v) in zip([col1, col2, col3, col4], stats.items()):
    col.metric(k, v)

# Fraud rate gauge
fig = go.Figure(go.Indicator(
    mode="gauge+number+delta",
    value=df["Class"].mean() * 100,
    delta={"reference": 0.172, "decreasing": {"color": "green"}, "increasing": {"color": "red"}},
    domain={"x": [0, 1], "y": [0, 1]},
    title={"text": "Fraud Rate (%)"},
    gauge={"axis": {"range": [0, 1]}, "bar": {"color": "#e74c3c"}},
))
fig.update_layout(paper_bgcolor="#0e1117", font_color="white", height=250)
st.plotly_chart(fig, use_container_width=True)

# Cost info
c1, c2 = st.columns(2)
c1.info(f"💰 **Cost per False Positive:** ${COST_FALSE_POSITIVE:.0f} (customer friction)")
c2.info(f"💸 **Cost per False Negative:** ${COST_FALSE_NEGATIVE:.0f} (fraud loss)")
