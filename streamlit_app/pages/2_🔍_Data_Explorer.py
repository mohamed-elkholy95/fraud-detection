"""Data Explorer page."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px

st.markdown("## 🔍 Data Explorer")

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

tab1, tab2, tab3 = st.tabs(["Amount Distribution", "V-Features", "Time Patterns"])

with tab1:
    col1, col2 = st.columns(2)
    with col1:
        fig = px.histogram(df, x="Amount", nbins=50, color="Class",
                           color_discrete_map={0: "#1f77b4", 1: "#e74c3c"},
                           title="Transaction Amount by Class")
        fig.update_layout(plot_bgcolor="#0e1117", paper_bgcolor="#0e1117", font_color="white")
        st.plotly_chart(fig, use_container_width=True)
    with col2:
        df_log = df.copy()
        df_log["log_amount"] = np.log1p(df_log["Amount"])
        fig = px.histogram(df_log, x="log_amount", nbins=50, color="Class",
                           color_discrete_map={0: "#1f77b4", 1: "#e74c3c"},
                           title="Log-Transformed Amount")
        fig.update_layout(plot_bgcolor="#0e1117", paper_bgcolor="#0e1117", font_color="white")
        st.plotly_chart(fig, use_container_width=True)

with tab2:
    v_features = [f"V{i}" for i in range(1, 29)]
    selected_v = st.multiselect("Select V-features", v_features, default=["V1", "V2", "V3", "V4"])
    if selected_v:
        fraud = df[df["Class"] == 1]
        normal = df[df["Class"] == 0]
        fig = go.Figure()
        for v in selected_v:
            fig.add_trace(go.Box(y=normal[v], name=f"{v} (Normal)", marker_color="#1f77b4"))
            fig.add_trace(go.Box(y=fraud[v], name=f"{v} (Fraud)", marker_color="#e74c3c"))
        fig.update_layout(plot_bgcolor="#0e1117", paper_bgcolor="#0e1117", font_color="white",
                          title="Feature Distributions: Normal vs Fraud")
        st.plotly_chart(fig, use_container_width=True)

with tab3:
    df_hour = df.copy()
    df_hour["hour"] = (df_hour["Time"] % 86400) / 3600
    hourly = df_hour.groupby("hour")["Class"].agg(["mean", "count"]).reset_index()
    hourly.columns = ["hour", "fraud_rate", "count"]

    fig = go.Figure()
    fig.add_trace(go.Bar(x=hourly["hour"], y=hourly["count"], name="Transactions",
                         marker_color="#1f77b4", opacity=0.5))
    fig.add_trace(go.Scatter(x=hourly["hour"], y=hourly["fraud_rate"] * hourly["count"].max() / hourly["fraud_rate"].max(),
                             name="Fraud Rate", marker_color="#e74c3c", yaxis="y2"))
    fig.update_layout(plot_bgcolor="#0e1117", paper_bgcolor="#0e1117", font_color="white",
                      title="Transaction Volume & Fraud Rate by Hour",
                      yaxis2={"overlaying": "y", "side": "right"})
    st.plotly_chart(fig, use_container_width=True)

st.markdown(f"**Data shape:** {df.shape[0]:,} rows × {df.shape[1]} columns")
