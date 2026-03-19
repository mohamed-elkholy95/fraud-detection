"""Real-Time Detection simulation page."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import streamlit as st
import numpy as np
import plotly.graph_objects as go

st.markdown("## ⚡ Real-Time Detection Simulation")

st.markdown("Simulate streaming transactions and watch the fraud detection system score them.")

col1, col2, col3 = st.columns(3)
threshold = col1.slider("Detection Threshold", 0.01, 0.99, 0.15, 0.01)
fraud_rate = col2.slider("Simulation Fraud Rate", 0.001, 0.05, 0.005, 0.001)
n_transactions = col3.number_input("Transactions to Simulate", 10, 500, 100)

if st.button("🎲 Run Simulation", type="primary"):
    rng = np.random.default_rng(42)
    n_fraud = max(1, int(n_transactions * fraud_rate))
    n_normal = n_transactions - n_fraud

    # Generate transaction features
    normal_scores = rng.normal(0.05, 0.08, n_normal)
    fraud_scores = rng.normal(0.7, 0.2, n_fraud)
    scores = np.concatenate([normal_scores, fraud_scores])
    scores = np.clip(scores, 0, 1)
    rng.shuffle(scores)

    decisions = ["🟢 Approve" if s < threshold else "🔴 Block" for s in scores]
    colors = ["#2ecc71" if s < threshold else "#e74c3c" for s in scores]

    # Results
    blocked = sum(1 for d in decisions if "Block" in d)
    approved = n_transactions - blocked

    m1, m2, m3 = st.columns(3)
    m1.metric("Total", n_transactions)
    m2.metric("Approved", approved, delta=f"{approved/n_transactions*100:.1f}%")
    m3.metric("Blocked", blocked, delta=f"{blocked/n_transactions*100:.1f}%", delta_color="inverse")

    # Score distribution
    fig = go.Figure()
    fig.add_trace(go.Histogram(x=scores, nbinsx=30, marker_color="#1f77b4", name="Scores"))
    fig.add_vline(x=threshold, line_dash="dash", line_color="#e74c3c", line_width=2,
                  annotation_text=f"Threshold={threshold}")
    fig.update_layout(plot_bgcolor="#0e1117", paper_bgcolor="#0e1117", font_color="white",
                      title="Fraud Score Distribution")
    st.plotly_chart(fig, use_container_width=True)

    # Transaction log
    with st.expander(f"📋 Transaction Log ({n_transactions} transactions)"):
        log_data = [{"#": i+1, "Score": round(float(s), 4), "Decision": decisions[i]}
                    for i, s in enumerate(scores)]
        st.dataframe(log_data, use_container_width=True, hide_index=True, height=300)
