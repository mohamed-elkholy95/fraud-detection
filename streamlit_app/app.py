"""Fraud Detection — Streamlit Dashboard."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import streamlit as st
from src.config import STREAMLIT_THEME

st.set_page_config(
    page_title="Fraud Detection Dashboard",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Apply dark theme
for key, val in STREAMLIT_THEME.items():
    if hasattr(st, key):
        setattr(st, key, val)

st.markdown(
    """<style>
    .stApp { background-color: #0e1117; color: #ffffff; }
    .stMetric { background-color: #262730; border-radius: 8px; padding: 12px; }
    </style>""",
    unsafe_allow_html=True,
)

st.title("🔍 Credit Card Fraud Detection")
st.markdown("Real-time fraud detection with ensemble ML models")

pg = st.navigation([
    st.Page("pages/1_📊_Overview.py", title="Overview", icon="📊"),
    st.Page("pages/2_🔍_Data_Explorer.py", title="Data Explorer", icon="🔍"),
    st.Page("pages/3_🤖_Model_Analysis.py", title="Model Analysis", icon="🤖"),
    st.Page("pages/4_⚡_Real_Time_Detection.py", title="Real-Time Detection", icon="⚡"),
    st.Page("pages/5_📈_Drift_Monitor.py", title="Drift Monitor", icon="📈"),
])
pg.run()
