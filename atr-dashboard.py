
import streamlit as st
import pandas as pd
from render_atr_chart import render_atr_chart

# Load the simulated data (or real one in production)
df = pd.read_csv("fake_atr_dashboard_data.csv")

st.set_page_config(page_title="ATR Roadmap", layout="wide")
st.title("ATR Levels Roadmap (Simulated Data)")

# Sidebar controls
st.sidebar.header("🔧 Select Scenario")
trigger_levels = sorted(df["trigger_level"].unique())
hours = sorted(df["trigger_hour"].unique())

trigger = st.sidebar.selectbox("Trigger Level", trigger_levels, index=trigger_levels.index(0.0))
hour = st.sidebar.selectbox("Trigger Time", hours)

# Direction option may be used later for retracement logic
_ = st.sidebar.radio("Price Approached Trigger From", ["Below", "Above"])

# Filtered Data Check
filtered = df[(df["trigger_level"] == trigger) & (df["trigger_hour"] == hour)]

if filtered.empty:
    st.warning("No data found for this combination.")
else:
    render_atr_chart(df, trigger, hour)
