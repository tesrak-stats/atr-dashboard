
import streamlit as st
import pandas as pd
from render_atr_chart import render_atr_chart

st.set_page_config(page_title="ATR Roadmap", layout="wide")
st.title("ATR Levels Roadmap (Simulated Data)")

# Load the fake data CSV
df = pd.read_csv("fake_atr_chart_data.csv")

# Sidebar filters
st.sidebar.header("🔧 Select Scenario")
trigger_levels = sorted(df["trigger_level"].unique())
hours = sorted(df["trigger_hour"].unique())
directions = sorted(df["direction"].unique())

trigger = st.sidebar.selectbox("Trigger Level", trigger_levels, index=trigger_levels.index(0.0))
hour = st.sidebar.selectbox("Trigger Time", hours)
direction = st.sidebar.radio("Price Approached Trigger From", directions)

# Filter data
filtered = df[
    (df["trigger_level"] == trigger) &
    (df["trigger_hour"] == hour) &
    (df["direction"] == direction)
]

if filtered.empty:
    st.warning("No data found for this combination.")
else:
    st.plotly_chart(render_atr_chart(filtered), use_container_width=True)
