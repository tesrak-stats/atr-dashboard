import streamlit as st
import pandas as pd
from render_atr_chart import render_atr_chart

st.set_page_config(page_title="ATR Levels Roadmap", layout="wide")
st.title("ATR Levels Roadmap (Simulated Data)")

# Load & prepare fake data
df = pd.read_csv("fake_atr_dashboard_data.csv")
df['hour'] = df['trigger_hour']

# Sidebar controls
trigger = st.sidebar.selectbox("Trigger Level", sorted(df['trigger_level'].unique()))
hour = st.sidebar.selectbox("Trigger Time", sorted(df['trigger_hour'].unique()))
direction = st.sidebar.radio("Price Approached Trigger From", ["Below", "Above"])

filtered = df[(df['trigger_level']==trigger) & (df['trigger_hour']==hour)]

if filtered.empty:
    st.warning("No data found for this combination.")
else:
    fig = render_atr_chart(filtered)
    st.plotly_chart(fig, use_container_width=True)
