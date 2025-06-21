import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

st.set_page_config(page_title="ATR Dashboard", layout="wide")

# Load data (you’ll need to update this with your actual CSV path or method)
@st.cache_data
def load_data():
    try:
        df = pd.read_csv("goal_matrix_percent.csv", index_col=0)
        df_raw = pd.read_csv("goal_matrix_raw.csv", index_col=0)
        return df, df_raw
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None, None

df, df_raw = load_data()

st.title("SPX ATR Level Roadmap")
st.caption("Data: 10 years of 10-minute SPX candles — Powered by tesrak")

# User inputs
direction = st.selectbox("Trigger Direction", ["Upside", "Downside"])
trigger_level = st.selectbox("Trigger Level", df.columns if df is not None else [])
trigger_hour = st.selectbox("Trigger Hour", df.index if df is not None else [])

# Validate
if df is not None and trigger_hour in df.index and trigger_level in df.columns:
    st.subheader(f"Triggered {trigger_level} at {trigger_hour} ({direction})")

    percent_value = df.loc[trigger_hour, trigger_level]
    raw_value = df_raw.loc[trigger_hour, trigger_level]

    # Visual
    st.metric(label="Goal Completion Rate", value=f"{percent_value:.1f}%", delta=f"{raw_value} Samples")

    # Heatmap or visual matrix
    st.subheader("Goal Completion Matrix")
    fig = px.imshow(df.astype(float), text_auto=True, aspect="auto", color_continuous_scale="Blues",
                    labels={"x": "Goal Level", "y": "Hour", "color": "Completion %"})
    st.plotly_chart(fig, use_container_width=True)

    st.caption("⚠️ Lighter samples (<30) may have reduced statistical confidence.")
else:
    st.warning("Please select a valid trigger level and hour to view results.")
