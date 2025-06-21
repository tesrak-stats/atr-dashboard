import streamlit as st
import pandas as pd
import plotly.express as px

# Load the simulated data
df = pd.read_csv("fake_atr_dashboard_data.csv")

st.set_page_config(page_title="ATR Roadmap", layout="wide")
st.title("ATR Levels Roadmap (Simulated Data)")

# Sidebar inputs
st.sidebar.header("🔧 Select Scenario")
trigger_levels = sorted(df["trigger_level"].unique())
hours = sorted(df["trigger_hour"].unique())

trigger = st.sidebar.selectbox("Trigger Level", trigger_levels, index=trigger_levels.index(0.0))
hour = st.sidebar.selectbox("Trigger Time", hours)
direction = st.sidebar.radio("Price Approached Trigger From", ["Below", "Above"])

# Filter based on selection
filtered = df[
    (df["trigger_level"] == trigger) &
    (df["trigger_hour"] == hour)
]

# Display a warning if nothing is found
if filtered.empty:
    st.warning("No data found for this combination.")
else:
    # Format display labels
    filtered["Goal Label"] = filtered["goal_level"].apply(lambda x: f"{x:+.3f}")
    filtered["Time Label"] = filtered["trigger_hour"]

    fig = px.scatter(
        filtered,
        x="Time Label",
        y="Goal Label",
        size="percent_complete",
        color="direction",
        hover_data=["percent_complete", "raw_count"],
        labels={"percent_complete": "% Complete"},
        title=f"Trigger: {trigger:+.3f} at {hour}"
    )

    fig.update_traces(marker=dict(line=dict(width=1, color='DarkSlateGrey')))
    fig.update_layout(height=600)

    # Annotate percentage values
    for i, row in filtered.iterrows():
        fig.add_annotation(
            x=row["Time Label"],
            y=row["Goal Label"],
            text=f"{row['percent_complete']}%",
            showarrow=False,
            yshift=15,
            font=dict(size=11, color="black")
        )

    st.plotly_chart(fig, use_container_width=True)
  
