
import streamlit as st
import pandas as pd
import plotly.graph_objects as go

st.set_page_config(page_title="ATR Levels Roadmap", layout="wide")
st.title("ATR Levels Roadmap (Simulated Data)")

# Load data
df = pd.read_csv("fake_atr_dashboard_data.csv")

# Sidebar inputs
st.sidebar.header("🔧 Select Scenario")
trigger_levels = sorted(df["trigger_level"].unique())
hours = sorted(df["trigger_hour"].unique(), key=lambda x: ["OPEN", "0900", "1000", "1100", "1200", "1300", "1400", "1500", "1600"].index(x) if x in ["OPEN", "0900", "1000", "1100", "1200", "1300", "1400", "1500", "1600"] else 99)
directions = ["Below", "Above"]

trigger = st.sidebar.selectbox("Trigger Level", trigger_levels, index=trigger_levels.index(0.0))
hour = st.sidebar.selectbox("Trigger Time", hours)
direction = st.sidebar.radio("Price Approached Trigger From", directions)

# Filter for scenario
valid_hours = ["OPEN", "0900", "1000", "1100", "1200", "1300", "1400", "1500", "1600", "TOTAL"]
filtered = df[
    (df["trigger_level"] == trigger) &
    (df["trigger_hour"] == hour) &
    (df["direction"] == direction) &
    (df["goal_hour"].isin(valid_hours))
]

# Format labels
filtered["Goal Label"] = filtered["goal_level"].apply(lambda x: f"{x:+.3f}")
filtered["Time Label"] = pd.Categorical(filtered["goal_hour"], categories=valid_hours, ordered=True)
filtered = filtered.sort_values("Time Label")

# Build chart
fig = go.Figure()

for _, row in filtered.iterrows():
    fig.add_trace(go.Scatter(
        x=[row["Time Label"]],
        y=[row["Goal Label"]],
        mode='markers+text',
        marker=dict(
            size=row["percent_complete"] / 1.2,
            color='cyan',
            line=dict(width=1, color='darkslategray')
        ),
        text=[f"{row['percent_complete']}%"],
        textposition="top center",
        hovertemplate=(
            f"Trigger: {trigger:+.3f}<br>"
            f"Goal Level: {row['goal_level']}<br>"
            f"Goal Hour: {row['goal_hour']}<br>"
            f"Completion: {row['percent_complete']}%<br>"
            f"Triggers: {row['raw_count']}"
        )
    ))

# Add ATR level reference lines
atr_levels = sorted(filtered["Goal Label"].unique(), reverse=True)
for lvl in atr_levels:
    color = "white"
    width = 1
    if lvl in ["+0.618", "-0.618"]: width = 2
    if lvl in ["+1.000", "-1.000"]: width = 3
    if lvl == "-0.236": color = "yellow"

    fig.add_hline(y=lvl, line=dict(color=color, width=width))

# Final layout
fig.update_layout(
    height=650,
    xaxis=dict(title="Hour of Day", tickmode="array", tickvals=valid_hours),
    yaxis=dict(title="ATR Level", autorange="reversed"),
    plot_bgcolor="#111111",
    paper_bgcolor="#111111",
    font=dict(color="white"),
    margin=dict(t=60, l=60, r=60, b=60)
)

st.plotly_chart(fig, use_container_width=True)
