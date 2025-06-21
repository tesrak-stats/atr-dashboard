
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import json
from pathlib import Path

st.set_page_config(layout="wide")
st.title("ATR Levels Roadmap")

# Load precalculated levels (fake values for now)
levels_path = Path("data/daily_atr_levels.json")
levels = {
    "+1.0": 5340, "+0.786": 5300, "+0.618": 5280, "+0.5": 5260, "+0.382": 5240, "+0.236": 5220,
    "0.0": 5200, "-0.236": 5180, "-0.382": 5160, "-0.5": 5140, "-0.618": 5120, "-0.786": 5100, "-1.0": 5080,
    "open": 5235
}
if levels_path.exists():
    with open(levels_path, "r") as f:
        levels = json.load(f)

# Load fake CSVs for standard and retracement
try:
    standard_df = pd.read_csv("standard_results.csv", index_col=0)
    retrace_df = pd.read_csv("retracement_results.csv", index_col=0)
except Exception:
    st.warning("Fake data not found. Please upload standard_results.csv and retracement_results.csv.")
    st.stop()

fib_levels = list(standard_df.index)
hours = list(standard_df.columns)

# --- UI ---
st.sidebar.header("Trigger Setup")
trigger_hour = st.sidebar.selectbox("Trigger Hour", hours, index=0)
trigger_level = st.sidebar.selectbox("Trigger Level", fib_levels, index=fib_levels.index("0.0"))
trigger_direction = st.sidebar.radio("Trigger Direction", ["Upward Move", "Downward Move"])

# --- Plot Setup ---
fig = go.Figure()

# Add ATR lines
for lvl in fib_levels:
    y = float(lvl)
    price = levels.get(lvl)
    color = "white"
    width = 1
    if lvl == "+0.236":
        color = "cyan"
    elif lvl == "-0.236":
        color = "yellow"
    if abs(y) == 0.618:
        width = 2
    elif abs(y) == 1.0:
        width = 3
    fig.add_shape(type="line", x0=-0.5, x1=8.5, y0=y, y1=y, line=dict(color=color, width=width), layer="below")
    if price:
        fig.add_trace(go.Scatter(
            x=[9], y=[y],
            mode="text",
            text=[f"{price:.0f}"],
            textposition="middle left",
            textfont=dict(color="gray", size=12),
            hoverinfo="skip",
            showlegend=False
        ))

# Add open price dot
open_price = levels.get("open")
if open_price:
    closest_lvl = min(fib_levels, key=lambda lvl: abs(levels.get(lvl, 0) - open_price))
    fig.add_trace(go.Scatter(
        x=[0], y=[float(closest_lvl)],
        mode="markers+text",
        marker=dict(size=16, color="cyan", line=dict(color="white", width=1.5)),
        text=[f"Open: {open_price:.2f}"],
        textposition="top center",
        hovertemplate="Open: " + f"{open_price:.2f}" + "<extra></extra>",
        name="Open"
    ))

# Plot bubbles for standard and retrace
for df, color, label in [(standard_df, "lime", "Standard"), (retrace_df, "orange", "Retracement")]:
    for j, hour in enumerate(hours):
        for i, lvl in enumerate(fib_levels):
            val = df.at[lvl, hour]
            if pd.notna(val) and float(val) > 0:
                fig.add_trace(go.Scatter(
                    x=[j], y=[float(lvl)],
                    mode="markers+text",
                    marker=dict(size=10 + float(val), color=color, opacity=0.7, line=dict(width=1, color="black")),
                    text=[f"{val:.0f}%"],
                    textposition="top center",
                    hovertemplate=f"{label}<br>Level: {lvl}<br>Hour: {hour}<br>{val:.0f}%<extra></extra>",
                    showlegend=False
                ))

# Final layout
fig.update_layout(
    plot_bgcolor="#111",
    paper_bgcolor="#111",
    font=dict(color="white"),
    xaxis=dict(tickmode="array", tickvals=list(range(len(hours))), ticktext=hours, title="Hour"),
    yaxis=dict(tickvals=[float(lvl) for lvl in fib_levels], ticktext=fib_levels, title="ATR Level"),
    title="ATR Levels Roadmap",
    margin=dict(t=50, b=40, l=60, r=100)
)

st.plotly_chart(fig, use_container_width=True)
