import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

st.set_page_config(page_title="ATR Dashboard", layout="wide")
st.title("SPX ATR Level Roadmap")
st.caption("Live default view using simulated SPX levels")

# === SIMULATED DATA === #
# Normally this would be loaded from real daily OHLC + ATR data
close = 5200
atr = 40
open_price = 5180

fibs = [-1.0, -0.786, -0.618, -0.5, -0.382, -0.236, 0.0, 0.236, 0.382, 0.5, 0.618, 0.786, 1.0]
levels = [close + atr * f for f in fibs]
labels = [f"{f:+.3f}" for f in fibs]

# === DETERMINE OPEN ZONE === #
trigger_zone = None
for i in range(len(levels) - 1):
    lower = min(levels[i], levels[i + 1])
    upper = max(levels[i], levels[i + 1])
    if lower <= open_price <= upper:
        trigger_zone = labels[i] if open_price > close else labels[i + 1]
        break

st.subheader("Today's Setup (Simulated)")

col1, col2 = st.columns(2)
with col1:
    st.metric("Previous Close", f"{close:.2f}")
    st.metric("ATR", f"{atr:.2f}")
    st.metric("Open Price", f"{open_price:.2f}")

with col2:
    if trigger_zone:
        st.success(f"Trigger Zone Detected: **{trigger_zone}**")
    else:
        st.warning("Open price did not fall within a defined ATR zone")

# === SIMULATED COMPLETION OUTPUT === #
st.subheader("Simulated Goal Completion Probabilities")
st.write("⚠️ Placeholder values until real data is loaded.")

simulated_percentages = {
    "+0.236": "62%",
    "+0.382": "48%",
    "+0.5": "39%",
    "+0.618": "29%",
    "+0.786": "21%",
    "+1.0": "13%"
}

st.write("Assuming upward trigger at zone:", trigger_zone)
st.table(pd.DataFrame.from_dict(simulated_percentages, orient="index", columns=["Completion %"]))

# === ATR LEVEL CHART === #
st.subheader("ATR Levels Chart (Simulated)")
fig = go.Figure()

for level, label in zip(levels, labels):
    color = "cyan" if ".236" in label else "white" if label == "0.000" else "gray"
    width = 3 if label in ["+0.618", "-0.618"] else 1
    fig.add_shape(type="line", x0=0, x1=1, y0=level, y1=level,
                  line=dict(color=color, width=width), xref="paper", yref="y")

fig.add_trace(go.Scatter(x=[0.5], y=[open_price], mode="markers+text",
                         marker=dict(color="red", size=10),
                         text=["Open"], textposition="bottom center"))

fig.update_layout(height=400, yaxis_title="Price", plot_bgcolor="#111", paper_bgcolor="#111",
                  font_color="white", margin=dict(l=50, r=50, t=30, b=30))
st.plotly_chart(fig, use_container_width=True)

# === REMINDER === #
st.info("🔄 This view will update automatically once real data is connected.")
