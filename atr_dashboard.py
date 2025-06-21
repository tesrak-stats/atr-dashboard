
import streamlit as st
import pandas as pd
from draw_roadmap_chart import draw_roadmap_chart

st.set_page_config(page_title="ATR Dashboard", layout="wide")
st.title("SPX ATR Level Roadmap")
st.caption("Simulated default view with preloaded ATR levels")

# === Simulated daily OHLC and ATR data === #
close = 5200
atr = 40
open_price = 5180

# === ATR Fib Levels and Labels === #
fibs = [1.0, 0.786, 0.618, 0.5, 0.382, 0.236, 0.0, -0.236, -0.382, -0.5, -0.618, -0.786, -1.0]
labels = [f"{f:+.3f}" for f in fibs]
levels = [close + atr * f for f in fibs]
level_annotations = [f"{val:.2f}" for val in levels]

# === Identify trigger zone (optional logic for future) === #
trigger_zone = None
for i in range(len(levels) - 1):
    lower = min(levels[i], levels[i + 1])
    upper = max(levels[i], levels[i + 1])
    if lower <= open_price <= upper:
        trigger_zone = labels[i] if open_price > close else labels[i + 1]
        break

# === Show chart === #
st.subheader("Today's Simulated Chart")
fig = draw_roadmap_chart(levels, labels, level_annotations, open_price=open_price, ticker="SPX")
st.plotly_chart(fig, use_container_width=True)

# === Placeholder completions === #
st.subheader("Simulated Goal Completion Probabilities")
st.write("⚠️ Placeholder values until real data is connected.")

simulated_percentages = {
    "+0.236": "62%",
    "+0.382": "48%",
    "+0.5": "39%",
    "+0.618": "29%",
    "+0.786": "21%",
    "+1.0": "13%"
}
if trigger_zone:
    st.write(f"Assuming upward trigger at zone: **{trigger_zone}**")
st.table(pd.DataFrame.from_dict(simulated_percentages, orient="index", columns=["Completion %"]))

# === Footer === #
st.caption("🔄 Chart will automatically update when live ATR data and probabilities are connected.")
