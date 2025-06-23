
import streamlit as st
import pandas as pd
import plotly.graph_objects as go

# --- Load and prepare summary data ---
df = pd.read_csv("atr_dashboard_summary.csv")
df["TriggerTime"] = df["TriggerTime"].astype(str)
df["GoalTime"] = df["GoalTime"].astype(str)

# --- Sidebar UI ---
st.title("📈 ATR Roadmap Dashboard")
col1, col2, col3 = st.columns(3)

direction = col1.selectbox("Direction", sorted(df["Direction"].unique()), index=0)
trigger_level = col2.selectbox("Trigger Level", sorted(df["TriggerLevel"].unique()))
available_times = df[(df["Direction"] == direction) & (df["TriggerLevel"] == trigger_level)]["TriggerTime"].unique()

time_order = ["OPEN", "0900", "1000", "1100", "1200", "1300", "1400", "1500", "1600"]
trigger_times_sorted = [t for t in time_order if t in available_times]
trigger_time = col3.selectbox("Trigger Time", trigger_times_sorted if trigger_times_sorted else available_times)

# --- Filter for selected scenario ---
filtered = df[
    (df["Direction"] == direction) &
    (df["TriggerLevel"] == trigger_level) &
    (df["TriggerTime"] == trigger_time)
].copy()

# --- Aggregate format ---
fib_levels = [1.0, 0.786, 0.618, 0.5, 0.382, 0.236, 0.0,
              -0.236, -0.382, -0.5, -0.618, -0.786, -1.0]

padded_times = []
for i, t in enumerate(time_order):
    padded_times.append(t)
    if i < len(time_order) - 1:
        padded_times.append(f"{t}_pad")

# --- Create Plotly heatmap-like chart ---
fig = go.Figure()

# Add hidden anchor trace for layout control
fig.add_trace(go.Scatter(
    x=padded_times,
    y=[fib_levels[0]] * len(padded_times),
    mode="markers",
    marker=dict(color="rgba(0,0,0,0)"),
    hoverinfo="skip",
    showlegend=False
))

# Add goal completion percentages as text
for level in fib_levels:
    for t in padded_times:
        if "_pad" in t or (real_index := time_order.index(t) if t in time_order else -1) == -1:
            continue
        if real_index < time_order.index(trigger_time):
            continue
        match = filtered[(filtered["GoalLevel"] == level) & (filtered["GoalTime"] == t)]
        if not match.empty:
            row = match.iloc[0]
            pct = row["PctCompletion"]
            hits = row["NumHits"]
            total = row["NumTriggers"]
            warn = " ⚠️" if total < 30 else ""
            text = f"{pct:.1f}%"
            hover = f"{pct:.1f}% ({hits}/{total}){warn}"
            fig.add_trace(go.Scatter(
                x=[t], y=[level + 0.02],
                mode="text", text=[text],
                hovertext=[hover],
                hoverinfo="text",
                textfont=dict(color="white", size=12),
                showlegend=False
            ))
        elif level != trigger_level:
            fig.add_trace(go.Scatter(
                x=[t], y=[level + 0.02],
                mode="text", text=["0.0%"],
                hovertext=["0.0% (0/0)"],
                hoverinfo="text",
                textfont=dict(color="gray", size=11),
                showlegend=False
            ))

# Shading direction to next level
def next_level(levels, current, direction):
    idx = levels.index(current)
    if direction == "up" and idx > 0:
        return levels[idx - 1]
    elif direction == "down" and idx < len(levels) - 1:
        return levels[idx + 1]
    return None

upper = next_level(fib_levels, trigger_level, "up")
lower = next_level(fib_levels, trigger_level, "down")
if upper:
    fig.add_shape(type="rect", x0=0, x1=1, xref="paper", y0=trigger_level, y1=upper, yref="y",
                  fillcolor="rgba(0,255,0,0.2)", line_width=0, layer="below")
if lower:
    fig.add_shape(type="rect", x0=0, x1=1, xref="paper", y0=trigger_level, y1=lower, yref="y",
                  fillcolor="rgba(255,255,0,0.2)", line_width=0, layer="below")

# Add horizontal fib level lines
fibo_styles = {
    1.0: ("white", 2), 0.786: ("white", 1), 0.618: ("white", 2), 0.5: ("white", 1),
    0.382: ("white", 1), 0.236: ("cyan", 2), 0.0: ("white", 1),
    -0.236: ("yellow", 2), -0.382: ("white", 1), -0.5: ("white", 1),
    -0.618: ("white", 2), -0.786: ("white", 1), -1.0: ("white", 2)
}
for level, (color, width) in fibo_styles.items():
    fig.add_shape(type="line", xref="paper", x0=0, x1=1, yref="y", y0=level, y1=level,
                  line=dict(color=color, width=width), layer="below")

# Add vertical grid lines
for t in time_order:
    fig.add_shape(type="line", x0=t, x1=t, xref="x", y0=min(fib_levels), y1=max(fib_levels), yref="y",
                  line=dict(color="gray", width=1, dash="dot"), layer="below")

fig.update_layout(
    title=f"{direction} | Trigger {trigger_level} at {trigger_time}",
    xaxis=dict(
        title="Hour Goal Was Reached",
        categoryorder="array",
        categoryarray=padded_times,
        tickmode="array",
        tickvals=time_order,
        ticktext=time_order,
        tickfont=dict(color="white")
    ),
    yaxis=dict(
        title="Goal Level",
        categoryorder="array",
        categoryarray=fib_levels,
        tickmode="array",
        tickvals=fib_levels,
        tickfont=dict(color="white")
    ),
    plot_bgcolor="black",
    paper_bgcolor="black",
    font=dict(color="white"),
    height=720,
    width=2800,
    margin=dict(l=60, r=60, t=60, b=60)
)

st.plotly_chart(fig, use_container_width=False)
