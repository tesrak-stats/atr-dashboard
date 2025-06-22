
import streamlit as st
import pandas as pd
import plotly.graph_objects as go

# Load data
df = pd.read_csv("combined_trigger_goal_results.csv")
df["TriggerTime"] = df["TriggerTime"].astype(str)
df["GoalTime"] = df["GoalTime"].astype(str)

# UI controls on top
st.title("ATR Roadmap Dashboard")
col1, col2, col3 = st.columns(3)
with col1:
    direction = st.radio("Direction", sorted(df["Direction"].unique()), index=1, horizontal=True)
with col2:
    trigger_level = st.selectbox("Trigger Level", sorted(df["TriggerLevel"].unique()), index=sorted(df["TriggerLevel"].unique()).index(0.0))
with col3:
    time_order = ["OPEN", "0900", "1000", "1100", "1200", "1300", "1400", "1500", "1600"]
    available_times = df[(df["Direction"] == direction) & (df["TriggerLevel"] == trigger_level)]["TriggerTime"].unique()
    trigger_times_sorted = [t for t in time_order if t in available_times]
    trigger_time = st.selectbox("Trigger Time", trigger_times_sorted, index=0)

# Filter
filtered = df[(df["Direction"] == direction) & (df["TriggerLevel"] == trigger_level) & (df["TriggerTime"] == trigger_time)].copy()

# Aggregate
grouped = filtered.groupby(["GoalLevel", "GoalTime"]).agg(
    NumHits=("GoalHit", lambda x: (x == "Yes").sum()),
    NumTriggers=("GoalHit", "count")
).reset_index()
grouped["PctCompletion"] = (grouped["NumHits"] / grouped["NumTriggers"]) * 100

# Count OPEN trigger failures
open_fails = df[
    (df["Direction"] == direction) &
    (df["TriggerLevel"] == trigger_level) &
    (df["TriggerTime"] == "OPEN") &
    (df["GoalHit"] == "No")
].shape[0]

# Time padding logic
visible_times = ["OPEN", "0900", "1000", "1100", "1200", "1300", "1400", "1500", "1600"]
padded_times = []
for t in visible_times:
    padded_times.append(t)
    padded_times.append(f"{t}_pad")
padded_times = padded_times[:-1]

# Fib levels
fib_levels = [1.0, 0.786, 0.618, 0.5, 0.382, 0.236, 0.0,
              -0.236, -0.382, -0.5, -0.618, -0.786, -1.0]

fig = go.Figure()

# Add percentages and OPEN tooltip
for level in fib_levels:
    for t in padded_times:
        if "_pad" in t:
            continue
        if t == "OPEN":
            if level != trigger_level:
                fig.add_trace(go.Scatter(
                    x=[t],
                    y=[level + 0.02],
                    mode="text",
                    text=[""],
                    hovertext=[f"🕒 OPEN triggers that failed: {open_fails}"],
                    hoverinfo="text",
                    showlegend=False
                ))
            continue
        if visible_times.index(t) < visible_times.index(trigger_time):
            continue
        match = grouped[(grouped["GoalLevel"] == level) & (grouped["GoalTime"] == t)]
        if not match.empty:
            row = match.iloc[0]
            pct = row["PctCompletion"]
            hits = row["NumHits"]
            total = row["NumTriggers"]
            warn = " ⚠️" if total < 30 else ""
            hover = f"{pct:.1f}% ({hits}/{total}){warn}"
            fig.add_trace(go.Scatter(
                x=[t],
                y=[level + 0.02],
                mode="text",
                text=[f"{pct:.1f}%"],
                hovertext=[hover],
                hoverinfo="text",
                textfont=dict(color="white", size=12),
                showlegend=False
            ))
        elif level != trigger_level:
            fig.add_trace(go.Scatter(
                x=[t],
                y=[level + 0.02],
                mode="text",
                text=["0.0%"],
                hovertext=["0.0% (0/0)"],
                hoverinfo="text",
                textfont=dict(color="white", size=12),
                showlegend=False
            ))

# Add fib lines
fibo_styles = {
    1.0: ("white", 2), 0.786: ("white", 1), 0.618: ("white", 2),
    0.5: ("white", 1), 0.382: ("white", 1), 0.236: ("cyan", 2),
    0.0: ("white", 1), -0.236: ("yellow", 2), -0.382: ("white", 1),
    -0.5: ("white", 1), -0.618: ("white", 2), -0.786: ("white", 1), -1.0: ("white", 2)
}
for level, (color, width) in fibo_styles.items():
    fig.add_shape(type="line", x0=0, x1=1, xref="paper", y0=level, y1=level, yref="y",
                  line=dict(color=color, width=width), layer="below")

# Add vertical gridlines
for t in visible_times:
    fig.add_vline(x=t, line=dict(color="gray", width=1, dash="dot"))

# Final layout
fig.update_layout(
    title=f"{direction} | Trigger {trigger_level} at {trigger_time}",
    xaxis=dict(
        title="Hour Goal Was Reached",
        categoryorder="array",
        categoryarray=padded_times,
        tickmode="array",
        tickvals=visible_times,
        ticktext=visible_times,
        tickfont=dict(color="white", size=12)
    ),
    yaxis=dict(
        title="Goal Level",
        categoryorder="array",
        categoryarray=fib_levels,
        tickmode="array",
        tickvals=fib_levels,
        tickfont=dict(color="white", size=12)
    ),
    plot_bgcolor="black",
    paper_bgcolor="black",
    font=dict(color="white"),
    height=720,
    width=3200,
    margin=dict(l=40, r=40, t=60, b=40)
)

# Show chart
st.plotly_chart(fig, use_container_width=False)
