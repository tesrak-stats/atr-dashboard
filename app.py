
import streamlit as st
import pandas as pd
import plotly.graph_objects as go

@st.cache_data
def load_data():
    df = pd.read_csv("combined_trigger_goal_results.csv")
    df["TriggerTime"] = df["TriggerTime"].astype(str)
    df["GoalTime"] = df["GoalTime"].astype(str)
    return df

df = load_data()

st.title("ATR Roadmap Dashboard")
direction = st.radio("Direction", sorted(df["Direction"].unique()), index=1, horizontal=True)
trigger_level = st.selectbox("Trigger Level", sorted(df["TriggerLevel"].unique()), index=sorted(df["TriggerLevel"].unique()).index(0.0))

# Real hour labels
real_times = ["OPEN", "0900", "1000", "1100", "1200", "1300", "1400", "1500"]
# Interleaved for spacing
category_order = []
for i, t in enumerate(real_times):
    category_order.append(t)
    if i < len(real_times) - 1:
        category_order.append(f"{t}_gap")

# Tick labels only for real times
tickvals = [t for t in category_order if "_gap" not in t]
ticktext = [t for t in real_times]

# Trigger time selector
trigger_times_sorted = [t for t in real_times if t in df["TriggerTime"].unique()]
trigger_time = st.selectbox("Trigger Time", trigger_times_sorted, index=0)

# Filter
filtered = df[
    (df["Direction"] == direction) &
    (df["TriggerLevel"] == trigger_level) &
    (df["TriggerTime"] == trigger_time)
].copy()

grouped = filtered.groupby(["GoalLevel", "GoalTime"]).agg(
    NumHits=("GoalHit", lambda x: (x == "Yes").sum()),
    NumTriggers=("GoalHit", "count")
).reset_index()
grouped["PctCompletion"] = (grouped["NumHits"] / grouped["NumTriggers"]) * 100

fib_levels = [1.0, 0.786, 0.618, 0.5, 0.382, 0.236, 0.0,
              -0.236, -0.382, -0.5, -0.618, -0.786, -1.0]

fig = go.Figure()

# Phantom anchor points
fig.add_trace(go.Scatter(
    x=category_order,
    y=[fib_levels[0]] * len(category_order),
    mode="markers",
    marker=dict(color="rgba(0,0,0,0)"),
    hoverinfo="skip",
    showlegend=False
))

# Plot real values only
for level in fib_levels:
    for t in real_times:
        if real_times.index(t) < real_times.index(trigger_time):
            continue

        match = grouped[(grouped["GoalLevel"] == level) & (grouped["GoalTime"] == t)]
        if not match.empty:
            row = match.iloc[0]
            pct = row["PctCompletion"]
            hits = int(row["NumHits"])
            total = int(row["NumTriggers"])
            warn = " ⚠️" if total < 30 else ""
            text = f"{pct:.1f}%"
            hover = f"{pct:.1f}% ({hits}/{total}){warn}"
        elif level != trigger_level:
            text = "0.0%"
            hover = "0.0% (0/0)"
        else:
            continue

        fig.add_trace(go.Scatter(
            x=[t],
            y=[level + 0.02],
            mode="text",
            text=[text],
            hovertext=[hover],
            hoverinfo="text",
            textfont=dict(color="white", size=14),
            showlegend=False
        ))

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
                  fillcolor="rgba(0,255,0,0.25)", line_width=0, layer="below")
if lower:
    fig.add_shape(type="rect", x0=0, x1=1, xref="paper", y0=trigger_level, y1=lower, yref="y",
                  fillcolor="rgba(255,255,0,0.25)", line_width=0, layer="below")

fibo_styles = {
    1.0: ("white", 2), 0.786: ("white", 1), 0.618: ("white", 2), 0.5: ("white", 1),
    0.382: ("white", 1), 0.236: ("cyan", 2), 0.0: ("white", 1),
    -0.236: ("yellow", 2), -0.382: ("white", 1), -0.5: ("white", 1),
    -0.618: ("white", 2), -0.786: ("white", 1), -1.0: ("white", 2)
}

for level, (color, width) in fibo_styles.items():
    fig.add_shape(type="line", xref="paper", x0=0, x1=1,
                  yref="y", y0=level, y1=level,
                  line=dict(color=color, width=width),
                  layer="below")

# Vertical grid for real hours
for t in real_times:
    fig.add_shape(type="line", x0=t, x1=t, xref="x",
                  y0=min(fib_levels), y1=max(fib_levels), yref="y",
                  line=dict(color="gray", width=1, dash="dot"), layer="below")

fig.update_layout(
    title=f"{direction} | Trigger {trigger_level} at {trigger_time}",
    xaxis=dict(
        title="Hour Goal Was Reached",
        categoryorder="array",
        categoryarray=category_order,
        tickmode="array",
        tickvals=tickvals,
        ticktext=ticktext,
        tickangle=0,
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
    width=3000,
    margin=dict(l=60, r=60, t=60, b=60)
)

st.plotly_chart(fig, use_container_width=False)
