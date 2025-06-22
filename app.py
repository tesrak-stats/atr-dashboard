
import streamlit as st
import pandas as pd
import plotly.graph_objects as go

# --- Load and prepare data ---
df = pd.read_csv("combined_trigger_goal_results.csv")
df["TriggerTime"] = df["TriggerTime"].astype(str)
df["GoalTime"] = df["GoalTime"].astype(str)

# --- Sidebar UI ---
st.sidebar.title("ATR Roadmap Matrix")
direction = st.sidebar.selectbox("Select Direction", sorted(df["Direction"].unique()), index=1)  # Default: Upside
trigger_level = st.sidebar.selectbox("Select Trigger Level", sorted(df["TriggerLevel"].unique()), index=6)  # Default: 0
time_order = ["OPEN", "0900", "1000", "1100", "1200", "1300", "1400", "1500"]
trigger_times_sorted = [t for t in time_order if t in df["TriggerTime"].unique()]
trigger_time = st.sidebar.selectbox("Select Trigger Time", trigger_times_sorted, index=0)  # Default: OPEN

# --- Filter for selected scenario ---
filtered = df[
    (df["Direction"] == direction) &
    (df["TriggerLevel"] == trigger_level) &
    (df["TriggerTime"] == trigger_time)
].copy()

# --- Aggregate data ---
grouped = (
    filtered.groupby(["GoalLevel", "GoalTime"])
    .agg(NumHits=("GoalHit", lambda x: (x == "Yes").sum()), NumTriggers=("GoalHit", "count"))
    .reset_index()
)
grouped["PctCompletion"] = (grouped["NumHits"] / grouped["NumTriggers"]) * 100

# --- Fixed fib levels (top to bottom) ---
fib_levels = [1.0, 0.786, 0.618, 0.5, 0.382, 0.236, 0.0,
              -0.236, -0.382, -0.5, -0.618, -0.786, -1.0]

fig = go.Figure()

# --- Add percentage text cells ---
for level in fib_levels:
    for t in time_order:
        if t > trigger_time:
            continue  # leave blank if before trigger
        match = grouped[(grouped["GoalLevel"] == level) & (grouped["GoalTime"] == t)]
        if not match.empty:
            row = match.iloc[0]
            pct = row["PctCompletion"]
            hits = int(row["NumHits"]) if pd.notna(row["NumHits"]) else 0
            total = int(row["NumTriggers"]) if pd.notna(row["NumTriggers"]) else 0
            warn = " ⚠️" if total < 30 else ""
            hover = f"{pct:.1f}% ({hits}/{total}){warn}"
            fig.add_trace(go.Scatter(
                x=[t],
                y=[level],
                mode="text",
                text=[f"{pct:.1f}%" if total > 0 or level == 0 else ""],
                hovertext=[hover],
                hoverinfo="text",
                textfont=dict(color="white", size=12),
                showlegend=False
            ))

# --- Horizontal guide lines ---
fibo_styles = {
    1.0: ("white", 2), 0.786: ("white", 1), 0.618: ("white", 2),
    0.5: ("white", 1), 0.382: ("white", 1), 0.236: ("cyan", 2),
    0.0: ("white", 1), -0.236: ("yellow", 2), -0.382: ("white", 1),
    -0.5: ("white", 1), -0.618: ("white", 2), -0.786: ("white", 1),
    -1.0: ("white", 2),
}

for level, (color, width) in fibo_styles.items():
    fig.add_shape(
        type="line", x0=0, x1=1, xref="paper", y0=level, y1=level, yref="y",
        line=dict(color=color, width=width), layer="below"
    )

# --- Shading between current level and next up/down ---
def get_next_level(levels, current, direction):
    idx = levels.index(current)
    if direction == "up" and idx > 0:
        return levels[idx - 1]
    if direction == "down" and idx < len(levels) - 1:
        return levels[idx + 1]
    return None

upper = get_next_level(fib_levels, trigger_level, "up")
lower = get_next_level(fib_levels, trigger_level, "down")

if upper:
    fig.add_shape(type="rect", xref="paper", yref="y", layer="below",
                  x0=0, x1=1, y0=trigger_level, y1=upper,
                  fillcolor="rgba(0,255,0,0.2)", line=dict(width=0))
if lower:
    fig.add_shape(type="rect", xref="paper", yref="y", layer="below",
                  x0=0, x1=1, y0=lower, y1=trigger_level,
                  fillcolor="rgba(255,255,0,0.2)", line=dict(width=0))

# --- Vertical hour lines ---
for t in time_order:
    fig.add_shape(type="line", xref="x", yref="paper", layer="below",
                  x0=t, x1=t, y0=0, y1=1,
                  line=dict(color="gray", width=1, dash="dot"))

# --- Layout ---
fig.update_layout(
    title=f"{direction} | Trigger {trigger_level} at {trigger_time}",
    xaxis=dict(
        title="Hour Goal Was Reached",
        categoryorder="array",
        categoryarray=time_order,
        tickmode="array",
        tickvals=time_order,
        tickangle=45,  # <-- rotate labels
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
    width=1000,
    margin=dict(l=40, r=40, t=60, b=60)
)

st.plotly_chart(fig, use_container_width=False)
