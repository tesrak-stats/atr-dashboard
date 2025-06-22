
import streamlit as st
import pandas as pd
import plotly.graph_objects as go

# --- Load and prepare data ---
df = pd.read_csv("combined_trigger_goal_results.csv")
df["TriggerTime"] = df["TriggerTime"].astype(str)
df["GoalTime"] = df["GoalTime"].astype(str)

# --- Sidebar UI ---
st.sidebar.title("ATR Roadmap Matrix")
direction = st.sidebar.selectbox("Select Direction", sorted(df["Direction"].unique()), index=0)
trigger_level = st.sidebar.selectbox("Select Trigger Level", sorted(df["TriggerLevel"].unique()), index=sorted(df["TriggerLevel"].unique()).index(0.0))
time_order = ["OPEN", "0900", "1000", "1100", "1200", "1300", "1400", "1500"]

available_times = df[
    (df["Direction"] == direction) & (df["TriggerLevel"] == trigger_level)
]["TriggerTime"].unique()
trigger_times_sorted = [t for t in time_order if t in available_times]
trigger_time = st.sidebar.selectbox("Select Trigger Time", trigger_times_sorted, index=0)

# --- Filter for selected scenario ---
filtered = df[
    (df["Direction"] == direction) &
    (df["TriggerLevel"] == trigger_level) &
    (df["TriggerTime"] == trigger_time)
].copy()

# --- Aggregate data ---
grouped = (
    filtered.groupby(["GoalLevel", "GoalTime"])
    .agg(
        NumHits=("GoalHit", lambda x: (x == "Yes").sum()),
        NumTriggers=("GoalHit", "count")
    )
    .reset_index()
)
grouped["PctCompletion"] = (grouped["NumHits"] / grouped["NumTriggers"]) * 100

# --- Define goal levels ---
fib_levels = [1.0, 0.786, 0.618, 0.5, 0.382, 0.236, 0.0,
              -0.236, -0.382, -0.5, -0.618, -0.786, -1.0]

# --- Plotly figure setup ---
fig = go.Figure()

# --- Add percent text cells ---
for level in fib_levels:
    for t in time_order:
        match = grouped[
            (grouped["GoalLevel"] == level) & (grouped["GoalTime"] == t)
        ]
        if t < trigger_time:
            continue
        if not match.empty:
            row = match.iloc[0]
            pct = row["PctCompletion"]
            hits = int(row["NumHits"])
            total = int(row["NumTriggers"])
            warn = " ⚠️" if total < 30 else ""
            hover = f"{pct:.1f}% ({hits}/{total}){warn}"
            display_text = f"{pct:.1f}%" if not pd.isna(pct) else ""
        elif t >= trigger_time and level != trigger_level:
            display_text = "0.0%"
            hover = "0.0% (0/0)"
        else:
            continue

        fig.add_trace(go.Scatter(
            x=[t],
            y=[level],
            mode="text",
            text=[display_text],
            hovertext=[hover],
            hoverinfo="text",
            textfont=dict(color="white", size=12),
            showlegend=False
        ))

# --- Draw shaded zones above and below the trigger ---
def get_adjacent_level(levels, value, direction):
    idx = levels.index(value)
    if direction == "up" and idx > 0:
        return levels[idx - 1]
    elif direction == "down" and idx < len(levels) - 1:
        return levels[idx + 1]
    return None

next_up = get_adjacent_level(fib_levels, trigger_level, "up")
next_down = get_adjacent_level(fib_levels, trigger_level, "down")

if next_up is not None:
    fig.add_shape(
        type="rect",
        xref="paper", yref="y",
        x0=0, x1=1,
        y0=min(trigger_level, next_up),
        y1=max(trigger_level, next_up),
        fillcolor="lightgreen",
        opacity=0.25,
        line_width=0,
        layer="below"
    )

if next_down is not None:
    fig.add_shape(
        type="rect",
        xref="paper", yref="y",
        x0=0, x1=1,
        y0=min(trigger_level, next_down),
        y1=max(trigger_level, next_down),
        fillcolor="lightyellow",
        opacity=0.25,
        line_width=0,
        layer="below"
    )

# --- Fib line styles ---
fibo_styles = {
    1.0: ("white", 2),
    0.786: ("white", 1),
    0.618: ("white", 2),
    0.5: ("white", 1),
    0.382: ("white", 1),
    0.236: ("cyan", 2),
    0.0: ("white", 1),
    -0.236: ("yellow", 2),
    -0.382: ("white", 1),
    -0.5: ("white", 1),
    -0.618: ("white", 2),
    -0.786: ("white", 1),
    -1.0: ("white", 2),
}

for level, (color, width) in fibo_styles.items():
    fig.add_shape(
        type="line",
        x0=0, x1=1, xref="paper",
        y0=level, y1=level, yref="y",
        line=dict(color=color, width=width),
        layer="below"
    )

# Vertical hour lines
for hour in time_order:
    fig.add_shape(
        type="line",
        xref="x", yref="paper",
        x0=hour, x1=hour,
        y0=0, y1=1,
        line=dict(color="gray", width=1, dash="dot"),
        layer="below"
    )

# --- Layout ---
fig.update_layout(
    title=f"{direction} | Trigger {trigger_level} at {trigger_time}",
    xaxis=dict(
        title="Hour Goal Was Reached",
        categoryorder="array",
        categoryarray=time_order,
        tickmode="array",
        tickvals=time_order,
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
    margin=dict(l=40, r=40, t=60, b=40)
)

# --- Display ---
st.plotly_chart(fig, use_container_width=True)
