
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import itertools

# --- Load and prepare data ---
df = pd.read_csv("combined_trigger_goal_results.csv")
df["TriggerTime"] = df["TriggerTime"].astype(str)
df["GoalTime"] = df["GoalTime"].astype(str)

# --- Sidebar UI ---
st.sidebar.title("ATR Roadmap Matrix")
direction = st.sidebar.selectbox("Select Direction", sorted(df["Direction"].unique()), index=0)
trigger_level = st.sidebar.selectbox("Select Trigger Level", sorted(df["TriggerLevel"].unique()), index=sorted(df["TriggerLevel"].unique()).index(0.0))
time_order = ["OPEN", "0900", "1000", "1100", "1200", "1300", "1400", "1500"]

# Sort available trigger times for the selected scenario
available_times = df[
    (df["Direction"] == direction) & (df["TriggerLevel"] == trigger_level)
]["TriggerTime"].unique()
trigger_times_sorted = [t for t in time_order if t in available_times]
trigger_time = st.sidebar.selectbox("Select Trigger Time", trigger_times_sorted, index=trigger_times_sorted.index("OPEN") if "OPEN" in trigger_times_sorted else 0)

# --- Filter for selected scenario ---
filtered = df[
    (df["Direction"] == direction) &
    (df["TriggerLevel"] == trigger_level) &
    (df["TriggerTime"] == trigger_time)
].copy()

# --- Grouping logic to preserve misses ---
# Count total triggers per GoalLevel (regardless of goal time)
attempts = (
    filtered.groupby("GoalLevel")
    .size()
    .reset_index(name="NumTriggers")
)

# Count hits per GoalLevel + GoalTime
hits = (
    filtered[filtered["GoalHit"] == "Yes"]
    .groupby(["GoalLevel", "GoalTime"])
    .size()
    .reset_index(name="NumHits")
)

# Full grid: every GoalLevel x GoalTime
fib_levels = [1.0, 0.786, 0.618, 0.5, 0.382, 0.236, 0.0,
              -0.236, -0.382, -0.5, -0.618, -0.786, -1.0]

all_combos = pd.DataFrame(
    list(itertools.product(fib_levels, time_order)),
    columns=["GoalLevel", "GoalTime"]
)

# Merge hits and trigger attempts
grouped = all_combos.merge(hits, on=["GoalLevel", "GoalTime"], how="left")
grouped = grouped.merge(attempts, on="GoalLevel", how="left")

# Define time index to blank earlier times
time_index = {t: i for i, t in enumerate(time_order)}
trigger_index = time_index[trigger_time]

# Compute percent completions — show 0.0% where appropriate, blank earlier + trigger level
def compute_pct(row):
    if row["GoalLevel"] == trigger_level:
        return None
    if time_index[row["GoalTime"]] < trigger_index:
        return None
    if pd.isna(row["NumTriggers"]) or row["NumTriggers"] == 0:
        return 0.0
    return (row["NumHits"] or 0) / row["NumTriggers"] * 100

grouped["PctCompletion"] = grouped.apply(compute_pct, axis=1)

# --- Plotly figure setup ---
fig = go.Figure()

for _, row in grouped.iterrows():
    level = row["GoalLevel"]
    time_label = row["GoalTime"]
    pct = row["PctCompletion"]
    hits = int(row["NumHits"]) if pd.notna(row["NumHits"]) else 0
    total = int(row["NumTriggers"]) if pd.notna(row["NumTriggers"]) else 0
    warn = " ⚠️" if total < 30 else ""
    
    if pd.notna(pct):
        fig.add_trace(go.Scatter(
            x=[time_label],
            y=[level],
            mode="text",
            text=[f"{pct:.1f}%"],
            hovertext=[f"{pct:.1f}% ({hits}/{total}){warn}"],
            hoverinfo="text",
            textfont=dict(color="white", size=12),
            textposition="top center",
            showlegend=False
        ))

# --- Add horizontal fib lines ---
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



# --- Highlight both trigger zones regardless of direction ---
if trigger_level in fib_levels:
    idx = fib_levels.index(trigger_level)

    # Green: UPWARD zone → next higher price level = lower index
    if idx - 1 >= 0:
        y0_up = fib_levels[idx]
        y1_up = fib_levels[idx - 1]
        fig.add_shape(
            type="rect",
            xref="paper", x0=0, x1=1,
            yref="y", y0=min(y0_up, y1_up), y1=max(y0_up, y1_up),
            fillcolor="rgba(0,255,0,0.3)",  # green zone above trigger
            layer="below",
            line_width=0,
        )

    # Yellow: DOWNWARD zone → next lower price level = higher index
    if idx + 1 < len(fib_levels):
        y0_down = fib_levels[idx]
        y1_down = fib_levels[idx + 1]
        fig.add_shape(
            type="rect",
            xref="paper", x0=0, x1=1,
            yref="y", y0=min(y0_down, y1_down), y1=max(y0_down, y1_down),
            fillcolor="rgba(255,255,0,0.3)",  # yellow zone below trigger
            layer="below",
            line_width=0,
        )

fig.add_shape(
    type="line",
    x0="OPEN", x1="OPEN",
    y0=min(fib_levels), y1=max(fib_levels),
    xref="x", yref="y",
    line=dict(color="gray", width=1, dash="dot"),
    layer="below"
)
fig.add_shape(
    type="line",
    x0="0900", x1="0900",
    y0=min(fib_levels), y1=max(fib_levels),
    xref="x", yref="y",
    line=dict(color="gray", width=1, dash="dot"),
    layer="below"
)
fig.add_shape(
    type="line",
    x0="1000", x1="1000",
    y0=min(fib_levels), y1=max(fib_levels),
    xref="x", yref="y",
    line=dict(color="gray", width=1, dash="dot"),
    layer="below"
)
fig.add_shape(
    type="line",
    x0="1100", x1="1100",
    y0=min(fib_levels), y1=max(fib_levels),
    xref="x", yref="y",
    line=dict(color="gray", width=1, dash="dot"),
    layer="below"
)
fig.add_shape(
    type="line",
    x0="1200", x1="1200",
    y0=min(fib_levels), y1=max(fib_levels),
    xref="x", yref="y",
    line=dict(color="gray", width=1, dash="dot"),
    layer="below"
)
fig.add_shape(
    type="line",
    x0="1300", x1="1300",
    y0=min(fib_levels), y1=max(fib_levels),
    xref="x", yref="y",
    line=dict(color="gray", width=1, dash="dot"),
    layer="below"
)
fig.add_shape(
    type="line",
    x0="1400", x1="1400",
    y0=min(fib_levels), y1=max(fib_levels),
    xref="x", yref="y",
    line=dict(color="gray", width=1, dash="dot"),
    layer="below"
)
fig.add_shape(
    type="line",
    x0="1500", x1="1500",
    y0=min(fib_levels), y1=max(fib_levels),
    xref="x", yref="y",
    line=dict(color="gray", width=1, dash="dot"),
    layer="below"
)

# --- Layout ---
fig.update_layout(
    title=f"{direction} | Trigger {trigger_level} at {trigger_time}",
    width=1000,
    height=720,
    margin=dict(l=40, r=40, t=60, b=40)
)

# --- Display ---
st.plotly_chart(fig, use_container_width=True)
