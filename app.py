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
direction = st.sidebar.selectbox("Select Direction", sorted(df["Direction"].unique()))
trigger_level = st.sidebar.selectbox("Select Trigger Level", sorted(df["TriggerLevel"].unique()))
time_order = ["OPEN", "0900", "1000", "1100", "1200", "1300", "1400", "1500"]

# Sort available trigger times for the selected scenario
available_times = df[
    (df["Direction"] == direction) & (df["TriggerLevel"] == trigger_level)
]["TriggerTime"].unique()
trigger_times_sorted = [t for t in time_order if t in available_times]
trigger_time = st.sidebar.selectbox("Select Trigger Time", trigger_times_sorted)

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

# Compute percent completions — show 0.0% when NumHits=0
def compute_pct(row):
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
    hits = int(row["NumHits"] or 0)
    total = int(row["NumTriggers"] or 0)
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
    margin=dict(l=40, r=40, t=60, b=40)
)

# --- Display ---
st.plotly_chart(fig, use_container_width=True)
