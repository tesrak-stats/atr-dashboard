import streamlit as st
import pandas as pd
import plotly.graph_objects as go

# --- Load and prepare data ---
df = pd.read_csv("combined_trigger_goal_results.csv")
df["TriggerTime"] = df["TriggerTime"].astype(str)
df["GoalTime"] = df["GoalTime"].astype(str)

# --- Sidebar UI ---
st.sidebar.title("ATR Roadmap Matrix")
direction = st.sidebar.selectbox("Select Direction", sorted(df["Direction"].unique()))
trigger_level = st.sidebar.selectbox("Select Trigger Level", sorted(df["TriggerLevel"].unique()))

# Fixed time order
time_order = ["OPEN", "0900", "1000", "1100", "1200", "1300", "1400", "1500"]

# Restrict time options to those that exist for selection
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

# --- Define goal levels from +1 to -1 (top to bottom) ---
fib_levels = [1.0, 0.786, 0.618, 0.5, 0.382, 0.236, 0.0,
              -0.236, -0.382, -0.5, -0.618, -0.786, -1.0]

# --- Plotly figure setup ---
fig = go.Figure()

# --- Add percentage text cells ---
for level in fib_levels:
    for t in time_order:
        match = grouped[
            (grouped["GoalLevel"] == level) &
            (grouped["GoalTime"] == t)
        ]
        if not match.empty:
            row = match.iloc[0]
            pct = row["PctCompletion"]
            hits = row["NumHits"]
            total = row["NumTriggers"]
            warn = " ⚠️" if total < 30 else ""
            hover = f"{pct:.1f}% ({hits}/{total}){warn}"
            fig.add_trace(go.Scatter(
                x=[t],
                y=[level],
                mode="text",
                text=[f"{pct:.1f}%"],
                hovertext=[hover],
                hoverinfo="text",
                textfont=dict(color="white", size=12),
                showlegend=False
            ))

# --- Add horizontal fib level guide lines ---
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
        categoryarray=fib_levels,  # 🔥 forces 1.0 at top, -1.0 at bottom
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
