
import streamlit as st
import pandas as pd
import plotly.graph_objects as go

# --- Load and prepare data ---
df = pd.read_csv("atr_dashboard_summary.csv")
df["TriggerTime"] = df["TriggerTime"].astype(str)
df["GoalTime"] = df["GoalTime"].astype(str)

# --- Fixed time and level order ---
time_order = ["OPEN", "0900", "1000", "1100", "1200", "1300", "1400", "1500", "1600"]
fib_levels = [1.0, 0.786, 0.618, 0.5, 0.382, 0.236, 0.0,
              -0.236, -0.382, -0.5, -0.618, -0.786, -1.0]

# --- Sidebar UI ---
st.sidebar.title("ATR Roadmap Matrix")

# Ensure all levels/times are present even if not in data
all_directions = sorted(df["Direction"].unique())
all_trigger_levels = sorted(set(df["TriggerLevel"]).union(fib_levels))
all_trigger_times = time_order

# Set defaults
default_dir = "Upside"
default_level = 0.0
default_time = "OPEN"

direction = st.sidebar.selectbox("Select Direction", all_directions, index=all_directions.index(default_dir))
trigger_level = st.sidebar.selectbox("Select Trigger Level", all_trigger_levels, index=all_trigger_levels.index(default_level))
trigger_time = st.sidebar.selectbox("Select Trigger Time", all_trigger_times, index=all_trigger_times.index(default_time))

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
        NumHits=("NumHits", "sum"),
        NumTriggers=("NumTriggers", "first")
    )
    .reset_index()
)

grouped["PctCompletion"] = (grouped["NumHits"] / grouped["NumTriggers"] * 100).round(1)

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
            text_display = f"{pct:.1f}%"
        else:
            # Leave early time blocks blank, others show 0%
            hover = "No data"
            text_display = ""

        fig.add_trace(go.Scatter(
            x=[t],
            y=[level],
            mode="text",
            text=[text_display],
            hovertext=[hover],
            hoverinfo="text",
            textfont=dict(color="white", size=12),
            showlegend=False
        ))

# --- Horizontal guide lines for fib levels ---
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
