import streamlit as st
import pandas as pd
import plotly.graph_objects as go

# --- Load summary data ---
df = pd.read_csv("atr_matrix_summary_by_goal_time.csv")

# --- Ensure GoalTime and TriggerTime are numeric ---
df["TriggerTime"] = pd.to_numeric(df["TriggerTime"], errors="coerce")
df["GoalTime"] = pd.to_numeric(df["GoalTime"], errors="coerce")
df = df.dropna(subset=["TriggerTime", "GoalTime"])

# --- Sidebar Controls ---
st.sidebar.title("ATR Roadmap Matrix")
direction = st.sidebar.selectbox("Select Direction", sorted(df["Direction"].unique()))
trigger_level = st.sidebar.selectbox("Select Trigger Level", sorted(df["TriggerLevel"].unique()))

# Only show valid trigger times for this combination
valid_times = df[
    (df["Direction"] == direction) &
    (df["TriggerLevel"] == trigger_level)
]["TriggerTime"].dropna().astype(int).unique()

if len(valid_times) == 0:
    st.error("No valid trigger times found for this setup.")
    st.stop()

trigger_time = st.sidebar.selectbox("Select Trigger Time", sorted(valid_times))

# --- Filter scenario ---
filtered = df[
    (df["Direction"] == direction) &
    (df["TriggerLevel"] == trigger_level) &
    (df["TriggerTime"] == trigger_time)
].copy()

# --- Separate continuation and retracement goals ---
if direction == "Upside":
    continuation = filtered[filtered["GoalLevel"] > trigger_level]
    retracement = filtered[filtered["GoalLevel"] < trigger_level]
else:
    continuation = filtered[filtered["GoalLevel"] < trigger_level]
    retracement = filtered[filtered["GoalLevel"] > trigger_level]

# --- Combine & filter by time ≥ trigger time ---
matrix_data = pd.concat([continuation, retracement])
matrix_data = matrix_data[matrix_data["GoalTime"] >= trigger_time]

# --- Prepare heatmap ---
matrix_data.set_index(["GoalLevel", "GoalTime"], inplace=True)
goal_levels = sorted(matrix_data.index.get_level_values(0).unique(), reverse=True)
goal_times = sorted(matrix_data.index.get_level_values(1).unique())

z = []
hover_text = []

for goal in goal_levels:
    z_row = []
    hover_row = []
    for time in goal_times:
        try:
            row = matrix_data.loc[(goal, time)]
            pct = row["PctCompletion"]
            hits = row["NumHits"]
            total = row["NumTriggers"]
            warning = " ⚠️ Low Sample" if total < 30 else ""
            hover = f"{pct:.1f}% ({hits}/{total}){warning}"
            z_row.append(pct)
            hover_row.append(hover)
        except:
            z_row.append(None)
            hover_row.append("No Data")
    z.append(z_row)
    hover_text.append(hover_row)

# --- Plotly Heatmap ---
fig = go.Figure(data=go.Heatmap(
    z=z,
    x=[str(t) for t in goal_times],
    y=goal_levels,
    text=hover_text,
    hoverinfo="text",
    colorscale=[
        [0.0, "#200026"],
        [0.01, "#440154"],
        [0.25, "#31688e"],
        [0.5, "#35b779"],
        [0.75, "#fde725"],
        [1.0, "#ffffe0"]
    ],
    zmin=0,
    zmax=100,
    colorbar=dict(title="% Completion")
))

fig.update_layout(
    title=f"{direction} | Trigger {trigger_level} at {trigger_time}",
    height=700,
    xaxis_title="Hour Goal Was Reached",
    yaxis_title="Goal Level",
    yaxis_autorange="reversed"
)

st.plotly_chart(fig, use_container_width=True)
