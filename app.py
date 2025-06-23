
import streamlit as st
import pandas as pd
import plotly.graph_objects as go

# --- Embedded summary generation ---
@st.cache_data
def generate_summary():
    df = pd.read_csv("combined_trigger_goal_results.csv")
    df["TriggerTime"] = df["TriggerTime"].astype(str)
    df["GoalTime"] = df["GoalTime"].astype(str)

    # Count unique triggers per day
    trigger_occurrences = df[['Date', 'TriggerLevel', 'TriggerTime', 'Direction']].drop_duplicates()
    trigger_counts = (
        trigger_occurrences
        .value_counts(subset=['TriggerLevel', 'TriggerTime', 'Direction'])
        .reset_index()
    )
    trigger_counts.columns = ['TriggerLevel', 'TriggerTime', 'Direction', 'NumTriggers']

    # Count successful goal hits by goal time
    goal_hits = df[df['GoalHit'] == 'Yes']
    goal_counts = (
        goal_hits
        .groupby(['TriggerLevel', 'TriggerTime', 'Direction', 'GoalLevel', 'GoalTime'])
        .size()
        .reset_index(name='NumHits')
    )

    summary = pd.merge(
        goal_counts,
        trigger_counts,
        on=['TriggerLevel', 'TriggerTime', 'Direction'],
        how='left'
    )
    summary['PctCompletion'] = (summary['NumHits'] / summary['NumTriggers'] * 100).round(2)

    return summary

df = generate_summary()

# --- Streamlit controls ---
st.title("ATR Roadmap Dashboard")

col1, col2, col3 = st.columns(3)
with col1:
    direction = st.selectbox("Direction", sorted(df["Direction"].unique()), index=0)
with col2:
    selected_trigger = st.selectbox("Trigger Level", sorted(df["TriggerLevel"].unique()))
with col3:
    all_hours = ["OPEN", "0900", "1000", "1100", "1200", "1300", "1400", "1500", "1600"]
    available_hours = df[(df["Direction"] == direction) & (df["TriggerLevel"] == selected_trigger)]["TriggerTime"].unique()
    trigger_hour = st.selectbox("Trigger Time", [h for h in all_hours if h in available_hours], index=0)

# Filter
filtered = df[(df["Direction"] == direction) & 
              (df["TriggerLevel"] == selected_trigger) & 
              (df["TriggerTime"] == trigger_hour)].copy()

# Time axis
real_times = ["OPEN", "0900", "1000", "1100", "1200", "1300", "1400", "1500", "1600"]
padded_times = []
for i, t in enumerate(real_times):
    padded_times.append(t)
    if i < len(real_times) - 1:
        padded_times.append(f"{t}_pad")

# Tick labels only for real hours
tickvals = real_times

# Fib levels
fib_levels = [1.0, 0.786, 0.618, 0.5, 0.382, 0.236, 0.0,
              -0.236, -0.382, -0.5, -0.618, -0.786, -1.0]

fig = go.Figure()

# Anchoring layout
fig.add_trace(go.Scatter(
    x=padded_times,
    y=[fib_levels[0]] * len(padded_times),
    mode="markers",
    marker=dict(color="rgba(0,0,0,0)"),
    hoverinfo="skip",
    showlegend=False
))

# Display logic
for level in fib_levels:
    for t in padded_times:
        if "_pad" in t:
            continue
        if t == "OPEN":
            fails = df[(df["TriggerTime"] == "OPEN") & (df["GoalHit"] == "No")].shape[0]
            if level != selected_trigger and fails > 0:
                fig.add_trace(go.Scatter(
                    x=[t], y=[level + 0.02],
                    mode="text", text=[""],
                    hovertext=[f"🕒 OPEN triggers that failed: {fails}"],
                    hoverinfo="text", showlegend=False
                ))
            continue
        if real_times.index(t) < real_times.index(trigger_hour):
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
        elif level != selected_trigger:
            text = "0.0%"
            hover = "0.0% (0/0)"
        else:
            continue
        fig.add_trace(go.Scatter(
            x=[t], y=[level + 0.02],
            mode="text", text=[text],
            hovertext=[hover],
            hoverinfo="text",
            textfont=dict(color="white", size=12),
            showlegend=False
        ))

# Fib shading
def next_level(levels, current, direction):
    idx = levels.index(current)
    if direction == "up" and idx > 0:
        return levels[idx - 1]
    elif direction == "down" and idx < len(levels) - 1:
        return levels[idx + 1]
    return None

upper = next_level(fib_levels, selected_trigger, "up")
lower = next_level(fib_levels, selected_trigger, "down")
if upper:
    fig.add_shape(type="rect", x0=0, x1=1, xref="paper", y0=selected_trigger, y1=upper, yref="y",
                  fillcolor="rgba(0,255,0,0.25)", line_width=0, layer="below")
if lower:
    fig.add_shape(type="rect", x0=0, x1=1, xref="paper", y0=selected_trigger, y1=lower, yref="y",
                  fillcolor="rgba(255,255,0,0.25)", line_width=0, layer="below")

# Fib lines
fibo_styles = {
    1.0: ("white", 2), 0.786: ("white", 1), 0.618: ("white", 2), 0.5: ("white", 1),
    0.382: ("white", 1), 0.236: ("cyan", 2), 0.0: ("white", 1),
    -0.236: ("yellow", 2), -0.382: ("white", 1), -0.5: ("white", 1),
    -0.618: ("white", 2), -0.786: ("white", 1), -1.0: ("white", 2)
}
for level, (color, width) in fibo_styles.items():
    fig.add_shape(type="line", xref="paper", x0=0, x1=1,
                  yref="y", y0=level, y1=level,
                  line=dict(color=color, width=width), layer="below")

# Grid lines
for t in real_times:
    fig.add_shape(type="line", x0=t, x1=t, xref="x",
                  y0=min(fib_levels), y1=max(fib_levels), yref="y",
                  line=dict(color="gray", width=1, dash="dot"), layer="below")

fig.update_layout(
    title=f"{direction} | Trigger {selected_trigger} at {trigger_hour}",
    xaxis=dict(
        title="Hour Goal Was Reached",
        categoryorder="array",
        categoryarray=padded_times,
        tickmode="array",
        tickvals=real_times,
        ticktext=real_times,
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
