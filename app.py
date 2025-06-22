
import streamlit as st
import pandas as pd
import plotly.graph_objects as go

# --- Load and prepare data ---
df = pd.read_csv("combined_trigger_goal_results.csv")
df["TriggerTime"] = df["TriggerTime"].astype(str)
df["GoalTime"] = df["GoalTime"].astype(str)

# --- Sidebar UI ---
st.sidebar.title("ATR Roadmap Matrix")
direction = st.sidebar.selectbox("Select Direction", sorted(df["Direction"].unique()), index=1)
trigger_level = st.sidebar.selectbox("Select Trigger Level", sorted(df["TriggerLevel"].unique()), index=6)
time_order = ["OPEN", "0900", "1000", "1100", "1200", "1300", "1400", "1500"]
trigger_time = st.sidebar.selectbox("Select Trigger Time", time_order, index=0)

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

# --- Find highlight zone bounds ---
def next_level_up(levels, current):
    try:
        idx = levels.index(current)
        return levels[idx - 1] if idx > 0 else current
    except ValueError:
        return current

def next_level_down(levels, current):
    try:
        idx = levels.index(current)
        return levels[idx + 1] if idx < len(levels) - 1 else current
    except ValueError:
        return current

upper = next_level_up(fib_levels, trigger_level)
lower = next_level_down(fib_levels, trigger_level)

# --- Plotly figure setup ---
fig = go.Figure()

# --- Force X-axis to spread by adding invisible dummy trace
fig.add_trace(go.Scatter(
    x=time_order,
    y=[None]*len(time_order),
    mode="lines",
    line=dict(color="rgba(0,0,0,0)"),
    showlegend=False,
    hoverinfo="skip"
))

# --- Add percent text cells ---
for level in fib_levels:
    for t in time_order:
        match = grouped[(grouped["GoalLevel"] == level) & (grouped["GoalTime"] == t)]
        level_is_trigger = (level == trigger_level)
        if t < trigger_time:
            continue
        if match.empty:
            if level_is_trigger:
                text = ""
                hover = "Trigger Level — no goal evaluation"
            elif t >= trigger_time:
                text = "0.0%"
                hover = "0.0% (0/0) ⚠️"
            else:
                text = ""
                hover = ""
        else:
            row = match.iloc[0]
            pct = row["PctCompletion"]
            hits = row["NumHits"]
            total = row["NumTriggers"]
            warn = " ⚠️" if total < 30 else ""
            hover = f"{pct:.1f}% ({hits}/{total}){warn}"
            text = "" if level_is_trigger else f"{pct:.1f}%"

        fig.add_trace(go.Scatter(
            x=[t], y=[level],
            mode="text",
            text=[text],
            hovertext=[hover],
            hoverinfo="text",
            textfont=dict(color="white", size=12),
            showlegend=False
        ))

# --- Horizontal fib lines ---
fibo_styles = {
    1.0: ("white", 2), 0.786: ("white", 1), 0.618: ("white", 2), 0.5: ("white", 1),
    0.382: ("white", 1), 0.236: ("cyan", 2), 0.0: ("white", 1),
    -0.236: ("yellow", 2), -0.382: ("white", 1), -0.5: ("white", 1),
    -0.618: ("white", 2), -0.786: ("white", 1), -1.0: ("white", 2)
}

for level, (color, width) in fibo_styles.items():
    fig.add_shape(type="line", x0=0, x1=1, xref="paper", y0=level, y1=level, yref="y",
                  line=dict(color=color, width=width), layer="below")

# --- Add shaded zones ---
fig.add_shape(type="rect", x0=0, x1=1, xref="paper", y0=trigger_level, y1=upper, yref="y",
              fillcolor="rgba(0,255,0,0.2)", line=dict(width=0), layer="below")

fig.add_shape(type="rect", x0=0, x1=1, xref="paper", y0=trigger_level, y1=lower, yref="y",
              fillcolor="rgba(255,255,0,0.2)", line=dict(width=0), layer="below")

# --- Add vertical hour lines ---
for t in time_order:
    fig.add_shape(type="line", x0=t, x1=t, xref="x", y0=-1, y1=1, yref="y",
                  line=dict(color="gray", width=1, dash="dot"), layer="below")

# --- Layout ---
fig.update_layout(
    title=f"{direction} | Trigger {trigger_level} at {trigger_time}",
    xaxis=dict(
        title="Hour Goal Was Reached",
        categoryorder="array",
        categoryarray=time_order,
        tickmode="array",
        tickvals=time_order,
        tickangle=0,
        tickfont=dict(color="white"),
        showgrid=False
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
    height=800,
    width=1200,
    margin=dict(l=60, r=60, t=60, b=60)
)

# --- Scrollable container ---
st.markdown("<div style='overflow-x:auto;'>", unsafe_allow_html=True)
st.plotly_chart(fig, use_container_width=False)
st.markdown("</div>", unsafe_allow_html=True)
