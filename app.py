import streamlit as st
import pandas as pd
import plotly.graph_objects as go

# --- Load and prepare data ---
df = pd.read_csv("atr_dashboard_summary.csv")
df["TriggerTime"] = df["TriggerTime"].astype(str)
df["GoalTime"] = df["GoalTime"].astype(str)

# --- Time blocks: real + spacing
visible_hours = ["OPEN", "0900", "1000", "1100", "1200", "1300", "1400", "1500", "1600"]
invisible_fillers = ["0930", "1030", "1130", "1230", "1330", "1430", "1530"]
time_order = []
for hour in visible_hours:
    time_order.append(hour)
    if hour != "1600":
        filler = f"{str(int(hour[:2])+1).zfill(2)}30" if hour != "OPEN" else "0930"
        if filler not in time_order:
            time_order.append(filler)

# --- Fixed goal levels ---
fib_levels = [1.0, 0.786, 0.618, 0.5, 0.382, 0.236, 0.0,
              -0.236, -0.382, -0.5, -0.618, -0.786, -1.0]

# --- Top Layout UI ---
st.title("📈 ATR Roadmap Matrix")
col1, col2, col3 = st.columns(3)

direction = col1.selectbox("Select Direction", sorted(df["Direction"].unique()), index=0)
trigger_level = col2.selectbox("Select Trigger Level", sorted(set(df["TriggerLevel"]).union(fib_levels)))
trigger_time = col3.selectbox("Select Trigger Time", visible_hours, index=0)

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

# --- Plotly setup ---
fig = go.Figure()

# --- Add matrix cells ---
for level in fib_levels:
    for t in time_order:
        match = grouped[(grouped["GoalLevel"] == level) & (grouped["GoalTime"] == t)]
        if not match.empty:
            row = match.iloc[0]
            pct = row["PctCompletion"]
            hits = row["NumHits"]
            total = row["NumTriggers"]
            warn = " ⚠️" if total < 30 else ""
            hover = f"{pct:.1f}% ({hits}/{total}){warn}"
            fig.add_trace(go.Scatter(
                x=[t], y=[level + 0.015],
                mode="text",
                text=[f"{pct:.1f}%"],
                hovertext=[hover],
                hoverinfo="text",
                textfont=dict(color="white", size=12),
                showlegend=False
            ))
        elif t not in invisible_fillers and time_order.index(t) >= time_order.index(trigger_time):
            fig.add_trace(go.Scatter(
                x=[t], y=[level + 0.015],
                mode="text",
                text=["0.0%"],
                hoverinfo="skip",
                textfont=dict(color="white", size=12),
                showlegend=False
            ))

# --- Horizontal fib lines ---
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

# --- Add green/yellow shading between trigger and next level ---
try:
    i = fib_levels.index(trigger_level)
    if direction == "Upside" and i > 0:
        upper_level = fib_levels[i - 1]
        fig.add_shape(
            type="rect",
            x0=0, x1=1, xref="paper",
            y0=trigger_level, y1=upper_level, yref="y",
            fillcolor="rgba(0,255,0,0.1)", line_width=0, layer="below"
        )
    elif direction == "Downside" and i < len(fib_levels) - 1:
        lower_level = fib_levels[i + 1]
        fig.add_shape(
            type="rect",
            x0=0, x1=1, xref="paper",
            y0=trigger_level, y1=lower_level, yref="y",
            fillcolor="rgba(255,255,0,0.1)", line_width=0, layer="below"
        )
except:
    pass

# --- Unified outer border with correct bounds and thickness ---
fig.add_shape(
    type="rect",
    xref="paper", yref="y",
    x0=0, x1=1,
    y0=-1.05, y1=1.05,
    line=dict(color="white", width=1),
    layer="above"
)

# --- Layout ---
fig.update_layout(
    title=f"{direction} | Trigger {trigger_level} at {trigger_time}",
    xaxis=dict(
        title="Hour Goal Was Reached",
        categoryorder="array",
        categoryarray=time_order,
        tickmode="array",
        tickvals=visible_hours,
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
    height=900,
    width=1400,
    margin=dict(l=80, r=60, t=60, b=60)
)

st.plotly_chart(fig, use_container_width=False)