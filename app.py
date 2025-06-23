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

# --- Top Layout UI ---
st.title("📈 ATR Roadmap Matrix")
col1, col2, col3 = st.columns(3)

direction = col1.selectbox("Select Direction", sorted(df["Direction"].unique()), index=0)
trigger_level = col2.selectbox("Select Trigger Level", sorted(set(df["TriggerLevel"]).union(fib_levels)))
trigger_time = col3.selectbox("Select Trigger Time", time_order, index=0)

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
                x=[t], y=[level + 0.015],  # bump text upward
                mode="text",
                text=[f"{pct:.1f}%"],
                hovertext=[hover],
                hoverinfo="text",
                textfont=dict(color="white", size=12),
                showlegend=False
            ))
        else:
            if time_order.index(t) < time_order.index(trigger_time):
                display = ""
            else:
                display = "0.0%"
            fig.add_trace(go.Scatter(
                x=[t], y=[level + 0.015],
                mode="text",
                text=[display],
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

# --- Outer white border ---
fig.add_shape(
    type="rect",
    xref="paper", yref="y",
    x0=0, x1=1,
    y0=min(fib_levels), y1=max(fib_levels),
    line=dict(color="white", width=2),
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
    height=800,
    width=1200,
    margin=dict(l=80, r=60, t=60, b=60)
)

st.plotly_chart(fig, use_container_width=False)