
import streamlit as st
import pandas as pd
import plotly.graph_objects as go

# Load CSV
@st.cache_data
def load_data():
    return pd.read_csv("combined_trigger_goal_results.csv")

df = load_data()

# Sidebar inputs
col1, col2, col3 = st.columns(3)
with col1:
    direction = st.selectbox("Direction", ["Upside", "Downside"], index=0)
with col2:
    selected_trigger = st.selectbox("Trigger Level", sorted(df["TriggerLevel"].unique()))
with col3:
    selected_hour = st.selectbox("Trigger Hour", ["OPEN", "0900", "1000", "1100", "1200", "1300", "1400", "1500", "1600"], index=0)

# Prep time blocks
time_blocks = [
    "OPEN", "0900", "0930", "1000", "1030", "1100", "1130",
    "1200", "1230", "1300", "1330", "1400", "1430", "1500", "1530", "1600"
]
tick_labels = ["OPEN", "0900", "1000", "1100", "1200", "1300", "1400", "1500", "1600"]

# Filter and pivot
filt = (df["Direction"] == direction) & (df["TriggerLevel"] == selected_trigger) & (df["TriggerHour"] == selected_hour)
data = df[filt]

pivot = pd.pivot_table(
    data, values="PctCompletion",
    index="GoalLevel", columns="GoalTime",
    aggfunc="first"
).reindex(columns=time_blocks).fillna("")

hits = pd.pivot_table(
    data, values="NumHits",
    index="GoalLevel", columns="GoalTime",
    aggfunc="first"
).reindex(columns=time_blocks).fillna("")

total = pd.pivot_table(
    data, values="NumTriggerTotal",
    index="GoalLevel", columns="GoalTime",
    aggfunc="first"
).reindex(columns=time_blocks).fillna("")

# Plotly chart
fig = go.Figure()

# Add empty heatmap for layout
fig.add_trace(go.Heatmap(
    z=[[None]*len(time_blocks) for _ in pivot.index],
    x=time_blocks, y=pivot.index,
    showscale=False, colorscale='Viridis',
    hoverinfo='none'
))

# Add annotations
for y_idx, level in enumerate(pivot.index):
    for x_idx, hour in enumerate(time_blocks):
        pct = pivot.loc[level, hour]
        if hour == "OPEN":
            trigger_count = total.loc[level, hour]
            if trigger_count and trigger_count > 0:
                fig.add_annotation(
                    text=f"{int(trigger_count)} trigger(s) failed at open",
                    x=hour, y=level,
                    showarrow=False, font=dict(color="gray", size=8)
                )
            continue
        if hour > selected_hour:
            if pct != "":
                fig.add_annotation(
                    text=f"{float(pct):.1f}%",
                    x=hour, y=level,
                    showarrow=False,
                    font=dict(color="white", size=10)
                )

# Highlight zones
levels = sorted(pivot.index.tolist())
if selected_trigger in levels:
    idx = levels.index(selected_trigger)
    if idx + 1 < len(levels):
        fig.add_shape(type="rect",
            x0=-0.5, x1=len(time_blocks)-0.5,
            y0=levels[idx], y1=levels[idx+1],
            fillcolor="green", opacity=0.2, layer="below", line_width=0)
    if idx - 1 >= 0:
        fig.add_shape(type="rect",
            x0=-0.5, x1=len(time_blocks)-0.5,
            y0=levels[idx-1], y1=levels[idx],
            fillcolor="yellow", opacity=0.2, layer="below", line_width=0)

# Layout
fig.update_layout(
    title=f"{direction} | Trigger {selected_trigger} at {selected_hour}",
    xaxis=dict(title="Hour Goal Was Reached", tickmode="array", tickvals=tick_labels),
    yaxis_title="Goal Level",
    plot_bgcolor="black",
    paper_bgcolor="black",
    font=dict(color="white"),
    width=1400,
    height=800,
    margin=dict(t=50, l=60, r=40, b=60)
)

st.plotly_chart(fig, use_container_width=False)
