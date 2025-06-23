
# Full updated app.py

import streamlit as st
import pandas as pd
import plotly.graph_objs as go

st.set_page_config(layout="wide")

st.title("📈 ATR Roadmap Matrix")

# Load summary data
@st.cache_data
def load_data():
    return pd.read_csv("atr_dashboard_summary.csv")

df = load_data()

# Sidebar controls
with st.sidebar:
    st.header("ATR Roadmap Matrix")
    direction = st.selectbox("Select Direction", df['Direction'].unique())
    level = st.selectbox("Select Trigger Level", sorted(df[df['Direction'] == direction]['TriggerLevel'].unique(), reverse=True))
    hour = st.selectbox("Select Trigger Time", sorted(df[df['Direction'] == direction]['TriggerTime'].unique()))

# Filter and reshape
filtered = df[(df['Direction'] == direction) & (df['TriggerLevel'] == level) & (df['TriggerTime'] == hour)]
pivot = filtered.pivot(index="GoalLevel", columns="GoalTime", values="PctCompletion").fillna("")

# Fill 0% explicitly, blank before trigger
time_order = ["OPEN", "0900", "1000", "1100", "1200", "1300", "1400", "1500", "1600"]
pivot = pivot.reindex(columns=time_order, fill_value="")

# Layout chart
fig = go.Figure()

goal_colors = {
    0.236: "cyan", -0.236: "yellow"
}
thick_levels = [0.618, -0.618]

for i, row in pivot.iterrows():
    for j, time in enumerate(pivot.columns):
        val = row[time]
        show_val = "" if val == "" else f"{val}%"
        fig.add_trace(go.Scatter(
            x=[time], y=[i], mode="text",
            text=[show_val],
            textposition="middle center",
            showlegend=False,
            hoverinfo="text",
            textfont=dict(size=10, color="white")
        ))

fig.update_layout(
    xaxis=dict(title="Hour Goal Was Reached", tickvals=time_order, tickangle=0),
    yaxis=dict(title="Goal Level", tickvals=sorted(pivot.index), autorange="reversed"),
    width=1000, height=700,
    plot_bgcolor="black",
    paper_bgcolor="black",
    margin=dict(l=60, r=40, t=40, b=60)
)

for lvl in pivot.index:
    if lvl in goal_colors:
        fig.add_shape(type="rect", x0=-0.5, x1=len(time_order)-0.5, y0=lvl-0.05, y1=lvl+0.05,
                      fillcolor=goal_colors[lvl], line=dict(width=0), opacity=0.2, layer="below")

    if lvl in thick_levels:
        fig.add_shape(type="line", x0=-0.5, x1=len(time_order)-0.5, y0=lvl, y1=lvl,
                      line=dict(color="white", width=2))

# Axis gridlines
fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor="gray")
fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor="white")

st.plotly_chart(fig, use_container_width=True)
