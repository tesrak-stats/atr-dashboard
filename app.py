import streamlit as st
import pandas as pd
import plotly.express as px

st.set_page_config(layout="wide")
st.title("📊 ATR Roadmap Matrix")

@st.cache_data
def load_data():
    return pd.read_csv("atr_dashboard_summary.csv")

df = load_data()

# User selections
direction = st.selectbox("Select Direction", df["Direction"].unique())
filtered = df[df["Direction"] == direction]
trigger_levels = sorted(filtered["TriggerLevel"].unique())
trigger_level = st.selectbox("Select Trigger Level", trigger_levels)
times = ["OPEN", "0900", "1000", "1100", "1200", "1300", "1400", "1500", "1600"]
trigger_time = st.selectbox("Select Trigger Time", times)

# Build matrix
goal_levels = [1, 0.786, 0.618, 0.5, 0.382, 0.236, 0, -0.236, -0.382, -0.5, -0.618, -0.786, -1]
matrix = pd.DataFrame(index=goal_levels, columns=times)

for t in times:
    for g in goal_levels:
        row = filtered[
            (filtered["TriggerLevel"] == trigger_level) &
            (filtered["TriggerTime"] == trigger_time) &
            (filtered["GoalLevel"] == g)
        ]
        if not row.empty:
            pct = row["PctCompletion"].values[0]
            matrix.at[g, t] = f"{pct:.1f}%"
        else:
            if t > trigger_time:
                matrix.at[g, t] = "0.0%"

# Plotting
z_data = matrix.replace('%','', regex=True).replace('None', None).astype(float)

fig = px.imshow(
    z_data.values,
    x=matrix.columns,
    y=matrix.index.astype(str),
    color_continuous_scale="Viridis",
    zmin=0,
    zmax=100,
    text_auto=True,
    labels=dict(x="Hour Goal Was Reached", y="Goal Level", color="% Complete")
)

fig.update_layout(
    autosize=True,
    height=700,
    width=1000,
    margin=dict(l=80, r=80, t=60, b=60),
    plot_bgcolor="black",
    paper_bgcolor="black",
    font=dict(color="white", size=10),
)

fig.update_xaxes(
    categoryorder="array",
    categoryarray=times,
    tickfont=dict(color="white")
)
fig.update_yaxes(
    categoryorder="array",
    categoryarray=[str(g) for g in goal_levels[::-1]],
    tickfont=dict(color="white")
)

st.plotly_chart(fig, use_container_width=True)