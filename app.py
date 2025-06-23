
import streamlit as st
import pandas as pd
import plotly.graph_objects as go

st.set_page_config(layout="wide")

st.title("📊 ATR Roadmap Matrix")

# --- Load summary data ---
@st.cache_data
def load_summary():
    return pd.read_csv("atr_dashboard_summary.csv")

df = load_summary()

# --- Dropdowns ---
col1, col2, col3 = st.columns(3)

with col1:
    direction = st.selectbox("Select Direction", sorted(df['Direction'].unique()))

with col2:
    trigger_level = st.selectbox("Select Trigger Level", sorted(df['TriggerLevel'].unique()))

with col3:
    trigger_time = st.selectbox("Select Trigger Time", sorted(df['TriggerTime'].unique(), key=lambda x: (x == 'OPEN', x)))

# --- Filter data ---
filtered = df[
    (df['Direction'] == direction) &
    (df['TriggerLevel'] == trigger_level) &
    (df['TriggerTime'] == trigger_time)
]

# --- Matrix formatting ---
goal_levels = [-1, -0.786, -0.618, -0.5, -0.382, -0.236, 0, 0.236, 0.382, 0.5, 0.618, 0.786, 1]
time_slots = ['OPEN', '0900', '1000', '1100', '1200', '1300', '1400', '1500', '1600']

matrix = pd.DataFrame(index=goal_levels, columns=time_slots)

# --- Fill matrix ---
for _, row in filtered.iterrows():
    goal = row['GoalLevel']
    time = row['GoalTime']
    pct = row['PctCompletion']
    hits = int(row['NumHits'])
    total = int(row['NumTriggers'])

    if time == 'OPEN':
        matrix.at[goal, time] = f"" if hits == 0 else f"⚠ {hits} triggers failed"
    else:
        label = f"{pct:.1f}%"
        if total < 30:
            label += " ⚠"
        matrix.at[goal, time] = label

# Fill pre-trigger cells with blank, others with 0.0% if NaN
time_index = {k: i for i, k in enumerate(time_slots)}
for goal in goal_levels:
    for time in time_slots:
        if pd.isna(matrix.at[goal, time]):
            if time_index[time] < time_index.get(trigger_time, 99):
                matrix.at[goal, time] = ""
            else:
                matrix.at[goal, time] = "0.0%"

# --- Plotly heatmap ---
fig = go.Figure()

z = []
hover = []

for level in goal_levels:
    row = []
    hover_row = []
    for time in time_slots:
        val = matrix.at[level, time]
        row.append(None if val in ["", "⚠"] else float(val.strip('%⚠')))
        hover_row.append(val if val != "" else "No data")
    z.append(row)
    hover.append(hover_row)

colorscale = [
    [0, "#000000"],
    [0.01, "#440154"],
    [0.5, "#238A8D"],
    [1.0, "#FDE725"]
]

fig.add_trace(go.Heatmap(
    z=z,
    x=time_slots,
    y=goal_levels,
    text=matrix.values,
    hoverinfo="text",
    colorscale=colorscale,
    colorbar=dict(title="% Completion"),
    showscale=True
))

# --- Highlight zones ---
for y in [0.236, -0.236]:
    fig.add_shape(type="rect",
                  x0=-0.5, x1=len(time_slots)-0.5,
                  y0=y - 0.001, y1=y + 0.001,
                  line=dict(color="#00FFFF" if y > 0 else "#FFFF00", width=2))

fig.update_layout(
    xaxis=dict(title="Hour Goal Was Reached", tickvals=list(range(len(time_slots))), ticktext=time_slots),
    yaxis=dict(title="Goal Level", tickvals=goal_levels),
    font=dict(color="white", size=11),
    plot_bgcolor="black",
    paper_bgcolor="black",
    height=800,
    width=1000,
    margin=dict(t=50, l=100)
)

st.plotly_chart(fig, use_container_width=True)
