import streamlit as st
import pandas as pd
import plotly.graph_objects as go

st.set_page_config(layout="wide")
st.title("📈 ATR Roadmap Matrix")

# Load the summary file
df = pd.read_csv("atr_dashboard_summary.csv")

# Dropdowns
directions = df['Direction'].unique()
trigger_levels = sorted(df['TriggerLevel'].unique(), reverse=True)
trigger_times = ['OPEN', '0900', '1000', '1100', '1200', '1300', '1400', '1500']

col1, col2, col3 = st.columns(3)
with col1:
    selected_direction = st.selectbox("Select Direction", directions)
with col2:
    selected_trigger = st.selectbox("Select Trigger Level", trigger_levels)
with col3:
    selected_time = st.selectbox("Select Trigger Time", trigger_times)

# Filtered Data
filtered = df[
    (df['Direction'] == selected_direction) &
    (df['TriggerLevel'] == selected_trigger) &
    (df['TriggerTime'] == selected_time)
]

# Ensure all goal levels appear (including 0% or missing)
goal_levels = sorted(df['GoalLevel'].unique(), reverse=True)
time_blocks = ['OPEN', '0900', '1000', '1100', '1200', '1300', '1400', '1500', '1600']

z = []
text = []
for level in goal_levels:
    row = []
    row_text = []
    for hour in time_blocks:
        match = filtered[(filtered['GoalLevel'] == level) & (filtered['GoalTime'] == hour)]
        if match.empty:
            if hour == 'OPEN':
                row.append(None)
                row_text.append("")
            else:
                row.append(0)
                row_text.append("0%")
        else:
            pct = match['PctCompletion'].values[0]
            row.append(pct)
            row_text.append(f"{pct:.1f}%")
    z.append(row)
    text.append(row_text)

# Color logic
def get_color(pct):
    if pct is None:
        return 'rgba(0,0,0,0)'
    elif pct == 0:
        return 'black'
    elif pct == 100:
        return 'darkgreen' if selected_direction == 'Upside' else 'darkred'
    else:
        return 'goldenrod'

shapes = []
annotations = []
cell_width = 1
cell_height = 1
for i, level in enumerate(goal_levels):
    for j, hour in enumerate(time_blocks):
        x0 = j * cell_width
        x1 = x0 + cell_width
        y0 = i * cell_height
        y1 = y0 + cell_height
        val = z[i][j]
        color = get_color(val)
        shapes.append(dict(type="rect", x0=x0, x1=x1, y0=y0, y1=y1,
                           line=dict(color="white", width=1),
                           fillcolor=color))
        if text[i][j]:
            annotations.append(dict(x=(x0 + x1)/2, y=(y0 + y1)/2,
                                    text=text[i][j],
                                    showarrow=False,
                                    font=dict(color="white", size=10)))

fig = go.Figure()
fig.update_layout(
    shapes=shapes,
    annotations=annotations,
    xaxis=dict(
        tickmode='array',
        tickvals=[i + 0.5 for i in range(len(time_blocks))],
        ticktext=time_blocks,
        showgrid=False,
        zeroline=False
    ),
    yaxis=dict(
        tickmode='array',
        tickvals=[i + 0.5 for i in range(len(goal_levels))],
        ticktext=[str(lvl) for lvl in goal_levels],
        autorange="reversed",
        showgrid=False,
        zeroline=False
    ),
    plot_bgcolor='black',
    margin=dict(l=80, r=40, t=40, b=40),
    height=600,
    width=900
)

fig.update_xaxes(showline=False, showticklabels=True)
fig.update_yaxes(showline=False, showticklabels=True)
st.plotly_chart(fig, use_container_width=False)