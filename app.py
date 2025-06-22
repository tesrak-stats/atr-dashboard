
import streamlit as st
import pandas as pd
import plotly.graph_objects as go

# Sample data structure: replace with real data as needed
df = pd.DataFrame({
    'Hour': ['OPEN', '0900', '1000', '1100', '1200', '1300', '1400', '1500', '1600'],
    '0.786': [None]*9,
    '0.618': [None]*9,
    '0.5': [None]*9,
    '0.382': [None]*9,
    '0.236': [30]*9,
    '0': [0]*9,
    '-0.236': [3]*9,
    '-0.382': [3]*9,
    '-0.5': [3]*9,
    '-0.618': [3]*9,
    '-0.786': [3]*9,
    '-1': [3]*9
})

# Settings
st.set_page_config(layout="wide")
st.title("ATR Roadmap Dashboard")

fib_levels = ['0.786', '0.618', '0.5', '0.382', '0.236', '0', '-0.236', '-0.382', '-0.5', '-0.618', '-0.786', '-1']
hours = df['Hour'].tolist()

# Trigger selector
col1, col2, col3 = st.columns(3)
with col1:
    direction = st.radio("Direction", ['Upside', 'Downside'], index=0, horizontal=True)
with col2:
    trigger_level = st.selectbox("Trigger Level", fib_levels, index=fib_levels.index('0'))
with col3:
    trigger_hour = st.selectbox("Trigger Hour", hours, index=0)

# Heatmap z-values
z = []
for level in fib_levels:
    row = []
    for h in hours:
        val = df.loc[df['Hour'] == h, level].values[0]
        if h < trigger_hour:
            row.append(None)
        elif level == trigger_level:
            row.append(0)
        else:
            row.append(val)
    z.append(row)

# Annotations
annotations = []
for i, level in enumerate(fib_levels):
    for j, h in enumerate(hours):
        val = z[i][j]
        if val is not None:
            annotations.append(dict(
                text=f"{val:.1f}%",
                x=j,
                y=i,
                xref='x',
                yref='y',
                showarrow=False,
                font=dict(color='white' if abs(val) > 50 else 'gray', size=12)
            ))

# Figure setup
fig = go.Figure(data=go.Heatmap(
    z=z,
    x=hours,
    y=fib_levels,
    colorscale='gray',
    showscale=False
))

# Add horizontal lines
for level in fib_levels:
    fig.add_shape(type="line",
                  x0=-0.5, x1=len(hours)-0.5,
                  y0=fib_levels.index(level), y1=fib_levels.index(level),
                  line=dict(color="white", width=1 if level not in ['0.236', '-0.236'] else 2, dash="solid" if abs(float(level)) not in [0.236, 0.618] else "dash"),
                  layer="below")

# Add vertical gridlines
for i in range(len(hours)):
    fig.add_shape(type="line",
                  x0=i, x1=i,
                  y0=-0.5, y1=len(fib_levels)-0.5,
                  line=dict(color="lightgray", width=1, dash="dot"),
                  layer="below")

# Green and yellow zone shading logic
trigger_idx = fib_levels.index(trigger_level)
if direction == 'Upside' or True:
    # Shade from trigger up to next higher level
    if trigger_idx > 0:
        fig.add_shape(type="rect",
                      x0=-0.5, x1=len(hours)-0.5,
                      y0=trigger_idx - 1 + 0.5, y1=trigger_idx + 0.5,
                      fillcolor="rgba(0,255,0,0.2)", line=dict(width=0), layer="below")

if direction == 'Downside' or True:
    # Shade from trigger down to next lower level
    if trigger_idx < len(fib_levels) - 1:
        fig.add_shape(type="rect",
                      x0=-0.5, x1=len(hours)-0.5,
                      y0=trigger_idx - 0.5, y1=trigger_idx + 1 - 0.5,
                      fillcolor="rgba(255,255,0,0.2)", line=dict(width=0), layer="below")

# Final layout
fig.update_layout(
    width=1000,
    height=600,
    margin=dict(l=60, r=40, t=40, b=60),
    xaxis=dict(title="Hour Goal Was Hit", tickmode='array', tickvals=list(range(len(hours))), ticktext=hours),
    yaxis=dict(title="Goal Level", tickmode='array', tickvals=list(range(len(fib_levels))), ticktext=fib_levels),
    annotations=annotations,
    plot_bgcolor="black",
    paper_bgcolor="black",
    font=dict(color="white")
)

st.plotly_chart(fig, use_container_width=False)
