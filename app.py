
import streamlit as st
import pandas as pd
import plotly.graph_objects as go

# Sample data for demonstration (replace with your actual data logic)
fib_levels = [1, 0.786, 0.618, 0.5, 0.382, 0.236, 0, -0.236, -0.382, -0.5, -0.618, -0.786, -1]
hours = ['OPEN', '0900', '1000', '1100', '1200', '1300', '1400', '1500', '1600']
data = pd.DataFrame(0.0, index=fib_levels, columns=hours)

# UI controls at the top
st.title("ATR Goal Completion Map")
direction = st.radio("Direction", ["Upside", "Downside"], horizontal=True)
trigger_level = st.selectbox("Trigger Level", fib_levels)
trigger_hour = st.selectbox("Trigger Hour", hours)

# Setup figure
fig = go.Figure()

# Add heatmap rectangles (transparent for now, just annotations)
for i, level in enumerate(fib_levels):
    for j, hour in enumerate(hours):
        value = data.loc[level, hour]
        text = f"{value:.1%}" if hour <= trigger_hour else ""
        fig.add_trace(go.Scatter(
            x=[hour], y=[level], text=[text],
            mode="text", textfont=dict(color="white", size=12),
            showlegend=False, hoverinfo="skip"
        ))

# Add horizontal grid lines
for level in fib_levels:
    fig.add_shape(type="line",
                  x0=hours[0], x1=hours[-1],
                  y0=level, y1=level,
                  line=dict(color="white", width=1))

# Add vertical grid lines
for hour in hours:
    fig.add_shape(type="line",
                  x0=hour, x1=hour,
                  y0=min(fib_levels), y1=max(fib_levels),
                  line=dict(color="gray", width=1, dash="dot"))

# Add horizontal highlight zones
def get_next_level(current, direction):
    idx = fib_levels.index(current)
    if direction == "up" and idx > 0:
        return fib_levels[idx - 1]
    elif direction == "down" and idx < len(fib_levels) - 1:
        return fib_levels[idx + 1]
    return current

upper_target = get_next_level(trigger_level, "up")
lower_target = get_next_level(trigger_level, "down")

# Yellow zone: down
fig.add_shape(type="rect",
              x0=hours[0], x1=hours[-1],
              y0=lower_target, y1=trigger_level,
              fillcolor="yellow", opacity=0.3, line_width=0)

# Green zone: up
fig.add_shape(type="rect",
              x0=hours[0], x1=hours[-1],
              y0=trigger_level, y1=upper_target,
              fillcolor="green", opacity=0.3, line_width=0)

fig.update_layout(
    yaxis=dict(title="Goal Level", tickvals=fib_levels),
    xaxis=dict(title="Hour Goal Was Hit", tickvals=hours, type="category"),
    plot_bgcolor="black",
    paper_bgcolor="black",
    margin=dict(l=40, r=40, t=40, b=40),
    width=1200,
    height=600
)

st.plotly_chart(fig, use_container_width=True)
