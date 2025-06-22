
import streamlit as st
import pandas as pd
import plotly.graph_objects as go

# Load your preprocessed data
@st.cache_data
def load_data():
    return pd.read_csv("output_matrix.csv")

data = load_data()

# Sidebar controls at the top
st.title("ATR Roadmap Dashboard")
st.sidebar.title("Settings")
direction = st.sidebar.selectbox("Direction", ["Upside", "Downside"], index=0)
goal_start_level = st.sidebar.selectbox("Start Goal Level", sorted(data["Goal Level"].unique()), index=list(sorted(data["Goal Level"].unique())).index(0))
trigger_hour = st.sidebar.selectbox("Trigger Hour", sorted(data["Trigger Hour"].unique()), index=0)

# Filter data based on sidebar
filtered_data = data[(data["Direction"] == direction) & (data["Trigger Hour"] == trigger_hour)]

# Pivot the matrix
pivot = filtered_data.pivot(index="Goal Level", columns="Goal Hour", values="Completion %")
raw_counts = filtered_data.pivot(index="Goal Level", columns="Goal Hour", values="Trigger Count")

# Apply 0.0 display rule, hide future hours
current_hour = pd.Timestamp.now().tz_localize("US/Pacific").hour
pivot.columns = pivot.columns.astype(str)
raw_counts.columns = raw_counts.columns.astype(str)
for col in pivot.columns:
    if col == "OPEN":
        continue
    col_hour = int(col[:2]) if col[:2].isdigit() else None
    if col_hour is not None:
        if col_hour > current_hour:
            pivot[col] = ""
        else:
            pivot[col] = pivot[col].fillna(0.0)
    else:
        pivot[col] = pivot[col].fillna(0.0)

# Setup the figure
fig = go.Figure()

# Add heatmap cells with text annotations
for i, y in enumerate(pivot.index):
    for j, x in enumerate(pivot.columns):
        val = pivot.loc[y, x]
        count = raw_counts.loc[y, x]
        color = "rgba(0,0,0,0)"
        if pd.notna(val) and val != "":
            if float(val) > 0:
                color = "rgba(255,255,255,0.15)"
        fig.add_trace(go.Scatter(
            x=[x], y=[y],
            mode="markers+text",
            text=[f"{val}%" if val != "" else ""],
            textposition="middle center",
            marker=dict(size=80, color=color),
            hovertemplate=f"{val}%<br>{int(count)} triggers" if pd.notna(count) else "",
            showlegend=False
        ))

# Add shaded regions for direction
levels = sorted(pivot.index)
if direction == "Upside":
    if 0.0 in levels:
        idx = levels.index(0.0)
        if idx + 1 < len(levels):
            y0, y1 = levels[idx], levels[idx + 1]
            fig.add_shape(type="rect", x0=-0.5, x1=len(pivot.columns) - 0.5,
                          y0=y0, y1=y1,
                          fillcolor="rgba(0,255,0,0.2)", line_width=0)
    if -0.236 in levels and 0.0 in levels:
        y0, y1 = -0.236, 0.0
        fig.add_shape(type="rect", x0=-0.5, x1=len(pivot.columns) - 0.5,
                      y0=y0, y1=y1,
                      fillcolor="rgba(255,255,0,0.2)", line_width=0)
else:
    if 0.0 in levels and -0.236 in levels:
        y0, y1 = -0.236, 0.0
        fig.add_shape(type="rect", x0=-0.5, x1=len(pivot.columns) - 0.5,
                      y0=y0, y1=y1,
                      fillcolor="rgba(0,255,0,0.2)", line_width=0)
    if 0.0 in levels and 0.236 in levels:
        y0, y1 = 0.0, 0.236
        fig.add_shape(type="rect", x0=-0.5, x1=len(pivot.columns) - 0.5,
                      y0=y0, y1=y1,
                      fillcolor="rgba(255,255,0,0.2)", line_width=0)

# Add horizontal and vertical lines
for level in levels:
    fig.add_shape(type="line", x0=-0.5, x1=len(pivot.columns) - 0.5, y0=level, y1=level,
                  line=dict(color="white", width=1))
for i in range(len(pivot.columns)):
    fig.add_shape(type="line", x0=i, x1=i, y0=min(levels), y1=max(levels),
                  line=dict(color="white", width=1, dash="dot"))

# Style and layout
fig.update_layout(
    xaxis=dict(title="Hour Goal Was Hit", tickvals=list(range(len(pivot.columns))), ticktext=list(pivot.columns)),
    yaxis=dict(title="Goal Level", tickvals=levels),
    plot_bgcolor="black",
    paper_bgcolor="black",
    font=dict(color="white"),
    height=700,
    margin=dict(t=20, b=40, l=60, r=20)
)

st.plotly_chart(fig, use_container_width=True)
