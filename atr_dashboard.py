import streamlit as st
import pandas as pd
import plotly.express as px

# Load data
df = pd.read_csv("fake_atr_dashboard_data.csv")

st.set_page_config(page_title="ATR Roadmap", layout="wide")
st.title("ATR Levels Roadmap (Simulated Data)")

# Sidebar inputs
st.sidebar.header("🔧 Select Scenario")
trigger_levels = sorted(df["trigger_level"].unique())
hours = sorted(df["trigger_hour"].unique())

trigger = st.sidebar.selectbox("Trigger Level", trigger_levels, index=trigger_levels.index(0.0))
hour = st.sidebar.selectbox("Trigger Time", hours)
direction = st.sidebar.radio("Price Approached Trigger From", ["Below", "Above"])

# Filter based on selected trigger, hour, direction
valid_hours = ["OPEN", "0900", "1000", "1100", "1200", "1300", "1400", "1500", "1600", "TOTAL"]
filtered = df[
    (df["trigger_level"] == trigger) &
    (df["trigger_hour"] == hour) &
    (df["direction"] == direction) &
    (df["goal_hour"].isin(valid_hours))
]

# Warn if no match
if filtered.empty:
    st.warning("No data found for this combination.")
else:
    # Add formatted labels
    filtered["Goal Label"] = filtered["goal_level"].apply(lambda x: f"{x:+.3f}")
    filtered["Hour Label"] = filtered["goal_hour"]

    # Sort so TOTAL appears last vertically
    hour_order = ["OPEN", "0900", "1000", "1100", "1200", "1300", "1400", "1500", "1600", "TOTAL"]
    filtered["Hour Label"] = pd.Categorical(filtered["Hour Label"], categories=hour_order, ordered=True)

    # Plot
    fig = px.scatter(
        filtered,
        x="Hour Label",
        y="Goal Label",
        size="percent_complete",
        color="direction",
        hover_data=["percent_complete", "raw_count"],
        labels={"percent_complete": "% Complete"},
        title=f"Trigger {trigger:+.3f} at {hour} ({direction})"
    )

    # Add black border around bubbles
    fig.update_traces(marker=dict(line=dict(width=1, color='black')))

    # Annotations
    for _, row in filtered.iterrows():
        label = f"{row['percent_complete']}%"
        if row["goal_hour"] == "TOTAL":
            fig.add_annotation(
                x=row["Hour Label"],
                y=row["Goal Label"],
                text=label,
                showarrow=False,
                yshift=0,
                font=dict(size=12, color="orange", family="Arial Black")
            )
        else:
            fig.add_annotation(
                x=row["Hour Label"],
                y=row["Goal Label"],
                text=label,
                showarrow=False,
                yshift=12,
                font=dict(size=11, color="white")
            )

    # Custom styling
    fig.update_layout(
        height=700,
        xaxis_title="Hour of Day",
        yaxis_title="ATR Level",
        plot_bgcolor="#111",
        paper_bgcolor="#111",
        font_color="white",
        title_font_size=22
    )

    fig.update_xaxes(showgrid=True, gridcolor='gray')
    fig.update_yaxes(showgrid=True, gridcolor='gray')

    st.plotly_chart(fig, use_container_width=True)
