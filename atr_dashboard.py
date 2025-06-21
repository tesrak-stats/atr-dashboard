
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

# ---- Chart Rendering Function ----
def render_atr_chart(df, trigger_level, trigger_hour):
    goal_levels = sorted(df["goal_level"].unique(), reverse=True)
    time_blocks = ["OPEN"] + [f"{h:02d}00" for h in range(9, 17)]
    level_spacing = 55
    filtered = df[(df["trigger_level"] == trigger_level) & (df["trigger_hour"] == trigger_hour)]

    fig, ax = plt.subplots(figsize=(14, 10))
    fig.patch.set_facecolor('black')
    ax.set_facecolor("black")

    for i in range(len(time_blocks)):
        ax.axvline(x=i, color="gray", linewidth=0.5, linestyle='--', alpha=0.3)

    for i, level in enumerate(goal_levels):
        y = (len(goal_levels) - 1 - i) * level_spacing
        lw = 3.5 if abs(level) == 1.0 else 2.5 if abs(level) == 0.618 else 1.2
        color = "cyan" if level == 0.236 else "yellow" if level == -0.236 else "white"
        ax.hlines(y=y, xmin=-0.15, xmax=len(time_blocks) - 0.5, color=color, linewidth=lw)
        ax.text(-0.6, y, f"{level:+.3f}", color="white", ha="right", va="center", fontsize=10)
        ax.text(len(time_blocks) + 0.2, y, f"{4000 + level * 100:.1f}", color="gray", ha="left", va="center", fontsize=10)

    for _, row in filtered.iterrows():
        x = time_blocks.index(row["trigger_hour"])
        y = (len(goal_levels) - 1 - goal_levels.index(row["goal_level"])) * level_spacing
        size = max(row["percent_complete"], 1)
        ax.scatter(x, y, s=size * 4, color="cyan", edgecolors="white", linewidth=0.6, zorder=3, marker='o')
        ax.text(x, y + 11, f"{row['percent_complete']}%", ha="center", va="bottom", color="white", fontsize=9, zorder=4)

    ax.set_xlim(-1, len(time_blocks))
    ax.set_ylim(-level_spacing * 0.5, level_spacing * len(goal_levels))
    ax.set_xticks(range(len(time_blocks)))
    ax.set_xticklabels(time_blocks, color="white", fontsize=11)
    ax.set_yticks([])

    ax.set_title("ATR Levels Roadmap", color="white", fontsize=18, pad=30)

    for spine in ax.spines.values():
        spine.set_visible(False)

    st.pyplot(fig)

# ---- Streamlit UI ----
st.set_page_config(page_title="ATR Roadmap", layout="wide")
st.title("ATR Levels Roadmap (Simulated Data)")

# Load CSV with error handling
try:
    df = pd.read_csv("fake_atr_dashboard_data.csv")
except Exception as e:
    st.error(f"Error loading data: {e}")
    st.stop()

# Sidebar input
st.sidebar.header("🔧 Select Scenario")
trigger_levels = sorted(df["trigger_level"].unique())
trigger_hours = sorted(df["trigger_hour"].unique())

trigger = st.sidebar.selectbox("Trigger Level", trigger_levels, index=trigger_levels.index(0.0))
hour = st.sidebar.selectbox("Trigger Time", trigger_hours)
_ = st.sidebar.radio("Price Approached Trigger From", ["Below", "Above"])  # placeholder

# Chart rendering
filtered = df[(df["trigger_level"] == trigger) & (df["trigger_hour"] == hour)]

if filtered.empty:
    st.warning("No data found for this combination.")
else:
    render_atr_chart(df, trigger, hour)
