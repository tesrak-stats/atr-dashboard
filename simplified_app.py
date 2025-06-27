import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from generate_daily_atr_levels import get_latest_atr_levels

# --- Load pre-calculated summary data ---
df = pd.read_csv("atr_dashboard_summary_SIMPLE.csv")

# --- Load current ATR-based price levels ---
try:
    atr_price_levels = get_latest_atr_levels()
except:
    atr_price_levels = {}

# --- Display configuration ---
visible_hours = ["0900", "1000", "1100", "1200", "1300", "1400", "1500"]
invisible_fillers = ["0930", "1030", "1130", "1230", "1330", "1430", "1530"]
time_order = ["OPEN"]
for hour in visible_hours:
    time_order.append(hour)
    filler = f"{str(int(hour[:2])+1).zfill(2)}30"
    time_order.append(filler)
time_order.append("1600")

fib_levels = [1.0, 0.786, 0.618, 0.5, 0.382, 0.236, 0.0,
              -0.236, -0.382, -0.5, -0.618, -0.786, -1.0]

# --- Page Layout ---
st.title("📈 ATR Levels Roadmap")

# --- Controls ---
col1, col2, col3 = st.columns(3)

direction = col1.selectbox("Direction", sorted(df["Direction"].unique()), 
                          index=sorted(df["Direction"].unique()).index("Above"))

trigger_level = col2.selectbox("Trigger Level", sorted(set(df["TriggerLevel"]).union(fib_levels)), 
                              index=sorted(set(df["TriggerLevel"]).union(fib_levels)).index(0.0))

trigger_time = col3.selectbox("Trigger Time", ["OPEN"] + visible_hours, index=0)

# --- Simple filter (no aggregation needed) ---
filtered = df[
    (df["Direction"] == direction) &
    (df["TriggerLevel"] == trigger_level) &
    (df["TriggerTime"] == trigger_time)
].copy()

# --- Create lookup dictionary from pre-calculated data ---
data_lookup = {}
for _, row in filtered.iterrows():
    # Convert GoalTime to string and handle numeric times (CRITICAL for lookup)
    goal_time = row["GoalTime"]
    if pd.notna(goal_time):
        if isinstance(goal_time, (int, float)):
            time_int = int(goal_time)
            if time_int == 900:
                goal_time_str = "0900"
            elif time_int < 1000:
                goal_time_str = f"0{time_int}"
            else:
                goal_time_str = str(time_int)
        else:
            goal_time_str = str(goal_time)
    else:
        goal_time_str = "Unknown"
    
    key = (float(row["GoalLevel"]), goal_time_str)
    data_lookup[key] = {
        "hits": row["NumHits"],
        "triggers": row["NumTriggers"], 
        "pct": row["PctCompletion"]
    }

# --- Build chart ---
fig = go.Figure()

# --- Matrix cells ---
for level in fib_levels:
    for t in time_order:
        if t in invisible_fillers or t in ["OPEN", "1600"]:
            continue
            
        key = (level, t)
        if key in data_lookup:
            data = data_lookup[key]
            pct = data["pct"]
            hits = data["hits"]
            total = data["triggers"]
            warn = " ⚠️" if total < 30 else ""
            hover = f"{pct:.1f}% ({hits}/{total}){warn}"
            
            fig.add_trace(go.Scatter(
                x=[t], y=[level + 0.015],
                mode="text", text=[f"{pct:.1f}%"],
                hovertext=[hover], hoverinfo="text",
                textfont=dict(color="white", size=12),
                showlegend=False
            ))
        else:
            # No data for this combination
            if time_order.index(t) < time_order.index(trigger_time):
                display = ""
            else:
                display = "0.0%"
                
            fig.add_trace(go.Scatter(
                x=[t], y=[level + 0.015],
                mode="text", text=[display],
                hoverinfo="skip",
                textfont=dict(color="white", size=12),
                showlegend=False
            ))

# --- Anchor invisible point for OPEN ---
fig.add_trace(go.Scatter(
    x=["OPEN"], y=[0.0],
    mode="markers",
    marker=dict(opacity=0),
    showlegend=False,
    hoverinfo="skip"
))

# --- Horizontal lines for Fibonacci levels ---
fibo_styles = {
    1.0: ("white", 2), 0.786: ("white", 1), 0.618: ("white", 2),
    0.5: ("white", 1), 0.382: ("white", 1), 0.236: ("cyan", 2),
    0.0: ("white", 1),
    -0.236: ("yellow", 2), -0.382: ("white", 1), -0.5: ("white", 1),
    -0.618: ("white", 2), -0.786: ("white", 1), -1.0: ("white", 2),
}

for level, (color, width) in fibo_styles.items():
    fig.add_shape(
        type="line", x0=0, x1=1, xref="paper", y0=level, y1=level, yref="y",
        line=dict(color=color, width=width), layer="below"
    )

# --- Chart layout ---
fig.update_layout(
    title=f"{direction} | Trigger {trigger_level} at {trigger_time}",
    xaxis=dict(
        title="Hour Goal Was Reached",
        categoryorder="array",
        categoryarray=time_order,
        tickmode="array",
        tickvals=["OPEN"] + visible_hours + ["1600"],
        tickfont=dict(color="white")
    ),
    yaxis=dict(
        title="Goal Level",
        categoryorder="array",
        categoryarray=fib_levels,
        tickmode="array",
        tickvals=fib_levels,
        tickfont=dict(color="white")
    ),
    plot_bgcolor="black",
    paper_bgcolor="black",
    font=dict(color="white"),
    height=900,
    width=1400,
    margin=dict(l=80, r=60, t=60, b=60)
)

# --- Price ladder on right Y-axis ---
if atr_price_levels:
    price_labels = [atr_price_levels["levels"].get(f"{level:+.3f}", "") for level in fib_levels]

    fig.update_layout(
        yaxis=dict(
            title="Fib Level",
            tickvals=fib_levels,
            ticktext=[f"{lvl:+.3f}" for lvl in fib_levels],
        ),
        yaxis2=dict(
            title="Price Level",
            overlaying="y",
            side="right",
            tickvals=fib_levels,
            ticktext=[
                f"{price:.2f}" if isinstance(price, (int, float)) else ""
                for price in price_labels
            ],
            tickfont=dict(color="lightgray"),
            showgrid=False
        )
    )

st.plotly_chart(fig, use_container_width=False)

# --- Optional: Show filtered data summary ---
if len(filtered) > 0:
    with st.expander("📊 Data Summary for Current Selection"):
        total_triggers = filtered['NumTriggers'].iloc[0] if len(filtered) > 0 else 0
        total_hits = filtered['NumHits'].sum()
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Triggers", total_triggers)
        with col2:
            st.metric("Total Hits", total_hits)
        with col3:
            avg_hit_rate = filtered['PctCompletion'].mean()
            st.metric("Avg Hit Rate", f"{avg_hit_rate:.1f}%")
        
        # Show top performing goals
        top_goals = filtered.nlargest(5, 'PctCompletion')[['GoalLevel', 'GoalTime', 'NumHits', 'PctCompletion']]
        st.write("**Top 5 Goal Combinations:**")
        st.dataframe(top_goals, hide_index=True)
else:
    st.info("No data available for this trigger combination.")
