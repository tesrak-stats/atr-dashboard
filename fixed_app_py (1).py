import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from generate_daily_atr_levels import get_latest_atr_levels

# --- Load pre-computed data ---
df = pd.read_csv("atr_dashboard_summary.csv")

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

# --- UI Controls ---
st.title("📈 ATR Roadmap Matrix")
col1, col2, col3 = st.columns(3)

direction = col1.selectbox("Select Direction", sorted(df["Direction"].unique()), index=0)
trigger_level = col2.selectbox("Select Trigger Level", sorted(set(df["TriggerLevel"]).union(fib_levels)), index=sorted(set(df["TriggerLevel"]).union(fib_levels)).index(0.0))
trigger_time = col3.selectbox("Select Trigger Time", ["OPEN"] + visible_hours, index=0)

# --- Filter Data ---
filtered = df[
    (df["Direction"] == direction) &
    (df["TriggerLevel"] == trigger_level) &
    (df["TriggerTime"] == trigger_time)
].copy()

st.write(f"🔍 Found {len(filtered)} records for {direction} | {trigger_level} | {trigger_time}")

# --- ORIGINAL WORKING AGGREGATION ---
if len(filtered) > 0:
    grouped = (
        filtered.groupby(["GoalLevel", "GoalTime"])
        .agg(NumHits=("NumHits", "sum"), NumTriggers=("NumTriggers", "first"))
        .reset_index()
    )
    grouped["PctCompletion"] = (grouped["NumHits"] / grouped["NumTriggers"] * 100).round(1)
    
    st.write(f"✅ Aggregated to {len(grouped)} combinations")
    
    # Show non-zero combinations
    non_zero = grouped[grouped['PctCompletion'] > 0]
    if len(non_zero) > 0:
        st.write(f"🎯 {len(non_zero)} combinations with hits:")
        st.dataframe(non_zero)
else:
    st.warning(f"No data found for {direction} triggers at level {trigger_level} and time {trigger_time}")
    grouped = pd.DataFrame(columns=["GoalLevel", "GoalTime", "NumHits", "NumTriggers", "PctCompletion"])

# --- Create lookup dictionary ---
data_lookup = {}
for _, row in grouped.iterrows():
    # Convert GoalTime to match chart time format
    goal_time = row["GoalTime"]
    if pd.notna(goal_time):
        if isinstance(goal_time, (int, float)):
            # Convert numeric times to 4-digit string format
            time_int = int(goal_time)
            if time_int == 900:
                goal_time_str = "0900"  # 900 -> "0900"
            elif time_int < 1000:
                goal_time_str = f"0{time_int}"  # Handle other < 1000 times
            else:
                goal_time_str = str(time_int)  # 1000 -> "1000", 1100 -> "1100"
        else:
            goal_time_str = str(goal_time)  # "OPEN" -> "OPEN"
    else:
        goal_time_str = "Unknown"
        
    key = (row["GoalLevel"], goal_time_str)
    data_lookup[key] = {
        "hits": row["NumHits"],
        "triggers": row["NumTriggers"], 
        "pct": row["PctCompletion"]
    }

# Debug: Show what keys we're creating
st.write("### Sample Lookup Keys:")
sample_keys = list(data_lookup.keys())[:10]
for key in sample_keys:
    data = data_lookup[key]
    st.write(f"  {key} → {data['pct']}%")

# --- Build Visualization ---
fig = go.Figure()

# --- Matrix cells ---
for level in fib_levels:
    for t in time_order:
        if t in invisible_fillers or t in ["OPEN", "1600"]:
            continue
            
        # Lookup pre-computed values
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
                display = ""  # Before trigger time
            else:
                display = "0.0%"  # After trigger time but no data
                
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

# --- Chart Layout ---
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
