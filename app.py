import streamlit as st
import pandas as pd
import plotly.graph_objects as go

# --- Load and prepare data with CONVERSION FIX ---
st.title("📈 ATR Roadmap Matrix")

try:
    df = pd.read_csv("atr_dashboard_summary.csv")
    
    # FIX DATA TYPES AND FORMATS
    # Convert float times to proper string format
    df["TriggerTime"] = df["TriggerTime"].astype(str).str.replace('.0', '', regex=False)
    df["GoalTime"] = df["GoalTime"].astype(str).str.replace('.0', '', regex=False)
    
    # Convert time buckets to hour format
    def convert_to_hour_bucket(time_str):
        if time_str == 'OPEN':
            return 'OPEN'
        elif time_str in ['930', '940', '950', '959']:
            return '0900'
        elif time_str.startswith('10'):
            return '1000'
        elif time_str.startswith('11'):
            return '1100'
        elif time_str.startswith('12'):
            return '1200'
        elif time_str.startswith('13'):
            return '1300'
        elif time_str.startswith('14'):
            return '1400'
        elif time_str.startswith('15'):
            return '1500'
        else:
            return time_str
    
    df["TriggerTime"] = df["TriggerTime"].apply(convert_to_hour_bucket)
    df["GoalTime"] = df["GoalTime"].apply(convert_to_hour_bucket)
    
    # Re-aggregate after time conversion
    df_reaggregated = df.groupby(['Direction', 'TriggerLevel', 'TriggerTime', 'GoalLevel', 'GoalTime']).agg({
        'NumHits': 'sum',
        'NumTriggers': 'first',  # Should be same for all rows in group
        'PctCompletion': 'first'
    }).reset_index()
    
    # Recalculate percentages
    df_reaggregated['PctCompletion'] = (df_reaggregated['NumHits'] / df_reaggregated['NumTriggers'] * 100).round(1)
    
    df = df_reaggregated
    
    st.success(f"✅ Data loaded and fixed: {df.shape}")
    
except Exception as e:
    st.error(f"❌ Error loading data: {e}")
    st.stop()

# --- Time structure ---
visible_hours = ["0900", "1000", "1100", "1200", "1300", "1400", "1500"]
invisible_fillers = ["0930", "1030", "1130", "1230", "1330", "1430", "1530"]
time_order = ["OPEN"]
for hour in visible_hours:
    time_order.append(hour)
    filler = f"{str(int(hour[:2])+1).zfill(2)}30"
    time_order.append(filler)
time_order.append("1600")

# --- Fib levels ---
fib_levels = [1.0, 0.786, 0.618, 0.5, 0.382, 0.236, 0.0,
              -0.236, -0.382, -0.5, -0.618, -0.786, -1.0]

# --- UI layout with FIXED DEFAULTS ---
col1, col2, col3 = st.columns(3)

# Set up correct defaults
directions = sorted(df["Direction"].unique())
upside_index = directions.index("Upside") if "Upside" in directions else 0

trigger_levels_available = sorted(set(df["TriggerLevel"]).union(fib_levels))
zero_index = trigger_levels_available.index(0.0) if 0.0 in trigger_levels_available else 0

direction = col1.selectbox("Select Direction", directions, index=upside_index)
trigger_level = col2.selectbox("Select Trigger Level", trigger_levels_available, index=zero_index)
trigger_time = col3.selectbox("Select Trigger Time", ["OPEN"] + visible_hours, index=0)

# --- DEBUG INFO ---
st.write(f"**Selected:** {direction}, {trigger_level}, {trigger_time}")

# --- Filter for selection ---
filtered = df[
    (df["Direction"] == direction) &
    (df["TriggerLevel"] == trigger_level) &
    (df["TriggerTime"] == trigger_time)
].copy()

st.write(f"**Filtered records:** {len(filtered)}")

if len(filtered) == 0:
    st.error("❌ No data found for this combination!")
    
    # Show available combinations
    st.write("**Available combinations for debugging:**")
    available = df[df["Direction"] == direction][["TriggerLevel", "TriggerTime"]].drop_duplicates().head(10)
    st.dataframe(available)
    st.stop()

# --- Aggregate ---
grouped = (
    filtered.groupby(["GoalLevel", "GoalTime"])
    .agg(NumHits=("NumHits", "sum"), NumTriggers=("NumTriggers", "first"))
    .reset_index()
)
grouped["PctCompletion"] = (grouped["NumHits"] / grouped["NumTriggers"] * 100).round(1)

st.write(f"**Grouped records:** {len(grouped)}")
st.write(f"**Non-zero completions:** {len(grouped[grouped['PctCompletion'] > 0])}")

# Show sample of grouped data
with st.expander("📊 Sample Results"):
    st.dataframe(grouped.head(10))

# --- Plot setup ---
fig = go.Figure()

# --- Matrix cells ---
for level in fib_levels:
    for t in time_order:
        if t in invisible_fillers or t in ["OPEN", "1600"]:
            continue
        match = grouped[(grouped["GoalLevel"] == level) & (grouped["GoalTime"] == t)]
        if not match.empty:
            row = match.iloc[0]
            pct = row["PctCompletion"]
            hits = row["NumHits"]
            total = row["NumTriggers"]
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

# --- Horizontal lines ---
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

st.plotly_chart(fig, use_container_width=False)