import streamlit as st
import pandas as pd
import plotly.graph_objects as go

# --- Load and prepare data ---
st.title("🔍 ATR Dashboard Debug Mode")

try:
    df = pd.read_csv("atr_dashboard_summary.csv")
    st.success(f"✅ Loaded data: {df.shape}")
    
    df["TriggerTime"] = df["TriggerTime"].astype(str)
    df["GoalTime"] = df["GoalTime"].astype(str)
    
    # Show data structure
    with st.expander("📊 Data Structure"):
        st.write(f"**Columns:** {list(df.columns)}")
        st.write(f"**Data types:** {df.dtypes.to_dict()}")
        st.dataframe(df.head())
    
    # Show unique values
    with st.expander("📋 Unique Values"):
        col1, col2, col3 = st.columns(3)
        with col1:
            st.write("**Directions:**")
            directions = sorted(df["Direction"].unique())
            st.write(directions)
        with col2:
            st.write("**TriggerLevels:**")
            trigger_levels = sorted(df["TriggerLevel"].unique())
            st.write(trigger_levels[:10])  # Show first 10
        with col3:
            st.write("**TriggerTimes:**")
            trigger_times = sorted(df["TriggerTime"].unique())
            st.write(trigger_times)
    
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

# --- UI layout ---
st.subheader("🎯 Filter Controls")
col1, col2, col3 = st.columns(3)

# Set up defaults with debugging
directions = sorted(df["Direction"].unique())
upside_index = directions.index("Upside") if "Upside" in directions else 0
st.write(f"**Debug:** Upside index = {upside_index}, Available directions = {directions}")

trigger_levels_available = sorted(set(df["TriggerLevel"]).union(fib_levels))
zero_index = trigger_levels_available.index(0.0) if 0.0 in trigger_levels_available else 0
st.write(f"**Debug:** Zero index = {zero_index}, 0.0 in data = {0.0 in df['TriggerLevel'].values}")

direction = col1.selectbox("Select Direction", directions, index=upside_index)
trigger_level = col2.selectbox("Select Trigger Level", trigger_levels_available, index=zero_index)
trigger_time = col3.selectbox("Select Trigger Time", ["OPEN"] + visible_hours, index=0)

st.write(f"**Selected:** Direction={direction}, TriggerLevel={trigger_level}, TriggerTime={trigger_time}")

# --- Filter for selection with debugging ---
st.subheader("🔍 Filtering Debug")

# Show step-by-step filtering
st.write("**Step 1: Direction filter**")
direction_filtered = df[df["Direction"] == direction]
st.write(f"Records after direction filter: {len(direction_filtered)}")

st.write("**Step 2: TriggerLevel filter**")
level_filtered = direction_filtered[direction_filtered["TriggerLevel"] == trigger_level]
st.write(f"Records after trigger level filter: {len(level_filtered)}")

st.write("**Step 3: TriggerTime filter**")
time_filtered = level_filtered[level_filtered["TriggerTime"] == trigger_time]
st.write(f"Records after trigger time filter: {len(time_filtered)}")

if len(time_filtered) == 0:
    st.error("❌ No records found after filtering!")
    
    # Show what combinations DO exist
    st.write("**Available combinations for this direction and level:**")
    available_for_dir_level = direction_filtered[direction_filtered["TriggerLevel"] == trigger_level]["TriggerTime"].unique()
    st.write(f"Available TriggerTimes: {available_for_dir_level}")
    
    # Show available combinations for this direction
    st.write("**All combinations for this direction:**")
    dir_combinations = direction_filtered[["TriggerLevel", "TriggerTime"]].drop_duplicates().head(10)
    st.dataframe(dir_combinations)
    
else:
    st.success(f"✅ Found {len(time_filtered)} records after filtering")
    
    # Show the filtered data
    with st.expander("📋 Filtered Data"):
        st.dataframe(time_filtered)

# --- Continue with aggregation if we have data ---
if len(time_filtered) > 0:
    st.subheader("📊 Aggregation Debug")
    
    # Show aggregation step
    st.write("**Grouping by GoalLevel and GoalTime:**")
    grouped = (
        time_filtered.groupby(["GoalLevel", "GoalTime"])
        .agg(NumHits=("NumHits", "sum"), NumTriggers=("NumTriggers", "first"))
        .reset_index()
    )
    
    st.write(f"Grouped records: {len(grouped)}")
    
    # Calculate percentages
    grouped["PctCompletion"] = (grouped["NumHits"] / grouped["NumTriggers"] * 100).round(1)
    
    # Show results
    st.write("**Aggregated Results:**")
    st.dataframe(grouped)
    
    # Show non-zero results
    nonzero_results = grouped[grouped["PctCompletion"] > 0]
    st.write(f"**Non-zero completion rates:** {len(nonzero_results)} out of {len(grouped)}")
    
    if len(nonzero_results) > 0:
        st.success("✅ Found non-zero completion rates!")
        st.dataframe(nonzero_results)
    else:
        st.warning("⚠️ All completion rates are zero - check NumHits and NumTriggers values")

# --- Show matrix visualization logic ---
st.subheader("🎨 Matrix Visualization Debug")

if len(time_filtered) > 0:
    # Test a few specific matrix cells
    st.write("**Testing specific matrix cells:**")
    
    for test_level in [0.236, 0.382, 0.5]:
        for test_time in ["0900", "1000", "1100"]:
            match = grouped[(grouped["GoalLevel"] == test_level) & (grouped["GoalTime"] == test_time)]
            if not match.empty:
                row = match.iloc[0]
                st.write(f"Level {test_level}, Time {test_time}: {row['PctCompletion']:.1f}% ({row['NumHits']}/{row['NumTriggers']})")
            else:
                st.write(f"Level {test_level}, Time {test_time}: No data")

st.markdown("""
---
**🎯 This debug version shows:**
- Data loading and structure
- Filter step-by-step results
- Aggregation process
- Matrix cell values
- Exactly where the process fails
""")
