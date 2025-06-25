import streamlit as st
import pandas as pd
import os

st.title('🔍 App.py Data Debug Tool')
st.write('Debug what data app.py is actually seeing')

# Check what files exist
st.subheader('📁 Available Files')
files = [f for f in os.listdir('.') if f.endswith('.csv')]
st.write(f"CSV files found: {files}")

# Try to load the summary file that app.py uses
summary_file = "atr_dashboard_summary.csv"
if os.path.exists(summary_file):
    st.success(f"✅ Found {summary_file}")
    
    # Load the data like app.py does
    df = pd.read_csv(summary_file)
    
    st.subheader('📊 Summary File Analysis')
    st.write(f"**Shape:** {df.shape}")
    st.write(f"**Columns:** {list(df.columns)}")
    
    # Check data types
    st.subheader('🔍 Data Types')
    st.write(df.dtypes)
    
    # Show sample data
    st.subheader('📋 Sample Data')
    st.dataframe(df.head(10))
    
    # Check unique values for filtering
    st.subheader('📊 Unique Values for Filtering')
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Directions:**")
        directions = df['Direction'].unique()
        st.write(directions)
        
        st.write("**TriggerTimes:**")
        trigger_times = df['TriggerTime'].unique()
        st.write(trigger_times)
    
    with col2:
        st.write("**TriggerLevels:**")
        trigger_levels = sorted(df['TriggerLevel'].unique())
        st.write(trigger_levels)
        
        st.write("**PctCompletion Range:**")
        st.write(f"Min: {df['PctCompletion'].min()}")
        st.write(f"Max: {df['PctCompletion'].max()}")
        st.write(f"Mean: {df['PctCompletion'].mean():.2f}")
    
    # Simulate app.py filtering
    st.subheader('🎯 Simulate App.py Default Filter')
    
    # Get default values (what app.py would select first)
    default_direction = sorted(df["Direction"].unique())[0] if len(df["Direction"].unique()) > 0 else None
    trigger_levels_available = sorted(set(df["TriggerLevel"]).union([1.0, 0.786, 0.618, 0.5, 0.382, 0.236, 0.0, -0.236, -0.382, -0.5, -0.618, -0.786, -1.0]))
    default_trigger_level = trigger_levels_available[trigger_levels_available.index(0.0)] if 0.0 in trigger_levels_available else trigger_levels_available[0]
    trigger_times_available = ["OPEN"] + [t for t in df["TriggerTime"].unique() if t != "OPEN"]
    default_trigger_time = "OPEN"
    
    st.write(f"**Default Direction:** {default_direction}")
    st.write(f"**Default TriggerLevel:** {default_trigger_level}")
    st.write(f"**Default TriggerTime:** {default_trigger_time}")
    
    # Apply the filter
    filtered = df[
        (df["Direction"] == default_direction) &
        (df["TriggerLevel"] == default_trigger_level) &
        (df["TriggerTime"] == default_trigger_time)
    ].copy()
    
    st.write(f"**Filtered Results:** {len(filtered)} records")
    
    if len(filtered) > 0:
        st.success("✅ Filter found matching records!")
        st.dataframe(filtered)
        
        # Check aggregation like app.py does
        grouped = (
            filtered.groupby(["GoalLevel", "GoalTime"])
            .agg(NumHits=("NumHits", "sum"), NumTriggers=("NumTriggers", "first"))
            .reset_index()
        )
        grouped["PctCompletion"] = (grouped["NumHits"] / grouped["NumTriggers"] * 100).round(1)
        
        st.subheader('📈 Aggregated Results (like app.py)')
        st.dataframe(grouped)
        
    else:
        st.error("❌ No records match the default filter!")
        st.write("**This is why app.py shows zeros!**")
        
        # Debug what values exist
        st.write("\n**Available combinations:**")
        sample_combinations = df[['Direction', 'TriggerLevel', 'TriggerTime']].drop_duplicates().head(10)
        st.dataframe(sample_combinations)
    
    # Test manual filter
    st.subheader('🧪 Test Manual Filter')
    
    col1, col2, col3 = st.columns(3)
    with col1:
        test_direction = st.selectbox("Test Direction", df['Direction'].unique())
    with col2:
        test_level = st.selectbox("Test TriggerLevel", sorted(df['TriggerLevel'].unique()))
    with col3:
        test_time = st.selectbox("Test TriggerTime", df['TriggerTime'].unique())
    
    if st.button('🔍 Test Filter'):
        test_filtered = df[
            (df["Direction"] == test_direction) &
            (df["TriggerLevel"] == test_level) &
            (df["TriggerTime"] == test_time)
        ]
        
        st.write(f"**Test Results:** {len(test_filtered)} records")
        if len(test_filtered) > 0:
            st.dataframe(test_filtered)
        else:
            st.warning("No matches found with this combination")

else:
    st.error(f"❌ {summary_file} not found!")
    st.write("Make sure you've run the summary generator and it created the file.")

st.markdown("""
---
**🎯 This will help identify:**
- File existence and structure
- Data type mismatches  
- Filtering logic issues
- Default value problems
""")
