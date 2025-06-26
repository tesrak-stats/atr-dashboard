import streamlit as st
import pandas as pd

st.title("🔧 Basic System Diagnostic")
st.write("Let's check if the fundamental data pipeline works AT ALL")

# Load the summary CSV
try:
    df = pd.read_csv("atr_dashboard_summary.csv")
    st.success(f"✅ Loaded CSV with {len(df)} rows")
    
    # Show basic info
    st.write("### CSV Columns:")
    st.write(list(df.columns))
    
    st.write("### First 5 rows:")
    st.dataframe(df.head())
    
    st.write("### Data Types:")
    st.write(df.dtypes)
    
    # Check for any non-zero percentages
    st.write("### Non-Zero Percentages Check:")
    non_zero = df[df['PctCompletion'] > 0]
    st.write(f"Rows with PctCompletion > 0: {len(non_zero)}")
    
    if len(non_zero) > 0:
        st.write("Sample non-zero rows:")
        st.dataframe(non_zero.head())
        
        # Test the original working aggregation approach
        st.write("### Testing Original Aggregation Approach:")
        
        # Pick a combination that has non-zero data
        test_row = non_zero.iloc[0]
        direction = test_row['Direction']
        trigger_level = test_row['TriggerLevel'] 
        trigger_time = test_row['TriggerTime']
        
        st.write(f"**Testing: {direction} | {trigger_level} | {trigger_time}**")
        
        # Filter like the original dashboard did
        filtered = df[
            (df["Direction"] == direction) &
            (df["TriggerLevel"] == trigger_level) &
            (df["TriggerTime"] == trigger_time)
        ]
        
        st.write(f"Filtered results: {len(filtered)} rows")
        st.dataframe(filtered)
        
        # Do the original aggregation
        if len(filtered) > 0:
            grouped = (
                filtered.groupby(["GoalLevel", "GoalTime"])
                .agg(NumHits=("NumHits", "sum"), NumTriggers=("NumTriggers", "first"))
                .reset_index()
            )
            grouped["PctCompletion"] = (grouped["NumHits"] / grouped["NumTriggers"] * 100).round(1)
            
            st.write("### After Original Aggregation:")
            st.dataframe(grouped)
            
            # Check if any are non-zero
            non_zero_grouped = grouped[grouped['PctCompletion'] > 0]
            st.write(f"Non-zero after aggregation: {len(non_zero_grouped)}")
            
            if len(non_zero_grouped) > 0:
                st.success("🎉 SUCCESS! Original aggregation approach works!")
            else:
                st.error("❌ Aggregation produces all zeros")
        
    else:
        st.error("❌ NO non-zero percentages found in entire CSV!")
        
        # Show some sample data anyway
        st.write("### Sample data (all zeros):")
        st.dataframe(df.head(10))
        
except Exception as e:
    st.error(f"❌ Failed to load CSV: {e}")
