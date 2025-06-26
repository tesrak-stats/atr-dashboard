import streamlit as st
import pandas as pd
import numpy as np

st.title("🔍 CSV Diagnostic Tool")
st.write("Let's examine exactly what's in your summary CSV file")

uploaded_file = st.file_uploader("Upload atr_dashboard_summary.csv", type="csv")

if uploaded_file is not None:
    # Read the CSV
    df = pd.read_csv(uploaded_file)
    
    st.write(f"📊 Total records: {len(df)}")
    st.write(f"📊 Columns: {list(df.columns)}")
    
    # Examine TriggerLevel column specifically
    st.write("### TriggerLevel Analysis")
    trigger_levels = df['TriggerLevel']
    st.write(f"Data type: {trigger_levels.dtype}")
    st.write(f"Unique values count: {len(trigger_levels.unique())}")
    
    # Show unique values
    unique_vals = sorted(trigger_levels.unique())
    st.write("### All Unique TriggerLevel Values:")
    for val in unique_vals:
        st.write(f"- {val} (type: {type(val).__name__})")
    
    # Check for 0.0 specifically
    st.write("### Zero Detection Tests:")
    
    # Test 1: Direct equality
    zero_exact = (trigger_levels == 0.0).sum()
    st.write(f"Records with TriggerLevel == 0.0: {zero_exact}")
    
    # Test 2: Integer zero
    zero_int = (trigger_levels == 0).sum()
    st.write(f"Records with TriggerLevel == 0: {zero_int}")
    
    # Test 3: String zero
    zero_str = (trigger_levels == '0.0').sum()
    st.write(f"Records with TriggerLevel == '0.0': {zero_str}")
    
    # Test 4: Near zero (floating point issues)
    zero_near = (np.abs(trigger_levels) < 0.0001).sum()
    st.write(f"Records with TriggerLevel near 0 (abs < 0.0001): {zero_near}")
    
    # Show sample records with zero-like values
    potential_zeros = df[np.abs(df['TriggerLevel']) < 0.0001]
    if len(potential_zeros) > 0:
        st.write("### Sample Zero Records:")
        st.dataframe(potential_zeros.head(10))
    else:
        st.error("❌ No zero records found at all!")
    
    # Check what pandas unique() actually returns
    st.write("### Raw Unique Values (as pandas sees them):")
    st.write(repr(trigger_levels.unique()))
    
    # Show first few rows of entire dataset
    st.write("### Sample Data:")
    st.dataframe(df.head(20))
