import streamlit as st
import pandas as pd
import os

def analyze_dashboard_data():
    """
    Analyze the dashboard summary data to debug zero completion rates
    """
    try:
        # Check if summary file exists
        summary_file = "atr_dashboard_summary.csv"
        if not os.path.exists(summary_file):
            st.error(f"❌ {summary_file} not found!")
            return
        
        # Load summary data
        st.write("📊 Loading dashboard summary data...")
        df = pd.read_csv(summary_file)
        
        st.success(f"✅ Loaded {len(df):,} summary records")
        
        # Basic info
        st.subheader("📋 Basic Data Info")
        st.write(f"**Columns:** {list(df.columns)}")
        st.write(f"**Shape:** {df.shape}")
        
        # Check for zeros
        zero_pct = (df['PctCompletion'] == 0).sum()
        nonzero_pct = (df['PctCompletion'] > 0).sum()
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Zero Completion", zero_pct)
        with col2:
            st.metric("Non-Zero Completion", nonzero_pct)
        with col3:
            st.metric("Max Completion", f"{df['PctCompletion'].max():.1f}%")
        
        # Check data types
        st.subheader("🔍 Data Types")
        st.write(df.dtypes)
        
        # Check unique values for key fields
        st.subheader("📊 Unique Values Analysis")
        
        col1, col2 = st.columns(2)
        with col1:
            st.write("**Directions:**")
            st.write(df['Direction'].value_counts())
            
            st.write("**Trigger Levels:**")
            st.write(df['TriggerLevel'].value_counts().head(10))
        
        with col2:
            st.write("**Trigger Times:**")
            st.write(df['TriggerTime'].value_counts())
            
            st.write("**Goal Times:**")
            st.write(df['GoalTime'].value_counts().head(10))
        
        # Sample data with non-zero completion
        st.subheader("🎯 Non-Zero Completion Examples")
        nonzero_data = df[df['PctCompletion'] > 0].head(10)
        if not nonzero_data.empty:
            st.dataframe(nonzero_data)
        else:
            st.warning("⚠️ NO NON-ZERO COMPLETION RATES FOUND!")
        
        # Sample data with zero completion
        st.subheader("❌ Zero Completion Examples")
        zero_data = df[df['PctCompletion'] == 0].head(10)
        if not zero_data.empty:
            st.dataframe(zero_data)
        
        # Check specific filter combinations
        st.subheader("🔍 Filter Testing")
        st.write("Testing specific combinations that app.py might use...")
        
        # Test default filter from app.py (typically first direction, level 0.0, time OPEN)
        directions = sorted(df['Direction'].unique())
        levels = sorted(df['TriggerLevel'].unique())
        times = sorted(df['TriggerTime'].unique())
        
        if directions and levels and times:
            test_dir = directions[0]
            test_level = 0.0 if 0.0 in levels else levels[0]
            test_time = 'OPEN' if 'OPEN' in times else times[0]
            
            st.write(f"**Testing filter:** Direction={test_dir}, TriggerLevel={test_level}, TriggerTime={test_time}")
            
            filtered = df[
                (df['Direction'] == test_dir) &
                (df['TriggerLevel'] == test_level) &
                (df['TriggerTime'] == test_time)
            ]
            
            st.write(f"**Filtered results:** {len(filtered)} records")
            if not filtered.empty:
                st.dataframe(filtered.head())
            else:
                st.warning("⚠️ No records match this filter combination!")
        
        # Check for data calculation issues
        st.subheader("🧮 Calculation Check")
        calc_issues = df[df['NumHits'] > df['NumTriggers']]
        if not calc_issues.empty:
            st.error("❌ Found records where NumHits > NumTriggers!")
            st.dataframe(calc_issues)
        else:
            st.success("✅ No calculation issues found")
        
        # Manual percentage calculation check
        st.write("**Manual calculation check (first 5 records):**")
        for idx, row in df.head().iterrows():
            manual_pct = (row['NumHits'] / row['NumTriggers'] * 100) if row['NumTriggers'] > 0 else 0
            st.write(f"Row {idx}: {row['NumHits']}/{row['NumTriggers']} = {manual_pct:.1f}% (stored: {row['PctCompletion']}%)")
        
        return df
        
    except Exception as e:
        st.error(f"❌ Error analyzing data: {str(e)}")
        import traceback
        st.text(traceback.format_exc())
        return None

# Streamlit App
st.title('🔍 Dashboard Data Diagnostic Tool')
st.write('Debug why app.py is showing zero completion rates')

if st.button('🔍 Analyze Dashboard Data'):
    df = analyze_dashboard_data()
    
    if df is not None:
        st.subheader('💡 Recommendations')
        
        zero_count = (df['PctCompletion'] == 0).sum()
        total_count = len(df)
        
        if zero_count == total_count:
            st.error("🚨 ALL completion rates are zero!")
            st.write("**Possible causes:**")
            st.write("- Issue in run_summary processing logic")
            st.write("- NumHits or NumTriggers calculation error")
            st.write("- Data type conversion issues")
        elif zero_count > total_count * 0.8:
            st.warning("⚠️ Most completion rates are zero")
            st.write("**Possible causes:**")
            st.write("- Filtering logic issues")
            st.write("- Time format mismatches")
            st.write("- Level format mismatches")
        else:
            st.success("✅ Some non-zero completion rates found")
            st.write("Check the filtering logic in app.py")

st.markdown("""
---
**🎯 This tool checks:**
- Summary data structure and completeness
- Zero vs non-zero completion rates
- Data types and unique values
- Specific filter combinations
- Calculation accuracy
""")
