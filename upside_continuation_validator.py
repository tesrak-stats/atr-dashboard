import streamlit as st
import pandas as pd
import numpy as np

st.title("🔍 Upside Continuation Validator")
st.write("Step-by-step validation of upside continuation logic against known Excel results")

uploaded_file = st.file_uploader("Upload combined_trigger_goal_results.csv", type="csv")

if uploaded_file is not None:
    # Load the raw data
    df = pd.read_csv(uploaded_file)
    st.success(f"📊 Loaded {len(df)} total records")
    
    # Step 1: Filter for Upside Continuation only
    st.write("## Step 1: Filter for Upside Continuation")
    
    upside_continuation = df[
        (df['Direction'] == 'Upside') & 
        (df['GoalClassification'] == 'Continuation')
    ].copy()
    
    st.write(f"✅ Found {len(upside_continuation)} Upside Continuation records")
    st.write(f"📊 That's {len(upside_continuation)/len(df)*100:.1f}% of total data")
    
    # Step 2: Examine the trigger levels
    st.write("## Step 2: Analyze Trigger Levels")
    
    trigger_summary = upside_continuation.groupby('TriggerLevel').size().reset_index(name='Count')
    trigger_summary = trigger_summary.sort_values('TriggerLevel')
    st.write("### Trigger Level Distribution:")
    st.dataframe(trigger_summary)
    
    # Focus on 0.0 level (Previous Close) - your main validation case
    st.write("## Step 3: Focus on 0.0 (Previous Close) Triggers")
    
    zero_triggers = upside_continuation[upside_continuation['TriggerLevel'] == 0.0].copy()
    st.write(f"🎯 Found {len(zero_triggers)} records with 0.0 triggers")
    
    if len(zero_triggers) > 0:
        # Check trigger times
        trigger_times = zero_triggers['TriggerTime'].value_counts()
        st.write("### 0.0 Trigger Time Distribution:")
        st.write(trigger_times)
        
        # Check goal levels for 0.0 triggers
        goal_levels = zero_triggers['GoalLevel'].value_counts().sort_index()
        st.write("### Goal Levels for 0.0 Triggers:")
        st.write(goal_levels)
        
        # Step 4: Validate against your Excel - 0.0 to 0.236
        st.write("## Step 4: Validate Key Scenario - 0.0 → 0.236")
        
        zero_to_236 = zero_triggers[zero_triggers['GoalLevel'] == 0.236].copy()
        st.write(f"📈 Found {len(zero_to_236)} records: 0.0 trigger → 0.236 goal")
        
        if len(zero_to_236) > 0:
            # Show sample records
            st.write("### Sample 0.0 → 0.236 Records:")
            st.dataframe(zero_to_236[['Date', 'TriggerTime', 'GoalTime', 'GoalHit']].head(10))
            
            # Calculate success rates by trigger time
            trigger_time_analysis = []
            
            for trigger_time in zero_to_236['TriggerTime'].unique():
                subset = zero_to_236[zero_to_236['TriggerTime'] == trigger_time]
                total_triggers = len(subset)
                successful = len(subset[subset['GoalHit'] == 'Yes'])
                success_rate = (successful / total_triggers * 100) if total_triggers > 0 else 0
                
                trigger_time_analysis.append({
                    'TriggerTime': trigger_time,
                    'TotalTriggers': total_triggers,
                    'Successful': successful,
                    'SuccessRate': round(success_rate, 2)
                })
            
            analysis_df = pd.DataFrame(trigger_time_analysis).sort_values('TriggerTime')
            st.write("### 0.0 → 0.236 Success Rates by Trigger Time:")
            st.dataframe(analysis_df)
            
            # Compare with your Excel results
            st.write("### 🔍 Validation Check:")
            open_result = analysis_df[analysis_df['TriggerTime'] == 'OPEN']
            if len(open_result) > 0:
                our_rate = open_result.iloc[0]['SuccessRate']
                our_triggers = open_result.iloc[0]['TotalTriggers']
                st.write(f"**Our calculation**: OPEN 0.0 → 0.236 = {our_rate}% ({our_triggers} triggers)")
                st.write(f"**Your Excel**: OPEN 0.0 → 0.236 = 83.81% (976 triggers)")
                
                if abs(our_rate - 83.81) < 5:  # Within 5%
                    st.success("✅ Results are close! Logic appears correct.")
                else:
                    st.error("❌ Significant difference - need to investigate logic")
            else:
                st.warning("⚠️ No OPEN trigger data found")
        
        # Step 5: Check for data quality issues
        st.write("## Step 5: Data Quality Checks")
        
        # Check for same-time scenarios (should be filtered out)
        same_time = zero_to_236[zero_to_236['TriggerTime'] == zero_to_236['GoalTime']]
        st.write(f"🕐 Same-time scenarios (should be 0): {len(same_time)}")
        
        # Check date range
        if len(zero_to_236) > 0:
            min_date = zero_to_236['Date'].min()
            max_date = zero_to_236['Date'].max()
            st.write(f"📅 Date range: {min_date} to {max_date}")
        
        # Check for missing goal times
        missing_goal_times = zero_to_236[zero_to_236['GoalTime'].isna()]
        st.write(f"❓ Missing goal times: {len(missing_goal_times)}")
        
    else:
        st.error("❌ No 0.0 trigger records found - check data generation logic")
    
    # Step 6: Generate focused validation CSV
    st.write("## Step 6: Generate Validation Dataset")
    
    if st.button("📥 Generate Upside Continuation Validation CSV"):
        # Create a focused dataset for manual validation
        validation_data = upside_continuation[
            upside_continuation['TriggerLevel'].isin([0.0, 0.236, 0.382]) &
            upside_continuation['GoalLevel'].isin([0.236, 0.382, 0.5, 0.618, 0.786, 1.0])
        ].copy()
        
        # Add helpful columns for validation
        validation_data['TriggerLevelName'] = validation_data['TriggerLevel'].map({
            0.0: 'Previous Close',
            0.236: '0.236 ATR',
            0.382: '0.382 ATR'
        })
        
        csv_data = validation_data.to_csv(index=False)
        st.download_button(
            label="📥 Download Validation Dataset",
            data=csv_data,
            file_name="upside_continuation_validation.csv",
            mime="text/csv"
        )
        
        st.write(f"✅ Created validation dataset with {len(validation_data)} records")
        st.write("This focused dataset contains only the scenarios you can verify against your Excel analysis")

else:
    st.info("👆 Upload your combined_trigger_goal_results.csv to begin validation")
