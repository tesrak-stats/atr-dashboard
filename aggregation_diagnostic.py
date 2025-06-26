import streamlit as st
import pandas as pd
import numpy as np

def bucket_time(time_value):
    """Convert numeric times to hour buckets for dashboard display"""
    if pd.isna(time_value):
        return "Unknown"
    
    # Handle string times (like "OPEN")
    if isinstance(time_value, str):
        if time_value.upper() == "OPEN":
            return "OPEN"
        try:
            time_value = float(time_value)
        except:
            return str(time_value)
    
    # Convert numeric times to hour buckets
    if time_value < 930:
        return "OPEN"
    elif time_value < 1000:
        return "0900"
    elif time_value < 1100:
        return "1000"
    elif time_value < 1200:
        return "1100"
    elif time_value < 1300:
        return "1200"
    elif time_value < 1400:
        return "1300"
    elif time_value < 1500:
        return "1400"
    elif time_value < 1600:
        return "1500"
    else:
        return "1600"

st.title("🔍 Aggregation Diagnostic Tool")
st.write("Let's trace exactly what happens during the aggregation step")

uploaded_file = st.file_uploader("Upload combined_trigger_goal_results.csv", type="csv")

if uploaded_file is not None:
    # Load data
    df = pd.read_csv(uploaded_file)
    
    # Apply time bucketing
    df['TriggerTimeBucket'] = df['TriggerTime'].apply(bucket_time)
    df['GoalTimeBucket'] = df['GoalTime'].apply(bucket_time)
    
    st.write(f"📊 Total records: {len(df)}")
    
    # Focus on 0.0 OPEN Upside specifically
    test_case = df[
        (df['TriggerLevel'] == 0.0) & 
        (df['TriggerTimeBucket'] == 'OPEN') & 
        (df['Direction'] == 'Upside')
    ].copy()
    
    st.write(f"🎯 Test case (0.0 OPEN Upside): {len(test_case)} records")
    
    if len(test_case) > 0:
        st.write("### Sample Test Case Records:")
        st.dataframe(test_case[['Date', 'TriggerLevel', 'TriggerTimeBucket', 'GoalLevel', 'GoalTimeBucket', 'GoalHit']].head(10))
        
        # Check GoalHit values
        goal_hit_values = test_case['GoalHit'].value_counts()
        st.write("### GoalHit Value Counts:")
        st.write(goal_hit_values)
        
        # Filter for successful hits
        successes = test_case[test_case['GoalHit'] == 'Yes']
        st.write(f"🎯 Successful hits: {len(successes)}")
        
        if len(successes) > 0:
            st.write("### Sample Successful Hits:")
            st.dataframe(successes[['Date', 'GoalLevel', 'GoalTimeBucket', 'GoalHit']].head(10))
            
            # Group successful hits
            st.write("### Grouping Successful Hits by GoalLevel and GoalTimeBucket:")
            goal_groups = successes.groupby(['GoalLevel', 'GoalTimeBucket']).size().reset_index(name='Count')
            st.dataframe(goal_groups)
            
            # Now let's see what happens in trigger_counts
            st.write("### Trigger Counts Logic:")
            
            # Mimic the trigger counting logic
            trigger_occurrences = df[
                (df['TriggerLevel'] == 0.0) & 
                (df['TriggerTimeBucket'] == 'OPEN') & 
                (df['Direction'] == 'Upside')
            ][['Date', 'TriggerLevel', 'TriggerTimeBucket', 'Direction']].drop_duplicates()
            
            st.write(f"Unique trigger occurrences: {len(trigger_occurrences)}")
            
            trigger_count = trigger_occurrences.groupby(['TriggerLevel', 'TriggerTimeBucket', 'Direction']).size().iloc[0]
            st.write(f"**Trigger count for 0.0 OPEN Upside: {trigger_count}**")
            
            # Now test the merge logic
            st.write("### Testing Merge Logic:")
            
            # Create mini versions of trigger_counts and goal_counts
            mini_trigger = pd.DataFrame({
                'TriggerLevel': [0.0],
                'TriggerTime': ['OPEN'], 
                'Direction': ['Upside'],
                'NumTriggers': [trigger_count]
            })
            
            mini_goals = (
                successes
                .groupby(['TriggerLevel', 'TriggerTimeBucket', 'Direction', 'GoalLevel', 'GoalTimeBucket'])
                .size()
                .reset_index(name='NumHits')
            )
            mini_goals.columns = ['TriggerLevel', 'TriggerTime', 'Direction', 'GoalLevel', 'GoalTime', 'NumHits']
            
            st.write("**Mini trigger counts:**")
            st.dataframe(mini_trigger)
            
            st.write("**Mini goal counts:**")
            st.dataframe(mini_goals)
            
            # Test the merge
            test_merge = pd.merge(
                mini_trigger,
                mini_goals,
                on=['TriggerLevel', 'TriggerTime', 'Direction'],
                how='left'
            )
            
            st.write("**After merge:**")
            st.dataframe(test_merge)
            
            # Calculate percentages
            test_merge['NumHits'] = test_merge['NumHits'].fillna(0)
            test_merge['PctCompletion'] = (test_merge['NumHits'] / test_merge['NumTriggers'] * 100).round(2)
            
            st.write("**Final result with percentages:**")
            st.dataframe(test_merge)
            
            if test_merge['PctCompletion'].sum() > 0:
                st.success("🎉 SUCCESS! The aggregation logic works!")
            else:
                st.error("❌ Still getting 0.0% - there's a bug in the merge logic")
                
        else:
            st.error("❌ No successful hits found - check GoalHit filtering")
    else:
        st.error("❌ No test case records found")
        
        # Show what we do have
        st.write("### Available combinations:")
        available = df.groupby(['TriggerLevel', 'TriggerTimeBucket', 'Direction']).size().reset_index(name='Count')
        st.dataframe(available.head(20))

else:
    st.info("👆 Upload your combined_trigger_goal_results.csv to run diagnostics")
