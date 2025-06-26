import streamlit as st
import pandas as pd
import numpy as np
import io

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

st.title("🔧 Fixed Summary Generator with Time Bucketing")
st.write("Converts raw trigger/goal data into dashboard-ready format with proper time bucketing")

uploaded_file = st.file_uploader("Upload combined_trigger_goal_results.csv", type="csv")

if uploaded_file is not None:
    # Load combined results
    df = pd.read_csv(uploaded_file)
    
    st.success(f"📊 Loaded {len(df)} total records")
    
    # Show sample of raw time data
    st.write("### Sample Raw Time Data:")
    sample_times = df[['TriggerTime', 'GoalTime']].head(10)
    st.dataframe(sample_times)
    
    # Apply time bucketing
    st.write("🕐 Applying time bucketing...")
    df['TriggerTimeBucket'] = df['TriggerTime'].apply(bucket_time)
    df['GoalTimeBucket'] = df['GoalTime'].apply(bucket_time)
    
    # Show sample of bucketed times
    st.write("### After Time Bucketing:")
    sample_bucketed = df[['TriggerTime', 'TriggerTimeBucket', 'GoalTime', 'GoalTimeBucket']].head(10)
    st.dataframe(sample_bucketed)
    
    # Count unique triggers per day (using bucketed times)
    trigger_occurrences = df[['Date', 'TriggerLevel', 'TriggerTimeBucket', 'Direction']].drop_duplicates()

    trigger_counts = (
        trigger_occurrences
        .value_counts(subset=['TriggerLevel', 'TriggerTimeBucket', 'Direction'])
        .reset_index()
    )
    trigger_counts.columns = ['TriggerLevel', 'TriggerTime', 'Direction', 'NumTriggers']
    
    st.write(f"✅ Found {len(trigger_counts)} unique trigger combinations")
    
    # Check 0.0 in trigger counts
    zero_triggers = trigger_counts[trigger_counts['TriggerLevel'] == 0.0]
    st.write(f"🎯 0.0 trigger combinations: {len(zero_triggers)}")
    
    if len(zero_triggers) > 0:
        st.write("### 0.0 Trigger Combinations:")
        st.dataframe(zero_triggers)

    # Count successful goal hits per group (using bucketed times)
    goal_hits = df[df['GoalHit'] == 'Yes']
    st.write(f"🎯 Total successful goal hits: {len(goal_hits)}")
    
    # Check 0.0 trigger successes
    zero_successes = goal_hits[goal_hits['TriggerLevel'] == 0.0]
    st.write(f"🎯 Successful hits FROM 0.0 triggers: {len(zero_successes)}")
    
    if len(zero_successes) > 0:
        st.write("### Sample 0.0 Trigger Successes:")
        st.dataframe(zero_successes[['Date', 'TriggerLevel', 'TriggerTimeBucket', 'GoalLevel', 'GoalTimeBucket']].head(10))

    goal_counts = (
        goal_hits
        .groupby(['TriggerLevel', 'TriggerTimeBucket', 'Direction', 'GoalLevel', 'GoalTimeBucket'])
        .size()
        .reset_index(name='NumHits')
    )
    goal_counts.columns = ['TriggerLevel', 'TriggerTime', 'Direction', 'GoalLevel', 'GoalTime', 'NumHits']
    
    st.write(f"✅ Found {len(goal_counts)} goal hit combinations")
    
    # Check 0.0 in goal counts  
    zero_goals = goal_counts[goal_counts['TriggerLevel'] == 0.0]
    st.write(f"🎯 0.0 goal hit combinations: {len(zero_goals)}")
    
    if len(zero_goals) > 0:
        st.write("### Sample 0.0 Goal Hit Combinations:")
        st.dataframe(zero_goals.head(10))

    # CORRECTED MERGE: Start with ALL triggers, add goal data where it exists
    summary = pd.merge(
        trigger_counts,   # ALL trigger combinations (including 0.0)
        goal_counts,      # Goal hit data
        on=['TriggerLevel', 'TriggerTime', 'Direction'],
        how='left'        # Keep ALL triggers, even with 0 hits
    )
    
    st.write(f"✅ Merged summary: {len(summary)} total combinations")
    
    # Fill missing values for combinations with no goal hits
    summary['NumHits'] = summary['NumHits'].fillna(0).astype(int)
    summary['GoalLevel'] = summary['GoalLevel'].fillna('No Goals Hit')
    summary['GoalTime'] = summary['GoalTime'].fillna('N/A')

    # Calculate % completion
    summary['PctCompletion'] = (summary['NumHits'] / summary['NumTriggers'] * 100).round(2)

    # Final column order
    summary = summary[[
        'Direction',
        'TriggerLevel', 
        'TriggerTime',
        'GoalLevel',
        'GoalTime',
        'NumTriggers',
        'NumHits',
        'PctCompletion'
    ]]
    
    # Check final result
    zero_in_final = summary[summary['TriggerLevel'] == 0.0]
    st.write(f"🎯 0.0 in final summary: {len(zero_in_final)} combinations")
    
    if len(zero_in_final) > 0:
        st.success("✅ SUCCESS: 0.0 level preserved in final summary!")
        st.write("### Sample 0.0 combinations in final summary:")
        
        # Show specifically OPEN triggers with actual hits
        zero_open_hits = zero_in_final[(zero_in_final['TriggerTime'] == 'OPEN') & (zero_in_final['NumHits'] > 0)]
        if len(zero_open_hits) > 0:
            st.write("🔥 **0.0 OPEN triggers with SUCCESSFUL hits:**")
            st.dataframe(zero_open_hits)
        else:
            st.write("Sample 0.0 combinations:")
            st.dataframe(zero_in_final.head(10))
    else:
        st.error("❌ PROBLEM: 0.0 level still missing from final summary")

    # Show unique trigger levels and times
    unique_levels = sorted([x for x in summary['TriggerLevel'].unique() if pd.notna(x)])
    unique_times = sorted([x for x in summary['TriggerTime'].unique() if pd.notna(x)])
    
    st.write(f"### Final Unique Trigger Levels ({len(unique_levels)}):")
    st.write(unique_levels)
    
    st.write(f"### Final Unique Times ({len(unique_times)}):")
    st.write(unique_times)
    
    # Highlight if 0.0 is included
    if 0.0 in unique_levels:
        st.success("✅ 0.0 level is included in final summary!")
    else:
        st.error("❌ 0.0 level is missing from final summary")

    # Show summary statistics
    st.write("### Summary Statistics:")
    total_hits = summary['NumHits'].sum()
    total_triggers = summary['NumTriggers'].sum()
    overall_pct = (total_hits / total_triggers * 100) if total_triggers > 0 else 0
    
    st.write(f"- **Total combinations**: {len(summary)}")
    st.write(f"- **Total triggers**: {total_triggers:,}")
    st.write(f"- **Total hits**: {total_hits:,}")
    st.write(f"- **Overall success rate**: {overall_pct:.2f}%")
    
    # Download button
    csv_buffer = io.StringIO()
    summary.to_csv(csv_buffer, index=False)
    
    st.download_button(
        label="📥 Download Fixed Summary CSV",
        data=csv_buffer.getvalue(),
        file_name="atr_dashboard_summary_fixed.csv", 
        mime="text/csv"
    )
    
    # Show sample of final data with successful hits
    successful_combos = summary[summary['NumHits'] > 0].head(20)
    if len(successful_combos) > 0:
        st.write("### Sample Successful Combinations:")
        st.dataframe(successful_combos)
    else:
        st.warning("No successful combinations found in sample data")

else:
    st.info("👆 Please upload your combined_trigger_goal_results.csv file to generate the fixed summary")
