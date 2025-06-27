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

st.title("🔧 Simple Summary Generator for Clean Data")
st.write("**For use with the new clean trigger data - no complex adjustments needed**")

uploaded_file = st.file_uploader("Upload combined_trigger_goal_results_CLEAN.csv", type="csv")

if uploaded_file is not None:
    # Load clean results
    df = pd.read_csv(uploaded_file)
    
    st.success(f"📊 Loaded {len(df)} clean records")
    
    # Verify this is clean data
    same_time_count = len(df[df['SameTime'] == True]) if 'SameTime' in df.columns else 0
    if same_time_count > 0:
        st.warning(f"⚠️ Found {same_time_count} same-time records - this may not be clean data!")
    else:
        st.success("✅ Confirmed clean data - no same-time scenarios")
    
    # Apply time bucketing
    st.write("🕐 Applying time bucketing...")
    df['TriggerTimeBucket'] = df['TriggerTime'].apply(bucket_time)
    df['GoalTimeBucket'] = df['GoalTime'].apply(lambda x: bucket_time(x) if pd.notna(x) and x != '' else 'N/A')
    
    # Show basic stats
    st.write("## Basic Statistics")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Records", len(df))
    with col2:
        st.metric("Unique Dates", df['Date'].nunique())
    with col3:
        total_hits = len(df[df['GoalHit'] == 'Yes'])
        st.metric("Total Goal Hits", total_hits)
    with col4:
        hit_rate = (total_hits / len(df) * 100) if len(df) > 0 else 0
        st.metric("Overall Hit Rate", f"{hit_rate:.1f}%")
    
    # STEP 1: Count unique triggers per trigger combination
    st.write("## Step 1: Count Unique Triggers")
    
    # Get unique trigger events (one per date/trigger combination)
    trigger_events = df[['Date', 'TriggerLevel', 'TriggerTimeBucket', 'Direction']].drop_duplicates()
    
    trigger_counts = (
        trigger_events
        .groupby(['TriggerLevel', 'TriggerTimeBucket', 'Direction'])
        .size()
        .reset_index(name='NumTriggers')
    )
    trigger_counts.columns = ['TriggerLevel', 'TriggerTime', 'Direction', 'NumTriggers']
    
    st.write(f"✅ Found {len(trigger_counts)} unique trigger combinations")
    st.write(f"✅ Total trigger events: {trigger_counts['NumTriggers'].sum():,}")
    
    # STEP 2: Count goal hits per trigger-goal-time combination
    st.write("## Step 2: Count Goal Hits")
    
    goal_hits = df[df['GoalHit'] == 'Yes'].copy()
    
    goal_counts = (
        goal_hits
        .groupby(['TriggerLevel', 'TriggerTimeBucket', 'Direction', 'GoalLevel', 'GoalTimeBucket'])
        .size()
        .reset_index(name='NumHits')
    )
    goal_counts.columns = ['TriggerLevel', 'TriggerTime', 'Direction', 'GoalLevel', 'GoalTime', 'NumHits']
    
    st.write(f"✅ Found {len(goal_counts)} goal hit combinations")
    st.write(f"✅ Total goal hits: {goal_counts['NumHits'].sum():,}")
    
    # STEP 3: Create complete summary with all combinations
    st.write("## Step 3: Create Complete Summary")
    
    # Start with trigger counts as base
    summary_base = trigger_counts.copy()
    
    # Get all possible goals from the data
    all_goals = df['GoalLevel'].unique()
    all_goal_times = df['GoalTimeBucket'].unique()
    all_goal_times = [t for t in all_goal_times if t != 'N/A']  # Remove N/A times
    
    # Create complete summary with all trigger-goal-time combinations
    summary_rows = []
    
    for _, trigger_row in summary_base.iterrows():
        trigger_level = trigger_row['TriggerLevel']
        trigger_time = trigger_row['TriggerTime']
        direction = trigger_row['Direction']
        num_triggers = trigger_row['NumTriggers']
        
        for goal_level in all_goals:
            if goal_level == trigger_level:  # Skip same level
                continue
                
            for goal_time in all_goal_times:
                # Look for hits for this combination
                hit_record = goal_counts[
                    (goal_counts['TriggerLevel'] == trigger_level) &
                    (goal_counts['TriggerTime'] == trigger_time) &
                    (goal_counts['Direction'] == direction) &
                    (goal_counts['GoalLevel'] == goal_level) &
                    (goal_counts['GoalTime'] == goal_time)
                ]
                
                num_hits = hit_record['NumHits'].iloc[0] if len(hit_record) > 0 else 0
                
                # Calculate percentage
                pct_completion = (num_hits / num_triggers * 100) if num_triggers > 0 else 0.0
                
                summary_rows.append({
                    'Direction': direction,
                    'TriggerLevel': trigger_level,
                    'TriggerTime': trigger_time,
                    'GoalLevel': goal_level,
                    'GoalTime': goal_time,
                    'NumTriggers': num_triggers,
                    'NumHits': num_hits,
                    'PctCompletion': round(pct_completion, 2)
                })
    
    summary = pd.DataFrame(summary_rows)
    
    # Remove combinations with 0 hits (optional - keeps file smaller)
    # summary = summary[summary['NumHits'] > 0]  # Uncomment to remove zero-hit combinations
    
    st.write(f"✅ Complete summary: {len(summary)} combinations")
    
    # Show validation examples
    st.write("## Validation Examples")
    
    # Show some high-percentage examples
    high_success = summary[summary['PctCompletion'] > 50].head(10)
    if len(high_success) > 0:
        st.write("### High Success Rate Examples:")
        st.dataframe(high_success[['Direction', 'TriggerLevel', 'TriggerTime', 'GoalLevel', 'GoalTime', 'NumTriggers', 'NumHits', 'PctCompletion']])
    
    # Check for any >100% (shouldn't exist with clean data)
    over_100 = summary[summary['PctCompletion'] > 100]
    if len(over_100) > 0:
        st.error(f"❌ Found {len(over_100)} combinations with >100% completion - clean data issue!")
        st.dataframe(over_100)
    else:
        st.success("✅ All completion rates ≤ 100% - clean data confirmed!")
    
    # Show breakdown by trigger type
    st.write("### Summary by Trigger Type:")
    trigger_summary = (
        summary.groupby(['Direction', 'TriggerTime'])
        .agg({
            'NumTriggers': 'first',  # Same for all goals of same trigger
            'NumHits': 'sum',
            'PctCompletion': 'mean'
        })
        .reset_index()
    )
    st.dataframe(trigger_summary)
    
    # Save summary
    csv_buffer = io.StringIO()
    summary.to_csv(csv_buffer, index=False)
    
    st.download_button(
        label="📥 Download Clean Summary CSV",
        data=csv_buffer.getvalue(),
        file_name="atr_dashboard_summary_SIMPLE.csv",
        mime="text/csv"
    )
    
    # Final statistics
    st.write("## Final Statistics")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Summary Records", len(summary))
    with col2:
        unique_triggers = summary.groupby(['Direction', 'TriggerLevel', 'TriggerTime'])['NumTriggers'].first().sum()
        st.metric("Total Triggers", f"{unique_triggers:,}")
    with col3:
        total_hits = summary['NumHits'].sum()
        st.metric("Total Hits", f"{total_hits:,}")
    with col4:
        overall_rate = (total_hits / unique_triggers * 100) if unique_triggers > 0 else 0
        st.metric("Overall Success Rate", f"{overall_rate:.1f}%")
    
    st.success("🎉 **Simple summary complete!** Ready for dashboard consumption.")

else:
    st.info("👆 Upload your clean trigger-goal results CSV to generate simple summary")
