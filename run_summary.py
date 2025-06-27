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

st.title("🎯 Corrected Summary with Goal-Specific Denominators")
st.write("**Calculates proper denominators by removing OPEN completions per goal**")

uploaded_file = st.file_uploader("Upload combined_trigger_goal_results_CLEAN.csv", type="csv")

if uploaded_file is not None:
    # Load clean results
    df = pd.read_csv(uploaded_file)
    
    st.success(f"📊 Loaded {len(df)} clean records")
    
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
    
    # STEP 1: Count total triggers per trigger combination
    st.write("## Step 1: Count Total Triggers")
    
    trigger_events = df[['Date', 'TriggerLevel', 'TriggerTimeBucket', 'Direction']].drop_duplicates()
    total_trigger_counts = (
        trigger_events
        .groupby(['TriggerLevel', 'TriggerTimeBucket', 'Direction'])
        .size()
        .reset_index(name='TotalTriggers')
    )
    
    st.write(f"✅ Found {len(total_trigger_counts)} unique trigger combinations")
    st.write(f"✅ Total trigger events: {total_trigger_counts['TotalTriggers'].sum():,}")
    
    # STEP 2: Count OPEN completions per trigger-goal combination
    st.write("## Step 2: Count OPEN Completions per Goal")
    
    open_completions = df[
        (df['GoalHit'] == 'Yes') & 
        (df['GoalTimeBucket'] == 'OPEN')
    ].copy()
    
    open_completion_counts = (
        open_completions
        .groupby(['TriggerLevel', 'TriggerTimeBucket', 'Direction', 'GoalLevel'])
        .size()
        .reset_index(name='OpenCompletions')
    )
    
    st.write(f"✅ Found {len(open_completion_counts)} trigger-goal combinations with OPEN completions")
    st.write(f"✅ Total OPEN completions: {open_completion_counts['OpenCompletions'].sum():,}")
    
    # Show sample OPEN completions
    if len(open_completion_counts) > 0:
        st.write("### Sample OPEN Completions:")
        sample_open = open_completion_counts.head(10)
        st.dataframe(sample_open)
    
    # STEP 3: Count non-OPEN goal hits per trigger-goal-time combination
    st.write("## Step 3: Count Non-OPEN Goal Hits")
    
    non_open_hits = df[
        (df['GoalHit'] == 'Yes') & 
        (df['GoalTimeBucket'] != 'OPEN') &
        (df['GoalTimeBucket'] != 'N/A')
    ].copy()
    
    goal_hit_counts = (
        non_open_hits
        .groupby(['TriggerLevel', 'TriggerTimeBucket', 'Direction', 'GoalLevel', 'GoalTimeBucket'])
        .size()
        .reset_index(name='NonOpenHits')
    )
    
    st.write(f"✅ Found {len(goal_hit_counts)} non-OPEN goal hit combinations")
    st.write(f"✅ Total non-OPEN hits: {goal_hit_counts['NonOpenHits'].sum():,}")
    
    # STEP 4: Calculate goal-specific denominators
    st.write("## Step 4: Calculate Goal-Specific Denominators")
    
    # Create summary with goal-specific denominators
    summary_rows = []
    
    # Get all unique combinations
    all_triggers = total_trigger_counts[['TriggerLevel', 'TriggerTimeBucket', 'Direction']].drop_duplicates()
    all_goals = df['GoalLevel'].unique()
    all_goal_times = [t for t in df['GoalTimeBucket'].unique() if t not in ['OPEN', 'N/A']]
    
    for _, trigger_row in all_triggers.iterrows():
        trigger_level = trigger_row['TriggerLevel']
        trigger_time = trigger_row['TriggerTimeBucket']
        direction = trigger_row['Direction']
        
        # Get total triggers for this combination
        total_triggers = total_trigger_counts[
            (total_trigger_counts['TriggerLevel'] == trigger_level) &
            (total_trigger_counts['TriggerTimeBucket'] == trigger_time) &
            (total_trigger_counts['Direction'] == direction)
        ]['TotalTriggers'].iloc[0]
        
        for goal_level in all_goals:
            if goal_level == trigger_level:  # Skip same level
                continue
            
            # Get OPEN completions for this specific trigger-goal combination
            open_comps = open_completion_counts[
                (open_completion_counts['TriggerLevel'] == trigger_level) &
                (open_completion_counts['TriggerTimeBucket'] == trigger_time) &
                (open_completion_counts['Direction'] == direction) &
                (open_completion_counts['GoalLevel'] == goal_level)
            ]
            
            open_completions_count = open_comps['OpenCompletions'].iloc[0] if len(open_comps) > 0 else 0
            
            # Calculate goal-specific denominator
            actionable_triggers = total_triggers - open_completions_count
            
            for goal_time in all_goal_times:
                # Get non-OPEN hits for this combination
                hits = goal_hit_counts[
                    (goal_hit_counts['TriggerLevel'] == trigger_level) &
                    (goal_hit_counts['TriggerTimeBucket'] == trigger_time) &
                    (goal_hit_counts['Direction'] == direction) &
                    (goal_hit_counts['GoalLevel'] == goal_level) &
                    (goal_hit_counts['GoalTimeBucket'] == goal_time)
                ]
                
                num_hits = hits['NonOpenHits'].iloc[0] if len(hits) > 0 else 0
                
                # Calculate percentage with goal-specific denominator
                if actionable_triggers > 0:
                    pct_completion = (num_hits / actionable_triggers * 100)
                else:
                    pct_completion = 0.0
                
                summary_rows.append({
                    'Direction': direction,
                    'TriggerLevel': trigger_level,
                    'TriggerTime': trigger_time,
                    'GoalLevel': goal_level,
                    'GoalTime': goal_time,
                    'TotalTriggers': total_triggers,
                    'OpenCompletions': open_completions_count,
                    'ActionableTriggers': actionable_triggers,
                    'NumHits': num_hits,
                    'PctCompletion': round(pct_completion, 2)
                })
    
    summary = pd.DataFrame(summary_rows)
    
    # Remove combinations with 0 actionable triggers
    summary = summary[summary['ActionableTriggers'] > 0]
    
    st.write(f"✅ Complete summary: {len(summary)} combinations with goal-specific denominators")
    
    # Validation: Check for impossible percentages
    over_100 = summary[summary['PctCompletion'] > 100]
    if len(over_100) > 0:
        st.error(f"❌ Found {len(over_100)} combinations with >100% completion!")
        st.dataframe(over_100)
    else:
        st.success("✅ All completion rates ≤ 100% - logic is correct!")
    
    # Show validation example
    st.write("## Validation Example")
    st.write("**Compare the 0.382 OPEN → 0.5 vs 0.618 example:**")
    
    example_filter = (
        (summary['Direction'] == 'Upside') &
        (summary['TriggerLevel'] == 0.382) &
        (summary['TriggerTime'] == 'OPEN') &
        (summary['GoalTime'] == '0900') &
        (summary['GoalLevel'].isin([0.5, 0.618]))
    )
    
    example_data = summary[example_filter][['GoalLevel', 'TotalTriggers', 'OpenCompletions', 'ActionableTriggers', 'NumHits', 'PctCompletion']]
    
    if len(example_data) > 0:
        st.dataframe(example_data)
        st.write("**Key insight:** Different ActionableTriggers denominators explain the counter-intuitive percentages!")
    
    # Show top performing combinations
    st.write("### Top Performing Combinations:")
    top_performers = summary[summary['ActionableTriggers'] >= 20].nlargest(10, 'PctCompletion')
    st.dataframe(top_performers[['Direction', 'TriggerLevel', 'TriggerTime', 'GoalLevel', 'GoalTime', 'ActionableTriggers', 'NumHits', 'PctCompletion']])
    
    # Create final summary for dashboard (simplified columns)
    dashboard_summary = summary[['Direction', 'TriggerLevel', 'TriggerTime', 'GoalLevel', 'GoalTime', 'ActionableTriggers', 'NumHits', 'PctCompletion']].copy()
    dashboard_summary = dashboard_summary.rename(columns={'ActionableTriggers': 'NumTriggers'})
    
    # Save corrected summary
    csv_buffer = io.StringIO()
    dashboard_summary.to_csv(csv_buffer, index=False)
    
    st.download_button(
        label="📥 Download Corrected Summary CSV",
        data=csv_buffer.getvalue(),
        file_name="atr_dashboard_summary_CORRECTED.csv",
        mime="text/csv"
    )
    
    # Final statistics
    st.write("## Final Statistics")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Summary Records", len(summary))
    with col2:
        total_actionable = summary.groupby(['Direction', 'TriggerLevel', 'TriggerTime'])['ActionableTriggers'].first().sum()
        st.metric("Total Actionable Triggers", f"{total_actionable:,}")
    with col3:
        total_hits = summary['NumHits'].sum()
        st.metric("Total Non-OPEN Hits", f"{total_hits:,}")
    with col4:
        overall_rate = (total_hits / total_actionable * 100) if total_actionable > 0 else 0
        st.metric("Overall Actionable Rate", f"{overall_rate:.1f}%")
    
    st.success("🎉 **Corrected summary complete!** Each goal now has its own proper denominator.")
    st.write("**Key improvement:** Denominators are now goal-specific, accounting for different OPEN completion rates per goal.")

else:
    st.info("👆 Upload your clean trigger-goal results CSV to generate corrected summary")
