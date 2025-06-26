import streamlit as st
import pandas as pd
import io

st.title("ATR Debug Summary Generator")
st.write("Debug tool to trace where 0.0 trigger level gets lost in the pipeline")

# File upload
uploaded_file = st.file_uploader("Upload combined_trigger_goal_results.csv", type="csv")

if uploaded_file is not None:
    # Load data
    df = pd.read_csv(uploaded_file)
    
    st.success(f"📊 Loaded {len(df)} total records")
    
    # Debug output container
    debug_output = []
    
    # Check if 0.0 exists in raw data
    zero_triggers = df[df['TriggerLevel'] == 0.0]
    debug_output.append(f"🔍 Found {len(zero_triggers)} records with TriggerLevel = 0.0")
    
    if len(zero_triggers) > 0:
        st.write("### Sample 0.0 trigger records:")
        st.dataframe(zero_triggers.head())
    
    # Count unique triggers per day
    trigger_occurrences = df[['Date', 'TriggerLevel', 'TriggerTime', 'Direction']].drop_duplicates()
    debug_output.append(f"📈 Total unique trigger occurrences: {len(trigger_occurrences)}")
    
    # Check if 0.0 survives the deduplication
    zero_after_dedup = trigger_occurrences[trigger_occurrences['TriggerLevel'] == 0.0]
    debug_output.append(f"🔍 Zero triggers after deduplication: {len(zero_after_dedup)}")
    
    if len(zero_after_dedup) > 0:
        st.write("### 0.0 triggers after deduplication:")
        st.dataframe(zero_after_dedup)
    
    trigger_counts = (
        trigger_occurrences
        .value_counts(subset=['TriggerLevel', 'TriggerTime', 'Direction'])
        .reset_index()
    )
    
    trigger_counts.columns = ['TriggerLevel', 'TriggerTime', 'Direction', 'NumTriggers']
    
    # Check if 0.0 is in trigger_counts
    zero_in_counts = trigger_counts[trigger_counts['TriggerLevel'] == 0.0]
    debug_output.append(f"🔍 Zero triggers in trigger_counts: {len(zero_in_counts)}")
    
    if len(zero_in_counts) > 0:
        st.write("### Zero trigger counts:")
        st.dataframe(zero_in_counts)
    
    # Count successful goal hits per group
    goal_hits = df[df['GoalHit'] == 'Yes']
    debug_output.append(f"🎯 Total goal hits: {len(goal_hits)}")
    
    # Check if 0.0 triggers ever hit goals
    zero_goals = goal_hits[goal_hits['TriggerLevel'] == 0.0]
    debug_output.append(f"🔍 Goal hits FROM 0.0 triggers: {len(zero_goals)}")
    
    if len(zero_goals) > 0:
        st.write("### Goal hits from 0.0 triggers:")
        st.dataframe(zero_goals.head())
    else:
        st.warning("⚠️ No goal hits found from 0.0 triggers - this might be why 0.0 disappears!")
    
    goal_counts = (
        goal_hits
        .groupby(['TriggerLevel', 'TriggerTime', 'Direction', 'GoalLevel', 'GoalTime'])
        .size()
        .reset_index(name='NumHits')
    )
    
    # Check if 0.0 survives goal grouping
    zero_in_goal_counts = goal_counts[goal_counts['TriggerLevel'] == 0.0]
    debug_output.append(f"🔍 Zero triggers in goal_counts: {len(zero_in_goal_counts)}")
    
    # Merge hits with trigger totals
    summary = pd.merge(
        goal_counts,
        trigger_counts,
        on=['TriggerLevel', 'TriggerTime', 'Direction'],
        how='left'
    )
    
    debug_output.append(f"📋 Final summary records: {len(summary)}")
    
    # Check final result
    zero_in_summary = summary[summary['TriggerLevel'] == 0.0]
    debug_output.append(f"🔍 Zero triggers in final summary: {len(zero_in_summary)}")
    
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
    
    # Final check of unique trigger levels in summary
    unique_levels = sorted(summary['TriggerLevel'].unique())
    debug_output.append(f"🎯 Final unique trigger levels: {unique_levels}")
    
    # Display all debug output
    st.write("### Debug Trace:")
    for line in debug_output:
        st.write(line)
    
    # Show final summary stats
    st.write("### Final Summary Statistics:")
    st.write(f"- Total summary records: {len(summary)}")
    st.write(f"- Unique trigger levels: {len(unique_levels)}")
    st.write(f"- Contains 0.0 level: {'✅ Yes' if 0.0 in unique_levels else '❌ No'}")
    
    # Download processed summary
    csv_buffer = io.StringIO()
    summary.to_csv(csv_buffer, index=False)
    
    st.download_button(
        label="📥 Download Summary CSV",
        data=csv_buffer.getvalue(),
        file_name="atr_dashboard_summary.csv",
        mime="text/csv"
    )
    
    # Show sample of final data
    st.write("### Sample of Final Summary:")
    st.dataframe(summary.head(20))

else:
    st.info("👆 Please upload your combined_trigger_goal_results.csv file to begin analysis")
