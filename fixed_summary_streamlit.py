import streamlit as st
import pandas as pd
import io

st.title("Fixed ATR Summary Generator")
st.write("Corrected version that preserves all trigger levels including 0.0")

uploaded_file = st.file_uploader("Upload your trigger/goal data file", type=["csv", "xlsx", "xls"])

if uploaded_file is not None:
    # Load combined results - handle both CSV and Excel
    if uploaded_file.name.endswith('.csv'):
        df = pd.read_csv(uploaded_file)
    else:
        df = pd.read_excel(uploaded_file)
    
    st.success(f"📊 Loaded {len(df)} total records")
    
    # Count unique triggers per day
    trigger_occurrences = df[['Date', 'TriggerLevel', 'TriggerTime', 'Direction']].drop_duplicates()

    trigger_counts = (
        trigger_occurrences
        .value_counts(subset=['TriggerLevel', 'TriggerTime', 'Direction'])
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

    # Count successful goal hits per group
    goal_hits = df[df['GoalHit'] == 'Yes']

    goal_counts = (
        goal_hits
        .groupby(['TriggerLevel', 'TriggerTime', 'Direction', 'GoalLevel', 'GoalTime'])
        .size()
        .reset_index(name='NumHits')
    )
    
    st.write(f"✅ Found {len(goal_counts)} goal hit combinations")
    
    # Check 0.0 in goal counts  
    zero_goals = goal_counts[goal_counts['TriggerLevel'] == 0.0]
    st.write(f"🎯 0.0 goal hit combinations: {len(zero_goals)}")

    # FIXED MERGE: Start with ALL triggers, add goal data where it exists
    summary = pd.merge(
        trigger_counts,   # ALL trigger combinations (including 0.0)
        goal_counts,      # Goal hit data
        on=['TriggerLevel', 'TriggerTime', 'Direction'],
        how='left'        # Keep ALL triggers, even with 0 hits
    )
    
    st.write(f"✅ Merged summary: {len(summary)} total combinations")
    
    # Fill missing values for combinations with no goal hits
    summary['NumHits'] = summary['NumHits'].fillna(0)
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
        st.dataframe(zero_in_final.head(10))
    else:
        st.error("❌ PROBLEM: 0.0 level still missing from final summary")

    # Show unique trigger levels
    unique_levels = sorted(summary['TriggerLevel'].unique())
    st.write(f"### Final Unique Trigger Levels ({len(unique_levels)}):")
    st.write(unique_levels)
    
    # Highlight if 0.0 is included
    if 0.0 in unique_levels:
        st.success("✅ 0.0 level is included in final summary!")
    else:
        st.error("❌ 0.0 level is missing from final summary")

    # Download button
    csv_buffer = io.StringIO()
    summary.to_csv(csv_buffer, index=False)
    
    st.download_button(
        label="📥 Download Fixed Summary CSV",
        data=csv_buffer.getvalue(),
        file_name="atr_dashboard_summary_fixed.csv", 
        mime="text/csv"
    )
    
    # Show sample of final data
    st.write("### Sample of Final Summary Data:")
    st.dataframe(summary.head(20))

else:
    st.info("👆 Please upload your combined_trigger_goal_results.csv file to generate the fixed summary")
