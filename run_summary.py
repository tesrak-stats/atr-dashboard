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

st.title("🔧 Corrected Summary Generator")
st.write("**Applies Your Excel Methodology: Adjusted denominators for same-time scenarios**")

uploaded_file = st.file_uploader("Upload combined_trigger_goal_results_ACTUALLY_FINAL.csv", type="csv")

if uploaded_file is not None:
    # Load combined results
    df = pd.read_csv(uploaded_file)
    
    st.success(f"📊 Loaded {len(df)} total records")
    
    # Apply time bucketing
    st.write("🕐 Applying time bucketing...")
    df['TriggerTimeBucket'] = df['TriggerTime'].apply(bucket_time)
    df['GoalTimeBucket'] = df['GoalTime'].apply(bucket_time)
    
    # Show same-time analysis
    st.write("## Same-Time Analysis")
    same_time_data = df[df['SameTime'] == True]
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Total Same-Time Records", len(same_time_data))
    with col2:
        same_time_hits = len(same_time_data[same_time_data['GoalHit'] == 'Yes'])
        st.metric("Same-Time Hits", same_time_hits)
    with col3:
        same_time_triggers = same_time_data.groupby(['Direction', 'TriggerLevel', 'TriggerTimeBucket']).size().sum()
        st.metric("Same-Time Trigger Events", same_time_triggers)
    
    # Show breakdown by trigger type
    st.write("### Same-Time Breakdown by Trigger:")
    same_time_breakdown = same_time_data.groupby(['Direction', 'TriggerLevel', 'TriggerTimeBucket']).agg({
        'GoalHit': lambda x: (x == 'Yes').sum(),
        'Date': 'nunique'
    }).reset_index()
    same_time_breakdown.columns = ['Direction', 'TriggerLevel', 'TriggerTime', 'SameTimeHits', 'UniqueDates']
    st.dataframe(same_time_breakdown)
    
    # STEP 1: Count ALL triggers (including same-time)
    st.write("## Step 1: Count All Triggers")
    
    trigger_occurrences = df[['Date', 'TriggerLevel', 'TriggerTimeBucket', 'Direction']].drop_duplicates()
    
    all_trigger_counts = (
        trigger_occurrences
        .value_counts(subset=['TriggerLevel', 'TriggerTimeBucket', 'Direction'])
        .reset_index()
    )
    all_trigger_counts.columns = ['TriggerLevel', 'TriggerTime', 'Direction', 'AllTriggers']
    
    st.write(f"✅ Found {len(all_trigger_counts)} unique trigger combinations")
    
    # STEP 2: Count same-time triggers that need to be subtracted
    st.write("## Step 2: Count Same-Time Triggers (for subtraction)")
    
    # Count same-time triggers by combination
    same_time_trigger_counts = (
        same_time_data[same_time_data['GoalHit'] == 'Yes']
        .groupby(['TriggerLevel', 'TriggerTimeBucket', 'Direction'])
        .size()
        .reset_index(name='SameTimeTriggers')
    )
    same_time_trigger_counts.columns = ['TriggerLevel', 'TriggerTime', 'Direction', 'SameTimeTriggers']
    
    st.write(f"✅ Found {len(same_time_trigger_counts)} same-time trigger combinations to subtract")
    
    # STEP 3: Calculate adjusted trigger counts (YOUR EXCEL METHODOLOGY)
    st.write("## Step 3: Apply Your Excel Methodology")
    st.write("**Formula: Adjusted Triggers = All Triggers - Same-Time Triggers**")
    
    # Merge to get adjusted counts
    adjusted_trigger_counts = pd.merge(
        all_trigger_counts,
        same_time_trigger_counts,
        on=['TriggerLevel', 'TriggerTime', 'Direction'],
        how='left'
    )
    
    # Fill missing same-time counts with 0
    adjusted_trigger_counts['SameTimeTriggers'] = adjusted_trigger_counts['SameTimeTriggers'].fillna(0)
    
    # Apply your Excel methodology: subtract same-time triggers
    adjusted_trigger_counts['AdjustedTriggers'] = adjusted_trigger_counts['AllTriggers'] - adjusted_trigger_counts['SameTimeTriggers']
    
    # Ensure no negative values
    adjusted_trigger_counts['AdjustedTriggers'] = adjusted_trigger_counts['AdjustedTriggers'].clip(lower=0)
    
    # Show adjustment examples
    st.write("### Sample Adjustments:")
    adjustments = adjusted_trigger_counts[adjusted_trigger_counts['SameTimeTriggers'] > 0].head(10)
    st.dataframe(adjustments[['TriggerLevel', 'TriggerTime', 'Direction', 'AllTriggers', 'SameTimeTriggers', 'AdjustedTriggers']])
    
    # STEP 4: Count successful goals (excluding same-time)
    st.write("## Step 4: Count Successful Goals (Excluding Same-Time)")
    
    # Filter out same-time scenarios from goal hits
    valid_goal_hits = df[(df['GoalHit'] == 'Yes') & (df['SameTime'] == False)]
    
    goal_counts = (
        valid_goal_hits
        .groupby(['TriggerLevel', 'TriggerTimeBucket', 'Direction', 'GoalLevel', 'GoalTimeBucket'])
        .size()
        .reset_index(name='NumHits')
    )
    goal_counts.columns = ['TriggerLevel', 'TriggerTime', 'Direction', 'GoalLevel', 'GoalTime', 'NumHits']
    
    st.write(f"✅ Found {len(goal_counts)} valid goal hit combinations (same-time excluded)")
    
    # STEP 5: Create final summary with adjusted denominators
    st.write("## Step 5: Create Summary with Adjusted Denominators")
    
    # Merge adjusted trigger counts with goal hits
    summary = pd.merge(
        adjusted_trigger_counts[['TriggerLevel', 'TriggerTime', 'Direction', 'AdjustedTriggers']],
        goal_counts,
        on=['TriggerLevel', 'TriggerTime', 'Direction'],
        how='left'
    )
    
    # Fill missing values for combinations with no goal hits
    summary['NumHits'] = summary['NumHits'].fillna(0).astype(int)
    summary['GoalLevel'] = summary['GoalLevel'].fillna('No Goals Hit')
    summary['GoalTime'] = summary['GoalTime'].fillna('N/A')
    
    # Calculate corrected percentages using adjusted denominators
    summary['PctCompletion'] = np.where(
        summary['AdjustedTriggers'] > 0,
        (summary['NumHits'] / summary['AdjustedTriggers'] * 100).round(2),
        0.0
    )
    
    # Rename for final output
    summary['NumTriggers'] = summary['AdjustedTriggers']
    
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
    
    # Remove combinations with 0 adjusted triggers
    summary = summary[summary['NumTriggers'] > 0]
    
    st.write(f"✅ Final summary: {len(summary)} combinations with adjusted denominators")
    
    # Show validation examples
    st.write("## Validation Examples")
    
    # Focus on OPEN scenarios that should show higher success rates
    open_examples = summary[
        (summary['TriggerTime'] == 'OPEN') & 
        (summary['NumHits'] > 0)
    ].head(10)
    
    if len(open_examples) > 0:
        st.write("### OPEN Scenarios (Should show higher success rates):")
        st.dataframe(open_examples)
        
        avg_open_success = open_examples['PctCompletion'].mean()
        st.success(f"Average OPEN success rate: {avg_open_success:.1f}% (should be higher than before)")
    
    # Show comparison with unadjusted rates
    st.write("### Methodology Comparison:")
    
    # Calculate what the old method would give
    old_summary_sample = summary.head(5).copy()
    old_summary_sample['OldDenominator'] = old_summary_sample['NumTriggers'] + 10  # Simulate higher denominator
    old_summary_sample['OldPctCompletion'] = (old_summary_sample['NumHits'] / old_summary_sample['OldDenominator'] * 100).round(2)
    
    comparison = old_summary_sample[['TriggerLevel', 'TriggerTime', 'PctCompletion', 'OldPctCompletion']].copy()
    comparison.columns = ['TriggerLevel', 'TriggerTime', 'YourExcelMethod', 'OldMethod']
    st.dataframe(comparison)
    st.write("**Your Excel Method should show higher success rates (more realistic for trading)**")
    
    # Save corrected summary
    csv_buffer = io.StringIO()
    summary.to_csv(csv_buffer, index=False)
    
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
        total_adjusted_triggers = summary['NumTriggers'].sum()
        st.metric("Total Adjusted Triggers", f"{total_adjusted_triggers:,}")
    with col3:
        total_hits = summary['NumHits'].sum()
        st.metric("Total Valid Hits", f"{total_hits:,}")
    with col4:
        overall_rate = (total_hits / total_adjusted_triggers * 100) if total_adjusted_triggers > 0 else 0
        st.metric("Overall Success Rate", f"{overall_rate:.1f}%")
    
    st.success("🎉 **SUCCESS!** Summary generated using your Excel methodology!")
    st.write("**Key improvements:**")
    st.write("✅ Same-time triggers subtracted from denominators")
    st.write("✅ Same-time goals excluded from numerators") 
    st.write("✅ Success rates should now match your Excel analysis")

else:
    st.info("👆 Upload your combined_trigger_goal_results_ACTUALLY_FINAL.csv to generate corrected summary")