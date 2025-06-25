import streamlit as st
import pandas as pd
import os

def process_summary_data():
    """
    Process the combined trigger/goal results into dashboard summary
    """
    try:
        st.write("📊 Loading combined trigger/goal results...")
        
        # Try different possible filenames
        possible_files = [
            "combined_trigger_goal_results.csv",
            "combined_trigger_goal_results 5.csv",
            "combined_trigger_goal_results_5.csv"
        ]
        
        df = None
        used_file = None
        
        for filename in possible_files:
            if os.path.exists(filename):
                df = pd.read_csv(filename)
                used_file = filename
                st.success(f"✅ Found and loaded: {filename}")
                break
        
        if df is None:
            st.error("❌ No combined results file found. Please ensure one of these files exists:")
            for filename in possible_files:
                st.write(f"  - {filename}")
            return None, []
        
        debug_info = []
        debug_info.append(f"Loaded file: {used_file}")
        debug_info.append(f"Total records: {len(df):,}")
        debug_info.append(f"Date range: {df['Date'].min()} to {df['Date'].max()}")
        debug_info.append(f"Columns: {list(df.columns)}")
        
        st.write(f"📈 Processing {len(df):,} trigger-goal records...")
        
        # Count unique triggers per day
        st.write("🔍 Counting unique triggers per day...")
        trigger_occurrences = df[['Date', 'TriggerLevel', 'TriggerTime', 'Direction']].drop_duplicates()
        
        # Simple time conversion function - FIXED VERSION
        def convert_time_to_hour(time_str):
            time_str = str(time_str).replace('.0', '')  # Remove .0 from floats
            if time_str == 'OPEN':
                return 'OPEN'
            elif time_str in ['930', '940', '950', '959']:  # Specific 9:30 hour times
                return '0900'
            elif time_str.startswith('10'):  # 1000-1059
                return '1000'
            elif time_str.startswith('11'):  # 1100-1159
                return '1100'
            elif time_str.startswith('12'):  # 1200-1259
                return '1200'
            elif time_str.startswith('13'):  # 1300-1359
                return '1300'
            elif time_str.startswith('14'):  # 1400-1459
                return '1400'
            elif time_str.startswith('15'):  # 1500-1559
                return '1500'
            else:
                return time_str  # Keep as-is if doesn't match
        
        # Convert time formats for app.py compatibility
        trigger_occurrences['TriggerTime'] = trigger_occurrences['TriggerTime'].astype(str)
        trigger_occurrences['TriggerTime'] = trigger_occurrences['TriggerTime'].apply(convert_time_to_hour)
        
        trigger_counts = (
            trigger_occurrences
            .value_counts(subset=['TriggerLevel', 'TriggerTime', 'Direction'])
            .reset_index()
        )
        
        trigger_counts.columns = ['TriggerLevel', 'TriggerTime', 'Direction', 'NumTriggers']
        debug_info.append(f"Unique trigger combinations: {len(trigger_counts):,}")
        
        # Count successful goal hits per group
        st.write("🎯 Counting successful goal hits...")
        goal_hits = df[df['GoalHit'] == 'Yes'].copy()
        debug_info.append(f"Total goal hits: {len(goal_hits):,}")
        
        # Convert time formats for both TriggerTime and GoalTime  
        goal_hits['TriggerTime'] = goal_hits['TriggerTime'].astype(str)
        goal_hits['GoalTime'] = goal_hits['GoalTime'].astype(str)
        
        # Apply the same time conversion function
        goal_hits['TriggerTime'] = goal_hits['TriggerTime'].apply(convert_time_to_hour)
        goal_hits['GoalTime'] = goal_hits['GoalTime'].apply(convert_time_to_hour)
        
        goal_counts = (
            goal_hits
            .groupby(['TriggerLevel', 'TriggerTime', 'Direction', 'GoalLevel', 'GoalTime'])
            .size()
            .reset_index(name='NumHits')
        )
        
        debug_info.append(f"Unique goal hit combinations: {len(goal_counts):,}")
        
        # Merge hits with trigger totals
        st.write("🔗 Merging trigger counts with goal hits...")
        summary = pd.merge(
            goal_counts,
            trigger_counts,
            on=['TriggerLevel', 'TriggerTime', 'Direction'],
            how='left'
        )
        
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
        
        debug_info.append(f"Final summary records: {len(summary):,}")
        
        # Basic statistics
        avg_completion = summary['PctCompletion'].mean()
        max_completion = summary['PctCompletion'].max()
        debug_info.append(f"Average completion rate: {avg_completion:.1f}%")
        debug_info.append(f"Maximum completion rate: {max_completion:.1f}%")
        
        st.success(f"✅ Summary processing complete! Generated {len(summary):,} summary records")
        
        return summary, debug_info
        
    except Exception as e:
        st.error(f"❌ Error processing data: {str(e)}")
        import traceback
        st.text(traceback.format_exc())
        return None, [f"Error: {str(e)}"]

# Streamlit App Interface
st.title('📊 ATR Dashboard Summary Generator - FIXED VERSION')
st.write('Processes trigger/goal results into dashboard-ready summary format (NO REGEX)')

if st.button('🚀 Generate Dashboard Summary'):
    with st.spinner('Processing trigger/goal data...'):
        summary_df, debug_messages = process_summary_data()
        
        # Show debug information
        with st.expander('📋 Processing Details'):
            for msg in debug_messages:
                st.write(msg)
        
        if summary_df is not None and not summary_df.empty:
            # Save the summary file
            output_filename = 'atr_dashboard_summary.csv'
            summary_df.to_csv(output_filename, index=False)
            
            st.success(f'✅ Dashboard summary saved as {output_filename}')
            
            # Show summary statistics
            st.subheader('📈 Summary Statistics')
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric('Total Combinations', f"{len(summary_df):,}")
            with col2:
                avg_completion = summary_df['PctCompletion'].mean()
                st.metric('Avg Completion', f"{avg_completion:.1f}%")
            with col3:
                total_triggers = summary_df['NumTriggers'].sum()
                st.metric('Total Triggers', f"{total_triggers:,}")
            with col4:
                total_hits = summary_df['NumHits'].sum()
                st.metric('Total Hits', f"{total_hits:,}")
            
            # Show breakdown by direction
            st.subheader('📊 Breakdown by Direction')
            direction_summary = summary_df.groupby('Direction').agg({
                'NumTriggers': 'sum',
                'NumHits': 'sum',
                'PctCompletion': 'mean'
            }).round(1)
            st.dataframe(direction_summary)
            
            # Show top performing combinations
            st.subheader('🏆 Top Performing Combinations (>80% completion)')
            top_performers = summary_df[summary_df['PctCompletion'] > 80].sort_values('PctCompletion', ascending=False)
            if not top_performers.empty:
                st.dataframe(top_performers.head(10))
            else:
                st.write("No combinations with >80% completion rate found")
            
            # Preview of summary data
            st.subheader('🔍 Preview of Summary Data')
            st.dataframe(summary_df.head(20))
            
            # Download button
            st.download_button(
                label='⬇️ Download Dashboard Summary CSV',
                data=summary_df.to_csv(index=False),
                file_name=output_filename,
                mime='text/csv'
            )
            
            st.info('💡 **Next Step:** Use this summary file with your ATR Dashboard (app.py) to visualize the trigger/goal matrix!')
            
        else:
            st.warning('⚠️ No summary data generated. Please check the processing details above.')

st.markdown("""
---
**📋 What This Does:**
1. 📊 **Loads** your trigger/goal results (110K+ records)
2. 🔍 **Counts** unique triggers per day
3. 🎯 **Aggregates** goal completion rates
4. 📈 **Calculates** percentage completion for each combination
5. 💾 **Saves** dashboard-ready summary file
6. ⬇️ **Provides** download button for remote access

**🎯 Output:** `atr_dashboard_summary.csv` ready for your dashboard visualization!

**✅ FIXED:** No regex patterns - uses simple string matching instead!
""")
