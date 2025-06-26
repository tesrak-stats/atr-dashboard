import streamlit as st
import pandas as pd
import os

def process_summary_data():
    """
    Process trigger/goal results with FORCED time conversion to hour buckets
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
            st.error("❌ No combined results file found.")
            return None, []
        
        debug_info = []
        debug_info.append(f"Loaded file: {used_file}")
        debug_info.append(f"Total records: {len(df):,}")
        
        # FORCED TIME CONVERSION - Multiple approaches
        def force_time_conversion(time_val):
            """Convert any time format to hour bucket - AGGRESSIVE approach"""
            
            # Convert to string and clean up
            time_str = str(time_val).replace('.0', '').replace('nan', '').strip()
            
            # Handle OPEN specially
            if time_str.upper() == 'OPEN':
                return 'OPEN'
            
            # Handle empty/null values
            if not time_str or time_str == '' or time_str == 'nan':
                return 'UNKNOWN'
            
            try:
                # Try to convert to integer for processing
                time_int = int(float(time_str))
                
                # Direct mapping approach
                if time_int in [930, 935, 940, 945, 950, 955, 959]:
                    return '0900'
                elif 1000 <= time_int <= 1059:
                    return '1000'
                elif 1100 <= time_int <= 1159:
                    return '1100'
                elif 1200 <= time_int <= 1259:
                    return '1200'
                elif 1300 <= time_int <= 1359:
                    return '1300'
                elif 1400 <= time_int <= 1459:
                    return '1400'
                elif 1500 <= time_int <= 1559:
                    return '1500'
                elif 1600 <= time_int <= 1659:
                    return '1600'
                else:
                    # Fallback: use first 2 digits + 00
                    hour = time_int // 100
                    if hour == 9:
                        return '0900'
                    elif 10 <= hour <= 15:
                        return f'{hour}00'
                    else:
                        return str(time_int)  # Keep as-is if weird
                        
            except (ValueError, TypeError):
                # If conversion fails, return as-is
                return time_str
        
        # Show original time distributions
        st.write("🕐 Original TriggerTime distribution:")
        original_trigger_times = df['TriggerTime'].value_counts().head(10)
        st.write(original_trigger_times.to_dict())
        
        st.write("🎯 Original GoalTime distribution:")
        original_goal_times = df['GoalTime'].value_counts().head(10)
        st.write(original_goal_times.to_dict())
        
        # Apply FORCED conversion
        st.write("🔧 Applying FORCED time conversion...")
        df['TriggerTime_Original'] = df['TriggerTime'].copy()  # Keep backup
        df['GoalTime_Original'] = df['GoalTime'].copy()       # Keep backup
        
        df['TriggerTime'] = df['TriggerTime'].apply(force_time_conversion)
        df['GoalTime'] = df['GoalTime'].apply(force_time_conversion)
        
        # Show converted time distributions
        st.write("✅ Converted TriggerTime distribution:")
        converted_trigger_times = df['TriggerTime'].value_counts()
        st.write(converted_trigger_times.to_dict())
        
        st.write("✅ Converted GoalTime distribution:")
        converted_goal_times = df['GoalTime'].value_counts()
        st.write(converted_goal_times.to_dict())
        
        debug_info.append(f"Time conversion applied to all records")
        debug_info.append(f"TriggerTime unique values: {df['TriggerTime'].unique()}")
        debug_info.append(f"GoalTime unique values: {df['GoalTime'].unique()}")
        
        # Count unique triggers per day AFTER time conversion
        st.write("🔍 Counting unique triggers per day (after conversion)...")
        trigger_occurrences = df[['Date', 'TriggerLevel', 'TriggerTime', 'Direction']].drop_duplicates()
        
        trigger_counts = (
            trigger_occurrences
            .value_counts(subset=['TriggerLevel', 'TriggerTime', 'Direction'])
            .reset_index()
        )
        
        trigger_counts.columns = ['TriggerLevel', 'TriggerTime', 'Direction', 'NumTriggers']
        debug_info.append(f"Unique trigger combinations: {len(trigger_counts):,}")
        
        # Count successful goal hits per group AFTER time conversion
        st.write("🎯 Counting successful goal hits (after conversion)...")
        goal_hits = df[df['GoalHit'] == 'Yes'].copy()
        debug_info.append(f"Total goal hits: {len(goal_hits):,}")
        
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
        
        # Show sample of final results
        st.write("📊 Sample final results:")
        sample_results = summary.head(10)
        st.dataframe(sample_results)
        
        # Basic statistics
        avg_completion = summary['PctCompletion'].mean()
        max_completion = summary['PctCompletion'].max()
        nonzero_completions = len(summary[summary['PctCompletion'] > 0])
        
        debug_info.append(f"Average completion rate: {avg_completion:.1f}%")
        debug_info.append(f"Maximum completion rate: {max_completion:.1f}%")
        debug_info.append(f"Non-zero completions: {nonzero_completions:,} out of {len(summary):,}")
        
        st.success(f"✅ Summary processing complete! Generated {len(summary):,} summary records")
        
        return summary, debug_info
        
    except Exception as e:
        st.error(f"❌ Error processing data: {str(e)}")
        import traceback
        st.text(traceback.format_exc())
        return None, [f"Error: {str(e)}"]

# Streamlit App Interface
st.title('🔧 FORCED Time Conversion Summary Generator')
st.write('Aggressively converts minute times to hour buckets: 930→0900, 1540→1500, etc.')

if st.button('🚀 Generate Summary with FORCED Time Conversion'):
    with st.spinner('Processing with forced time conversion...'):
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
            
            # Show time conversion verification
            st.subheader('🕐 Time Conversion Verification')
            trigger_time_summary = summary_df['TriggerTime'].value_counts()
            st.write("**TriggerTime values in final summary:**")
            st.write(trigger_time_summary.to_dict())
            
            goal_time_summary = summary_df['GoalTime'].value_counts().head(10)
            st.write("**GoalTime values in final summary (top 10):**")
            st.write(goal_time_summary.to_dict())
            
            # Check for 0.0 level presence
            st.subheader('🎯 Level 0.0 Check')
            trigger_0_count = len(summary_df[summary_df['TriggerLevel'] == 0.0])
            goal_0_count = len(summary_df[summary_df['GoalLevel'] == 0.0])
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric('0.0 as TriggerLevel', trigger_0_count)
            with col2:
                st.metric('0.0 as GoalLevel', goal_0_count)
            
            if trigger_0_count > 0:
                st.success("✅ Level 0.0 found as trigger!")
            else:
                st.warning("⚠️ Level 0.0 not found as trigger")
                
            if goal_0_count > 0:
                st.success("✅ Level 0.0 found as goal!")
            else:
                st.warning("⚠️ Level 0.0 not found as goal")
            
            # Preview of summary data
            st.subheader('🔍 Preview of Summary Data')
            st.dataframe(summary_df.head(20))
            
            # Download button
            st.download_button(
                label='⬇️ Download FIXED Dashboard Summary CSV',
                data=summary_df.to_csv(index=False),
                file_name=output_filename,
                mime='text/csv'
            )
            
            st.info('💡 **Next Step:** Use this summary file with your dashboard. Should now show real completion percentages!')
            
        else:
            st.warning('⚠️ No summary data generated. Please check the processing details above.')

st.markdown("""
---
**🔧 This FORCED version:**
1. 📊 Shows original time distributions before conversion
2. 🔧 Applies aggressive time conversion (930/940/950 → 0900)
3. ✅ Shows converted time distributions after conversion  
4. 🎯 Verifies 0.0 level presence in final data
5. 💾 Saves properly formatted summary for dashboard
6. ⬇️ Provides download button for remote access

**Expected result:** Dashboard should finally show real completion percentages instead of zeros!
""")