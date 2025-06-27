import streamlit as st
import pandas as pd
import numpy as np

st.title("🔍 Detailed Logic Diagnostic")
st.write("Debug why results are too small - trace through specific examples")

# Upload the corrected results
uploaded_file = st.file_uploader("Upload corrected combined_trigger_goal_results.csv", type="csv")

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.success(f"📊 Loaded {len(df)} records")
    
    # Basic statistics
    st.write("## Basic Statistics")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Records", len(df))
    with col2:
        total_hits = len(df[df['GoalHit'] == 'Yes'])
        st.metric("Total Hits", total_hits)
    with col3:
        hit_rate = (total_hits / len(df) * 100) if len(df) > 0 else 0
        st.metric("Overall Hit Rate", f"{hit_rate:.1f}%")
    with col4:
        unique_dates = df['Date'].nunique()
        st.metric("Unique Dates", unique_dates)
    
    # Focus on specific problematic scenarios
    st.write("## Focus on Problematic Scenarios")
    
    # Check Downside 0.0 OPEN (should have data)
    downside_zero_open = df[
        (df['Direction'] == 'Downside') & 
        (df['TriggerLevel'] == 0.0) & 
        (df['TriggerTime'] == 'OPEN')
    ]
    
    st.write(f"### Downside 0.0 OPEN: {len(downside_zero_open)} records")
    
    if len(downside_zero_open) > 0:
        hits = len(downside_zero_open[downside_zero_open['GoalHit'] == 'Yes'])
        hit_rate = (hits / len(downside_zero_open) * 100)
        st.write(f"**Hit rate**: {hit_rate:.1f}% ({hits}/{len(downside_zero_open)})")
        
        # Show sample records
        st.dataframe(downside_zero_open[['Date', 'GoalLevel', 'GoalTime', 'GoalHit', 'TriggerPrice', 'GoalPrice']].head(10))
        
        # Check goal distribution
        goal_stats = downside_zero_open['GoalLevel'].value_counts().sort_index()
        st.write("**Goal Level Distribution:**")
        st.write(goal_stats)
        
        # Check if any goals are being hit
        successful_goals = downside_zero_open[downside_zero_open['GoalHit'] == 'Yes']
        if len(successful_goals) > 0:
            st.write("**Successful Goals:**")
            st.dataframe(successful_goals[['Date', 'GoalLevel', 'GoalTime', 'TriggerPrice', 'GoalPrice']].head())
        else:
            st.error("❌ NO successful goals found - this is the problem!")
    else:
        st.error("❌ NO Downside 0.0 OPEN records found - trigger detection issue")
    
    # Check Upside 0.0 OPEN (your reference case)
    st.write("### Upside 0.0 OPEN Comparison")
    upside_zero_open = df[
        (df['Direction'] == 'Upside') & 
        (df['TriggerLevel'] == 0.0) & 
        (df['TriggerTime'] == 'OPEN')
    ]
    
    if len(upside_zero_open) > 0:
        hits = len(upside_zero_open[upside_zero_open['GoalHit'] == 'Yes'])
        hit_rate = (hits / len(upside_zero_open) * 100)
        st.write(f"**Upside 0.0 OPEN Hit rate**: {hit_rate:.1f}% ({hits}/{len(upside_zero_open)})")
        
        # Compare with your Excel expectation
        if hit_rate > 50:
            st.success("✅ Upside 0.0 OPEN looks reasonable")
        else:
            st.error(f"❌ Upside 0.0 OPEN hit rate too low - expected ~80%")
    
    # Detailed analysis of a specific date
    st.write("## Detailed Date Analysis")
    
    # Pick a recent date with data
    available_dates = sorted(df['Date'].unique())[-20:]  # Last 20 dates
    selected_date = st.selectbox("Select date for detailed analysis:", available_dates)
    
    if selected_date:
        date_data = df[df['Date'] == selected_date]
        st.write(f"### Analysis for {selected_date}")
        st.write(f"**Total records for this date**: {len(date_data)}")
        
        # Show all triggers for this date
        triggers = date_data[['Direction', 'TriggerLevel', 'TriggerTime', 'TriggerPrice', 'PreviousClose']].drop_duplicates()
        st.write("**Triggers detected:**")
        st.dataframe(triggers)
        
        # Check if PreviousClose makes sense
        if len(triggers) > 0:
            prev_close = triggers.iloc[0]['PreviousClose']
            zero_trigger = triggers[triggers['TriggerLevel'] == 0.0]
            
            st.write(f"**Previous Close**: {prev_close}")
            
            if len(zero_trigger) > 0:
                zero_price = zero_trigger.iloc[0]['TriggerPrice']
                st.write(f"**0.0 Trigger Price**: {zero_price}")
                
                if abs(zero_price - prev_close) < 0.01:
                    st.success("✅ 0.0 trigger price matches previous close")
                else:
                    st.error(f"❌ 0.0 trigger price ({zero_price}) doesn't match previous close ({prev_close})")
        
        # Focus on 0.0 downside triggers for this date
        zero_downside = date_data[
            (date_data['TriggerLevel'] == 0.0) & 
            (date_data['Direction'] == 'Downside')
        ]
        
        if len(zero_downside) > 0:
            st.write("**0.0 Downside scenarios for this date:**")
            st.dataframe(zero_downside[['TriggerTime', 'GoalLevel', 'GoalPrice', 'GoalHit', 'GoalTime']])
            
            # Check if goals make logical sense
            successful = zero_downside[zero_downside['GoalHit'] == 'Yes']
            if len(successful) > 0:
                st.write(f"✅ {len(successful)} successful downside goals found")
            else:
                st.write("❌ NO successful downside goals - this suggests:")
                st.write("1. Price never went below any negative ATR levels")
                st.write("2. Goal detection logic is too strict") 
                st.write("3. Intraday data doesn't match daily data")
        
        # Check goal price ranges
        st.write("### Goal Price Analysis")
        goal_prices = date_data[['GoalLevel', 'GoalPrice']].drop_duplicates().sort_values('GoalLevel')
        st.dataframe(goal_prices)
        
        # Calculate expected ranges
        if len(triggers) > 0:
            prev_close = triggers.iloc[0]['PreviousClose']
            prev_atr = date_data.iloc[0]['PreviousATR'] if 'PreviousATR' in date_data.columns else "Unknown"
            
            st.write(f"**Previous Close**: {prev_close}")
            st.write(f"**Previous ATR**: {prev_atr}")
            
            if prev_atr != "Unknown":
                st.write("**Expected level ranges:**")
                st.write(f"- **+1 ATR**: ~{prev_close + prev_atr:.2f}")
                st.write(f"- **+0.236 ATR**: ~{prev_close + (0.236 * prev_atr):.2f}")
                st.write(f"- **-0.236 ATR**: ~{prev_close - (0.236 * prev_atr):.2f}")
                st.write(f"- **-1 ATR**: ~{prev_close - prev_atr:.2f}")
    
    # Summary of potential issues
    st.write("## Potential Issues to Investigate")
    
    # Check for common problems
    issues_found = []
    
    # Issue 1: Too few triggers overall
    avg_triggers_per_day = len(df) / df['Date'].nunique() if df['Date'].nunique() > 0 else 0
    if avg_triggers_per_day < 50:  # Should be ~156 (13 levels × 12 goals)
        issues_found.append(f"❌ Too few triggers per day: {avg_triggers_per_day:.1f} (expected ~156)")
    
    # Issue 2: Hit rate too low
    if hit_rate < 10:
        issues_found.append(f"❌ Overall hit rate too low: {hit_rate:.1f}% (expected 15-30%)")
    
    # Issue 3: Missing downside 0.0 data
    if len(downside_zero_open) == 0:
        issues_found.append("❌ No Downside 0.0 OPEN triggers found")
    
    # Issue 4: Check for same-time filtering
    same_time_count = len(df[df['TriggerTime'] == df['GoalTime']])
    if same_time_count > 0:
        issues_found.append(f"⚠️ {same_time_count} same-time scenarios found (should be filtered)")
    
    if issues_found:
        for issue in issues_found:
            st.write(issue)
    else:
        st.success("✅ No obvious structural issues found")
    
    st.write("---")
    st.write("**Next steps**: Based on the issues identified above, we can focus debugging on the specific problems.")

else:
    st.info("👆 Upload your corrected combined_trigger_goal_results.csv to begin diagnosis")
