import streamlit as st
import pandas as pd
import numpy as np
import os

def calculate_atr(df, period=14):
    """
    Calculate Wilder's ATR (Average True Range) - VALIDATED
    """
    df = df.copy()
    
    # Calculate True Range
    df['H-L'] = df['High'] - df['Low']
    df['H-PC'] = abs(df['High'] - df['Close'].shift(1))
    df['L-PC'] = abs(df['Low'] - df['Close'].shift(1))
    df['TR'] = df[['H-L', 'H-PC', 'L-PC']].max(axis=1)
    
    # Calculate Wilder's ATR (exponential moving average with alpha = 1/period)
    df['ATR'] = df['TR'].ewm(alpha=1/period, adjust=False).mean()
    
    # Clean up temporary columns
    df.drop(['H-L', 'H-PC', 'L-PC', 'TR'], axis=1, inplace=True)
    
    return df

def generate_atr_levels(close_price, atr_value):
    """
    Generate Fibonacci-based ATR levels - VALIDATED
    """
    fib_ratios = [0.236, 0.382, 0.500, 0.618, 0.786, 1.000, 
                  -0.236, -0.382, -0.500, -0.618, -0.786, -1.000, 0.000]
    
    levels = {}
    for ratio in fib_ratios:
        level_price = close_price + (ratio * atr_value)
        levels[ratio] = level_price
    
    return levels

def detect_triggers_and_goals(daily, intraday):
    """
    PERFECT SYSTEMATIC LOGIC:
    For each trigger level:
    1. Check if LOW <= trigger (Below direction) → check all 12 goals
    2. Check if HIGH >= trigger (Above direction) → check all 12 goals
    
    For goals:
    - Above goals: check HIGH >= goal
    - Below goals: check LOW <= goal
    """
    fib_levels = [0.236, 0.382, 0.500, 0.618, 0.786, 1.000, 
                 -0.236, -0.382, -0.500, -0.618, -0.786, -1.000, 0.000]
    
    results = []
    
    for i in range(1, len(daily)):
        try:
            # Use PREVIOUS day's data for level calculation
            previous_row = daily.iloc[i-1]  
            current_row = daily.iloc[i]     
            
            previous_close = previous_row['Close']  
            previous_atr = previous_row['ATR']      
            trading_date = current_row['Date']
            
            # Date filtering
            if hasattr(trading_date, 'strftime'):
                date_str = trading_date.strftime('%Y-%m-%d')
            elif isinstance(trading_date, str):
                date_str = trading_date[:10]
            else:
                date_str = str(trading_date)[:10]
            
            if date_str < '2014-01-02':
                continue
            
            if pd.isna(previous_atr) or pd.isna(previous_close):
                continue
            
            # Generate levels using PREVIOUS day's close + ATR
            level_map = generate_atr_levels(previous_close, previous_atr)
            
            # Get intraday data for trading date
            day_data = intraday[intraday['Date'] == pd.to_datetime(trading_date).date()].copy()
            if day_data.empty:
                continue

            day_data['Time'] = day_data['Datetime'].dt.strftime('%H%M')
            day_data.reset_index(drop=True, inplace=True)

            open_candle = day_data.iloc[0]
            open_price = open_candle['Open']
            
            # PERFECT SYSTEMATIC APPROACH: Each level checked in both directions
            for trigger_level in fib_levels:
                trigger_price = level_map[trigger_level]
                
                # 1. CHECK BELOW DIRECTION: LOW <= trigger level
                below_triggered = False
                below_trigger_time = None
                below_trigger_row = None
                
                # Check OPEN candle for below trigger
                if open_price <= trigger_price:
                    below_triggered = True
                    below_trigger_time = 'OPEN'
                    below_trigger_row = 0
                
                # Check intraday candles for below trigger
                if not below_triggered:
                    for idx, row in day_data.iloc[1:].iterrows():
                        if row['Low'] <= trigger_price:
                            below_triggered = True
                            below_trigger_time = row['Time']
                            below_trigger_row = idx
                            break
                
                # Process all goals for BELOW trigger
                if below_triggered:
                    trigger_candle = day_data.iloc[below_trigger_row]
                    
                    for goal_level in fib_levels:
                        if goal_level == trigger_level:  # Skip same level
                            continue
                        
                        goal_price = level_map[goal_level]
                        goal_hit = False
                        goal_time = ''
                        is_same_time = False
                        
                        # Determine goal type for BELOW trigger
                        if goal_level < trigger_level:
                            goal_type = 'Continuation'  # Further below
                        else:
                            goal_type = 'Retracement'   # Back above (includes cross-zero)
                        
                        # Check for goal completion
                        if below_trigger_time == 'OPEN':
                            # Check if goal completes at OPEN (same-time scenario)
                            if goal_level > trigger_level:  # Above goal
                                if open_price >= goal_price:
                                    goal_hit = True
                                    goal_time = 'OPEN'
                                    is_same_time = True
                            else:  # Below goal
                                if open_price <= goal_price:
                                    goal_hit = True
                                    goal_time = 'OPEN'
                                    is_same_time = True
                            
                            # Check subsequent candles if not completed at OPEN
                            if not goal_hit:
                                for _, row in day_data.iloc[1:].iterrows():
                                    if goal_level > trigger_level:  # Above goal
                                        if row['High'] >= goal_price:
                                            goal_hit = True
                                            goal_time = row['Time']
                                            break
                                    else:  # Below goal
                                        if row['Low'] <= goal_price:
                                            goal_hit = True
                                            goal_time = row['Time']
                                            break
                        
                        else:  # Intraday below trigger
                            # Check if goal completes on same candle as trigger
                            if goal_level > trigger_level:  # Above goal
                                if trigger_candle['High'] >= goal_price:
                                    goal_hit = True
                                    goal_time = below_trigger_time
                            else:  # Below goal
                                if trigger_candle['Low'] <= goal_price:
                                    goal_hit = True
                                    goal_time = below_trigger_time
                            
                            # Check subsequent candles if not completed on trigger candle
                            if not goal_hit:
                                for _, row in day_data.iloc[below_trigger_row + 1:].iterrows():
                                    if goal_level > trigger_level:  # Above goal
                                        if row['High'] >= goal_price:
                                            goal_hit = True
                                            goal_time = row['Time']
                                            break
                                    else:  # Below goal
                                        if row['Low'] <= goal_price:
                                            goal_hit = True
                                            goal_time = row['Time']
                                            break
                        
                        # Record this BELOW trigger-goal combination
                        results.append({
                            'Date': trading_date,
                            'Direction': 'Below',
                            'TriggerLevel': trigger_level,
                            'TriggerTime': below_trigger_time,
                            'TriggerPrice': round(trigger_price, 2),
                            'GoalLevel': goal_level,
                            'GoalPrice': round(goal_price, 2),
                            'GoalHit': 'Yes' if goal_hit else 'No',
                            'GoalTime': goal_time if goal_hit else '',
                            'GoalClassification': goal_type,
                            'PreviousClose': round(previous_close, 2),
                            'PreviousATR': round(previous_atr, 2),
                            'SameTime': is_same_time,
                            'RetestedTrigger': 'No'
                        })
                
                # 2. CHECK ABOVE DIRECTION: HIGH >= trigger level
                above_triggered = False
                above_trigger_time = None
                above_trigger_row = None
                
                # Check OPEN candle for above trigger
                if open_price >= trigger_price:
                    above_triggered = True
                    above_trigger_time = 'OPEN'
                    above_trigger_row = 0
                
                # Check intraday candles for above trigger (only if not already triggered at OPEN)
                if not above_triggered:
                    for idx, row in day_data.iloc[1:].iterrows():
                        if row['High'] >= trigger_price:
                            above_triggered = True
                            above_trigger_time = row['Time']
                            above_trigger_row = idx
                            break
                
                # Process all goals for ABOVE trigger
                if above_triggered:
                    trigger_candle = day_data.iloc[above_trigger_row]
                    
                    for goal_level in fib_levels:
                        if goal_level == trigger_level:  # Skip same level
                            continue
                        
                        goal_price = level_map[goal_level]
                        goal_hit = False
                        goal_time = ''
                        is_same_time = False
                        
                        # Determine goal type for ABOVE trigger
                        if goal_level > trigger_level:
                            goal_type = 'Continuation'  # Further above
                        else:
                            goal_type = 'Retracement'   # Back below (includes cross-zero)
                        
                        # Check for goal completion
                        if above_trigger_time == 'OPEN':
                            # Check if goal completes at OPEN (same-time scenario)
                            if goal_level > trigger_level:  # Above goal
                                if open_price >= goal_price:
                                    goal_hit = True
                                    goal_time = 'OPEN'
                                    is_same_time = True
                            else:  # Below goal
                                if open_price <= goal_price:
                                    goal_hit = True
                                    goal_time = 'OPEN'
                                    is_same_time = True
                            
                            # Check subsequent candles if not completed at OPEN
                            if not goal_hit:
                                for _, row in day_data.iloc[1:].iterrows():
                                    if goal_level > trigger_level:  # Above goal
                                        if row['High'] >= goal_price:
                                            goal_hit = True
                                            goal_time = row['Time']
                                            break
                                    else:  # Below goal
                                        if row['Low'] <= goal_price:
                                            goal_hit = True
                                            goal_time = row['Time']
                                            break
                        
                        else:  # Intraday above trigger
                            # Check if goal completes on same candle as trigger
                            if goal_level > trigger_level:  # Above goal
                                if trigger_candle['High'] >= goal_price:
                                    goal_hit = True
                                    goal_time = above_trigger_time
                            else:  # Below goal
                                if trigger_candle['Low'] <= goal_price:
                                    goal_hit = True
                                    goal_time = above_trigger_time
                            
                            # Check subsequent candles if not completed on trigger candle
                            if not goal_hit:
                                for _, row in day_data.iloc[above_trigger_row + 1:].iterrows():
                                    if goal_level > trigger_level:  # Above goal
                                        if row['High'] >= goal_price:
                                            goal_hit = True
                                            goal_time = row['Time']
                                            break
                                    else:  # Below goal
                                        if row['Low'] <= goal_price:
                                            goal_hit = True
                                            goal_time = row['Time']
                                            break
                        
                        # Record this ABOVE trigger-goal combination
                        results.append({
                            'Date': trading_date,
                            'Direction': 'Above',
                            'TriggerLevel': trigger_level,
                            'TriggerTime': above_trigger_time,
                            'TriggerPrice': round(trigger_price, 2),
                            'GoalLevel': goal_level,
                            'GoalPrice': round(goal_price, 2),
                            'GoalHit': 'Yes' if goal_hit else 'No',
                            'GoalTime': goal_time if goal_hit else '',
                            'GoalClassification': goal_type,
                            'PreviousClose': round(previous_close, 2),
                            'PreviousATR': round(previous_atr, 2),
                            'SameTime': is_same_time,
                            'RetestedTrigger': 'No'
                        })

        except Exception as e:
            st.write(f"⚠️ Error processing {trading_date}: {str(e)}")
            continue

    return pd.DataFrame(results)

def main():
    """
    PERFECT SYSTEMATIC: Every level checked in both directions (Above and Below)
    """
    debug_info = []
    
    try:
        debug_info.append("📊 Loading daily OHLC data...")
        daily = pd.read_excel('SPXdailycandles.xlsx', header=4)
        debug_info.append(f"Daily data loaded: {daily.shape}")
        
        # Check required columns
        required_cols = ['Date', 'Open', 'High', 'Low', 'Close']
        missing_cols = [col for col in required_cols if col not in daily.columns]
        
        if missing_cols:
            debug_info.append(f"❌ Missing required columns: {missing_cols}")
            return pd.DataFrame(), debug_info
        
        debug_info.append("🧮 Calculating ATR using validated Wilder's method...")
        daily = calculate_atr(daily, period=14)
        debug_info.append(f"ATR calculated successfully. Sample recent values: {daily['ATR'].tail(3).round(2).tolist()}")
        
        debug_info.append("📈 Loading intraday data...")
        intraday = pd.read_csv('SPX_10min.csv', parse_dates=['Datetime'])
        intraday['Date'] = intraday['Datetime'].dt.date
        debug_info.append(f"Intraday data loaded: {intraday.shape}")
        debug_info.append(f"Intraday date range: {intraday['Date'].min()} to {intraday['Date'].max()}")
        
        # Test level generation
        if len(daily) >= 2:
            prev_row = daily.iloc[-2]
            curr_row = daily.iloc[-1]
            if not pd.isna(prev_row['ATR']):
                test_levels = generate_atr_levels(prev_row['Close'], prev_row['ATR'])
                debug_info.append(f"✅ Level generation test:")
                debug_info.append(f"Previous day ({prev_row['Date']}): Close={prev_row['Close']:.2f}, ATR={prev_row['ATR']:.2f}")
                debug_info.append(f"0.0 level for current day: {test_levels[0.0]:.2f} (should equal previous close)")
        
        debug_info.append("🎯 Running PERFECT SYSTEMATIC trigger and goal detection...")
        df = detect_triggers_and_goals(daily, intraday)
        debug_info.append(f"✅ Detection complete: {len(df)} trigger-goal combinations found")
        
        # Additional validation
        if not df.empty:
            same_time_count = len(df[df['SameTime'] == True])
            debug_info.append(f"✅ Same-time scenarios found: {same_time_count}")
            
            same_time_hits = len(df[(df['SameTime'] == True) & (df['GoalHit'] == 'Yes')])
            debug_info.append(f"✅ Same-time hits: {same_time_hits}")
            
            open_goals = len(df[df['GoalTime'] == 'OPEN'])
            debug_info.append(f"✅ Records with GoalTime=OPEN: {open_goals}")
            
            # Check cross-zero scenarios (corrected logic)
            cross_zero_below_to_above = len(df[(df['Direction'] == 'Below') & (df['GoalLevel'] > df['TriggerLevel'])])
            cross_zero_above_to_below = len(df[(df['Direction'] == 'Above') & (df['GoalLevel'] < df['TriggerLevel'])])
            debug_info.append(f"✅ Cross-zero scenarios - Below triggers to above goals: {cross_zero_below_to_above}")
            debug_info.append(f"✅ Cross-zero scenarios - Above triggers to below goals: {cross_zero_above_to_below}")
            
            # Check specific levels
            zero_level_above = len(df[(df['TriggerLevel'] == 0.0) & (df['Direction'] == 'Above')])
            zero_level_below = len(df[(df['TriggerLevel'] == 0.0) & (df['Direction'] == 'Below')])
            debug_info.append(f"✅ Level 0.0 - Above triggers: {zero_level_above}, Below triggers: {zero_level_below}")
            
            # Direction breakdown
            above_triggers = len(df[df['Direction'] == 'Above'])
            below_triggers = len(df[df['Direction'] == 'Below'])
            debug_info.append(f"✅ Above triggers: {above_triggers}, Below triggers: {below_triggers}")
            
            open_triggers = len(df[df['TriggerTime'] == 'OPEN'])
            intraday_triggers = len(df[df['TriggerTime'] != 'OPEN'])
            debug_info.append(f"✅ OPEN triggers: {open_triggers}, Intraday triggers: {intraday_triggers}")
        
        return df, debug_info
        
    except Exception as e:
        debug_info.append(f"❌ Error: {str(e)}")
        import traceback
        debug_info.append(f"Traceback: {traceback.format_exc()}")
        return pd.DataFrame(), debug_info

# Streamlit Interface
st.title('🎯 PERFECT SYSTEMATIC ATR Trigger & Goal Generator')
st.write('**PERFECT: Every level checked in both Above and Below directions**')

output_path = 'combined_trigger_goal_results_PERFECT.csv'

if st.button('🚀 Generate PERFECT Results'):
    with st.spinner('Calculating with PERFECT systematic logic...'):
        try:
            result_df, debug_messages = main()
            
            # Show debug info
            with st.expander('📋 Debug Information'):
                for msg in debug_messages:
                    st.write(msg)
            
            if not result_df.empty:
                result_df['Source'] = 'Perfect_Systematic_Detection'
                result_df.to_csv(output_path, index=False)
                st.success('✅ PERFECT Results generated!')
                
                # Show summary stats
                st.subheader('📊 Summary Statistics')
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric('Total Records', len(result_df))
                with col2:
                    st.metric('Unique Dates', result_df['Date'].nunique())
                with col3:
                    st.metric('Goals Hit', len(result_df[result_df['GoalHit'] == 'Yes']))
                with col4:
                    hit_rate = len(result_df[result_df['GoalHit'] == 'Yes']) / len(result_df) * 100
                    st.metric('Hit Rate', f'{hit_rate:.1f}%')
                
                # Show direction analysis
                st.subheader('📊 Direction Analysis')
                col1, col2 = st.columns(2)
                with col1:
                    above_count = len(result_df[result_df['Direction'] == 'Above'])
                    st.metric('Above Triggers', above_count)
                with col2:
                    below_count = len(result_df[result_df['Direction'] == 'Below'])
                    st.metric('Below Triggers', below_count)
                
                # Show level 0.0 analysis
                st.subheader('🎯 Level 0.0 Analysis')
                zero_above = len(result_df[(result_df['TriggerLevel'] == 0.0) & (result_df['Direction'] == 'Above')])
                zero_below = len(result_df[(result_df['TriggerLevel'] == 0.0) & (result_df['Direction'] == 'Below')])
                
                col1, col2 = st.columns(2)
                with col1:
                    st.metric('0.0 Level Above Triggers', zero_above)
                with col2:
                    st.metric('0.0 Level Below Triggers', zero_below)
                
                # Show cross-zero analysis (corrected)
                st.subheader('🔄 Cross-Zero Analysis')
                cross_zero_below_to_above = result_df[(result_df['Direction'] == 'Below') & (result_df['GoalLevel'] > result_df['TriggerLevel'])]
                cross_zero_above_to_below = result_df[(result_df['Direction'] == 'Above') & (result_df['GoalLevel'] < result_df['TriggerLevel'])]
                
                col1, col2 = st.columns(2)
                with col1:
                    st.metric('Below→Above Cross-Zero', len(cross_zero_below_to_above))
                with col2:
                    st.metric('Above→Below Cross-Zero', len(cross_zero_above_to_below))
                
                # Show hits for cross-zero
                if len(cross_zero_below_to_above) > 0:
                    below_to_above_hits = len(cross_zero_below_to_above[cross_zero_below_to_above['GoalHit'] == 'Yes'])
                    below_to_above_rate = below_to_above_hits / len(cross_zero_below_to_above) * 100
                    st.metric('Below→Above Hit Rate', f'{below_to_above_rate:.1f}%')
                
                if len(cross_zero_above_to_below) > 0:
                    above_to_below_hits = len(cross_zero_above_to_below[cross_zero_above_to_below['GoalHit'] == 'Yes'])  
                    above_to_below_rate = above_to_below_hits / len(cross_zero_above_to_below) * 100
                    st.metric('Above→Below Hit Rate', f'{above_to_below_rate:.1f}%')
                
                # Show sample cross-zero scenarios
                if len(cross_zero_below_to_above) > 0:
                    st.write("### Sample Below→Above Cross-Zero Scenarios:")
                    sample_cross = cross_zero_below_to_above.head(10)[['Direction', 'TriggerLevel', 'TriggerTime', 'GoalLevel', 'GoalTime', 'GoalHit']]
                    st.dataframe(sample_cross)
                
                # Show same-time analysis
                st.subheader('🕐 Same-Time Analysis')
                same_time_data = result_df[result_df['SameTime'] == True]
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric('Same-Time Records', len(same_time_data))
                with col2:
                    same_time_hits = len(same_time_data[same_time_data['GoalHit'] == 'Yes'])
                    st.metric('Same-Time Hits', same_time_hits)
                with col3:
                    open_goal_records = len(result_df[result_df['GoalTime'] == 'OPEN'])
                    st.metric('OPEN Goal Records', open_goal_records)
                
                # Download button
                st.download_button(
                    '⬇️ Download PERFECT Results CSV', 
                    data=result_df.to_csv(index=False), 
                    file_name=output_path, 
                    mime='text/csv'
                )
                
                st.success('🎉 **PERFECT DATA READY!** Every level checked in both directions with full cross-zero support!')
                
            else:
                st.warning('⚠️ No results generated - check debug info above')
                
        except Exception as e:
            st.error(f'❌ Error: {e}')

st.markdown("""
---
**🔧 PERFECT SYSTEMATIC Logic Applied:**
- ✅ **Every level checked in BOTH directions** (Above and Below)
- ✅ **Level 0.0 works in both directions** 
- ✅ **Cross-zero scenarios fully supported** (Below→Above, Above→Below)
- ✅ **Proper high/low checking** for goal completion
- ✅ **Same-time flagging** preserved for summary processing

**🎯 This Should Finally Give Complete Cross-Zero Coverage!**
""")
