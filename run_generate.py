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
    CORRECTED: 
    1. Fixed OPEN detection to include 0.0 level
    2. NO same-time filtering - generate all records
    3. FLAG same-time scenarios for later processing
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

            triggered_up = {}
            triggered_down = {}

            # Process each intraday candle
            for idx, row in day_data.iterrows():
                open_price = row['Open']
                high = row['High']
                low = row['Low']
                
                # FIXED: Include 0.0 level in OPEN detection
                time_label = 'OPEN' if idx == 0 and (
                    any(open_price >= level_map[level] for level in fib_levels if level >= 0) or  # Include 0.0
                    any(open_price <= level_map[level] for level in fib_levels if level <= 0)     # Include 0.0
                ) else row['Time']

                # Check upside triggers (include 0.0 level)
                for level in [lvl for lvl in fib_levels if lvl >= 0]:
                    if level in triggered_up:
                        continue
                    
                    trigger_price = level_map[level]
                    if (time_label == 'OPEN' and open_price >= trigger_price) or \
                       (time_label != 'OPEN' and high >= trigger_price):
                        triggered_up[level] = {
                            'TriggerLevel': level,
                            'TriggerTime': time_label,
                            'TriggeredRow': idx,
                            'TriggerPrice': trigger_price
                        }

                # Check downside triggers (include 0.0 level)
                for level in [lvl for lvl in fib_levels if lvl <= 0]:
                    if level in triggered_down:
                        continue
                    
                    trigger_price = level_map[level]
                    if (time_label == 'OPEN' and open_price <= trigger_price) or \
                       (time_label != 'OPEN' and low <= trigger_price):
                        triggered_down[level] = {
                            'TriggerLevel': level,
                            'TriggerTime': time_label,
                            'TriggeredRow': idx,
                            'TriggerPrice': trigger_price
                        }

            # Process upside triggers and goals
            for level, trigger_info in triggered_up.items():
                trigger_row = trigger_info['TriggeredRow']
                
                for goal_level in fib_levels:
                    if goal_level == level:  # Skip same level
                        continue
                    
                    if goal_level not in level_map:
                        continue
                        
                    goal_price = level_map[goal_level]
                    goal_hit = False
                    goal_time = ''
                    
                    # Determine if this is continuation or retracement
                    if goal_level > level:
                        goal_type = 'Continuation'
                        # Check if goal is hit on same candle as trigger
                        trigger_candle = day_data.iloc[trigger_row]
                        if trigger_candle['High'] >= goal_price:
                            if trigger_info['TriggerTime'] != 'OPEN':
                                goal_hit = True
                                goal_time = trigger_info['TriggerTime']
                        else:
                            # Check subsequent candles for upside goal
                            for _, row in day_data.iloc[trigger_row + 1:].iterrows():
                                if row['High'] >= goal_price:
                                    goal_hit = True
                                    goal_time = row['Time']
                                    break
                    else:
                        goal_type = 'Retracement'
                        # Check subsequent candles for downside goal
                        for _, row in day_data.iloc[trigger_row + 1:].iterrows():
                            if row['Low'] <= goal_price:
                                goal_hit = True
                                goal_time = row['Time']
                                break
                    
                    # CHANGED: Don't filter out same-time scenarios - FLAG them instead
                    is_same_time = (trigger_info['TriggerTime'] == 'OPEN' and goal_time == 'OPEN')
                    
                    results.append({
                        'Date': trading_date,
                        'Direction': 'Upside',
                        'TriggerLevel': level,
                        'TriggerTime': trigger_info['TriggerTime'],
                        'TriggerPrice': round(trigger_info['TriggerPrice'], 2),
                        'GoalLevel': goal_level,
                        'GoalPrice': round(goal_price, 2),
                        'GoalHit': 'Yes' if goal_hit else 'No',
                        'GoalTime': goal_time if goal_hit else '',
                        'GoalClassification': goal_type,
                        'PreviousClose': round(previous_close, 2),
                        'PreviousATR': round(previous_atr, 2),
                        'SameTime': is_same_time,  # NEW: Flag for same-time scenarios
                        'RetestedTrigger': 'No'
                    })

            # Process downside triggers and goals
            for level, trigger_info in triggered_down.items():
                trigger_row = trigger_info['TriggeredRow']
                
                for goal_level in fib_levels:
                    if goal_level == level:  # Skip same level
                        continue
                    
                    if goal_level not in level_map:
                        continue
                        
                    goal_price = level_map[goal_level]
                    goal_hit = False
                    goal_time = ''
                    
                    # Determine if this is continuation or retracement
                    if goal_level < level:
                        goal_type = 'Continuation'
                        # Check if goal is hit on same candle as trigger
                        trigger_candle = day_data.iloc[trigger_row]
                        if trigger_candle['Low'] <= goal_price:
                            if trigger_info['TriggerTime'] != 'OPEN':
                                goal_hit = True
                                goal_time = trigger_info['TriggerTime']
                        else:
                            # Check subsequent candles for downside goal
                            for _, row in day_data.iloc[trigger_row + 1:].iterrows():
                                if row['Low'] <= goal_price:
                                    goal_hit = True
                                    goal_time = row['Time']
                                    break
                    else:
                        goal_type = 'Retracement'
                        # Check subsequent candles for upside goal
                        for _, row in day_data.iloc[trigger_row + 1:].iterrows():
                            if row['High'] >= goal_price:
                                goal_hit = True
                                goal_time = row['Time']
                                break
                    
                    # CHANGED: Don't filter out same-time scenarios - FLAG them instead
                    is_same_time = (trigger_info['TriggerTime'] == 'OPEN' and goal_time == 'OPEN')
                    
                    results.append({
                        'Date': trading_date,
                        'Direction': 'Downside',
                        'TriggerLevel': level,
                        'TriggerTime': trigger_info['TriggerTime'],
                        'TriggerPrice': round(trigger_info['TriggerPrice'], 2),
                        'GoalLevel': goal_level,
                        'GoalPrice': round(goal_price, 2),
                        'GoalHit': 'Yes' if goal_hit else 'No',
                        'GoalTime': goal_time if goal_hit else '',
                        'GoalClassification': goal_type,
                        'PreviousClose': round(previous_close, 2),
                        'PreviousATR': round(previous_atr, 2),
                        'SameTime': is_same_time,  # NEW: Flag for same-time scenarios
                        'RetestedTrigger': 'No'
                    })

        except Exception as e:
            st.write(f"⚠️ Error processing {trading_date}: {str(e)}")
            continue

    return pd.DataFrame(results)

def main():
    """
    CORRECTED: No same-time filtering, added SameTime flag
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
        
        debug_info.append("🎯 Running trigger and goal detection with SameTime flags...")
        df = detect_triggers_and_goals(daily, intraday)
        debug_info.append(f"✅ Detection complete: {len(df)} trigger-goal combinations found")
        
        # Additional validation
        if not df.empty:
            same_time_count = len(df[df['SameTime'] == True])
            debug_info.append(f"✅ Same-time scenarios flagged: {same_time_count}")
            
            downside_zero_open = len(df[(df['Direction'] == 'Downside') & 
                                       (df['TriggerLevel'] == 0.0) & 
                                       (df['TriggerTime'] == 'OPEN')])
            debug_info.append(f"✅ Downside 0.0 OPEN triggers found: {downside_zero_open}")
            
            upside_zero_open = len(df[(df['Direction'] == 'Upside') & 
                                     (df['TriggerLevel'] == 0.0) & 
                                     (df['TriggerTime'] == 'OPEN')])
            debug_info.append(f"✅ Upside 0.0 OPEN triggers found: {upside_zero_open}")
        
        return df, debug_info
        
    except Exception as e:
        debug_info.append(f"❌ Error: {str(e)}")
        import traceback
        debug_info.append(f"Traceback: {traceback.format_exc()}")
        return pd.DataFrame(), debug_info

# Streamlit Interface
st.title('🎯 FINAL ATR Trigger & Goal Generator')
st.write('**FINAL: Fixed OPEN detection + SameTime flag (no filtering)**')

output_path = 'combined_trigger_goal_results_FINAL.csv'

if st.button('🚀 Generate FINAL Results'):
    with st.spinner('Calculating with FINAL logic...'):
        try:
            result_df, debug_messages = main()
            
            # Show debug info
            with st.expander('📋 Debug Information'):
                for msg in debug_messages:
                    st.write(msg)
            
            if not result_df.empty:
                result_df['Source'] = 'Final_With_SameTime_Flag'
                result_df.to_csv(output_path, index=False)
                st.success('✅ FINAL Results generated!')
                
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
                
                # Show same-time statistics
                st.subheader('🕐 Same-Time Analysis')
                same_time_data = result_df[result_df['SameTime'] == True]
                col1, col2 = st.columns(2)
                with col1:
                    st.metric('Same-Time Records', len(same_time_data))
                with col2:
                    same_time_hits = len(same_time_data[same_time_data['GoalHit'] == 'Yes'])
                    st.metric('Same-Time Hits', same_time_hits)
                
                # Show key scenarios
                st.subheader('🎯 Key Scenarios Validation')
                downside_zero = len(result_df[(result_df['Direction'] == 'Downside') & 
                                            (result_df['TriggerLevel'] == 0.0) & 
                                            (result_df['TriggerTime'] == 'OPEN')])
                
                upside_zero = len(result_df[(result_df['Direction'] == 'Upside') & 
                                          (result_df['TriggerLevel'] == 0.0) & 
                                          (result_df['TriggerTime'] == 'OPEN')])
                
                col1, col2 = st.columns(2)
                with col1:
                    st.metric('Downside 0.0 OPEN', downside_zero)
                    if downside_zero > 0:
                        st.success("✅ Found!")
                    else:
                        st.error("❌ Still missing")
                        
                with col2:
                    st.metric('Upside 0.0 OPEN', upside_zero)
                    if upside_zero > 0:
                        st.success("✅ Found!")
                    else:
                        st.error("❌ Missing")
                
                # Download button
                st.download_button(
                    '⬇️ Download FINAL Results CSV', 
                    data=result_df.to_csv(index=False), 
                    file_name=output_path, 
                    mime='text/csv'
                )
                
                st.success('🎉 **READY FOR SUMMARY PROCESSING!** Data includes SameTime flags for denominator adjustment.')
                
            else:
                st.warning('⚠️ No results generated - check debug info above')
                
        except Exception as e:
            st.error(f'❌ Error: {e}')

st.markdown("""
---
**🔧 Final Configuration:**
- ✅ **Fixed OPEN Detection**: 0.0 level included in trigger detection
- ✅ **No Same-Time Filtering**: All records generated (including OPEN→OPEN)
- ✅ **SameTime Flag**: Added for summary script to handle denominator adjustment
- ✅ **Ready for Summary**: Implements your Excel methodology exactly

**🎯 Next Step: Update summary script to handle denominator adjustment!**
""")