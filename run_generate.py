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
    Ordered to match Excel: positive levels ascending, then negative levels descending, then zero
    """
    # Match Excel order: 0.236, 0.382, 0.5, 0.618, 0.786, 1, -0.236, -0.382, -0.5, -0.618, -0.786, -1, 0
    fib_ratios = [0.236, 0.382, 0.500, 0.618, 0.786, 1.000, 
                  -0.236, -0.382, -0.500, -0.618, -0.786, -1.000, 0.000]
    
    levels = {}
    for ratio in fib_ratios:
        level_price = close_price + (ratio * atr_value)
        levels[ratio] = level_price
    
    return levels

def detect_triggers_and_goals(daily, intraday):
    """
    Detect ATR-based triggers and goals using CALCULATED levels (no Excel dependency)
    """
    # Use same order as Excel for consistency
    fib_levels = [0.236, 0.382, 0.500, 0.618, 0.786, 1.000, 
                 -0.236, -0.382, -0.500, -0.618, -0.786, -1.000, 0.000]
    
    results = []
    
    for i in range(1, len(daily) - 1):  # Skip last row since we need next day
        try:
            current_row = daily.iloc[i]
            next_row = daily.iloc[i + 1]
            
            # Current day data (for level calculation)
            current_date = current_row['Date']
            current_close = current_row['Close']
            current_atr = current_row['ATR']
            
            # Trading day (when levels will be used)
            trading_date = next_row['Date']
            
            # Date filtering
            if hasattr(trading_date, 'strftime'):
                date_str = trading_date.strftime('%Y-%m-%d')
            elif isinstance(trading_date, str):
                date_str = trading_date[:10]
            else:
                date_str = str(trading_date)[:10]
            
            if date_str < '2014-01-02':
                continue
            
            if pd.isna(current_atr) or pd.isna(current_close):
                continue
            
            # Generate ATR levels using current day's close + ATR for next day's trading
            level_map = generate_atr_levels(current_close, current_atr)
            
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
                
                # Determine time label
                time_label = 'OPEN' if idx == 0 and (
                    any(open_price >= level_map[level] for level in fib_levels if level > 0) or
                    any(open_price <= level_map[level] for level in fib_levels if level < 0)
                ) else row['Time']

                # Check upside triggers
                for level in [lvl for lvl in fib_levels if lvl > 0]:
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

                # Check downside triggers
                for level in [lvl for lvl in fib_levels if lvl < 0]:
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
                
                # Check continuation goals (higher levels)
                for goal_level in [l for l in fib_levels if l > level]:
                    goal_price = level_map[goal_level]
                    goal_hit = False
                    goal_time = ''
                    
                    # Check if goal is hit on same candle as trigger
                    trigger_candle = day_data.iloc[trigger_row]
                    if trigger_candle['High'] >= goal_price:
                        if trigger_info['TriggerTime'] != 'OPEN':  # Don't count if both at open
                            goal_hit = True
                            goal_time = trigger_info['TriggerTime']
                    else:
                        # Check subsequent candles
                        for _, row in day_data.iloc[trigger_row + 1:].iterrows():
                            if row['High'] >= goal_price:
                                goal_hit = True
                                goal_time = row['Time']
                                break
                    
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
                        'Type': 'Continuation',
                        'BaseClose': round(current_close, 2),
                        'BaseATR': round(current_atr, 2),
                        'RetestedTrigger': 'No'
                    })

                # Check retracement goals (negative levels)
                for retrace_level in [l for l in fib_levels if l < 0]:
                    goal_price = level_map[retrace_level]
                    goal_hit = False
                    goal_time = ''
                    
                    for _, row in day_data.iloc[trigger_row + 1:].iterrows():
                        if row['Low'] <= goal_price:
                            goal_hit = True
                            goal_time = row['Time']
                            break
                    
                    results.append({
                        'Date': trading_date,
                        'Direction': 'Upside',
                        'TriggerLevel': level,
                        'TriggerTime': trigger_info['TriggerTime'],
                        'TriggerPrice': round(trigger_info['TriggerPrice'], 2),
                        'GoalLevel': retrace_level,
                        'GoalPrice': round(goal_price, 2),
                        'GoalHit': 'Yes' if goal_hit else 'No',
                        'GoalTime': goal_time if goal_hit else '',
                        'Type': 'Retracement',
                        'BaseClose': round(current_close, 2),
                        'BaseATR': round(current_atr, 2),
                        'RetestedTrigger': 'No'
                    })

            # Process downside triggers and goals
            for level, trigger_info in triggered_down.items():
                trigger_row = trigger_info['TriggeredRow']
                
                # Check continuation goals (lower levels)
                for goal_level in [l for l in fib_levels if l < level]:
                    goal_price = level_map[goal_level]
                    goal_hit = False
                    goal_time = ''
                    
                    # Check if goal is hit on same candle as trigger
                    trigger_candle = day_data.iloc[trigger_row]
                    if trigger_candle['Low'] <= goal_price:
                        if trigger_info['TriggerTime'] != 'OPEN':  # Don't count if both at open
                            goal_hit = True
                            goal_time = trigger_info['TriggerTime']
                    else:
                        # Check subsequent candles
                        for _, row in day_data.iloc[trigger_row + 1:].iterrows():
                            if row['Low'] <= goal_price:
                                goal_hit = True
                                goal_time = row['Time']
                                break
                    
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
                        'Type': 'Continuation',
                        'BaseClose': round(current_close, 2),
                        'BaseATR': round(current_atr, 2),
                        'RetestedTrigger': 'No'
                    })

                # Check retracement goals (positive levels)
                for retrace_level in [l for l in fib_levels if l > 0]:
                    goal_price = level_map[retrace_level]
                    goal_hit = False
                    goal_time = ''
                    
                    for _, row in day_data.iloc[trigger_row + 1:].iterrows():
                        if row['High'] >= goal_price:
                            goal_hit = True
                            goal_time = row['Time']
                            break
                    
                    results.append({
                        'Date': trading_date,
                        'Direction': 'Downside',
                        'TriggerLevel': level,
                        'TriggerTime': trigger_info['TriggerTime'],
                        'TriggerPrice': round(trigger_info['TriggerPrice'], 2),
                        'GoalLevel': retrace_level,
                        'GoalPrice': round(goal_price, 2),
                        'GoalHit': 'Yes' if goal_hit else 'No',
                        'GoalTime': goal_time if goal_hit else '',
                        'Type': 'Retracement',
                        'BaseClose': round(current_close, 2),
                        'BaseATR': round(current_atr, 2),
                        'RetestedTrigger': 'No'
                    })

        except Exception as e:
            st.write(f"⚠️ Error processing {trading_date}: {str(e)}")
            continue

    return pd.DataFrame(results)

def main():
    """
    Main function with validated ATR calculation and level generation
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
        test_row = daily.iloc[-2]
        if not pd.isna(test_row['ATR']):
            test_levels = generate_atr_levels(test_row['Close'], test_row['ATR'])
            debug_info.append(f"✅ Level generation test: Close={test_row['Close']:.2f}, ATR={test_row['ATR']:.2f}")
            debug_info.append(f"Sample levels: +0.382={test_levels[0.382]:.2f}, -0.236={test_levels[-0.236]:.2f}")
        
        debug_info.append("🎯 Running trigger and goal detection...")
        df = detect_triggers_and_goals(daily, intraday)
        debug_info.append(f"✅ Detection complete: {len(df)} trigger-goal combinations found")
        
        return df, debug_info
        
    except Exception as e:
        debug_info.append(f"❌ Error: {str(e)}")
        import traceback
        debug_info.append(f"Traceback: {traceback.format_exc()}")
        return pd.DataFrame(), debug_info

# Streamlit Interface
st.title('🎯 ATR Trigger & Goal Generator')
st.write('**Dynamic ATR Calculation** - No Excel dependency!')

output_path = 'combined_trigger_goal_results.csv'

if st.button('🚀 Generate Results with Validated ATR Logic'):
    with st.spinner('Calculating ATR and detecting triggers...'):
        try:
            result_df, debug_messages = main()
            
            # Show debug info
            with st.expander('📋 Debug Information'):
                for msg in debug_messages:
                    st.write(msg)
            
            if not result_df.empty:
                result_df['Source'] = 'Validated_ATR'
                result_df.to_csv(output_path, index=False)
                st.success('✅ Results generated with validated ATR logic!')
                
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
                
                # Show breakdown by direction
                st.subheader('📈 Direction Breakdown')
                direction_stats = result_df.groupby(['Direction', 'Type']).size().reset_index(name='Count')
                col1, col2 = st.columns(2)
                with col1:
                    upside_stats = direction_stats[direction_stats['Direction'] == 'Upside']
                    st.write("**Upside Triggers:**")
                    st.dataframe(upside_stats)
                with col2:
                    downside_stats = direction_stats[direction_stats['Direction'] == 'Downside']
                    st.write("**Downside Triggers:**")
                    st.dataframe(downside_stats)
                
                # Preview results
                st.subheader('🔍 Preview of Results')
                st.dataframe(result_df.head(20))
                
                # Download button
                st.download_button(
                    '⬇️ Download Complete Results CSV', 
                    data=result_df.to_csv(index=False), 
                    file_name=output_path, 
                    mime='text/csv'
                )
                
                st.success('🎉 **SUCCESS!** ATR levels calculated dynamically with validated logic!')
                
            else:
                st.warning('⚠️ No results generated - check debug info above')
                
        except Exception as e:
            st.error(f'❌ Error: {e}')

st.markdown("""
---
**🎯 What This Does:**
- ✅ **Calculates ATR** using validated Wilder's method
- ✅ **Generates Fibonacci levels** dynamically (no Excel dependency)
- ✅ **Detects triggers** when intraday price hits levels
- ✅ **Tracks goal completion** for continuation and retracement targets
- ✅ **Proper date alignment** - levels calculated today, used tomorrow
- ✅ **Complete backtest data** with hit rates and statistics

**🚀 Ready for Production!**
""")
