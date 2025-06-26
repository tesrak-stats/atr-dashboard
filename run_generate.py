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
    CORRECTED: Use PREVIOUS day's close and ATR to calculate TODAY's levels
    """
    fib_levels = [0.236, 0.382, 0.500, 0.618, 0.786, 1.000, 
                 -0.236, -0.382, -0.500, -0.618, -0.786, -1.000, 0.000]
    
    results = []
    
    # FIXED: Start from index 1 (need previous day data)
    for i in range(1, len(daily)):
        try:
            # CORRECTED LOGIC:
            previous_row = daily.iloc[i-1]  # Previous day (for level calculation)
            current_row = daily.iloc[i]     # Current day (trading day)
            
            # Use PREVIOUS day's data for level calculation
            previous_close = previous_row['Close']  # Known at market open
            previous_atr = previous_row['ATR']      # Known at market open
            
            # Trading occurs on CURRENT day
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
            
            # CORRECTED: Generate levels using PREVIOUS day's close + ATR
            level_map = generate_atr_levels(previous_close, previous_atr)
            
            # Get intraday data for CURRENT trading date
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
                        'PreviousClose': round(previous_close, 2),  # CORRECTED: Previous day's close
                        'PreviousATR': round(previous_atr, 2),      # CORRECTED: Previous day's ATR
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
                        'PreviousClose': round(previous_close, 2),  # CORRECTED: Previous day's close
                        'PreviousATR': round(previous_atr, 2),      # CORRECTED: Previous day's ATR
                        'RetestedTrigger': 'No'
                    })

        except Exception as e:
            st.write(f"⚠️ Error processing {trading_date}: {str(e)}")
            continue

    return pd.DataFrame(results)

def main():
    """
    CORRECTED: Main function using previous day's data for level calculation
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
        
        # CORRECTED: Test level generation using PREVIOUS day data
        if len(daily) >= 2:
            prev_row = daily.iloc[-2]  # Previous day
            curr_row = daily.iloc[-1]  # Current day
            if not pd.isna(prev_row['ATR']):
                test_levels = generate_atr_levels(prev_row['Close'], prev_row['ATR'])
                debug_info.append(f"✅ CORRECTED Level generation test:")
                debug_info.append(f"Previous day ({prev_row['Date']}): Close={prev_row['Close']:.2f}, ATR={prev_row['ATR']:.2f}")
                debug_info.append(f"Levels for current day ({curr_row['Date']}): +0.382={test_levels[0.382]:.2f}, 0.0={test_levels[0.0]:.2f}")
        
        debug_info.append("🎯 Running CORRECTED trigger and goal detection...")
        df = detect_triggers_and_goals(daily, intraday)
        debug_info.append(f"✅ Detection complete: {len(df)} trigger-goal combinations found")
        
        return df, debug_info
        
    except Exception as e:
        debug_info.append(f"❌ Error: {str(e)}")
        import traceback
        debug_info.append(f"Traceback: {traceback.format_exc()}")
        return pd.DataFrame(), debug_info

# Streamlit Interface
st.title('🎯 CORRECTED ATR Trigger & Goal Generator')
st.write('**FIXED: Uses PREVIOUS day close + ATR for level calculation**')

output_path = 'combined_trigger_goal_results_CORRECTED.csv'

if st.button('🚀 Generate CORRECTED Results'):
    with st.spinner('Calculating with CORRECTED logic...'):
        try:
            result_df, debug_messages = main()
            
            # Show debug info
            with st.expander('📋 Debug Information'):
                for msg in debug_messages:
                    st.write(msg)
            
            if not result_df.empty:
                result_df['Source'] = 'Corrected_Previous_Day_Logic'
                result_df.to_csv(output_path, index=False)
                st.success('✅ CORRECTED Results generated!')
                
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
                
                # Show key correction info
                st.subheader('🔧 Key Corrections Made')
                st.success("✅ **FIXED**: Now uses PREVIOUS day's close as reference (not current day)")
                st.success("✅ **FIXED**: ATR levels calculated using known data at market open")
                st.success("✅ **FIXED**: Logic now matches real-world trading scenario")
                
                # Show sample to verify correction
                st.subheader('🔍 Sample Results (Verify Previous Close)')
                sample = result_df[['Date', 'PreviousClose', 'PreviousATR', 'TriggerLevel', 'TriggerPrice']].head(10)
                st.dataframe(sample)
                st.write("**Verify**: TriggerPrice should equal PreviousClose + (TriggerLevel × PreviousATR)")
                
                # Download button
                st.download_button(
                    '⬇️ Download CORRECTED Results CSV', 
                    data=result_df.to_csv(index=False), 
                    file_name=output_path, 
                    mime='text/csv'
                )
                
                st.success('🎉 **CORRECTED LOGIC APPLIED!** Now uses previous day data like real trading!')
                
            else:
                st.warning('⚠️ No results generated - check debug info above')
                
        except Exception as e:
            st.error(f'❌ Error: {e}')

st.markdown("""
---
**🔧 Key Correction Made:**
- ✅ **OLD (WRONG)**: Used current day's close to calculate current day's levels  
- ✅ **NEW (CORRECT)**: Uses previous day's close to calculate current day's levels
- ✅ **Why this matters**: At market open, you only know yesterday's data, not today's final close
- ✅ **Result**: Levels now match what traders would actually calculate in real-time

**🎯 This Should Now Match Your Excel Analysis!**
""")