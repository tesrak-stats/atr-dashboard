import streamlit as st
import pandas as pd
import numpy as np

# Copy your exact functions for validation
def calculate_atr(df, period=14):
    """Calculate Wilder's ATR (Average True Range) - VALIDATED"""
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
    """Generate Fibonacci-based ATR levels - VALIDATED"""
    fib_ratios = [0.236, 0.382, 0.500, 0.618, 0.786, 1.000, 
                  -0.236, -0.382, -0.500, -0.618, -0.786, -1.000, 0.000]
    
    levels = {}
    for ratio in fib_ratios:
        level_price = close_price + (ratio * atr_value)
        levels[ratio] = level_price
    
    return levels

st.title("🔍 Logic Validator - Using Your Exact Code")
st.write("Step-by-step validation of your run_generate.py logic")

# File uploads
col1, col2 = st.columns(2)
with col1:
    daily_file = st.file_uploader("Upload SPXdailycandles.xlsx", type="xlsx")
with col2:
    intraday_file = st.file_uploader("Upload SPX_10min.csv", type="csv")

if daily_file is not None and intraday_file is not None:
    
    # Load data using your exact logic
    st.write("## Step 1: Load Data (Using Your Exact Logic)")
    
    try:
        # Your exact loading logic
        daily = pd.read_excel(daily_file, header=4)
        st.write(f"✅ Daily data loaded: {daily.shape}")
        st.write("Daily columns:", list(daily.columns))
        
        # Your exact ATR calculation
        daily = calculate_atr(daily, period=14)
        st.write("✅ ATR calculated using your exact method")
        
        # Your exact intraday loading
        intraday = pd.read_csv(intraday_file, parse_dates=['Datetime'])
        intraday['Date'] = intraday['Datetime'].dt.date
        st.write(f"✅ Intraday data loaded: {intraday.shape}")
        
    except Exception as e:
        st.error(f"Error loading data: {e}")
        st.stop()
    
    # Step 2: Pick a test date
    st.write("## Step 2: Pick Test Date")
    
    # Show recent dates
    recent_dates = daily['Date'].tail(20).tolist()
    test_date = st.selectbox("Select test date:", recent_dates)
    
    if test_date:
        st.write(f"### Testing: {test_date}")
        
        # Find the corresponding daily row (like your logic)
        daily_idx = daily[daily['Date'] == test_date].index
        
        if len(daily_idx) > 0 and daily_idx[0] > 0:
            # Your logic: use previous day for level calculation
            calc_idx = daily_idx[0] - 1
            calc_row = daily.iloc[calc_idx]
            trading_row = daily.iloc[daily_idx[0]]
            
            st.write(f"**Calculation day**: {calc_row['Date']} (Close: {calc_row['Close']:.2f}, ATR: {calc_row['ATR']:.2f})")
            st.write(f"**Trading day**: {trading_row['Date']}")
            
            # Generate levels using your exact logic
            level_map = generate_atr_levels(calc_row['Close'], calc_row['ATR'])
            
            st.write("### Generated Levels (Your Exact Logic):")
            levels_df = pd.DataFrame([
                {'Ratio': k, 'Price': v} for k, v in level_map.items()
            ]).sort_values('Ratio')
            st.dataframe(levels_df)
            
            # Get intraday data for trading date
            trading_date_obj = pd.to_datetime(trading_row['Date']).date()
            day_data = intraday[intraday['Date'] == trading_date_obj].copy()
            
            if not day_data.empty:
                st.write(f"✅ Found {len(day_data)} intraday candles for {trading_date_obj}")
                
                # Add time column like your logic
                day_data['Time'] = day_data['Datetime'].dt.strftime('%H%M')
                day_data.reset_index(drop=True, inplace=True)
                
                st.write("### Intraday Data Sample:")
                st.dataframe(day_data[['Datetime', 'Time', 'Open', 'High', 'Low', 'Close']].head(10))
                
                # Step 3: Test trigger detection (your exact logic)
                st.write("## Step 3: Test Trigger Detection")
                
                # Your exact fib_levels list
                fib_levels = [0.236, 0.382, 0.500, 0.618, 0.786, 1.000, 
                             -0.236, -0.382, -0.500, -0.618, -0.786, -1.000, 0.000]
                
                triggered_up = {}
                triggered_down = {}
                
                # Process each candle (your exact logic)
                for idx, row in day_data.iterrows():
                    open_price = row['Open']
                    high = row['High']
                    low = row['Low']
                    
                    # Your exact time label logic
                    time_label = 'OPEN' if idx == 0 and (
                        any(open_price >= level_map[level] for level in fib_levels if level > 0) or
                        any(open_price <= level_map[level] for level in fib_levels if level < 0)
                    ) else row['Time']
                    
                    # Your exact upside trigger logic (FIXED: include 0.0)
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
                                'TriggerPrice': trigger_price,
                                'ActualPrice': open_price if time_label == 'OPEN' else high
                            }
                    
                    # Your exact downside trigger logic
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
                                'TriggerPrice': trigger_price,
                                'ActualPrice': open_price if time_label == 'OPEN' else low
                            }
                
                # Show triggers found
                st.write("### Upside Triggers Found:")
                if triggered_up:
                    upside_df = pd.DataFrame(triggered_up.values())
                    st.dataframe(upside_df)
                else:
                    st.write("No upside triggers found")
                
                st.write("### Downside Triggers Found:")
                if triggered_down:
                    downside_df = pd.DataFrame(triggered_down.values())
                    st.dataframe(downside_df)
                else:
                    st.write("No downside triggers found")
                
                # Step 4: Test goal detection for 0.0 upside triggers
                st.write("## Step 4: Test Goal Detection for 0.0 Upside Triggers")
                
                if 0.0 in triggered_up:
                    zero_trigger = triggered_up[0.0]
                    trigger_row = zero_trigger['TriggeredRow']
                    
                    st.write(f"**0.0 Trigger Found**: {zero_trigger['TriggerTime']} at price {zero_trigger['TriggerPrice']:.2f}")
                    
                    # Test your exact goal logic for upside continuation
                    continuation_goals = []
                    retracement_goals = []
                    
                    for goal_level in fib_levels:
                        if goal_level == 0.0:  # Skip same level
                            continue
                        
                        goal_price = level_map[goal_level]
                        goal_hit = False
                        goal_time = ''
                        
                        # Your exact classification logic
                        if goal_level > 0.0:  # Higher than trigger
                            goal_type = 'Continuation'
                            # Check subsequent candles for upside goal
                            for _, row in day_data.iloc[trigger_row + 1:].iterrows():
                                if row['High'] >= goal_price:
                                    goal_hit = True
                                    goal_time = row['Time']
                                    break
                        else:  # Lower than trigger
                            goal_type = 'Retracement'
                            # Check subsequent candles for downside goal
                            for _, row in day_data.iloc[trigger_row + 1:].iterrows():
                                if row['Low'] <= goal_price:
                                    goal_hit = True
                                    goal_time = row['Time']
                                    break
                        
                        goal_result = {
                            'GoalLevel': goal_level,
                            'GoalPrice': goal_price,
                            'GoalType': goal_type,
                            'GoalHit': 'Yes' if goal_hit else 'No',
                            'GoalTime': goal_time
                        }
                        
                        if goal_type == 'Continuation':
                            continuation_goals.append(goal_result)
                        else:
                            retracement_goals.append(goal_result)
                    
                    # Show results
                    st.write("### Continuation Goals (0.0 → Higher Levels):")
                    if continuation_goals:
                        cont_df = pd.DataFrame(continuation_goals)
                        st.dataframe(cont_df)
                        
                        # Highlight successful ones
                        successful_cont = cont_df[cont_df['GoalHit'] == 'Yes']
                        if len(successful_cont) > 0:
                            st.success(f"✅ {len(successful_cont)} continuation goals hit!")
                        
                    st.write("### Retracement Goals (0.0 → Lower Levels):")
                    if retracement_goals:
                        ret_df = pd.DataFrame(retracement_goals)
                        st.dataframe(ret_df)
                        
                        # Highlight successful ones
                        successful_ret = ret_df[ret_df['GoalHit'] == 'Yes']
                        if len(successful_ret) > 0:
                            st.success(f"✅ {len(successful_ret)} retracement goals hit!")
                
                else:
                    st.warning("⚠️ No 0.0 upside trigger found for this date")
                
                # Summary for validation
                st.write("## Step 5: Validation Summary")
                
                total_upside = len(triggered_up)
                total_downside = len(triggered_down)
                has_zero_trigger = 0.0 in triggered_up
                
                st.write(f"**Date**: {test_date}")
                st.write(f"**Previous Close**: {calc_row['Close']:.2f}")
                st.write(f"**ATR**: {calc_row['ATR']:.2f}")
                st.write(f"**0.0 Level Price**: {level_map[0.0]:.2f}")
                st.write(f"**Total Upside Triggers**: {total_upside}")
                st.write(f"**Total Downside Triggers**: {total_downside}")
                st.write(f"**0.0 Trigger Found**: {'✅ Yes' if has_zero_trigger else '❌ No'}")
                
                if has_zero_trigger:
                    st.success("🎯 Your logic successfully detects 0.0 triggers!")
                else:
                    st.error("❌ 0.0 trigger not detected - may need investigation")
                
            else:
                st.error(f"No intraday data found for {trading_date_obj}")
        else:
            st.error("Cannot find daily data for selected date or it's the first date")

else:
    st.info("👆 Upload both files to validate your exact logic")

st.write("---")
st.write("**Goal**: Verify your run_generate.py logic is working correctly for the scenarios you validated in Excel")
