import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta, date

class AssetConfig:
    """Configuration for different asset classes"""
    
    @staticmethod
    def get_config(asset_type, extended_hours=False):
        configs = {
            'STOCKS': {
                'market_open': '04:00' if extended_hours else '09:30',
                'market_close': '20:00' if extended_hours else '16:00',
                'has_open_special': True,
                'weekends_closed': True,
                'session_types': ['PM', 'R', 'AH'] if extended_hours else ['R'],
                'default_session': ['PM', 'R', 'AH'] if extended_hours else ['R'],
                'description': f'US Stocks ({"Extended Hours 4AM-8PM" if extended_hours else "Regular Hours 9:30AM-4PM"})',
                'extended_hours': extended_hours
            },
            'CRYPTO': {
                'market_open': '00:00',
                'market_close': '23:59',
                'has_open_special': False,
                'weekends_closed': False,
                'session_types': ['24H'],
                'default_session': ['24H'],
                'description': 'Cryptocurrency (24/7 trading)',
                'extended_hours': True
            },
            'FOREX': {
                'market_open': '17:00',
                'market_close': '17:00',
                'has_open_special': False,
                'weekends_closed': True,
                'session_types': ['ASIA', 'EUROPE', 'US', '24H'],
                'default_session': ['24H'],
                'description': 'Foreign Exchange (Sun 5PM - Fri 5PM EST)',
                'extended_hours': True
            },
            'FUTURES': {
                'market_open': '18:00',
                'market_close': '17:00',
                'has_open_special': True,
                'weekends_closed': True,
                'session_types': ['GLOBEX', 'RTH'],
                'default_session': ['GLOBEX', 'RTH'],
                'description': 'Futures (nearly 24/5 trading)',
                'extended_hours': True
            }
        }
        return configs.get(asset_type, configs['STOCKS'])

def load_preformatted_data(uploaded_file):
    """Load pre-formatted data from the CSV Data Handler"""
    try:
        # Read the CSV file
        df = pd.read_csv(uploaded_file)
        
        # Validate required columns
        required_cols = ['Datetime', 'Open', 'High', 'Low', 'Close', 'Date', 'Time', 'ATR', 'Prior_Base_Close', 'Trading_Days_Count']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            st.error(f"Missing required columns: {missing_cols}")
            st.info(f"Available columns: {list(df.columns)}")
            return None
        
        # Convert datetime and date columns
        df['Datetime'] = pd.to_datetime(df['Datetime'])
        df['Date'] = pd.to_datetime(df['Date']).dt.date
        
        # Sort by datetime
        df = df.sort_values('Datetime').reset_index(drop=True)
        
        st.success(f"Loaded pre-formatted data: {len(df):,} records")
        st.info(f"Date range: {df['Date'].min()} to {df['Date'].max()}")
        
        return df
        
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        return None

def generate_atr_levels(close_price, atr_value, custom_ratios=None):
    """
    Generate ATR levels with customizable ratios
    """
    if custom_ratios is None:
        fib_ratios = [0.236, 0.382, 0.500, 0.618, 0.786, 1.000, 
                      -0.236, -0.382, -0.500, -0.618, -0.786, -1.000, 0.000]
    else:
        fib_ratios = custom_ratios
    
    levels = {}
    for ratio in fib_ratios:
        level_price = close_price + (ratio * atr_value)
        levels[ratio] = level_price
    
    return levels

# ==============================================================================================
# CRITICAL SECTION: DO NOT MODIFY THE CORE TRIGGER AND GOAL DETECTION LOGIC
# This section contains the validated systematic logic from run_generate.py
# ==============================================================================================

def detect_triggers_and_goals_systematic(daily, intraday, custom_ratios=None):
    """
    PERFECT SYSTEMATIC LOGIC FROM run_generate.py:
    For each trigger level:
    1. Check if LOW <= trigger (Below direction) → check all 12 goals
    2. Check if HIGH >= trigger (Above direction) → check all 12 goals
    
    For goals:
    - Above goals: check HIGH >= goal
    - Below goals: check LOW <= goal
    
    FIXED: 0930 candle goal completion logic
    """
    if custom_ratios is None:
        fib_levels = [0.236, 0.382, 0.500, 0.618, 0.786, 1.000, 
                     -0.236, -0.382, -0.500, -0.618, -0.786, -1.000, 0.000]
    else:
        fib_levels = custom_ratios
    
    # Standard trading hours for zero-fill
    standard_hours = ['OPEN', '0930', '0940', '0950', '1000', '1010', '1020', '1030', 
                      '1040', '1050', '1100', '1110', '1120', '1130', '1140', '1150',
                      '1200', '1210', '1220', '1230', '1240', '1250', '1300', '1310', 
                      '1320', '1330', '1340', '1350', '1400', '1410', '1420', '1430',
                      '1440', '1450', '1500', '1510', '1520', '1530', '1540', '1550', '1600']
    
    results = []
    
    # Progress tracking
    total_days = len(daily)
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    for i in range(len(daily)):
        try:
            # Update progress
            progress = (i + 1) / total_days
            progress_bar.progress(progress)
            status_text.text(f"Processing day {i+1}/{total_days}...")
            
            # Use CURRENT day's Pre-calculated Prior_Base_Close and ATR
            current_row = daily.iloc[i]     
            
            previous_close = current_row['Prior_Base_Close']  # Pre-calculated previous close
            previous_atr = current_row['ATR']                 # Pre-calculated ATR
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
            
            # Skip if no valid ATR (early days before period completion)
            if pd.isna(previous_atr) or pd.isna(previous_close):
                continue
            
            # Generate levels using PREVIOUS day's close + ATR
            level_map = generate_atr_levels(previous_close, previous_atr, custom_ratios)
            
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
                
                # If OPEN didn't trigger, check 0930 candle High/Low
                elif day_data.iloc[0]['Low'] <= trigger_price:
                    below_triggered = True
                    below_trigger_time = '0930'
                    below_trigger_row = 0
                
                # Check intraday candles for below trigger (only if neither OPEN nor 0930 triggered)
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
                        goal_price = level_map[goal_level]
                        goal_hit = False
                        goal_time = ''
                        is_same_time = False
                        
                        # Determine goal type for BELOW trigger
                        if goal_level == trigger_level:
                            goal_type = 'Retest'  # Same level retest
                        elif goal_level < trigger_level:
                            goal_type = 'Continuation'  # Further below
                        else:
                            goal_type = 'Retracement'   # Back above (includes cross-zero)
                        
                        # Check for goal completion - FIXED LOGIC (including same-level retests)
                        if below_trigger_time == 'OPEN':
                            # Step 1: Check if goal completes at OPEN price first (takes precedence)
                            if goal_level == trigger_level:  # Same level retest
                                # For same-level retest, we need opposite direction movement
                                # Below trigger at OPEN, so retest needs Above movement
                                if open_price >= goal_price:
                                    goal_hit = True
                                    goal_time = 'OPEN'
                                    is_same_time = True
                            elif goal_level > trigger_level:  # Above goal (RETRACEMENT)
                                if open_price >= goal_price:
                                    goal_hit = True
                                    goal_time = 'OPEN'
                                    is_same_time = True
                            else:  # Below goal (CONTINUATION)
                                if open_price <= goal_price:
                                    goal_hit = True
                                    goal_time = 'OPEN'
                                    is_same_time = True
                            
                            # Step 2: Only if OPEN missed, check candles based on goal type
                            if not goal_hit:
                                # CRITICAL: Different logic for CONTINUATION vs RETRACEMENT vs RETEST
                                if goal_level == trigger_level:  # RETEST - must skip same candle (like retracement)
                                    start_candles = day_data.iloc[1:].iterrows()  # Skip 0930, start from 0940
                                elif goal_level > trigger_level:  # RETRACEMENT - must skip same candle (0930), start from 0940
                                    start_candles = day_data.iloc[1:].iterrows()  # Skip 0930, start from 0940
                                else:  # CONTINUATION - can check same candle (0930)
                                    start_candles = day_data.iterrows()  # Include 0930 candle
                                
                                for _, row in start_candles:
                                    if goal_level == trigger_level:  # Same level retest (opposite direction)
                                        if row['High'] >= goal_price:  # Below trigger needs High to retest
                                            goal_hit = True
                                            goal_time = row['Time']
                                            break
                                    elif goal_level > trigger_level:  # Above goal
                                        if row['High'] >= goal_price:  # Use High, not Open
                                            goal_hit = True
                                            goal_time = row['Time']
                                            break
                                    else:  # Below goal  
                                        if row['Low'] <= goal_price:  # Use Low, not Open
                                            goal_hit = True
                                            goal_time = row['Time']
                                            break
                        
                        else:  # Intraday below trigger (e.g., 1000, 1100, etc.)
                            if goal_level == trigger_level:  # RETEST - Skip same candle entirely
                                # DO NOT check trigger candle - start from next candle only
                                pass  # Skip same-candle check for retests
                            elif goal_level > trigger_level:  # RETRACEMENT - Skip same candle entirely  
                                # DO NOT check trigger candle - start from next candle only
                                pass  # Skip same-candle check for retracements
                            else:  # CONTINUATION - Can check same candle
                                if goal_level < trigger_level:  # Below goal
                                    if trigger_candle['Low'] <= goal_price:
                                        goal_hit = True
                                        goal_time = below_trigger_time
                            
                            # Check subsequent candles if not completed on trigger candle
                            if not goal_hit:
                                for _, row in day_data.iloc[below_trigger_row + 1:].iterrows():
                                    if goal_level == trigger_level:  # Same level retest (opposite direction)
                                        if row['High'] >= goal_price:  # Below trigger needs High to retest
                                            goal_hit = True
                                            goal_time = row['Time']
                                            break
                                    elif goal_level > trigger_level:  # Above goal
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
                
                # If OPEN didn't trigger, check 0930 candle High/Low
                elif day_data.iloc[0]['High'] >= trigger_price:
                    above_triggered = True
                    above_trigger_time = '0930'
                    above_trigger_row = 0
                
                # Check intraday candles for above trigger (only if neither OPEN nor 0930 triggered)
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
                        goal_price = level_map[goal_level]
                        goal_hit = False
                        goal_time = ''
                        is_same_time = False
                        
                        # Determine goal type for ABOVE trigger
                        if goal_level == trigger_level:
                            goal_type = 'Retest'  # Same level retest
                        elif goal_level > trigger_level:
                            goal_type = 'Continuation'  # Further above
                        else:
                            goal_type = 'Retracement'   # Back below (includes cross-zero)
                        
                        # Check for goal completion - FIXED LOGIC (including same-level retests)
                        if above_trigger_time == 'OPEN':
                            # Step 1: Check if goal completes at OPEN price first (takes precedence)
                            if goal_level == trigger_level:  # Same level retest
                                # For same-level retest, we need opposite direction movement
                                # Above trigger at OPEN, so retest needs Below movement
                                if open_price <= goal_price:
                                    goal_hit = True
                                    goal_time = 'OPEN'
                                    is_same_time = True
                            elif goal_level > trigger_level:  # Above goal (CONTINUATION)
                                if open_price >= goal_price:
                                    goal_hit = True
                                    goal_time = 'OPEN'
                                    is_same_time = True
                            else:  # Below goal (RETRACEMENT)
                                if open_price <= goal_price:
                                    goal_hit = True
                                    goal_time = 'OPEN'
                                    is_same_time = True
                            
                            # Step 2: Only if OPEN missed, check ALL candles including 0930 (but use High/Low, not Open)
                            if not goal_hit:
                                for _, row in day_data.iterrows():  # FIXED: Include 0930 candle
                                    if goal_level == trigger_level:  # Same level retest (opposite direction)
                                        if row['Low'] <= goal_price:  # Above trigger needs Low to retest
                                            goal_hit = True
                                            goal_time = row['Time']
                                            break
                                    elif goal_level > trigger_level:  # Above goal
                                        if row['High'] >= goal_price:  # Use High, not Open
                                            goal_hit = True
                                            goal_time = row['Time']
                                            break
                                    else:  # Below goal
                                        if row['Low'] <= goal_price:  # Use Low, not Open
                                            goal_hit = True
                                            goal_time = row['Time']
                                            break
                        
                        else:  # Intraday above trigger (e.g., 1000, 1100, etc.)
                            if goal_level == trigger_level:  # RETEST - Skip same candle entirely
                                # DO NOT check trigger candle - start from next candle only
                                pass  # Skip same-candle check for retests
                            elif goal_level < trigger_level:  # RETRACEMENT - Skip same candle entirely
                                # DO NOT check trigger candle - start from next candle only
                                pass  # Skip same-candle check for retracements
                            else:  # CONTINUATION - Can check same candle
                                if goal_level > trigger_level:  # Above goal
                                    if trigger_candle['High'] >= goal_price:
                                        goal_hit = True
                                        goal_time = above_trigger_time
                            
                            # Check subsequent candles if not completed on trigger candle
                            if not goal_hit:
                                for _, row in day_data.iloc[above_trigger_row + 1:].iterrows():
                                    if goal_level == trigger_level:  # Same level retest (opposite direction)
                                        if row['Low'] <= goal_price:  # Above trigger needs Low to retest
                                            goal_hit = True
                                            goal_time = row['Time']
                                            break
                                    elif goal_level > trigger_level:  # Above goal
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
            st.warning(f"Error processing {trading_date}: {str(e)}")
            continue
    
    # Clear progress indicators
    progress_bar.empty()
    status_text.empty()
    
    return pd.DataFrame(results)

def debug_single_day_analysis(daily, intraday, debug_date, custom_ratios=None):
    """Quick debug mode: Analyze a single day with detailed 10-minute breakdown"""
    if custom_ratios is None:
        fib_levels = [0.236, 0.382, 0.500, 0.618, 0.786, 1.000, 
                     -0.236, -0.382, -0.500, -0.618, -0.786, -1.000, 0.000]
    else:
        fib_levels = custom_ratios
    
    st.subheader(f"Debug Analysis for {debug_date}")
    
    # Find the debug date in daily data
    daily_debug = daily[daily['Date'].dt.date == debug_date]
    if daily_debug.empty:
        st.error(f"Date {debug_date} not found in daily data")
        return
    
    # Get the row for debug date
    debug_row = daily_debug.iloc[0]
    
    previous_close = debug_row['Prior_Base_Close']
    previous_atr = debug_row['ATR']
    
    if pd.isna(previous_atr):
        st.error(f"No valid ATR for debug date")
        return
    
    # Generate ATR levels
    level_map = generate_atr_levels(previous_close, previous_atr, custom_ratios)
    
    # Get intraday data for debug date
    day_data = intraday[intraday['Date'] == debug_date].copy()
    if day_data.empty:
        st.error(f"No intraday data found for {debug_date}")
        return
    
    day_data['Time'] = day_data['Datetime'].dt.strftime('%H%M')
    day_data.reset_index(drop=True, inplace=True)
    
    # Display setup info
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Prior Base Close", f"{previous_close:.2f}")
    with col2:
        st.metric("ATR", f"{previous_atr:.2f}")
    with col3:
        st.metric("Intraday Candles", len(day_data))
    
    # Show ATR levels
    st.subheader("ATR Levels for Debug Date")
    levels_df = pd.DataFrame([
        {"Level": level, "Price": f"{price:.2f}"} 
        for level, price in sorted(level_map.items(), key=lambda x: x[1], reverse=True)
    ])
    st.dataframe(levels_df, use_container_width=True)
    
    # Detailed analysis
    st.subheader("10-Minute Candle Analysis")
    
    open_price = day_data.iloc[0]['Open']
    st.info(f"Opening Price: {open_price:.2f}")
    
    # Analyze each candle
    candle_analysis = []
    
    for idx, candle in day_data.iterrows():
        time_str = candle['Time']
        open_val = candle['Open']
        high_val = candle['High']
        low_val = candle['Low']
        close_val = candle['Close']
        
        # Check what levels this candle interacts with
        triggered_levels = []
        
        for level, price in level_map.items():
            level_triggered = False
            trigger_type = None
            
            # Check if this candle triggers the level
            if idx == 0:  # First candle (0930)
                # Check OPEN trigger first
                if (level >= 0 and open_val >= price) or (level < 0 and open_val <= price):
                    level_triggered = True
                    trigger_type = "OPEN"
                # Check High/Low trigger if OPEN didn't trigger
                elif (level >= 0 and high_val >= price) or (level < 0 and low_val <= price):
                    level_triggered = True
                    trigger_type = "0930"
            else:
                # Regular intraday candle
                if (level >= 0 and high_val >= price) or (level < 0 and low_val <= price):
                    level_triggered = True
                    trigger_type = time_str
            
            if level_triggered:
                direction = "Above" if level >= 0 else "Below"
                triggered_levels.append({
                    "Level": level,
                    "Price": price,
                    "Direction": direction,
                    "Type": trigger_type
                })
        
        candle_analysis.append({
            "Time": time_str,
            "Open": f"{open_val:.2f}",
            "High": f"{high_val:.2f}",
            "Low": f"{low_val:.2f}",
            "Close": f"{close_val:.2f}",
            "Triggered_Levels": len(triggered_levels),
            "Details": triggered_levels
        })
    
    # Display candle analysis
    for analysis in candle_analysis:
        if analysis["Triggered_Levels"] > 0:
            with st.expander(f"{analysis['Time']} - {analysis['Triggered_Levels']} triggers"):
                col1, col2 = st.columns([1, 2])
                
                with col1:
                    st.write("**OHLC:**")
                    st.write(f"Open: {analysis['Open']}")
                    st.write(f"High: {analysis['High']}")
                    st.write(f"Low: {analysis['Low']}")
                    st.write(f"Close: {analysis['Close']}")
                
                with col2:
                    st.write("**Triggered Levels:**")
                    for detail in analysis["Details"]:
                        st.write(f"• **{detail['Level']}** ({detail['Direction']}) @ {detail['Price']:.2f} - Type: {detail['Type']}")
        else:
            st.write(f"**{analysis['Time']}**: O:{analysis['Open']} H:{analysis['High']} L:{analysis['Low']} C:{analysis['Close']} - No triggers")
    
    # Summary
    total_triggers = sum(len(a["Details"]) for a in candle_analysis)
    st.success(f"**Debug Summary**: {total_triggers} total level triggers detected across {len(day_data)} candles")

# ==============================================================================================
# END OF CRITICAL SECTION
# ==============================================================================================

# Main analysis function - now properly handles the CSV handler format
def main_analysis(ticker, asset_type, data_file, custom_ratios=None, debug_mode=False, debug_date=None):
    """Main function for pre-formatted CSV analysis"""
    debug_info = []
    
    try:
        # Debug mode check
        if debug_mode and debug_date:
            st.success(f"DEBUG MODE - Will process ONLY {debug_date}")
        else:
            st.info("FULL MODE - Will process all days")
        
        # Get asset configuration
        asset_config = AssetConfig.get_config(asset_type, False)
        debug_info.append(f"Asset Type: {asset_config['description']}")
        
        # Load pre-formatted data
        df = load_preformatted_data(data_file)
        if df is None:
            debug_info.append("Failed to load pre-formatted data")
            return pd.DataFrame(), debug_info
        
        debug_info.append(f"Data loaded: {df.shape}")
        debug_info.append(f"Date range: {df['Date'].min()} to {df['Date'].max()}")
        
        # Validate required data
        if df['ATR'].isna().all():
            debug_info.append("No valid ATR values found")
            return pd.DataFrame(), debug_info
        
        recent_atr = df['ATR'].tail(3).round(2).tolist()
        debug_info.append(f"ATR values found. Recent values: {recent_atr}")
        
        # Prepare data for systematic analysis
        # Create daily data - one record per date with ATR and Prior_Base_Close
        daily_data = df.groupby('Date').agg({
            'Open': 'first',
            'High': 'max', 
            'Low': 'min',
            'Close': 'last',
            'ATR': 'first',
            'Prior_Base_Close': 'first'
        }).reset_index()
        daily_data['Date'] = pd.to_datetime(daily_data['Date'])
        
        # Intraday data is the full dataframe
        intraday_data = df.copy()
        
        debug_info.append(f"Daily data prepared: {len(daily_data)} days")
        debug_info.append(f"Intraday data prepared: {len(intraday_data)} records")
        
        # Debug Mode - single day analysis
        if debug_mode and debug_date:
            st.info(f"Debug Mode Active - Analyzing single day: {debug_date}")
            debug_single_day_analysis(daily_data, intraday_data, debug_date, custom_ratios)
            return pd.DataFrame(), debug_info + [f"Debug analysis completed for {debug_date}"]
        
        # Run full systematic analysis
        if not debug_mode:
            debug_info.append("Running SYSTEMATIC trigger and goal detection...")
            result_df = detect_triggers_and_goals_systematic(daily_data, intraday_data, custom_ratios)
            debug_info.append(f"Detection complete: {len(result_df)} trigger-goal combinations found")
            
            # Add Trading_Days_Count to results (from last record in the dataset)
            if not result_df.empty and 'Trading_Days_Count' in df.columns:
                final_trading_days = df['Trading_Days_Count'].iloc[-1]
                result_df['Trading_Days_Count'] = final_trading_days
                debug_info.append(f"Added Trading_Days_Count: {final_trading_days}")
            
            # Additional statistics
            if not result_df.empty:
                above_triggers = len(result_df[result_df['Direction'] == 'Above'])
                below_triggers = len(result_df[result_df['Direction'] == 'Below'])
                debug_info.append(f"Above triggers: {above_triggers}, Below triggers: {below_triggers}")
                
                goals_hit = len(result_df[result_df['GoalHit'] == 'Yes'])
                hit_rate = goals_hit / len(result_df) * 100 if len(result_df) > 0 else 0
                debug_info.append(f"Goals hit: {goals_hit}/{len(result_df)} ({hit_rate:.1f}%)")
                
                # Validation metrics
                same_time_count = len(result_df[result_df['SameTime'] == True])
                debug_info.append(f"Same-time scenarios found: {same_time_count}")
                
                open_triggers = len(result_df[result_df['TriggerTime'] == 'OPEN'])
                intraday_triggers = len(result_df[result_df['TriggerTime'] != 'OPEN'])
                debug_info.append(f"OPEN triggers: {open_triggers}, Intraday triggers: {intraday_triggers}")
            
            return result_df, debug_info
        else:
            debug_info.append("Debug mode enabled but no debug date selected")
            return pd.DataFrame(), debug_info
        
    except Exception as e:
        debug_info.append(f"Error: {str(e)}")
        import traceback
        debug_info.append(f"Traceback: {traceback.format_exc()}")
        return pd.DataFrame(), debug_info

def display_results(result_df, debug_messages, ticker, asset_type):
    """Display analysis results with enhanced statistics"""
    # Show debug info
    with st.expander('Processing Information'):
        for msg in debug_messages:
            st.write(msg)
    
    if not result_df.empty:
        result_df['Ticker'] = ticker
        result_df['AssetType'] = asset_type
        
        # Enhanced summary stats
        st.subheader('Summary Statistics')
        
        # Top row metrics
        col1, col2, col3, col4, col5 = st.columns(5)
        with col1:
            st.metric('Total Records', f"{len(result_df):,}")
        with col2:
            st.metric('Unique Dates', result_df['Date'].nunique())
        with col3:
            goals_hit = len(result_df[result_df['GoalHit'] == 'Yes'])
            st.metric('Goals Hit', goals_hit)
        with col4:
            hit_rate = goals_hit / len(result_df) * 100 if len(result_df) > 0 else 0
            st.metric('Hit Rate', f'{hit_rate:.1f}%')
        with col5:
            avg_atr = result_df['PreviousATR'].mean()
            st.metric('Avg ATR', f'{avg_atr:.2f}')
        
        # Detailed breakdowns
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader('Direction Analysis')
            direction_stats = result_df.groupby('Direction').agg({
                'GoalHit': lambda x: (x == 'Yes').sum(),
                'TriggerLevel': 'count'
            }).rename(columns={'TriggerLevel': 'Total'})
            direction_stats['Hit Rate %'] = (direction_stats['GoalHit'] / direction_stats['Total'] * 100).round(1)
            st.dataframe(direction_stats)
        
        with col2:
            st.subheader('Goal Classification')
            goal_stats = result_df.groupby('GoalClassification').agg({
                'GoalHit': lambda x: (x == 'Yes').sum(),
                'TriggerLevel': 'count'
            }).rename(columns={'TriggerLevel': 'Total'})
            goal_stats['Hit Rate %'] = (goal_stats['GoalHit'] / goal_stats['Total'] * 100).round(1)
            st.dataframe(goal_stats)
        
        # Show ATR validation
        if 'PreviousATR' in result_df.columns:
            latest_atr = result_df['PreviousATR'].iloc[-1]
            st.subheader('ATR Validation')
            st.write(f"**Latest ATR: {latest_atr:.2f}** (Pre-calculated)")
            
            # ATR trend chart
            atr_by_date = result_df.groupby('Date')['PreviousATR'].first().tail(20)
            if len(atr_by_date) > 1:
                st.line_chart(atr_by_date)
        
        # Show systematic validation metrics
        st.subheader('Systematic Logic Validation')
        col1, col2, col3 = st.columns(3)
        with col1:
            same_time_count = len(result_df[result_df['SameTime'] == True])
            st.metric('Same-Time Scenarios', same_time_count)
        with col2:
            open_triggers = len(result_df[result_df['TriggerTime'] == 'OPEN'])
            st.metric('OPEN Triggers', open_triggers)
        with col3:
            cross_zero = len(result_df[(result_df['Direction'] == 'Below') & (result_df['GoalLevel'] > result_df['TriggerLevel'])]) + \
                        len(result_df[(result_df['Direction'] == 'Above') & (result_df['GoalLevel'] < result_df['TriggerLevel'])])
            st.metric('Cross-Zero Scenarios', cross_zero)
        
        # Show data preview
        st.subheader('Results Preview')
        preview_df = result_df.head(10).copy()
        # Format numeric columns
        numeric_cols = ['TriggerPrice', 'GoalPrice', 'PreviousClose', 'PreviousATR']
        for col in numeric_cols:
            if col in preview_df.columns:
                preview_df[col] = preview_df[col].apply(lambda x: f"{x:.2f}")
        
        st.dataframe(preview_df, use_container_width=True)
        
        # Download options
        st.subheader('Download Results')
        col1, col2 = st.columns(2)
        
        with col1:
            # Full results
            ticker_clean = ticker.replace("^", "").replace("=", "_")
            output_filename = f'{ticker_clean}_{asset_type}_ATR_analysis_{datetime.now().strftime("%Y%m%d")}.csv'
            st.download_button(
                'Download Full Results CSV',
                data=result_df.to_csv(index=False),
                file_name=output_filename,
                mime='text/csv'
            )
        
        with col2:
            # Summary only
            summary_data = {
                'Metric': ['Total Records', 'Unique Dates', 'Goals Hit', 'Hit Rate %', 'Avg ATR', 'Same-Time Scenarios', 'OPEN Triggers', 'Cross-Zero'],
                'Value': [len(result_df), result_df['Date'].nunique(), goals_hit, f"{hit_rate:.1f}%", f"{avg_atr:.2f}", same_time_count, open_triggers, cross_zero]
            }
            summary_df = pd.DataFrame(summary_data)
            summary_filename = f'{ticker_clean}_{asset_type}_summary_{datetime.now().strftime("%Y%m%d")}.csv'
            st.download_button(
                'Download Summary CSV',
                data=summary_df.to_csv(index=False),
                file_name=summary_filename,
                mime='text/csv'
            )
        
        st.success(f'Analysis complete for {ticker} using SYSTEMATIC logic!')
        
    else:
        st.warning('No results generated - check processing information above')

# Streamlit Interface
st.title('🎯 Simplified ATR Analysis Generator')
st.write('**Clean, focused ATR analysis using pre-formatted CSV data**')
st.write('**Upload your CSV file from the CSV Data Handler to get started**')

# File upload section
st.header("📁 Data Upload")

data_file = st.file_uploader(
    "Upload Pre-formatted Data File",
    type=['csv'],
    help="CSV file that has been processed by the CSV Data Handler",
    key="data_upload"
)

if data_file:
    st.success(f"✅ Data file uploaded: {data_file.name}")
    st.info("📊 File should contain both daily and intraday data with ATR already calculated")

# Configuration section
if data_file:
    st.header("⚙️ Analysis Configuration")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🏷️ Basic Settings")
        
        ticker = st.text_input(
            "Ticker Symbol (for labeling)",
            value="TICKER",
            help="Enter ticker symbol for output file naming"
        )
        
        asset_type = st.selectbox(
            "Asset Class",
            options=['STOCKS', 'CRYPTO', 'FOREX', 'FUTURES'],
            help="Select asset type for appropriate market handling"
        )
        
        # Extended hours for stocks
        extended_hours = False
        if asset_type == 'STOCKS':
            extended_hours = st.checkbox(
                "Include Extended Hours",
                value=False,
                help="Include pre-market (4AM) and after-hours (8PM) data"
            )
        
        atr_period = st.number_input(
            "ATR Period (Reference Only)", 
            min_value=1, 
            max_value=50, 
            value=14,
            help="ATR is pre-calculated in the data file, this is for reference only"
        )
    
    with col2:
        st.subheader("🔧 Advanced Settings")
        
        # Custom ratios
        use_custom_ratios = st.checkbox("Use Custom Ratios")
        if use_custom_ratios:
            custom_ratios_text = st.text_area(
                "Custom Ratios (comma-separated)",
                value="0.236, 0.382, 0.5, 0.618, 0.786, 1.0, -0.236, -0.382, -0.5, -0.618, -0.786, -1.0, 0.0",
                help="Enter custom ratios separated by commas"
            )
            try:
                custom_ratios = [float(x.strip()) for x in custom_ratios_text.split(',')]
            except:
                st.error("Invalid custom ratios format")
                custom_ratios = None
        else:
            custom_ratios = None
        
        # Debug mode
        debug_mode = st.checkbox("Debug Mode", help="Analyze just one specific day with detailed breakdown")
        debug_date = None
        if debug_mode:
            debug_date = st.date_input(
                "Debug Date",
                value=pd.to_datetime("2024-01-03").date(),
                help="Enter a specific date to analyze in detail"
            )
    
    # Session filtering (simplified)
    config = AssetConfig.get_config(asset_type, extended_hours)
    session_filter = None  # Will be handled by CSV handler if needed
    
    # Run analysis button
    st.markdown("---")
    
    if st.button('🚀 Generate ATR Analysis', type="primary", use_container_width=True):
        with st.spinner('Processing with SYSTEMATIC logic...'):
            try:
                result_df, debug_messages = main_analysis(
                    ticker=ticker,
                    asset_type=asset_type,
                    data_file=data_file,
                    custom_ratios=custom_ratios,
                    debug_mode=debug_mode,
                    debug_date=debug_date
                )
                
                display_results(result_df, debug_messages, ticker, asset_type)
                    
            except Exception as e:
                st.error(f'Error: {e}')
                import traceback
                st.error(traceback.format_exc())

else:
    # Show requirements when files aren't uploaded
    st.info("👆 **Please upload a pre-formatted CSV file to proceed**")
    
    # Show file format requirements
    with st.expander("📋 Required File Format", expanded=True):
        st.markdown("""
        **📊 Pre-formatted Data Requirements:**
        - **Source**: File processed by the CSV Data Handler
        - **Columns**: Datetime, Open, High, Low, Close, Volume, Date, Time, ATR, Prior_Base_Close, Trading_Days_Count
        - **Format**: CSV file with proper datetime formatting
        - **Content**: Combined daily and intraday data with ATR pre-calculated
        
        **✅ Expected Format:**
        - ATR column already calculated
        - Prior_Base_Close for level calculations
        - Date and Time columns properly formatted
        - Trading_Days_Count for validation
        - No missing OHLC data
        """)
    
    # Show workflow
    with st.expander("🔧 Analysis Workflow", expanded=False):
        st.markdown("""
        **🎯 Step-by-Step Process:**
        
        1. **Use CSV Data Handler** - Process your raw data files first
        2. **Upload Pre-formatted File** - Upload the output from CSV Data Handler
        3. **Configure Settings** - Ticker, asset type, analysis options
        4. **Systematic Detection** - Trigger and goal analysis
        5. **Results Export** - Download full analysis or summary
        
        **🔍 What You Get:**
        - Complete trigger/goal combinations for each day
        - Hit rates and success statistics
        - Goal classifications (Continuation, Retracement, Retest)
        - Same-time scenario analysis
        - Cross-zero detection
        - Debug mode for detailed single-day analysis
        
        **💾 Perfect for:**
        - Systematic trading strategy development
        - ATR-based level analysis
        - Intraday goal completion studies
        - Trading system backtesting
        """)

# Help section
st.markdown("---")
st.subheader("📚 Key Features")

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("""
    **🎯 Systematic Logic**
    - Validated trigger/goal detection
    - Uses pre-calculated ATR values
    - Prior_Base_Close for accuracy
    - Same-candle completion rules
    - Cross-zero scenario handling
    """)

with col2:
    st.markdown("""
    **📊 Simplified Input**
    - Single CSV file upload
    - Pre-formatted data expected
    - No complex data handling
    - Uses CSV Data Handler output
    - Streamlined processing
    """)

with col3:
    st.markdown("""
    **🔧 Analysis Features**
    - Multi-asset class support
    - Custom ratio definitions
    - Debug mode for single days
    - Comprehensive statistics
    - Clean results export
    """)

st.info("💡 **Tip**: Use the CSV Data Handler tool to prepare and clean your data files before analysis!")

st.markdown("""
---
### 🎯 About This Tool

This is a **simplified ATR analysis tool** that works with pre-formatted CSV data from the CSV Data Handler.

**Key Features:**
- ✅ **Single CSV input** - Uses pre-formatted data with ATR and Prior_Base_Close
- ✅ **Simplified interface** - Focus on the core analysis
- ✅ **Clean data flow** - No complex parsing or validation needed
- ✅ **Fast processing** - Streamlined for pre-calculated data
- ✅ **Systematic logic** - Validated trigger/goal detection remains untouched

**Perfect workflow:**
1. **CSV Data Handler** → Process and calculate ATR/Prior_Base_Close
2. **This ATR Tool** → Run systematic trigger/goal analysis  
3. **Export Results** → Get clean CSV files for further analysis

Clean, focused, and efficient!
""")
