def create_rolling_result_record(base_record, goal_level, goal_price, goal_hit, goal_time, goal_type, is_same_time):
    """
    Create a rolling analysis result record based on daily record template
    """
    rolling_record = base_record.copy()
    rolling_record.update({
        'GoalLevel': goal_level,
        'GoalPrice': round(goal_price, 2),
        'GoalHit': 'Yes' if goal_hit else 'No',
        'GoalTime': goal_time if goal_hit else '',
        'GoalClassification': goal_type,
        'SameTime': is_same_time,
        'AnalysisType': 'Rolling'
    })
    return rolling_record

def identify_zone(price, atr_levels):
    """
    Identify which ATR zone the price falls into
    Returns zone identifier like "0.236_to_0.382" or "above_1.0"
    """
    sorted_levels = sorted(atr_levels.values())
    
    # Check if price falls between consecutive levels
    for i in range(len(sorted_levels) - 1):
        if sorted_levels[i] <= price < sorted_levels[i + 1]:
            return f"{sorted_levels[i]:.3f}_to_{sorted_levels[i + 1]:.3f}"
    
    # Handle edge cases
    if price >= sorted_levels[-1]:
        return f"above_{sorted_levels[-1]:.3f}"
    else:
        return f"below_{sorted_levels[0]:.3f}"

def get_zones_crossed(high, low, atr_levels):
    """
    Return all zones that the candle's high/low range crosses
    Creates comprehensive zone coverage for gradient mapping
    """
    zones_touched = []
    sorted_levels = sorted(atr_levels.values())
    
    # Check each zone to see if candle's range overlaps
    for i in range(len(sorted_levels) - 1):
        zone_low = sorted_levels[i]
        zone_high = sorted_levels[i + 1]
        
        # Check if candle's range overlaps with this zone
        # Overlap exists if: low <= zone_high AND high >= zone_low
        if low <= zone_high and high >= zone_low:
            zones_touched.append(f"{zone_low:.3f}_to_{zone_high:.3f}")
    
    # Handle edge cases - price beyond our ATR levels
    if high >= sorted_levels[-1]:
        zones_touched.append(f"above_{sorted_levels[-1]:.3f}")
    if low <= sorted_levels[0]:
        zones_touched.append(f"below_{sorted_levels[0]:.3f}")
    
    return zones_touched

def zone_baseline_analysis(daily, intraday, custom_ratios=None):
    """
    ZoneBaseline Analysis: Comprehensive price behavior mapping
    For every candle, identify ALL ATR zones that price touches (High/Low range)
    Creates multiple records per candle if it crosses multiple zones
    Perfect for detailed gradient mapping
    """
    if custom_ratios is None:
        fib_levels = [0.236, 0.382, 0.500, 0.618, 0.786, 1.000, 
                     -0.236, -0.382, -0.500, -0.618, -0.786, -1.000, 0.000]
    else:
        fib_levels = custom_ratios

    results = []
    
    # Progress tracking
    total_periods = len(daily)
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    for i in range(len(daily)):
        try:
            # Update progress
            progress = (i + 1) / total_periods
            progress_bar.progress(progress)
            status_text.text(f"ZoneBaseline: Processing period {i+1}/{total_periods}...")
            
            # Get current day's data
            current_row = daily.iloc[i]
            previous_close = current_row['Prior_Base_Close']
            previous_atr = current_row['ATR']
            trading_date = current_row['Date']
            
            # Skip if no valid ATR
            if pd.isna(previous_atr) or pd.isna(previous_close):
                continue
            
            # Generate ATR levels
            level_map = generate_atr_levels(previous_close, previous_atr, custom_ratios)
            
            # Get intraday data for trading date
            day_data = intraday[intraday['Date'] == pd.to_datetime(trading_date).date()].copy()
            if day_data.empty:
                continue
            
            day_data['Time'] = day_data['Datetime'].dt.strftime('%H%M')
            day_data.reset_index(drop=True, inplace=True)
            
            # For every candle, identify ALL zones it crosses
            for idx, candle in day_data.iterrows():
                high_price = candle['High']
                low_price = candle['Low']
                close_price = candle['Close']
                
                # Get all zones this candle touches
                zones_crossed = get_zones_crossed(high_price, low_price, level_map)
                
                # Create a record for EACH zone crossed
                for zone in zones_crossed:
                    results.append({
                        'Date': trading_date,
                        'Datetime': candle['Datetime'],
                        'Time': candle['Time'],
                        'High': round(high_price, 2),
                        'Low': round(low_price, 2),
                        'Close': round(close_price, 2),
                        'Zone': zone,
                        'PreviousClose': round(previous_close, 2),
                        'PreviousATR': round(previous_atr, 2),
                        'AnalysisType': 'ZoneBaseline'
                    })
                
        except Exception as e:
            st.warning(f"ZoneBaseline error processing {trading_date}: {str(e)}")
            continue
    
    # Clear progress indicators
    progress_bar.empty()
    status_text.empty()
    
    return pd.DataFrame(results)

def state_check_analysis(daily, intraday, custom_ratios=None):
    """
    StateCheck Analysis: Conditional zone behavior mapping
    For every candle: IF price is in Zone X, THEN what zones does price visit for rest of session
    Like trigger analysis but condition is "price is between two ATR levels"
    """
    if custom_ratios is None:
        fib_levels = [0.236, 0.382, 0.500, 0.618, 0.786, 1.000, 
                     -0.236, -0.382, -0.500, -0.618, -0.786, -1.000, 0.000]
    else:
        fib_levels = custom_ratios

    results = []
    
    # Progress tracking
    total_periods = len(daily)
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    for i in range(len(daily)):
        try:
            # Update progress
            progress = (i + 1) / total_periods
            progress_bar.progress(progress)
            status_text.text(f"StateCheck: Processing period {i+1}/{total_periods}...")
            
            # Get current day's data
            current_row = daily.iloc[i]
            previous_close = current_row['Prior_Base_Close']
            previous_atr = current_row['ATR']
            trading_date = current_row['Date']
            
            # Skip if no valid ATR
            if pd.isna(previous_atr) or pd.isna(previous_close):
                continue
            
            # Generate ATR levels
            level_map = generate_atr_levels(previous_close, previous_atr, custom_ratios)
            
            # Get intraday data for trading date
            day_data = intraday[intraday['Date'] == pd.to_datetime(trading_date).date()].copy()
            if day_data.empty:
                continue
            
            day_data['Time'] = day_data['Datetime'].dt.strftime('%H%M')
            day_data.reset_index(drop=True, inplace=True)
            
            # For every candle, check if it's in a zone, then analyze rest of session
            for idx, candle in day_data.iterrows():
                high_price = candle['High']
                low_price = candle['Low']
                close_price = candle['Close']
                
                # Get all zones this candle touches (initial state)
                initial_zones = get_zones_crossed(high_price, low_price, level_map)
                
                # For each initial zone, analyze rest of session
                for initial_zone in initial_zones:
                    # Get rest of session data (from current candle to end)
                    rest_of_session = day_data.iloc[idx:].copy()
                    
                    # Count zone frequencies for rest of session
                    zone_frequencies = {}
                    total_candles_remaining = len(rest_of_session)
                    
                    if total_candles_remaining > 0:
                        # Analyze each remaining candle
                        for _, future_candle in rest_of_session.iterrows():
                            future_high = future_candle['High']
                            future_low = future_candle['Low']
                            
                            # Get zones this future candle touches
                            future_zones = get_zones_crossed(future_high, future_low, level_map)
                            
                            # Count each zone
                            for zone in future_zones:
                                if zone not in zone_frequencies:
                                    zone_frequencies[zone] = 0
                                zone_frequencies[zone] += 1
                        
                        # Convert counts to percentages
                        for zone, count in zone_frequencies.items():
                            percentage = (count / total_candles_remaining) * 100
                            
                            results.append({
                                'Date': trading_date,
                                'InitialDatetime': candle['Datetime'],
                                'InitialTime': candle['Time'],
                                'InitialZone': initial_zone,
                                'InitialHigh': round(high_price, 2),
                                'InitialLow': round(low_price, 2),
                                'InitialClose': round(close_price, 2),
                                'RestOfSessionZone': zone,
                                'ZoneFrequency': count,
                                'ZonePercentage': round(percentage, 2),
                                'TotalCandlesRemaining': total_candles_remaining,
                                'PreviousClose': round(previous_close, 2),
                                'PreviousATR': round(previous_atr, 2),
                                'AnalysisType': 'StateCheck'
                            })
                
        except Exception as e:
            st.warning(f"StateCheck error processing {trading_date}: {str(e)}")
            continue
    
    # Clear progress indicators
    progress_bar.empty()
    status_text.empty()
    
    return pd.DataFrame(results)import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta, date
import time

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

def calculate_rolling_candles(candle_interval_minutes, period_type, period_count):
    """
    Calculate how many candles = desired number of periods for rolling analysis
    """
    period_durations = {
        'hourly': 60,
        '4hour': 240,
        'half_day': 195,        # 3.25 trading hours
        'daily': 390,           # 6.5 trading hours  
        'weekly': 1950          # 5 days × 6.5 hours
    }
    
    minutes_per_period = period_durations.get(period_type, 60)  # Default to hourly
    total_minutes = minutes_per_period * period_count
    candles_to_count = total_minutes / candle_interval_minutes
    return int(candles_to_count)

def create_rolling_result_record(base_record, goal_level, goal_price, goal_hit, goal_time, goal_type, is_same_time):
    """
    Create a rolling analysis result record based on daily record template
    """
    rolling_record = base_record.copy()
    rolling_record.update({
        'GoalLevel': goal_level,
        'GoalPrice': round(goal_price, 2),
        'GoalHit': 'Yes' if goal_hit else 'No',
        'GoalTime': goal_time if goal_hit else '',
        'GoalClassification': goal_type,
        'SameTime': is_same_time,
        'AnalysisType': 'Rolling'
    })
    return rolling_record

def detect_triggers_and_goals_batch(daily, intraday, custom_ratios=None, start_index=0, batch_size=50):
    """
    Batch version of systematic analysis with session state checkpointing
    Processes batch_size periods at a time to avoid timeouts
    """
    if custom_ratios is None:
        fib_levels = [0.236, 0.382, 0.500, 0.618, 0.786, 1.000,
                     -0.236, -0.382, -0.500, -0.618, -0.786, -1.000, 0.000]
    else:
        fib_levels = custom_ratios

    results = []

    # Progress tracking
    total_periods = len(daily)
    end_index = min(start_index + batch_size, total_periods)

    progress_bar = st.progress(start_index / total_periods)
    status_text = st.empty()

    for i in range(start_index, end_index):
        try:
            # Update progress
            progress = (i + 1) / total_periods
            progress_bar.progress(progress)
            status_text.text(f"Processing period {i+1}/{total_periods}...")
            
            # Use CURRENT day's Pre-calculated Prior_Base_Close and ATR
            current_row = daily.iloc[i]     
            
            previous_close = current_row['Prior_Base_Close']  # Pre-calculated previous close
            previous_atr = current_row['ATR']                 # Pre-calculated ATR
            trading_date = current_row['Date']
            
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
            
            # Check if rolling analysis is configured
            has_rolling_config = False
            rolling_candles = 0
            if 'Candle_Interval_Minutes' in intraday.columns:
                candle_interval = intraday['Candle_Interval_Minutes'].iloc[0]
                period_type = intraday['Rolling_Period_Type'].iloc[0] if 'Rolling_Period_Type' in intraday.columns else None
                period_count = intraday['Rolling_Period_Count'].iloc[0] if 'Rolling_Period_Count' in intraday.columns else None
                analysis_timeframe = intraday['Analysis_Timeframe'].iloc[0] if 'Analysis_Timeframe' in intraday.columns else None
                
                if period_type and period_type != 'None' and analysis_timeframe and analysis_timeframe != 'Other':
                    has_rolling_config = True
                    rolling_candles = calculate_rolling_candles(candle_interval, period_type, period_count)
            
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
                    
                    # Base record template for both daily and rolling
                    base_record = {
                        'Date': trading_date,
                        'Direction': 'Below',
                        'TriggerLevel': trigger_level,
                        'TriggerTime': below_trigger_time,
                        'TriggerPrice': round(trigger_price, 2),
                        'PreviousClose': round(previous_close, 2),
                        'PreviousATR': round(previous_atr, 2),
                        'RetestedTrigger': 'No'
                    }
                    
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
                        
                        # Record this BELOW trigger-goal combination (SESSION)
                        session_record = base_record.copy()
                        session_record.update({
                            'GoalLevel': goal_level,
                            'GoalPrice': round(goal_price, 2),
                            'GoalHit': 'Yes' if goal_hit else 'No',
                            'GoalTime': goal_time if goal_hit else '',
                            'GoalClassification': goal_type,
                            'SameTime': is_same_time,
                            'AnalysisType': 'Session'
                        })
                        results.append(session_record)
                        
                        # ROLLING ANALYSIS: Process same trigger-goal combination for rolling window
                        if has_rolling_config:
                            rolling_goal_hit = False
                            rolling_goal_time = ''
                            rolling_is_same_time = False
                            
                            # Use identical logic but with rolling candle boundary
                            if below_trigger_time == 'OPEN':
                                # Step 1: Check if goal completes at OPEN price first (takes precedence)
                                if goal_level == trigger_level:  # Same level retest
                                    # For same-level retest, we need opposite direction movement
                                    # Below trigger at OPEN, so retest needs Above movement
                                    if open_price >= goal_price:
                                        rolling_goal_hit = True
                                        rolling_goal_time = 'OPEN'
                                        rolling_is_same_time = True
                                elif goal_level > trigger_level:  # Above goal (RETRACEMENT)
                                    if open_price >= goal_price:
                                        rolling_goal_hit = True
                                        rolling_goal_time = 'OPEN'
                                        rolling_is_same_time = True
                                else:  # Below goal (CONTINUATION)
                                    if open_price <= goal_price:
                                        rolling_goal_hit = True
                                        rolling_goal_time = 'OPEN'
                                        rolling_is_same_time = True
                                
                                # Step 2: Only if OPEN missed, check candles based on goal type (ROLLING BOUNDARY)
                                if not rolling_goal_hit:
                                    # CRITICAL: Different logic for CONTINUATION vs RETRACEMENT vs RETEST
                                    if goal_level == trigger_level:  # RETEST - must skip same candle (like retracement)
                                        rolling_end_idx = min(1 + rolling_candles, len(day_data))
                                        start_candles = day_data.iloc[1:rolling_end_idx].iterrows()
                                    elif goal_level > trigger_level:  # RETRACEMENT - must skip same candle (0930), start from 0940
                                        rolling_end_idx = min(1 + rolling_candles, len(day_data))
                                        start_candles = day_data.iloc[1:rolling_end_idx].iterrows()
                                    else:  # CONTINUATION - can check same candle (0930)
                                        rolling_end_idx = min(rolling_candles, len(day_data))
                                        start_candles = day_data.iloc[0:rolling_end_idx].iterrows()
                                    
                                    for _, row in start_candles:
                                        if goal_level == trigger_level:  # Same level retest (opposite direction)
                                            if row['High'] >= goal_price:  # Below trigger needs High to retest
                                                rolling_goal_hit = True
                                                rolling_goal_time = row['Time']
                                                break
                                        elif goal_level > trigger_level:  # Above goal
                                            if row['High'] >= goal_price:  # Use High, not Open
                                                rolling_goal_hit = True
                                                rolling_goal_time = row['Time']
                                                break
                                        else:  # Below goal  
                                            if row['Low'] <= goal_price:  # Use Low, not Open
                                                rolling_goal_hit = True
                                                rolling_goal_time = row['Time']
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
                                            rolling_goal_hit = True
                                            rolling_goal_time = below_trigger_time
                                
                                # Check subsequent candles if not completed on trigger candle (ROLLING BOUNDARY)
                                if not rolling_goal_hit:
                                    rolling_end_idx = min(below_trigger_row + 1 + rolling_candles, len(day_data))
                                    for _, row in day_data.iloc[below_trigger_row + 1:rolling_end_idx].iterrows():
                                        if goal_level == trigger_level:  # Same level retest (opposite direction)
                                            if row['High'] >= goal_price:  # Below trigger needs High to retest
                                                rolling_goal_hit = True
                                                rolling_goal_time = row['Time']
                                                break
                                        elif goal_level > trigger_level:  # Above goal
                                            if row['High'] >= goal_price:
                                                rolling_goal_hit = True
                                                rolling_goal_time = row['Time']
                                                break
                                        else:  # Below goal
                                            if row['Low'] <= goal_price:
                                                rolling_goal_hit = True
                                                rolling_goal_time = row['Time']
                                                break
                            
                            # Record this BELOW trigger-goal combination (ROLLING)
                            rolling_record = create_rolling_result_record(
                                base_record, goal_level, goal_price, rolling_goal_hit, 
                                rolling_goal_time, goal_type, rolling_is_same_time
                            )
                            results.append(rolling_record)
                
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
                    
                    # Base record template for both daily and rolling
                    base_record = {
                        'Date': trading_date,
                        'Direction': 'Above',
                        'TriggerLevel': trigger_level,
                        'TriggerTime': above_trigger_time,
                        'TriggerPrice': round(trigger_price, 2),
                        'PreviousClose': round(previous_close, 2),
                        'PreviousATR': round(previous_atr, 2),
                        'RetestedTrigger': 'No'
                    }
                    
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
                        
                        # Record this ABOVE trigger-goal combination (SESSION)
                        session_record = base_record.copy()
                        session_record.update({
                            'GoalLevel': goal_level,
                            'GoalPrice': round(goal_price, 2),
                            'GoalHit': 'Yes' if goal_hit else 'No',
                            'GoalTime': goal_time if goal_hit else '',
                            'GoalClassification': goal_type,
                            'SameTime': is_same_time,
                            'AnalysisType': 'Session'
                        })
                        results.append(session_record)
                        
                        # ROLLING ANALYSIS: Process same trigger-goal combination for rolling window
                        if has_rolling_config:
                            rolling_goal_hit = False
                            rolling_goal_time = ''
                            rolling_is_same_time = False
                            
                            # Use identical logic but with rolling candle boundary
                            if above_trigger_time == 'OPEN':
                                # Step 1: Check if goal completes at OPEN price first (takes precedence)
                                if goal_level == trigger_level:  # Same level retest
                                    # For same-level retest, we need opposite direction movement
                                    # Above trigger at OPEN, so retest needs Below movement
                                    if open_price <= goal_price:
                                        rolling_goal_hit = True
                                        rolling_goal_time = 'OPEN'
                                        rolling_is_same_time = True
                                elif goal_level > trigger_level:  # Above goal (CONTINUATION)
                                    if open_price >= goal_price:
                                        rolling_goal_hit = True
                                        rolling_goal_time = 'OPEN'
                                        rolling_is_same_time = True
                                else:  # Below goal (RETRACEMENT)
                                    if open_price <= goal_price:
                                        rolling_goal_hit = True
                                        rolling_goal_time = 'OPEN'
                                        rolling_is_same_time = True
                                
                                # Step 2: Only if OPEN missed, check ALL candles including 0930 (but use High/Low, not Open) (ROLLING BOUNDARY)
                                if not rolling_goal_hit:
                                    rolling_end_idx = min(rolling_candles, len(day_data))
                                    for _, row in day_data.iloc[0:rolling_end_idx].iterrows():
                                        if goal_level == trigger_level:  # Same level retest (opposite direction)
                                            if row['Low'] <= goal_price:  # Above trigger needs Low to retest
                                                rolling_goal_hit = True
                                                rolling_goal_time = row['Time']
                                                break
                                        elif goal_level > trigger_level:  # Above goal
                                            if row['High'] >= goal_price:  # Use High, not Open
                                                rolling_goal_hit = True
                                                rolling_goal_time = row['Time']
                                                break
                                        else:  # Below goal
                                            if row['Low'] <= goal_price:  # Use Low, not Open
                                                rolling_goal_hit = True
                                                rolling_goal_time = row['Time']
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
                                            rolling_goal_hit = True
                                            rolling_goal_time = above_trigger_time
                                
                                # Check subsequent candles if not completed on trigger candle (ROLLING BOUNDARY)
                                if not rolling_goal_hit:
                                    rolling_end_idx = min(above_trigger_row + 1 + rolling_candles, len(day_data))
                                    for _, row in day_data.iloc[above_trigger_row + 1:rolling_end_idx].iterrows():
                                        if goal_level == trigger_level:  # Same level retest (opposite direction)
                                            if row['Low'] <= goal_price:  # Above trigger needs Low to retest
                                                rolling_goal_hit = True
                                                rolling_goal_time = row['Time']
                                                break
                                        elif goal_level > trigger_level:  # Above goal
                                            if row['High'] >= goal_price:
                                                rolling_goal_hit = True
                                                rolling_goal_time = row['Time']
                                                break
                                        else:  # Below goal
                                            if row['Low'] <= goal_price:
                                                rolling_goal_hit = True
                                                rolling_goal_time = row['Time']
                                                break
                            
                            # Record this ABOVE trigger-goal combination (ROLLING)
                            rolling_record = create_rolling_result_record(
                                base_record, goal_level, goal_price, rolling_goal_hit, 
                                rolling_goal_time, goal_type, rolling_is_same_time
                            )
                            results.append(rolling_record)

        except Exception as e:
            st.warning(f"Error processing {trading_date}: {str(e)}")
            continue

    # Update session state
    st.session_state.atr_processing['last_processed_index'] = end_index

    # Check if we're done
    if end_index >= total_periods:
        st.session_state.atr_processing['is_complete'] = True
        status_text.text("Processing complete!")
    else:
        status_text.text(f"Batch complete. Next batch will start at period {end_index + 1}")

    # Clear progress bar after batch
    progress_bar.empty()

    return pd.DataFrame(results)

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
    total_periods = len(daily)
    progress_bar = st.progress(0)
    status_text = st.empty()

    for i in range(len(daily)):
        try:
            # Update progress
            progress = (i + 1) / total_periods
            progress_bar.progress(progress)
            status_text.text(f"Processing period {i+1}/{total_periods}...")
            
            # Keep Streamlit alive - output every 100 periods
            if i % 100 == 0:
                st.write(f"Processed {i} periods...")
            
            # Use CURRENT day's Pre-calculated Prior_Base_Close and ATR
            current_row = daily.iloc[i]     
            
            previous_close = current_row['Prior_Base_Close']  # Pre-calculated previous close
            previous_atr = current_row['ATR']                 # Pre-calculated ATR
            trading_date = current_row['Date']
            
            # Date filtering - start from first available intraday data
            if hasattr(trading_date, 'strftime'):
                date_str = trading_date.strftime('%Y-%m-%d')
            elif isinstance(trading_date, str):
                date_str = trading_date[:10]
            else:
                date_str = str(trading_date)[:10]
            
            # No hardcoded date filter - process all available data
            
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

# Main analysis function with session state resume capability
def main_analysis(ticker, asset_type, data_file, custom_ratios=None, debug_mode=False, debug_date=None, resume_from_period=0, extended_hours=False):
    """Main function for pre-formatted CSV analysis with auto-resume"""
    debug_info = []
    
    try:
        # Initialize or get session state
        if 'atr_processing' not in st.session_state:
            st.session_state.atr_processing = {
                'results': [],
                'last_processed_index': 0,
                'is_complete': False,
                'daily_data': None,
                'intraday_data': None,
                'custom_ratios': None,
                'ticker': '',
                'asset_type': '',
                'extended_hours': False,
                'total_periods': 0,
                'zone_baseline_results': [],
                'zone_baseline_complete': False,
                'state_check_results': [],
                'state_check_complete': False
            }
        
        # Debug mode check
        if debug_mode and debug_date:
            st.success(f"DEBUG MODE - Will process ONLY {debug_date}")
        else:
            st.info("FULL MODE - Will process all periods with auto-resume capability")
        
        # Get asset configuration
        asset_config = AssetConfig.get_config(asset_type, False)
        debug_info.append(f"Asset Type: {asset_config['description']}")
        
        # Load pre-formatted data (only if not already loaded)
        if st.session_state.atr_processing['daily_data'] is None:
            df = load_preformatted_data(data_file)
            if df is None:
                debug_info.append("Failed to load pre-formatted data")
                return pd.DataFrame(), debug_info
            
            debug_info.append(f"Data loaded: {df.shape}")
            debug_info.append(f"Date range: {df['Date'].min()} to {df['Date'].max()}")
            
            # Check for rolling analysis configuration
            if 'Candle_Interval_Minutes' in df.columns:
                candle_interval = df['Candle_Interval_Minutes'].iloc[0]
                period_type = df['Rolling_Period_Type'].iloc[0] if 'Rolling_Period_Type' in df.columns else 'None'
                period_count = df['Rolling_Period_Count'].iloc[0] if 'Rolling_Period_Count' in df.columns else 0
                analysis_timeframe = df['Analysis_Timeframe'].iloc[0] if 'Analysis_Timeframe' in df.columns else 'Other'
                
                debug_info.append(f"Rolling Config: {candle_interval}min candles, {period_type} periods, {period_count} count, {analysis_timeframe} timeframe")
                
                if period_type != 'None' and analysis_timeframe != 'Other':
                    rolling_candles = calculate_rolling_candles(candle_interval, period_type, period_count)
                    debug_info.append(f"Rolling Analysis: Enabled ({rolling_candles} candles ahead)")
                else:
                    debug_info.append("Rolling Analysis: Disabled")
            else:
                debug_info.append("Rolling Analysis: No configuration found")
            
            # Validate required data
            if df['ATR'].isna().all():
                debug_info.append("No valid ATR values found")
                return pd.DataFrame(), debug_info
            
            recent_atr = df['ATR'].tail(3).round(2).tolist()
            debug_info.append(f"ATR values found. Recent values: {recent_atr}")
            
            # Prepare data for systematic analysis
            daily_data = df.groupby('Date').agg({
                'Open': 'first',
                'High': 'max', 
                'Low': 'min',
                'Close': 'last',
                'ATR': 'first',
                'Prior_Base_Close': 'first'
            }).reset_index()
            daily_data['Date'] = pd.to_datetime(daily_data['Date'])
            
            intraday_data = df.copy()
            
            # Store in session state
            st.session_state.atr_processing['daily_data'] = daily_data
            st.session_state.atr_processing['intraday_data'] = intraday_data
            st.session_state.atr_processing['custom_ratios'] = custom_ratios
            st.session_state.atr_processing['ticker'] = ticker
            st.session_state.atr_processing['asset_type'] = asset_type
            st.session_state.atr_processing['extended_hours'] = extended_hours
            st.session_state.atr_processing['total_periods'] = len(daily_data)
            
            # Handle resume from specific period
            if resume_from_period > 0:
                st.session_state.atr_processing['last_processed_index'] = resume_from_period
                debug_info.append(f"Resuming from period {resume_from_period}")
            
            debug_info.append(f"Daily data prepared: {len(daily_data)} periods")
            debug_info.append(f"Intraday data prepared: {len(intraday_data)} records")
        else:
            # Resume with existing data
            daily_data = st.session_state.atr_processing['daily_data']
            intraday_data = st.session_state.atr_processing['intraday_data']
            debug_info.append(f"Resuming from period {st.session_state.atr_processing['last_processed_index']}")
            debug_info.append(f"Total periods: {st.session_state.atr_processing['total_periods']}")
        
        # Debug Mode - single day analysis
        if debug_mode and debug_date:
            st.info(f"Debug Mode Active - Analyzing single day: {debug_date}")
            debug_single_day_analysis(daily_data, intraday_data, debug_date, custom_ratios)
            return pd.DataFrame(), debug_info + [f"Debug analysis completed for {debug_date}"]
        
        # Run batch systematic analysis
        if not debug_mode:
            if not st.session_state.atr_processing['is_complete']:
                debug_info.append("Running ALL analysis types in batches...")
                
                # Process trigger/goal analysis (Daily + Rolling)
                batch_results = detect_triggers_and_goals_batch(
                    daily_data, 
                    intraday_data, 
                    custom_ratios,
                    start_index=st.session_state.atr_processing['last_processed_index']
                )
                
                # Process ZoneBaseline analysis (if not already done)
                if 'zone_baseline_complete' not in st.session_state.atr_processing:
                    st.info("Running ZoneBaseline Analysis...")
                    zone_baseline_results = zone_baseline_analysis(daily_data, intraday_data, custom_ratios)
                    st.session_state.atr_processing['zone_baseline_results'] = zone_baseline_results.to_dict('records')
                    st.session_state.atr_processing['zone_baseline_complete'] = True
                    debug_info.append(f"ZoneBaseline analysis complete: {len(zone_baseline_results)} records")
                
                # Process StateCheck analysis (if not already done)
                if 'state_check_complete' not in st.session_state.atr_processing:
                    st.info("Running StateCheck Analysis...")
                    state_check_results = state_check_analysis(daily_data, intraday_data, custom_ratios)
                    st.session_state.atr_processing['state_check_results'] = state_check_results.to_dict('records')
                    st.session_state.atr_processing['state_check_complete'] = True
                    debug_info.append(f"StateCheck analysis complete: {len(state_check_results)} records")
                
                if batch_results is not None:
                    # Add new trigger/goal results to session state
                    if len(batch_results) > 0:
                        st.session_state.atr_processing['results'].extend(batch_results.to_dict('records'))
                    
                    debug_info.append(f"Batch complete. Total trigger/goal combinations found: {len(st.session_state.atr_processing['results'])}")
                
                # Check if processing is complete
                if st.session_state.atr_processing['is_complete']:
                    # Combine all analysis results
                    all_results = []
                    
                    # Add trigger/goal results (Daily + Rolling)
                    if st.session_state.atr_processing['results']:
                        all_results.extend(st.session_state.atr_processing['results'])
                    
                    # Add ZoneBaseline results
                    if 'zone_baseline_results' in st.session_state.atr_processing:
                        all_results.extend(st.session_state.atr_processing['zone_baseline_results'])
                    
                    # Add StateCheck results
                    if 'state_check_results' in st.session_state.atr_processing:
                        all_results.extend(st.session_state.atr_processing['state_check_results'])
                    
                    # Convert to DataFrame
                    result_df = pd.DataFrame(all_results)
                    
                    # Add Trading_Days_Count to results
                    if not result_df.empty and 'Trading_Days_Count' in intraday_data.columns:
                        final_trading_days = intraday_data['Trading_Days_Count'].iloc[-1]
                        result_df['Trading_Days_Count'] = final_trading_days
                        debug_info.append(f"Added Trading_Days_Count: {final_trading_days}")
                    
                    # Add Analysis_Timeframe to results for display app
                    if not result_df.empty and 'Analysis_Timeframe' in intraday_data.columns:
                        analysis_timeframe = intraday_data['Analysis_Timeframe'].iloc[0]
                        result_df['Analysis_Timeframe'] = analysis_timeframe
                        debug_info.append(f"Added Analysis_Timeframe: {analysis_timeframe}")
                    
                    # Add Candle_Interval_Minutes (for summarizer)
                    if not result_df.empty and 'Candle_Interval_Minutes' in intraday_data.columns:
                        candle_interval = intraday_data['Candle_Interval_Minutes'].iloc[0]
                        result_df['Candle_Interval_Minutes'] = candle_interval
                        debug_info.append(f"Added Candle_Interval_Minutes: {candle_interval}")
                    
                    # Add Base_Interval_Minutes (for summarizer)
                    if not result_df.empty and 'Base_Interval_Minutes' in intraday_data.columns:
                        base_interval = intraday_data['Base_Interval_Minutes'].iloc[0]
                        result_df['Base_Interval_Minutes'] = base_interval
                        debug_info.append(f"Added Base_Interval_Minutes: {base_interval}")
                    
                    # Add Ticker and Asset Type for identification
                    if not result_df.empty:
                        result_df['Ticker'] = ticker
                        result_df['AssetType'] = asset_type
                        
                        # Add detailed asset class info
                        if asset_type == 'STOCKS':
                            if extended_hours:
                                result_df['AssetClass'] = 'STOCKS_ETH'  # Extended Trading Hours
                            else:
                                result_df['AssetClass'] = 'STOCKS_RTH'  # Regular Trading Hours
                        else:
                            result_df['AssetClass'] = asset_type  # CRYPTO, FOREX, FUTURES as-is
                        
                        debug_info.append(f"Added AssetClass: {result_df['AssetClass'].iloc[0]}")
                    
                    # Comprehensive statistics
                    if not result_df.empty:
                        # Analysis type breakdown
                        analysis_type_counts = result_df['AnalysisType'].value_counts()
                        debug_info.append(f"Analysis breakdown: {analysis_type_counts.to_dict()}")
                        
                        # Traditional trigger/goal stats
                        trigger_goal_results = result_df[result_df['AnalysisType'].isin(['Session', 'Rolling'])]
                        if len(trigger_goal_results) > 0:
                            above_triggers = len(trigger_goal_results[trigger_goal_results['Direction'] == 'Above'])
                            below_triggers = len(trigger_goal_results[trigger_goal_results['Direction'] == 'Below'])
                            debug_info.append(f"Trigger/Goal: Above {above_triggers}, Below {below_triggers}")
                            
                            goals_hit = len(trigger_goal_results[trigger_goal_results['GoalHit'] == 'Yes'])
                            hit_rate = goals_hit / len(trigger_goal_results) * 100 if len(trigger_goal_results) > 0 else 0
                            debug_info.append(f"Goals hit: {goals_hit}/{len(trigger_goal_results)} ({hit_rate:.1f}%)")
                        
                        # Zone analysis stats
                        zone_results = result_df[result_df['AnalysisType'].isin(['ZoneBaseline', 'StateCheck'])]
                        if len(zone_results) > 0:
                            unique_zones = zone_results['Zone'].nunique() if 'Zone' in zone_results.columns else 0
                            debug_info.append(f"Zone analysis: {len(zone_results)} records, {unique_zones} unique zones")
                        
                        # Validation metrics
                        if 'SameTime' in result_df.columns:
                            same_time_count = len(result_df[result_df['SameTime'] == True])
                            debug_info.append(f"Same-time scenarios found: {same_time_count}")
                        
                        if 'TriggerTime' in result_df.columns:
                            open_triggers = len(result_df[result_df['TriggerTime'] == 'OPEN'])
                            intraday_triggers = len(result_df[result_df['TriggerTime'] != 'OPEN'])
                            debug_info.append(f"OPEN triggers: {open_triggers}, Intraday triggers: {intraday_triggers}")
                    
                    return result_df, debug_info
                else:
                    # Still processing - return partial results and auto-continue
                    partial_df = pd.DataFrame(st.session_state.atr_processing['results'])
                    progress_pct = (st.session_state.atr_processing['last_processed_index'] / 
                                  st.session_state.atr_processing['total_periods']) * 100
                    debug_info.append(f"Processing in progress: {progress_pct:.1f}% complete")
                    debug_info.append("Auto-continuing processing...")
                    
                    # Force immediate continuation
                    st.write(f"🔄 Auto-continuing... {progress_pct:.1f}% complete")
                    time.sleep(0.1)  # Brief pause
                    st.rerun()  # Force restart
                    
                    return partial_df, debug_info
            else:
                # Processing already complete
                result_df = pd.DataFrame(st.session_state.atr_processing['results'])
                debug_info.append("Processing already complete - showing final results")
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
        # Ensure metadata columns are present for downstream processing
        if 'Ticker' not in result_df.columns:
            result_df['Ticker'] = ticker
        if 'AssetType' not in result_df.columns:
            result_df['AssetType'] = asset_type
        
        # Enhanced summary stats
        st.subheader('Summary Statistics')
        
        # Check if we have rolling analysis
        has_rolling = 'AnalysisType' in result_df.columns and 'Rolling' in result_df['AnalysisType'].values
        
        # Top row metrics
        if has_rolling:
            col1, col2, col3, col4, col5, col6 = st.columns(6)
            with col1:
                st.metric('Total Records', f"{len(result_df):,}")
            with col2:
                session_count = len(result_df[result_df['AnalysisType'] == 'Session'])
                st.metric('Session Records', f"{session_count:,}")
            with col3:
                rolling_count = len(result_df[result_df['AnalysisType'] == 'Rolling'])
                st.metric('Rolling Records', f"{rolling_count:,}")
            with col4:
                st.metric('Unique Dates', result_df['Date'].nunique())
            with col5:
                goals_hit = len(result_df[result_df['GoalHit'] == 'Yes'])
                st.metric('Goals Hit', goals_hit)
            with col6:
                hit_rate = goals_hit / len(result_df) * 100 if len(result_df) > 0 else 0
                st.metric('Overall Hit Rate', f'{hit_rate:.1f}%')
        else:
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
        
        # Rolling vs Session comparison
        if has_rolling:
            st.subheader('Session vs Rolling Analysis Comparison')
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("**Session Analysis**")
                session_results = result_df[result_df['AnalysisType'] == 'Session']
                session_hit_rate = len(session_results[session_results['GoalHit'] == 'Yes']) / len(session_results) * 100 if len(session_results) > 0 else 0
                st.metric('Session Hit Rate', f'{session_hit_rate:.1f}%')
                
                session_direction_stats = session_results.groupby('Direction').agg({
                    'GoalHit': lambda x: (x == 'Yes').sum(),
                    'TriggerLevel': 'count'
                }).rename(columns={'TriggerLevel': 'Total'})
                session_direction_stats['Hit Rate %'] = (session_direction_stats['GoalHit'] / session_direction_stats['Total'] * 100).round(1)
                st.write("Session Direction Stats:")
                st.dataframe(session_direction_stats, use_container_width=True)
            
            with col2:
                st.write("**Rolling Analysis**")
                rolling_results = result_df[result_df['AnalysisType'] == 'Rolling']
                rolling_hit_rate = len(rolling_results[rolling_results['GoalHit'] == 'Yes']) / len(rolling_results) * 100 if len(rolling_results) > 0 else 0
                st.metric('Rolling Hit Rate', f'{rolling_hit_rate:.1f}%')
                
                rolling_direction_stats = rolling_results.groupby('Direction').agg({
                    'GoalHit': lambda x: (x == 'Yes').sum(),
                    'TriggerLevel': 'count'
                }).rename(columns={'TriggerLevel': 'Total'})
                rolling_direction_stats['Hit Rate %'] = (rolling_direction_stats['GoalHit'] / rolling_direction_stats['Total'] * 100).round(1)
                st.write("Rolling Direction Stats:")
                st.dataframe(rolling_direction_stats, use_container_width=True)
        
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
        preview_df = result_df.head(20).copy()
        # Format numeric columns
        numeric_cols = ['TriggerPrice', 'GoalPrice', 'PreviousClose', 'PreviousATR']
        for col in numeric_cols:
            if col in preview_df.columns:
                preview_df[col] = preview_df[col].apply(lambda x: f"{x:.2f}")
        
        st.dataframe(preview_df, use_container_width=True)
        
        # Download options
        st.subheader('Download Results')
        
        # Create download data
        csv_data = result_df.to_csv(index=False)
        ticker_clean = ticker.replace("^", "").replace("=", "_")
        analysis_type_suffix = "_combined" if has_rolling else "_session"
        output_filename = f'{ticker_clean}_{asset_type}_ATR_analysis{analysis_type_suffix}_{datetime.now().strftime("%Y%m%d")}.csv'
        
        # Summary data
        summary_data = {
            'Metric': ['Total Records', 'Unique Dates', 'Goals Hit', 'Hit Rate %', 'Same-Time Scenarios', 'OPEN Triggers', 'Cross-Zero'],
            'Value': [len(result_df), result_df['Date'].nunique(), goals_hit, f"{hit_rate:.1f}%", same_time_count, open_triggers, cross_zero]
        }
        
        if has_rolling:
            summary_data['Metric'].extend(['Session Records', 'Rolling Records', 'Session Hit Rate %', 'Rolling Hit Rate %'])
            summary_data['Value'].extend([session_count, rolling_count, f"{session_hit_rate:.1f}%", f"{rolling_hit_rate:.1f}%"])
        
        summary_df = pd.DataFrame(summary_data)
        summary_csv_data = summary_df.to_csv(index=False)
        summary_filename = f'{ticker_clean}_{asset_type}_summary{analysis_type_suffix}_{datetime.now().strftime("%Y%m%d")}.csv'
        
        # Display file info and persistent download buttons
        col1, col2, col3 = st.columns([2, 1, 1])
        
        with col1:
            st.write(f"**📊 Full Results:** {len(result_df):,} records ({len(csv_data):,} bytes)")
            st.write(f"**📈 Summary:** {len(summary_df)} metrics ({len(summary_csv_data):,} bytes)")
            st.write(f"**💾 Make sure you have enough disk space before downloading!**")
        
        with col2:
            # Full results download - always available
            st.download_button(
                label='📊 Download Full Results',
                data=csv_data,
                file_name=output_filename,
                mime='text/csv',
                key=f'download_full_{hash(output_filename)}',
                help=f'Download {len(result_df):,} records'
            )
        
        with col3:
            # Summary download - always available
            st.download_button(
                label='📈 Download Summary',
                data=summary_csv_data,
                file_name=summary_filename,
                mime='text/csv',
                key=f'download_summary_{hash(summary_filename)}',
                help=f'Download {len(summary_df)} metrics'
            )
        
        # Additional download options
        st.write("---")
        st.subheader("🔄 Alternative Download Methods")
        
        col1, col2 = st.columns(2)
        with col1:
            st.write("**If download buttons fail:**")
            st.write("1. Check available disk space")
            st.write("2. Try downloading summary first (smaller)")
            st.write("3. Use 'Save As' and choose location")
            st.write("4. Refresh page - results will still be here!")
        
        with col2:
            # Show file details
            st.write("**File Details:**")
            st.write(f"📁 Full Results: `{output_filename}`")
            st.write(f"📁 Summary: `{summary_filename}`")
            st.write(f"📅 Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        # Backup download section
        with st.expander("💾 Backup Download Options"):
            st.write("**If regular download fails, try these:**")
            
            # Raw CSV display for copy/paste
            st.subheader("📋 Copy/Paste CSV Data")
            st.text_area(
                "Full Results CSV (copy and save as .csv file):",
                value=csv_data[:5000] + "..." if len(csv_data) > 5000 else csv_data,
                height=200,
                key="csv_backup"
            )
            
            st.text_area(
                "Summary CSV (copy and save as .csv file):",
                value=summary_csv_data,
                height=100,
                key="summary_backup"
            )
            
            # Additional download buttons with different keys
            st.write("**Alternative Download Buttons:**")
            col1, col2 = st.columns(2)
            with col1:
                st.download_button(
                    '🔄 Retry Full Download',
                    data=csv_data,
                    file_name=f'backup_{output_filename}',
                    mime='text/csv',
                    key=f'backup_full_{datetime.now().strftime("%H%M%S")}'
                )
            with col2:
                st.download_button(
                    '🔄 Retry Summary Download',
                    data=summary_csv_data,
                    file_name=f'backup_{summary_filename}',
                    mime='text/csv',
                    key=f'backup_summary_{datetime.now().strftime("%H%M%S")}'
                )
        
        analysis_description = "SYSTEMATIC logic with Rolling Analysis" if has_rolling else "SYSTEMATIC logic"
        st.success(f'Analysis complete for {ticker} using {analysis_description}!')
        
    else:
        st.warning('No results generated - check processing information above')

# Streamlit Interface
st.title('🎯 Comprehensive ATR Analysis Generator')
st.write('**Complete ATR analysis suite: Session, Rolling, ZoneBaseline, and StateCheck**')
st.write('**Upload your CSV file from the CSV Data Handler to get started**')

# Auto-continue processing if incomplete
if ('atr_processing' in st.session_state and
    st.session_state.atr_processing['total_periods'] > 0 and
    not st.session_state.atr_processing['is_complete'] and
    st.session_state.atr_processing['daily_data'] is not None):
    
    st.info("🔄 Resuming processing automatically...")
    
    # PROGRESSIVE SAVE SECTION - SEPARATE FROM MAIN PROCESSING
    current_progress = st.session_state.atr_processing.get('last_processed_index', 0)
    total_progress = st.session_state.atr_processing.get('total_periods', 1)
    current_records = len(st.session_state.atr_processing.get('results', []))
    
    if current_progress > 0:
        st.markdown("---")
        st.subheader("📊 Live Progress Monitor")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            progress_pct = (current_progress / total_progress) * 100
            st.metric("Progress", f"{progress_pct:.1f}%", f"Period {current_progress:,}")
        with col2:
            st.metric("Records Generated", f"{current_records:,}")
        with col3:
            # Add zone records if available
            zone_records_count = 0
            if 'zone_baseline_results' in st.session_state.atr_processing:
                zone_records_count += len(st.session_state.atr_processing['zone_baseline_results'])
            if 'state_check_results' in st.session_state.atr_processing:
                zone_records_count += len(st.session_state.atr_processing['state_check_results'])
            
            total_records = current_records + zone_records_count
            st.metric("Total Records", f"{total_records:,}")
        
        # Progressive download in separate container
        with st.container():
            st.subheader("💾 Emergency Download")
            st.write(f"**Save current progress:** {total_records:,} records through period {current_progress:,}")
            
            if total_records > 0:
                # Prepare download data
                all_current_data = []
                
                # Add session/rolling results
                if st.session_state.atr_processing.get('results'):
                    all_current_data.extend(st.session_state.atr_processing['results'])
                
                # Add zone results if available
                if 'zone_baseline_results' in st.session_state.atr_processing:
                    all_current_data.extend(st.session_state.atr_processing['zone_baseline_results'])
                if 'state_check_results' in st.session_state.atr_processing:
                    all_current_data.extend(st.session_state.atr_processing['state_check_results'])
                
                if all_current_data:
                    partial_df = pd.DataFrame(all_current_data)
                    
                    # ADD CRITICAL METADATA COLUMNS for downstream apps
                    if not partial_df.empty:
                        # Get original intraday data to extract metadata
                        if st.session_state.atr_processing.get('intraday_data') is not None:
                            intraday_data = st.session_state.atr_processing['intraday_data']
                            
                            # Add Trading_Days_Count (for final display)
                            if 'Trading_Days_Count' in intraday_data.columns:
                                final_trading_days = intraday_data['Trading_Days_Count'].iloc[-1]
                                partial_df['Trading_Days_Count'] = final_trading_days
                            
                            # Add Analysis_Timeframe (for summarizer logic)
                            if 'Analysis_Timeframe' in intraday_data.columns:
                                analysis_timeframe = intraday_data['Analysis_Timeframe'].iloc[0]
                                partial_df['Analysis_Timeframe'] = analysis_timeframe
                            
                            # Add Candle_Interval_Minutes (for summarizer)
                            if 'Candle_Interval_Minutes' in intraday_data.columns:
                                candle_interval = intraday_data['Candle_Interval_Minutes'].iloc[0]
                                partial_df['Candle_Interval_Minutes'] = candle_interval
                            
                            # Add Base_Interval_Minutes (for summarizer)
                            if 'Base_Interval_Minutes' in intraday_data.columns:
                                base_interval = intraday_data['Base_Interval_Minutes'].iloc[0]
                                partial_df['Base_Interval_Minutes'] = base_interval
                            
                            # Add Ticker and Asset Type for identification
                            partial_df['Ticker'] = st.session_state.atr_processing.get('ticker', 'UNKNOWN')
                            
                            # Add detailed asset class info from UI
                            base_asset_type = st.session_state.atr_processing.get('asset_type', 'UNKNOWN')
                            partial_df['AssetType'] = base_asset_type
                            
                            # Add extended hours info for stocks
                            extended_hours_info = st.session_state.atr_processing.get('extended_hours', False)
                            if base_asset_type == 'STOCKS':
                                if extended_hours_info:
                                    partial_df['AssetClass'] = 'STOCKS_ETH'  # Extended Trading Hours
                                else:
                                    partial_df['AssetClass'] = 'STOCKS_RTH'  # Regular Trading Hours
                            else:
                                partial_df['AssetClass'] = base_asset_type  # CRYPTO, FOREX, FUTURES as-is
                    
                    csv_data = partial_df.to_csv(index=False)
                    
                    # Create unique filename
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    ticker_clean = st.session_state.atr_processing.get('ticker', 'TICKER').replace("^", "").replace("=", "_")
                    asset_type = st.session_state.atr_processing.get('asset_type', 'ASSET')
                    
                    emergency_filename = f'{ticker_clean}_{asset_type}_EMERGENCY_{current_progress}of{total_progress}_{timestamp}.csv'
                    
                    # Multiple download options to prevent button disappearing
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.download_button(
                            label=f'💾 Save Progress',
                            data=csv_data,
                            file_name=emergency_filename,
                            mime='text/csv',
                            key=f'emergency_save_1_{current_progress}_{len(all_current_data)}',
                            help=f'Download {len(all_current_data):,} records'
                        )
                    
                    with col2:
                        st.download_button(
                            label=f'🔄 Backup Download',
                            data=csv_data,
                            file_name=f'backup_{emergency_filename}',
                            mime='text/csv',
                            key=f'emergency_save_2_{current_progress}_{len(all_current_data)}',
                            help=f'Alternative download option'
                        )
                    
                    with col3:
                        # Show file info
                        st.write(f"**File:** `{emergency_filename}`")
                        st.write(f"**Size:** {len(csv_data):,} bytes")
                        st.write(f"**Progress:** {progress_pct:.1f}%")
        
        st.markdown("---")
    
    # Continue processing immediately
    with st.spinner('Auto-continuing systematic analysis...'):
        try:
            batch_results = detect_triggers_and_goals_batch(
                st.session_state.atr_processing['daily_data'], 
                st.session_state.atr_processing['intraday_data'], 
                st.session_state.atr_processing['custom_ratios'],
                start_index=st.session_state.atr_processing['last_processed_index']
            )
            
            if batch_results is not None and len(batch_results) > 0:
                st.session_state.atr_processing['results'].extend(batch_results.to_dict('records'))
            
            # Auto-restart if not complete
            if not st.session_state.atr_processing['is_complete']:
                time.sleep(0.1)
                st.rerun()
                
        except Exception as e:
            st.error(f"Error in auto-continue: {e}")

# Check if we have completed results to display
if ('atr_processing' in st.session_state and 
    'final_results' in st.session_state.atr_processing and 
    st.session_state.atr_processing['final_results'] is not None):
    
    st.success("✅ **ANALYSIS COMPLETE!** Results ready for download.")
    
    # Display the final results
    final_df = st.session_state.atr_processing['final_results']
    final_debug = st.session_state.atr_processing['final_debug']
    final_ticker = st.session_state.atr_processing['final_ticker']
    final_asset_type = st.session_state.atr_processing['final_asset_type']
    
    display_results(final_df, final_debug, final_ticker, final_asset_type)
    
    # Clear results button
    if st.button("🗑️ Clear Results and Start New Analysis"):
        st.session_state.atr_processing = {
            'results': [],
            'last_processed_index': 0,
            'is_complete': False,
            'daily_data': None,
            'intraday_data': None,
            'custom_ratios': None,
            'ticker': '',
            'asset_type': '',
            'total_periods': 0,
            'zone_baseline_results': [],
            'zone_baseline_complete': False,
            'state_check_results': [],
            'state_check_complete': False
        }
        st.rerun()

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
        
        # Resume processing option
        st.subheader("🔄 Resume Processing")
        resume_processing = st.checkbox("Resume from specific period", help="Continue processing from a previous point")
        resume_from_period = 0
        if resume_processing:
            resume_from_period = st.number_input(
                "Start from period:",
                min_value=0,
                max_value=10000,
                value=0,
                help="Enter the period number to resume from (0 = start from beginning)"
            )
            st.info(f"Will skip to period {resume_from_period} and continue from there.")
            
            if st.button("🗑️ Clear Previous Session Data"):
                if 'atr_processing' in st.session_state:
                    del st.session_state.atr_processing
                st.success("Session data cleared. Ready for fresh start or resume.")
                st.rerun()
    
    # Session filtering (simplified)
    config = AssetConfig.get_config(asset_type, extended_hours)
    session_filter = None  # Will be handled by CSV handler if needed
    
    # Run analysis button
    st.markdown("---")
    
    if st.button('🚀 Generate Complete ATR Analysis Suite', type="primary", use_container_width=True):
        with st.spinner('Processing ALL analysis types: Session, Rolling, ZoneBaseline, and StateCheck...'):
            try:
                result_df, debug_messages = main_analysis(
                    ticker=ticker,
                    asset_type=asset_type,
                    data_file=data_file,
                    custom_ratios=custom_ratios,
                    debug_mode=debug_mode,
                    debug_date=debug_date,
                    resume_from_period=resume_from_period,
                    extended_hours=extended_hours
                )
                
                display_results(result_df, debug_messages, ticker, asset_type)
                
                # DON'T clear processing state - keep results for download
                # Store final results in session state for persistence
                st.session_state.atr_processing['final_results'] = result_df
                st.session_state.atr_processing['final_debug'] = debug_messages
                st.session_state.atr_processing['final_ticker'] = ticker
                st.session_state.atr_processing['final_asset_type'] = asset_type
                    
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
        - **Rolling Config**: Candle_Interval_Minutes, Rolling_Period_Type, Rolling_Period_Count, Analysis_Timeframe
        - **Format**: CSV file with proper datetime formatting
        - **Content**: Combined daily and intraday data with ATR pre-calculated
        
        **✅ Expected Format:**
        - ATR column already calculated
        - Prior_Base_Close for level calculations
        - Date and Time columns properly formatted
        - Trading_Days_Count for validation
        - Rolling analysis configuration columns
        - No missing OHLC data
        """)
    
    # Show workflow
    with st.expander("🔧 Analysis Workflow", expanded=False):
        st.markdown("""
        **🎯 Step-by-Step Process:**
        
        1. **Use CSV Data Handler** - Process your raw data files first
        2. **Upload Pre-formatted File** - Upload the output from CSV Data Handler
        3. **Configure Settings** - Ticker, asset type, analysis options
        4. **Enhanced Analysis** - Both Daily and Rolling trigger/goal analysis
        5. **Results Export** - Download comprehensive analysis results
        
        **🔍 What You Get:**
        - **Daily Analysis**: Complete trigger/goal combinations until end of session
        - **Rolling Analysis**: Fixed-period lookahead windows (e.g., 8 hours)
        - **Comparative Statistics**: Hit rates for both timeframes
        - **Goal Classifications**: Continuation, Retracement, Retest
        - **Same-time Scenario Analysis**: Cross-zero detection
        - **Debug Mode**: Detailed single-day breakdown
        
        **💾 Perfect for:**
        - Multi-timeframe trading strategy development
        - Short-term vs long-term goal completion comparison
        - Systematic trading system backtesting
        - Time-based performance analysis
        """)

# Help section
st.markdown("---")
st.subheader("📚 Key Features")

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("""
    **🎯 Enhanced Analysis**
    - Daily AND Rolling analysis
    - Identical systematic logic
    - Dynamic period calculation
    - Cross-session goal tracking
    - Comparative hit rate analysis
    """)

with col2:
    st.markdown("""
    **📊 Dual Time Windows**
    - Daily: Until end of session
    - Rolling: Fixed period ahead
    - Configurable rolling periods
    - Automatic candle calculation
    - Session-aware boundaries
    """)

with col3:
    st.markdown("""
    **🔧 Advanced Features**
    - Multi-asset class support
    - Custom ratio definitions
    - Debug mode for single days
    - Comprehensive statistics
    - Enhanced results export
    """)

st.info("💡 **Enhanced Tool**: Now provides both daily session analysis AND rolling period analysis for comprehensive market behavior insights!")

st.markdown("""
### 🎯 About This Enhanced Tool

This is an **enhanced ATR analysis tool** that provides both **Daily** and **Rolling** time window analysis.

**Key Enhancements:**

- ✅ **Dual Analysis** - Daily (until end of session) + Rolling (fixed periods ahead)
- ✅ **Same Critical Logic** - Identical systematic trigger/goal detection for both
- ✅ **Dynamic Configuration** - Automatically calculates rolling periods from CSV config
- ✅ **Comparative Results** - Side-by-side hit rate analysis
- ✅ **Enhanced Statistics** - Comprehensive breakdown of both analysis types

**Perfect workflow:**

1. **CSV Data Handler** → Process data and configure rolling analysis
2. **This Enhanced ATR Tool** → Run both daily and rolling analysis
3. **Export Results** → Get comprehensive CSV files with both analysis types

**Example Output:**
- Daily Analysis: 15,000 trigger/goal combinations
- Rolling Analysis: 15,000 additional combinations with fixed time windows
- Total: 30,000 records for comprehensive analysis

Enhanced, comprehensive, and powerful!
""")
