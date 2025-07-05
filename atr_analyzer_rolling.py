import streamlit as st
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

def detect_triggers_and_goals_unified_batch(daily, intraday, custom_ratios=None, start_index=0, batch_size=50):
    """
    UNIFIED batch analysis: Session, Rolling, ZoneBaseline, and StateCheck in single loop
    Processes batch_size periods at a time with all 4 analysis types per period
    """
    if custom_ratios is None:
        fib_levels = [0.236, 0.382, 0.500, 0.618, 0.786, 1.000,
                     -0.236, -0.382, -0.500, -0.618, -0.786, -1.000, 0.000]
    else:
        fib_levels = custom_ratios

    all_results = []

    # Progress tracking
    total_periods = len(daily)
    end_index = min(start_index + batch_size, total_periods)

    progress_bar = st.progress(start_index / total_periods)
    status_text = st.empty()

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

    for i in range(start_index, end_index):
        try:
            # Update progress
            progress = (i + 1) / total_periods
            progress_bar.progress(progress)
            status_text.text(f"Unified Processing: period {i+1}/{total_periods} (All 4 analysis types)...")
            
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
            
            # ==================================================================================
            # 1. SESSION + ROLLING ANALYSIS (Using existing critical section logic)
            # ==================================================================================
            
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
                    
                    # Base record template for both session and rolling
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
                        
                        # Determine goal type for BELOW trigger
                        if goal_level == trigger_level:
                            goal_type = 'Retest'  # Same level retest
                        elif goal_level < trigger_level:
                            goal_type = 'Continuation'  # Further below
                        else:
                            goal_type = 'Retracement'   # Back above (includes cross-zero)
                        
                        # SESSION ANALYSIS: Check until end of session
                        session_goal_hit = False
                        session_goal_time = ''
                        session_is_same_time = False
                        
                        # [CRITICAL SECTION LOGIC FOR SESSION - UNCHANGED]
                        if below_trigger_time == 'OPEN':
                            # Step 1: Check if goal completes at OPEN price first (takes precedence)
                            if goal_level == trigger_level:  # Same level retest
                                if open_price >= goal_price:
                                    session_goal_hit = True
                                    session_goal_time = 'OPEN'
                                    session_is_same_time = True
                            elif goal_level > trigger_level:  # Above goal (RETRACEMENT)
                                if open_price >= goal_price:
                                    session_goal_hit = True
                                    session_goal_time = 'OPEN'
                                    session_is_same_time = True
                            else:  # Below goal (CONTINUATION)
                                if open_price <= goal_price:
                                    session_goal_hit = True
                                    session_goal_time = 'OPEN'
                                    session_is_same_time = True
                            
                            # Step 2: Only if OPEN missed, check candles based on goal type
                            if not session_goal_hit:
                                if goal_level == trigger_level:  # RETEST - must skip same candle
                                    start_candles = day_data.iloc[1:].iterrows()
                                elif goal_level > trigger_level:  # RETRACEMENT - must skip same candle
                                    start_candles = day_data.iloc[1:].iterrows()
                                else:  # CONTINUATION - can check same candle
                                    start_candles = day_data.iterrows()
                                
                                for _, row in start_candles:
                                    if goal_level == trigger_level:  # Same level retest (opposite direction)
                                        if row['High'] >= goal_price:
                                            session_goal_hit = True
                                            session_goal_time = row['Time']
                                            break
                                    elif goal_level > trigger_level:  # Above goal
                                        if row['High'] >= goal_price:
                                            session_goal_hit = True
                                            session_goal_time = row['Time']
                                            break
                                    else:  # Below goal  
                                        if row['Low'] <= goal_price:
                                            session_goal_hit = True
                                            session_goal_time = row['Time']
                                            break
                        
                        else:  # Intraday below trigger
                            if goal_level == trigger_level:  # RETEST - Skip same candle entirely
                                pass
                            elif goal_level > trigger_level:  # RETRACEMENT - Skip same candle entirely  
                                pass
                            else:  # CONTINUATION - Can check same candle
                                if goal_level < trigger_level:
                                    if trigger_candle['Low'] <= goal_price:
                                        session_goal_hit = True
                                        session_goal_time = below_trigger_time
                            
                            # Check subsequent candles if not completed on trigger candle
                            if not session_goal_hit:
                                for _, row in day_data.iloc[below_trigger_row + 1:].iterrows():
                                    if goal_level == trigger_level:  # Same level retest (opposite direction)
                                        if row['High'] >= goal_price:
                                            session_goal_hit = True
                                            session_goal_time = row['Time']
                                            break
                                    elif goal_level > trigger_level:  # Above goal
                                        if row['High'] >= goal_price:
                                            session_goal_hit = True
                                            session_goal_time = row['Time']
                                            break
                                    else:  # Below goal
                                        if row['Low'] <= goal_price:
                                            session_goal_hit = True
                                            session_goal_time = row['Time']
                                            break
                        
                        # Record SESSION result
                        session_record = base_record.copy()
                        session_record.update({
                            'GoalLevel': goal_level,
                            'GoalPrice': round(goal_price, 2),
                            'GoalHit': 'Yes' if session_goal_hit else 'No',
                            'GoalTime': session_goal_time if session_goal_hit else '',
                            'GoalClassification': goal_type,
                            'SameTime': session_is_same_time,
                            'AnalysisType': 'Session'
                        })
                        all_results.append(session_record)
                        
                        # ROLLING ANALYSIS: Same logic but with rolling boundary
                        if has_rolling_config:
                            rolling_goal_hit = False
                            rolling_goal_time = ''
                            rolling_is_same_time = False
                            
                            # [SAME LOGIC AS SESSION BUT WITH ROLLING BOUNDARY]
                            if below_trigger_time == 'OPEN':
                                # Step 1: Check if goal completes at OPEN price first
                                if goal_level == trigger_level:  # Same level retest
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
                                
                                # Step 2: Only if OPEN missed, check candles with rolling boundary
                                if not rolling_goal_hit:
                                    if goal_level == trigger_level:  # RETEST
                                        rolling_end_idx = min(1 + rolling_candles, len(day_data))
                                        start_candles = day_data.iloc[1:rolling_end_idx].iterrows()
                                    elif goal_level > trigger_level:  # RETRACEMENT
                                        rolling_end_idx = min(1 + rolling_candles, len(day_data))
                                        start_candles = day_data.iloc[1:rolling_end_idx].iterrows()
                                    else:  # CONTINUATION
                                        rolling_end_idx = min(rolling_candles, len(day_data))
                                        start_candles = day_data.iloc[0:rolling_end_idx].iterrows()
                                    
                                    for _, row in start_candles:
                                        if goal_level == trigger_level:  # Same level retest
                                            if row['High'] >= goal_price:
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
                            
                            else:  # Intraday below trigger
                                if goal_level == trigger_level:  # RETEST - Skip same candle
                                    pass
                                elif goal_level > trigger_level:  # RETRACEMENT - Skip same candle
                                    pass
                                else:  # CONTINUATION - Can check same candle
                                    if goal_level < trigger_level:
                                        if trigger_candle['Low'] <= goal_price:
                                            rolling_goal_hit = True
                                            rolling_goal_time = below_trigger_time
                                
                                # Check subsequent candles with rolling boundary
                                if not rolling_goal_hit:
                                    rolling_end_idx = min(below_trigger_row + 1 + rolling_candles, len(day_data))
                                    for _, row in day_data.iloc[below_trigger_row + 1:rolling_end_idx].iterrows():
                                        if goal_level == trigger_level:  # Same level retest
                                            if row['High'] >= goal_price:
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
                            
                            # Record ROLLING result
                            rolling_record = create_rolling_result_record(
                                base_record, goal_level, goal_price, rolling_goal_hit, 
                                rolling_goal_time, goal_type, rolling_is_same_time
                            )
                            all_results.append(rolling_record)
                
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
                
                # Check intraday candles for above trigger
                if not above_triggered:
                    for idx, row in day_data.iloc[1:].iterrows():
                        if row['High'] >= trigger_price:
                            above_triggered = True
                            above_trigger_time = row['Time']
                            above_trigger_row = idx
                            break
                
                # Process all goals for ABOVE trigger (same pattern as below)
                if above_triggered:
                    trigger_candle = day_data.iloc[above_trigger_row]
                    
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
                        
                        # Determine goal type for ABOVE trigger
                        if goal_level == trigger_level:
                            goal_type = 'Retest'  # Same level retest
                        elif goal_level > trigger_level:
                            goal_type = 'Continuation'  # Further above
                        else:
                            goal_type = 'Retracement'   # Back below (includes cross-zero)
                        
                        # SESSION ANALYSIS for ABOVE trigger
                        session_goal_hit = False
                        session_goal_time = ''
                        session_is_same_time = False
                        
                        # [Same logic as below trigger but for above direction]
                        if above_trigger_time == 'OPEN':
                            if goal_level == trigger_level:  # Same level retest
                                if open_price <= goal_price:
                                    session_goal_hit = True
                                    session_goal_time = 'OPEN'
                                    session_is_same_time = True
                            elif goal_level > trigger_level:  # Above goal (CONTINUATION)
                                if open_price >= goal_price:
                                    session_goal_hit = True
                                    session_goal_time = 'OPEN'
                                    session_is_same_time = True
                            else:  # Below goal (RETRACEMENT)
                                if open_price <= goal_price:
                                    session_goal_hit = True
                                    session_goal_time = 'OPEN'
                                    session_is_same_time = True
                            
                            if not session_goal_hit:
                                for _, row in day_data.iterrows():
                                    if goal_level == trigger_level:  # Same level retest
                                        if row['Low'] <= goal_price:
                                            session_goal_hit = True
                                            session_goal_time = row['Time']
                                            break
                                    elif goal_level > trigger_level:  # Above goal
                                        if row['High'] >= goal_price:
                                            session_goal_hit = True
                                            session_goal_time = row['Time']
                                            break
                                    else:  # Below goal
                                        if row['Low'] <= goal_price:
                                            session_goal_hit = True
                                            session_goal_time = row['Time']
                                            break
                        
                        else:  # Intraday above trigger
                            if goal_level == trigger_level:  # RETEST - Skip same candle
                                pass
                            elif goal_level < trigger_level:  # RETRACEMENT - Skip same candle
                                pass
                            else:  # CONTINUATION - Can check same candle
                                if goal_level > trigger_level:
                                    if trigger_candle['High'] >= goal_price:
                                        session_goal_hit = True
                                        session_goal_time = above_trigger_time
                            
                            if not session_goal_hit:
                                for _, row in day_data.iloc[above_trigger_row + 1:].iterrows():
                                    if goal_level == trigger_level:  # Same level retest
                                        if row['Low'] <= goal_price:
                                            session_goal_hit = True
                                            session_goal_time = row['Time']
                                            break
                                    elif goal_level > trigger_level:  # Above goal
                                        if row['High'] >= goal_price:
                                            session_goal_hit = True
                                            session_goal_time = row['Time']
                                            break
                                    else:  # Below goal
                                        if row['Low'] <= goal_price:
                                            session_goal_hit = True
                                            session_goal_time = row['Time']
                                            break
                        
                        # Record SESSION result for ABOVE
                        session_record = base_record.copy()
                        session_record.update({
                            'GoalLevel': goal_level,
                            'GoalPrice': round(goal_price, 2),
                            'GoalHit': 'Yes' if session_goal_hit else 'No',
                            'GoalTime': session_goal_time if session_goal_hit else '',
                            'GoalClassification': goal_type,
                            'SameTime': session_is_same_time,
                            'AnalysisType': 'Session'
                        })
                        all_results.append(session_record)
                        
                        # ROLLING ANALYSIS for ABOVE trigger (same pattern)
                        if has_rolling_config:
                            rolling_goal_hit = False
                            rolling_goal_time = ''
                            rolling_is_same_time = False
                            
                            # [Rolling logic for above trigger - similar to session but with boundary]
                            if above_trigger_time == 'OPEN':
                                if goal_level == trigger_level:  # Same level retest
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
                                
                                if not rolling_goal_hit:
                                    rolling_end_idx = min(rolling_candles, len(day_data))
                                    for _, row in day_data.iloc[0:rolling_end_idx].iterrows():
                                        if goal_level == trigger_level:  # Same level retest
                                            if row['Low'] <= goal_price:
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
                            
                            else:  # Intraday above trigger
                                if goal_level == trigger_level:  # RETEST - Skip same candle
                                    pass
                                elif goal_level < trigger_level:  # RETRACEMENT - Skip same candle
                                    pass
                                else:  # CONTINUATION - Can check same candle
                                    if goal_level > trigger_level:
                                        if trigger_candle['High'] >= goal_price:
                                            rolling_goal_hit = True
                                            rolling_goal_time = above_trigger_time
                                
                                if not rolling_goal_hit:
                                    rolling_end_idx = min(above_trigger_row + 1 + rolling_candles, len(day_data))
                                    for _, row in day_data.iloc[above_trigger_row + 1:rolling_end_idx].iterrows():
                                        if goal_level == trigger_level:  # Same level retest
                                            if row['Low'] <= goal_price:
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
                            
                            # Record ROLLING result for ABOVE
                            rolling_record = create_rolling_result_record(
                                base_record, goal_level, goal_price, rolling_goal_hit, 
                                rolling_goal_time, goal_type, rolling_is_same_time
                            )
                            all_results.append(rolling_record)
            
            # ==================================================================================
            # 2. ZONEBASELINE ANALYSIS (For this single period)
            # ==================================================================================
            
            # For every candle in this period, identify ALL zones it crosses
            for idx, candle in day_data.iterrows():
                high_price = candle['High']
                low_price = candle['Low']
                close_price = candle['Close']
                
                # Get all zones this candle touches
                zones_crossed = get_zones_crossed(high_price, low_price, level_map)
                
                # Create a record for EACH zone crossed
                for zone in zones_crossed:
                    all_results.append({
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
            
            # ==================================================================================
            # 3. STATECHECK ANALYSIS (For this single period)
            # ==================================================================================
            
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
                        
                        # Convert counts to percentages and record
                        for zone, count in zone_frequencies.items():
                            percentage = (count / total_candles_remaining) * 100
                            
                            all_results.append({
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
            st.warning(f"Error processing {trading_date}: {str(e)}")
            continue

    # Update session state
    st.session_state.atr_processing['last_processed_index'] = end_index

    # Check if we're done
    if end_index >= total_periods:
        st.session_state.atr_processing['is_complete'] = True
        status_text.text("All 4 analysis types complete!")
    else:
        status_text.text(f"Unified batch complete. Next batch will start at period {end_index + 1}")

    # Clear progress bar after batch
    progress_bar.empty()

    return pd.DataFrame(all_results)

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
                'total_periods': 0
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
        
        # Run unified systematic analysis
        if not debug_mode:
            if not st.session_state.atr_processing['is_complete']:
                debug_info.append("Running UNIFIED analysis: Session, Rolling, ZoneBaseline, and StateCheck together...")
                
                # Process ALL 4 analysis types in unified batches
                batch_results = detect_triggers_and_goals_unified_batch(
                    daily_data, 
                    intraday_data, 
                    custom_ratios,
                    start_index=st.session_state.atr_processing['last_processed_index']
                )
                
                if batch_results is not None:
                    # Add new results to session state (all 4 analysis types)
                    if len(batch_results) > 0:
                        st.session_state.atr_processing['results'].extend(batch_results.to_dict('records'))
                    
                    debug_info.append(f"Unified batch complete. Total records found: {len(st.session_state.atr_processing['results'])}")
                
                # Check if processing is complete
                if st.session_state.atr_processing['is_complete']:
                    # Convert results back to DataFrame (all 4 analysis types included)
                    result_df = pd.DataFrame(st.session_state.atr_processing['results'])
                    
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
        
        # Session vs Rolling comparison
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
                    'GoalHit': lambda x: (x == 'Yes
