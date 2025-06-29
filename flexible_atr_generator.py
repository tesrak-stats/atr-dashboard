def calculate_atr(df, period=14):
    """
    Calculate TRUE Wilder's ATR - validated implementation
    """
    df = df.copy()
    
    # Calculate True Range
    df['H-L'] = df['High'] - df['Low']
    df['H-PC'] = abs(df['High'] - df['Close'].shift(1))
    df['L-PC'] = abs(df['Low'] - df['Close'].shift(1))
    df['TR'] = df[['H-L', 'H-PC', 'L-PC']].max(axis=1)
    
    # Calculate TRUE Wilder's ATR
    atr_values = [None] * len(df)
    
    for i in range(len(df)):
        if i < period:
            atr_values[i] = None
        elif i == period:
            atr_values[i] = df['TR'].iloc[i-period+1:i+1].mean()
        else:
            prev_atr = atr_values[i-1]
            current_tr = df['TR'].iloc[i]
            atr_values[i] = (1/period) * current_tr + ((period-1)/period) * prev_atr
    
    df['ATR'] = atr_values
    df.drop(['H-L', 'H-PC', 'L-PC', 'TR'], axis=1, inplace=True)
    
    return df

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
    
    return levelsimport streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
import os

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
                'example_tickers': ['AAPL', 'GOOGL', 'SPY', '^GSPC'],
                'extended_hours': extended_hours
            },
            'STOCKS_24H': {
                'market_open': '00:00',
                'market_close': '23:59',
                'has_open_special': False,  # 24-hour stock data (rare but exists)
                'weekends_closed': False,
                'session_types': ['24H'],
                'default_session': ['24H'],
                'description': 'US Stocks (24-Hour Data - if available)',
                'example_tickers': ['AAPL', 'GOOGL', 'SPY'],
                'extended_hours': True
            },
            'CRYPTO': {
                'market_open': '00:00',
                'market_close': '23:59',
                'has_open_special': False,
                'weekends_closed': False,
                'session_types': ['24H'],
                'default_session': ['24H'],
                'description': 'Cryptocurrency (24/7 trading)',
                'example_tickers': ['BTC-USD', 'ETH-USD', 'ADA-USD'],
                'extended_hours': True
            },
            'FOREX': {
                'market_open': '17:00',  # Sunday 5 PM EST
                'market_close': '17:00',  # Friday 5 PM EST
                'has_open_special': False,
                'weekends_closed': True,
                'session_types': ['ASIA', 'EUROPE', 'US', '24H'],
                'default_session': ['24H'],
                'description': 'Foreign Exchange (Sun 5PM - Fri 5PM EST)',
                'example_tickers': ['EURUSD=X', 'GBPUSD=X', 'USDJPY=X'],
                'extended_hours': True
            },
            'FUTURES': {
                'market_open': '18:00',  # Sunday 6 PM EST
                'market_close': '17:00',  # Friday 5 PM EST
                'has_open_special': True,
                'weekends_closed': True,
                'session_types': ['GLOBEX', 'RTH'],
                'default_session': ['GLOBEX', 'RTH'],
                'description': 'Futures (nearly 24/5 trading)',
                'example_tickers': ['ES=F', 'NQ=F', 'YM=F', 'CL=F'],
                'extended_hours': True
            },
            'COMMODITIES': {
                'market_open': '09:30',
                'market_close': '16:30',
                'has_open_special': True,
                'weekends_closed': True,
                'session_types': ['R', 'AH'],
                'default_session': ['R'],
                'description': 'Commodities (varies by commodity)',
                'example_tickers': ['GC=F', 'SI=F', 'CL=F'],
                'extended_hours': extended_hours
            }
        }
        return configs.get(asset_type, configs['STOCKS'])

def load_daily_data(uploaded_file=None, ticker=None, start_date=None, end_date=None):
    """
    Load daily data from uploaded file or Yahoo Finance
    """
    if uploaded_file is not None:
        try:
            # Handle different file types
            if uploaded_file.name.endswith('.csv'):
                daily = pd.read_csv(uploaded_file)
            elif uploaded_file.name.endswith(('.xlsx', '.xls')):
                # Try to detect header row for Excel files
                daily = pd.read_excel(uploaded_file, header=0)
                # If that doesn't work, try other common header positions
                if 'Date' not in daily.columns and 'Close' not in daily.columns:
                    for header_row in [1, 2, 3, 4, 5]:
                        try:
                            daily = pd.read_excel(uploaded_file, header=header_row)
                            if 'Date' in daily.columns or 'Close' in daily.columns:
                                st.info(f"✅ Found headers at row {header_row + 1}")
                                break
                        except:
                            continue
            else:
                st.error("Unsupported file format. Please use CSV or Excel files.")
                return None
            
            # Standardize column names
            daily = standardize_columns(daily)
            
            st.success(f"✅ Loaded daily data: {len(daily)} records")
            st.info(f"Columns found: {list(daily.columns)}")
            
            return daily
            
        except Exception as e:
            st.error(f"Error loading daily data: {str(e)}")
            return None
    
    elif ticker and start_date and end_date:
        # Fallback to Yahoo Finance
        try:
            st.info(f"📈 Fetching daily data from Yahoo Finance for {ticker}...")
            daily_data = yf.download(ticker, start=start_date, end=end_date, interval='1d')
            
            if daily_data.empty:
                st.error(f"No daily data found for {ticker}")
                return None
            
            daily_data.reset_index(inplace=True)
            daily_data = standardize_columns(daily_data)
            
            st.success(f"✅ Loaded daily data from Yahoo: {len(daily_data)} records")
            return daily_data
            
        except Exception as e:
            st.error(f"Error fetching from Yahoo Finance: {str(e)}")
            return None
    
    return None

def load_intraday_data(uploaded_file):
    """
    Load intraday data from uploaded file
    """
    try:
        if uploaded_file.name.endswith('.csv'):
            intraday = pd.read_csv(uploaded_file)
        elif uploaded_file.name.endswith(('.xlsx', '.xls')):
            intraday = pd.read_excel(uploaded_file, header=0)
        else:
            st.error("Unsupported file format. Please use CSV or Excel files.")
            return None
        
        # Standardize columns
        intraday = standardize_columns(intraday)
        
        # Ensure we have a datetime column
        if 'Datetime' not in intraday.columns:
            # Try to create from Date and Time columns
            if 'Date' in intraday.columns and 'Time' in intraday.columns:
                intraday['Datetime'] = pd.to_datetime(intraday['Date'].astype(str) + ' ' + intraday['Time'].astype(str))
            elif 'Date' in intraday.columns:
                # Assume Date column contains full datetime
                intraday['Datetime'] = pd.to_datetime(intraday['Date'])
            else:
                st.error("Could not find datetime information in intraday data")
                return None
        else:
            intraday['Datetime'] = pd.to_datetime(intraday['Datetime'])
        
        # Create Date column for matching
        intraday['Date'] = intraday['Datetime'].dt.date
        
        st.success(f"✅ Loaded intraday data: {len(intraday)} records")
        st.info(f"Date range: {intraday['Date'].min()} to {intraday['Date'].max()}")
        
        return intraday
        
    except Exception as e:
        st.error(f"Error loading intraday data: {str(e)}")
        return None

def standardize_columns(df):
    """
    Standardize column names across different data sources
    """
    # Common column mappings
    column_mappings = {
        # Date columns
        'date': 'Date',
        'timestamp': 'Date', 
        'datetime': 'Datetime',
        # OHLC columns
        'open': 'Open',
        'high': 'High', 
        'low': 'Low',
        'close': 'Close',
        'adj close': 'Close',
        'adjusted_close': 'Close',
        # Volume
        'volume': 'Volume',
        'vol': 'Volume',
        # Session
        'session': 'Session'
    }
    
    # Apply mappings (case insensitive)
    for old_name, new_name in column_mappings.items():
        for col in df.columns:
            if col.lower() == old_name:
                df.rename(columns={col: new_name}, inplace=True)
                break
    
    return df

def filter_by_session_and_hours(intraday_data, date, asset_config, session_filter=None):
    """
    Filter intraday data based on sessions and trading hours
    """
    day_data = intraday_data[intraday_data['Date'] == date].copy()
    
    if day_data.empty:
        return day_data
    
    # Session-based filtering (if session column exists)
    if session_filter and 'Session' in day_data.columns:
        day_data = day_data[day_data['Session'].isin(session_filter)]
        st.info(f"Filtered by sessions: {session_filter}")
    
    # Time-based filtering for traditional markets
    if asset_config['weekends_closed'] and asset_config['has_open_special']:
        # Apply time filtering
        market_open = asset_config['market_open']
        market_close = asset_config['market_close']
        
        if market_open != '00:00' or market_close != '23:59':
            day_data['Time_obj'] = day_data['Datetime'].dt.time
            open_time = pd.to_datetime(market_open, format='%H:%M').time()
            close_time = pd.to_datetime(market_close, format='%H:%M').time()
            
            # Filter by time range
            time_mask = (day_data['Time_obj'] >= open_time) & (day_data['Time_obj'] <= close_time)
            day_data = day_data[time_mask]
            day_data.drop('Time_obj', axis=1, inplace=True)
    
    # Create time string for display
    day_data['Time'] = day_data['Datetime'].dt.strftime('%H%M')
    
    return day_data.reset_index(drop=True)

def detect_triggers_and_goals_flexible(daily, intraday, asset_config, custom_ratios=None, session_filter=None):
    """
    Flexible trigger and goal detection for different asset classes
    """
    if custom_ratios is None:
        fib_levels = [0.236, 0.382, 0.500, 0.618, 0.786, 1.000, 
                     -0.236, -0.382, -0.500, -0.618, -0.786, -1.000, 0.000]
    else:
        fib_levels = custom_ratios
    
    results = []
    has_open_special = asset_config['has_open_special']
    
    for i in range(1, len(daily)):
        try:
            previous_row = daily.iloc[i-1]
            current_row = daily.iloc[i]
            
            previous_close = previous_row['Close']
            previous_atr = previous_row['ATR']
            trading_date = current_row['Date']
            
            # Skip if no valid ATR
            if pd.isna(previous_atr) or pd.isna(previous_close):
                continue
            
            # Generate levels
            level_map = generate_atr_levels(previous_close, previous_atr, custom_ratios)
            
            # Get session-appropriate intraday data
            if isinstance(trading_date, str):
                date_obj = pd.to_datetime(trading_date).date()
            else:
                date_obj = trading_date.date() if hasattr(trading_date, 'date') else trading_date
            
            # Get trading session data using correct function name
        day_data = filter_by_session_and_hours(intraday, date_obj, asset_config, session_filter)
            
            if day_data.empty:
                continue
            
            # For markets with gaps, there's a special "OPEN" - use first candle
            # For 24/7 markets, there's no gap - just continuous trading
            if has_open_special and len(day_data) > 0:
                open_candle = day_data.iloc[0]
                open_price = open_candle['Open']
            else:
                open_price = None
            
            # Process each trigger level
            for trigger_level in fib_levels:
                trigger_price = level_map[trigger_level]
                
                # BELOW DIRECTION
                below_triggered = False
                below_trigger_time = None
                below_trigger_row = None
                
                # Check OPEN for markets with gaps only
                if has_open_special and open_price is not None and open_price <= trigger_price:
                    below_triggered = True
                    below_trigger_time = 'OPEN'
                    below_trigger_row = 0
                
                # Check intraday candles
                if not below_triggered:
                    start_idx = 1 if has_open_special else 0
                    for idx, row in day_data.iloc[start_idx:].iterrows():
                        if row['Low'] <= trigger_price:
                            below_triggered = True
                            below_trigger_time = row['Time']
                            below_trigger_row = idx
                            break
                
                # Process goals for BELOW trigger
                if below_triggered:
                    process_goals_for_trigger(
                        results, day_data, fib_levels, level_map, trigger_level, 
                        'Below', below_trigger_time, below_trigger_row, trigger_price,
                        trading_date, previous_close, previous_atr, has_open_special, open_price
                    )
                
                # ABOVE DIRECTION
                above_triggered = False
                above_trigger_time = None
                above_trigger_row = None
                
                # Check OPEN for markets with gaps only
                if has_open_special and open_price is not None and open_price >= trigger_price:
                    above_triggered = True
                    above_trigger_time = 'OPEN'
                    above_trigger_row = 0
                
                # Check intraday candles
                if not above_triggered:
                    start_idx = 1 if has_open_special else 0
                    for idx, row in day_data.iloc[start_idx:].iterrows():
                        if row['High'] >= trigger_price:
                            above_triggered = True
                            above_trigger_time = row['Time']
                            above_trigger_row = idx
                            break
                
                # Process goals for ABOVE trigger
                if above_triggered:
                    process_goals_for_trigger(
                        results, day_data, fib_levels, level_map, trigger_level,
                        'Above', above_trigger_time, above_trigger_row, trigger_price,
                        trading_date, previous_close, previous_atr, has_open_special, open_price
                    )
                    
        except Exception as e:
            st.warning(f"Error processing {trading_date}: {str(e)}")
            continue
    
    return pd.DataFrame(results)

def process_goals_for_trigger(results, day_data, fib_levels, level_map, trigger_level, 
                            direction, trigger_time, trigger_row, trigger_price,
                            trading_date, previous_close, previous_atr, has_open_special, open_price):
    """
    Process all goals for a given trigger (separated for cleaner code)
    """
    trigger_candle = day_data.iloc[trigger_row] if trigger_row is not None else None
    
    for goal_level in fib_levels:
        if goal_level == trigger_level:
            continue
        
        goal_price = level_map[goal_level]
        goal_hit = False
        goal_time = ''
        is_same_time = False
        
        # Determine goal type
        if direction == 'Below':
            goal_type = 'Continuation' if goal_level < trigger_level else 'Retracement'
        else:  # Above
            goal_type = 'Continuation' if goal_level > trigger_level else 'Retracement'
        
        # Check for goal completion
        if trigger_time == 'OPEN' and has_open_special:
            # Check if goal completes at OPEN
            if direction == 'Below':
                if goal_level > trigger_level and open_price >= goal_price:
                    goal_hit = True
                    goal_time = 'OPEN'
                    is_same_time = True
                elif goal_level < trigger_level and open_price <= goal_price:
                    goal_hit = True
                    goal_time = 'OPEN'
                    is_same_time = True
            else:  # Above
                if goal_level > trigger_level and open_price >= goal_price:
                    goal_hit = True
                    goal_time = 'OPEN'
                    is_same_time = True
                elif goal_level < trigger_level and open_price <= goal_price:
                    goal_hit = True
                    goal_time = 'OPEN'
                    is_same_time = True
            
            # Check subsequent candles if not completed at OPEN
            if not goal_hit:
                start_idx = 1 if has_open_special else 0
                for _, row in day_data.iloc[start_idx:].iterrows():
                    if check_goal_hit(row, goal_level, trigger_level, goal_price):
                        goal_hit = True
                        goal_time = row['Time']
                        break
        
        else:  # Intraday trigger or 24/7 market
            # Check if goal completes on same candle as trigger
            if trigger_candle is not None and check_goal_hit(trigger_candle, goal_level, trigger_level, goal_price):
                goal_hit = True
                goal_time = trigger_time
                is_same_time = True
            
            # Check subsequent candles if not completed on trigger candle
            if not goal_hit and trigger_row is not None:
                for _, row in day_data.iloc[trigger_row + 1:].iterrows():
                    if check_goal_hit(row, goal_level, trigger_level, goal_price):
                        goal_hit = True
                        goal_time = row['Time']
                        break
        
        # Record result
        results.append({
            'Date': trading_date,
            'Direction': direction,
            'TriggerLevel': trigger_level,
            'TriggerTime': trigger_time,
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

def check_goal_hit(candle, goal_level, trigger_level, goal_price):
    """
    Check if a goal is hit on a given candle
    """
    if goal_level > trigger_level:  # Above goal
        return candle['High'] >= goal_price
    else:  # Below goal
        return candle['Low'] <= goal_price

def main_flexible(ticker=None, asset_type='STOCKS', daily_file=None, intraday_file=None, 
                 start_date=None, end_date=None, atr_period=14, custom_ratios=None, 
                 session_filter=None, extended_hours=False):
    """
    Main function for flexible ATR analysis with file inputs
    """
    debug_info = []
    
    try:
        # Get asset configuration
        asset_config = AssetConfig.get_config(asset_type, extended_hours)
        debug_info.append(f"📊 Asset Type: {asset_config['description']}")
        debug_info.append(f"Market Hours: {asset_config['market_open']} - {asset_config['market_close']}")
        debug_info.append(f"Special OPEN handling: {asset_config['has_open_special']}")
        debug_info.append(f"Extended Hours: {extended_hours}")
        
        # Load daily data
        daily = load_daily_data(daily_file, ticker, start_date, end_date)
        
        if daily is None:
            debug_info.append("❌ Failed to load daily data")
            return pd.DataFrame(), debug_info
        
        debug_info.append(f"Daily data loaded: {daily.shape}")
        
        # Calculate ATR
        debug_info.append(f"🧮 Calculating ATR with period {atr_period}...")
        daily = calculate_atr(daily, period=atr_period)
        
        # Validate ATR
        valid_atr = daily[daily['ATR'].notna()]
        if not valid_atr.empty:
            recent_atr = valid_atr['ATR'].tail(3).round(2).tolist()
            debug_info.append(f"ATR calculated successfully. Recent values: {recent_atr}")
        else:
            debug_info.append("⚠️ No valid ATR values calculated")
            return pd.DataFrame(), debug_info
        
        # Load intraday data
        if intraday_file is None:
            debug_info.append("⚠️ No intraday data provided - analysis cannot proceed")
            return pd.DataFrame(), debug_info
        
        intraday = load_intraday_data(intraday_file)
        
        if intraday is None:
            debug_info.append("❌ Failed to load intraday data")
            return pd.DataFrame(), debug_info
        
        debug_info.append(f"Intraday data loaded: {intraday.shape}")
        debug_info.append(f"Intraday date range: {intraday['Date'].min()} to {intraday['Date'].max()}")
        
        # Check for session column
        if 'Session' in intraday.columns:
            unique_sessions = intraday['Session'].unique()
            debug_info.append(f"Session types found: {list(unique_sessions)}")
        
        # Run analysis
        debug_info.append("🎯 Running trigger and goal detection...")
        df = detect_triggers_and_goals_flexible(daily, intraday, asset_config, custom_ratios, session_filter)
        debug_info.append(f"✅ Detection complete: {len(df)} trigger-goal combinations found")
        
        # Additional statistics
        if not df.empty:
            above_triggers = len(df[df['Direction'] == 'Above'])
            below_triggers = len(df[df['Direction'] == 'Below'])
            debug_info.append(f"✅ Above triggers: {above_triggers}, Below triggers: {below_triggers}")
            
            goals_hit = len(df[df['GoalHit'] == 'Yes'])
            hit_rate = goals_hit / len(df) * 100 if len(df) > 0 else 0
            debug_info.append(f"✅ Goals hit: {goals_hit}/{len(df)} ({hit_rate:.1f}%)")
            
             # Session analysis if available
            if session_filter:
                debug_info.append(f"✅ Session filter applied: {session_filter}")
        
        return df, debug_info
        
    except Exception as e:
        debug_info.append(f"❌ Error: {str(e)}")
        import traceback
        debug_info.append(f"Traceback: {traceback.format_exc()}")
        return pd.DataFrame(), debug_info

# Streamlit Interface
st.title('🎯 Flexible ATR Generator - File Input Support')
st.write('**Upload your own data files - No Yahoo Finance limitations!**')

# Data source selection
st.sidebar.header("📁 Data Input")
data_source = st.sidebar.radio(
    "Data Source",
    options=["Upload Files", "Yahoo Finance (Limited)"],
    index=0,
    help="Upload files for full historical data, or use Yahoo Finance for recent data only"
)

if data_source == "Upload Files":
    st.sidebar.subheader("📊 File Uploads")
    
    # Daily data upload
    daily_file = st.sidebar.file_uploader(
        "Daily OHLC Data",
        type=['csv', 'xlsx', 'xls'],
        help="Upload daily OHLC data (CSV or Excel)"
    )
    
    # Intraday data upload  
    intraday_file = st.sidebar.file_uploader(
        "Intraday OHLC Data",
        type=['csv', 'xlsx', 'xls'],
        help="Upload intraday OHLC data (CSV or Excel)"
    )
    
    st.sidebar.info("""
    📋 **Required Columns:**
    - **Daily**: Date, Open, High, Low, Close
    - **Intraday**: Datetime (or Date+Time), Open, High, Low, Close
    - **Optional**: Volume, Session (PM/R/AH)
    """)
    
    # Asset configuration
    st.sidebar.subheader("🏷️ Asset Configuration")
    asset_type = st.sidebar.selectbox(
        "Asset Class",
        options=['STOCKS', 'STOCKS_24H', 'CRYPTO', 'FOREX', 'FUTURES', 'COMMODITIES'],
        help="Select asset type for appropriate market handling"
    )
    
    # Extended hours for stocks
    extended_hours = False
    if asset_type == 'STOCKS':
        extended_hours = st.sidebar.checkbox(
            "Include Extended Hours",
            value=False,
            help="Include pre-market (4AM) and after-hours (8PM) data"
        )
        
        # Update asset type based on extended hours
        if extended_hours:
            config = AssetConfig.get_config('STOCKS', extended_hours=True)
        else:
            config = AssetConfig.get_config('STOCKS', extended_hours=False)
    else:
        config = AssetConfig.get_config(asset_type)
    
    # Session filtering
    if 'Session' in config['session_types'] and len(config['session_types']) > 1:
        session_filter = st.sidebar.multiselect(
            "Filter by Sessions",
            options=config['session_types'],
            default=config['default_session'],
            help="Select trading sessions to include in analysis"
        )
    else:
        session_filter = None
    
    ticker = st.sidebar.text_input(
        "Ticker Symbol (for labeling)",
        value="",
        help="Optional: Enter ticker symbol for output labeling"
    )

else:  # Yahoo Finance
    st.sidebar.subheader("📈 Yahoo Finance")
    st.sidebar.warning("⚠️ Limited to ~60 days of intraday data")
    
    asset_type = st.sidebar.selectbox(
        "Asset Class",
        options=['STOCKS', 'CRYPTO', 'FOREX', 'FUTURES', 'COMMODITIES']
    )
    
    config = AssetConfig.get_config(asset_type)
    
    ticker = st.sidebar.text_input(
        "Ticker Symbol",
        value=config['example_tickers'][0],
        help=f"Enter ticker (examples: {', '.join(config['example_tickers'])})"
    ).upper()
    
    # Date range
    col1, col2 = st.sidebar.columns(2)
    with col1:
        start_date = st.date_input(
            "Start Date",
            value=datetime.now().date() - timedelta(days=365),
            help="Start date for analysis"
        )
    with col2:
        end_date = st.date_input(
            "End Date",
            value=datetime.now().date(),
            help="End date for analysis"
        )
    
    daily_file = None
    intraday_file = None
    session_filter = None
    extended_hours = False

# Show asset configuration
st.sidebar.info(f"""
📋 **Data Requirements:**
- **Daily**: Date, Open, High, Low, Close (Yahoo or file)
- **Intraday**: Datetime, Open, High, Low, Close (file upload)
- **Optional**: Volume, Session (PM/R/AH)
""")

# Asset configuration
st.sidebar.subheader("🏷️ Asset Configuration")
asset_type = st.sidebar.selectbox(
    "Asset Class",
    options=['STOCKS', 'STOCKS_24H', 'CRYPTO', 'FOREX', 'FUTURES', 'COMMODITIES'],
    help="Select asset type for appropriate market handling"
)

# Extended hours for stocks
extended_hours = False
if asset_type == 'STOCKS':
    extended_hours = st.sidebar.checkbox(
        "Include Extended Hours",
        value=False,
        help="Include pre-market (4AM) and after-hours (8PM) data"
    )
    
    # Update asset type based on extended hours
    if extended_hours:
        config = AssetConfig.get_config('STOCKS', extended_hours=True)
    else:
        config = AssetConfig.get_config('STOCKS', extended_hours=False)
else:
    config = AssetConfig.get_config(asset_type)

# Session filtering
if len(config['session_types']) > 1:
    session_filter = st.sidebar.multiselect(
        "Filter by Sessions",
        options=config['session_types'],
        default=config['default_session'],
        help="Select trading sessions to include in analysis"
    )
else:
    session_filter = None

# Advanced settings
with st.sidebar.expander("⚙️ Advanced Settings"):
    atr_period = st.number_input("ATR Period", min_value=1, max_value=50, value=14)
    
    # Custom ratio input
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

# Generate button
if st.button('🚀 Generate ATR Analysis'):
    if data_source == "Upload Both Files":
        if daily_file is None or intraday_file is None:
            st.error("Please upload both daily and intraday data files")
        else:
            with st.spinner(f'Analyzing uploaded data ({config["description"]})...'):
                try:
                    result_df, debug_messages = main_flexible_hybrid(
                        ticker=ticker or "UPLOADED_DATA",
                        asset_type=asset_type,
                        daily_file=daily_file,
                        intraday_file=intraday_file,
                        atr_period=atr_period,
                        custom_ratios=custom_ratios,
                        session_filter=session_filter,
                        extended_hours=extended_hours,
                        data_source=data_source
                    )
                    
                    display_results(result_df, debug_messages, ticker or "UPLOADED_DATA", asset_type, "Both Files Uploaded")
                        
                except Exception as e:
                    st.error(f'❌ Error: {e}')
    
    elif data_source == "Yahoo Daily + Upload Intraday":
        if not ticker:
            st.error("Please enter a ticker symbol")
        elif intraday_file is None:
            st.error("Please upload intraday data file")
        elif start_date >= end_date:
            st.error("Start date must be before end date")
        else:
            with st.spinner(f'Analyzing {ticker} with hybrid data sources...'):
                try:
                    result_df, debug_messages = main_flexible_hybrid(
                        ticker=ticker,
                        asset_type=asset_type,
                        daily_file=None,  # Will fetch from Yahoo
                        intraday_file=intraday_file,
                        start_date=start_date,
                        end_date=end_date,
                        atr_period=atr_period,
                        custom_ratios=custom_ratios,
                        session_filter=session_filter,
                        extended_hours=extended_hours,
                        data_source=data_source
                    )
                    
                    display_results(result_df, debug_messages, ticker, asset_type, "Yahoo Daily + Uploaded Intraday")
                        
                except Exception as e:
                    st.error(f'❌ Error: {e}')
    
    elif data_source == "Yahoo Finance Only (Limited)":
        if not ticker:
            st.error("Please enter a ticker symbol")
        elif start_date >= end_date:
            st.error("Start date must be before end date")
        else:
            st.warning("⚠️ Using Yahoo Finance only - limited to ~60 days of intraday data")
            with st.spinner(f'Fetching all data from Yahoo Finance for {ticker}...'):
                try:
                    result_df, debug_messages = main_flexible_hybrid(
                        ticker=ticker,
                        asset_type=asset_type,
                        daily_file=None,
                        intraday_file=None,
                        start_date=start_date,
                        end_date=end_date,
                        atr_period=atr_period,
                        custom_ratios=custom_ratios,
                        session_filter=session_filter,
                        extended_hours=extended_hours,
                        data_source=data_source
                    )
                    
                    display_results(result_df, debug_messages, ticker, asset_type, "Yahoo Finance Only")
                        
                except Exception as e:
                    st.error(f'❌ Error: {e}')

def display_results(result_df, debug_messages, ticker, asset_type, data_source_label):
    """Helper function to display analysis results"""
    # Show debug info
    with st.expander('📋 Processing Information'):
        for msg in debug_messages:
            st.write(msg)
    
    if not result_df.empty:
        result_df['Ticker'] = ticker
        result_df['AssetType'] = asset_type
        result_df['DataSource'] = data_source_label
        
        # Show summary stats
        st.subheader('📊 Summary Statistics')
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric('Total Records', len(result_df))
        with col2:
            st.metric('Unique Dates', result_df['Date'].nunique())
        with col3:
            goals_hit = len(result_df[result_df['GoalHit'] == 'Yes'])
            st.metric('Goals Hit', goals_hit)
        with col4:
            hit_rate = goals_hit / len(result_df) * 100 if len(result_df) > 0 else 0
            st.metric('Hit Rate', f'{hit_rate:.1f}%')
        
        # Show data preview
        st.subheader('📋 Results Preview')
        st.dataframe(result_df.head(10))
        
        # Download button
        ticker_clean = ticker.replace("^", "").replace("=", "_")
        output_filename = f'{ticker_clean}_{asset_type}_ATR_analysis.csv'
        st.download_button(
            '⬇️ Download Results CSV',
            data=result_df.to_csv(index=False),
            file_name=output_filename,
            mime='text/csv'
        )
        
        st.success(f'🎉 Analysis complete for {ticker}!')
        
    else:
        st.warning('⚠️ No results generated - check processing information above')

# Help section
st.markdown("""
---
### 📁 File Upload Benefits

**🚀 No Data Limitations**
- Analyze **years** of historical data
- **Any timeframe** - 1-minute to daily
- **Complete control** over data quality

**📊 Supported Formats**
- **CSV files** - most common format
- **Excel files** (.xlsx, .xls) - automatic header detection
- **Flexible columns** - automatic standardization

**🕐 Extended Hours Support for Stocks**
- **Regular Hours**: 9:30 AM - 4:00 PM ET
- **Extended Hours**: 4:00 AM - 8:00 PM ET  
- **24-Hour**: Full day (if data available)

**🔧 Session Filtering**
- **PM** - Pre-market (4:00 AM - 9:30 AM)
- **R** - Regular hours (9:30 AM - 4:00 PM)  
- **AH** - After-hours (4:00 PM - 8:00 PM)

### 🎯 Asset-Specific Features

**STOCKS** - Traditional equity analysis
- OPEN gap handling for traditional market hours
- Extended hours support (4AM-8PM)
- Session filtering (PM/R/AH)

**CRYPTO** - 24/7 cryptocurrency 
- No OPEN gaps (continuous trading)
- Weekends included
- Perfect for BTC, ETH analysis

**FOREX** - Foreign exchange
- Near 24/5 trading
- Sunday 5PM - Friday 5PM
- Multiple session support

**FUTURES** - Futures contracts
- Extended hours (Sunday 6PM - Friday 5PM)
- Electronic vs Regular Trading Hours
- ES, NQ, commodity futures

### ✅ Key Improvements
- **File uploads** instead of Yahoo Finance limitations
- **True Wilder's ATR** - matches Excel calculations
- **Flexible market hours** - adapts to asset type
- **Session filtering** - analyze specific trading sessions
- **Custom ratios** - define your own levels
- **Smart column detection** - handles various data formats

### 📋 Required File Formats

**Daily Data File:**
```
Date,Open,High,Low,Close,Volume
2023-01-03,100.50,101.25,99.75,100.80,1000000
2023-01-04,100.80,102.00,100.50,101.50,1200000
...
```

**Intraday Data File:**
```
Datetime,Open,High,Low,Close,Volume,Session
2023-01-03 09:30:00,100.50,100.75,100.25,100.60,50000,R
2023-01-03 09:40:00,100.60,100.90,100.40,100.80,45000,R
...
```

**Session Codes (Optional):**
- **PM** = Pre-market (4:00-9:30 AM)
- **R** = Regular hours (9:30 AM-4:00 PM)
- **AH** = After-hours (4:00-8:00 PM)
""")
m', '15m', '30m', '1h'],
        index=3,
        help="Interval for intraday data (limited to ~60 days)"
    )
    
    # Custom ratio input
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

# Generate button
if st.button('🚀 Generate ATR Analysis'):
    if not ticker:
        st.error("Please enter a ticker symbol")
    elif start_date >= end_date:
        st.error("Start date must be before end date")
    else:
        with st.spinner(f'Analyzing {ticker} ({config["description"]})...'):
            try:
                result_df, debug_messages = main_flexible(
                    ticker=ticker,
                    asset_type=asset_type,
                    start_date=start_date,
                    end_date=end_date,
                    atr_period=atr_period,
                    daily_interval=daily_interval,
                    intraday_interval=intraday_interval,
                    custom_ratios=custom_ratios
                )
                
                # Show debug info
                with st.expander('📋 Debug Information'):
                    for msg in debug_messages:
                        st.write(msg)
                
                if not result_df.empty:
                    result_df['Ticker'] = ticker
                    result_df['AssetType'] = asset_type
                    
                    # Show summary stats
                    st.subheader('📊 Summary Statistics')
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric('Total Records', len(result_df))
                    with col2:
                        st.metric('Unique Dates', result_df['Date'].nunique())
                    with col3:
                        goals_hit = len(result_df[result_df['GoalHit'] == 'Yes'])
                        st.metric('Goals Hit', goals_hit)
                    with col4:
                        hit_rate = goals_hit / len(result_df) * 100 if len(result_df) > 0 else 0
                        st.metric('Hit Rate', f'{hit_rate:.1f}%')
                    
                    # Show data preview
                    st.subheader('📋 Results Preview')
                    st.dataframe(result_df.head(10))
                    
                    # Download button
                    output_filename = f'{ticker}_{asset_type}_ATR_analysis_{start_date}_to_{end_date}.csv'
                    st.download_button(
                        '⬇️ Download Results CSV',
                        data=result_df.to_csv(index=False),
                        file_name=output_filename,
                        mime='text/csv'
                    )
                    
                    st.success(f'🎉 Analysis complete for {ticker}!')
                    
                else:
                    st.warning('⚠️ No results generated - check debug info above')
                    
            except Exception as e:
                st.error(f'❌ Error: {e}')

# Help section
st.markdown("""
---
### 🔧 Asset Class Features

**STOCKS** - Traditional equity markets
- Market hours: 9:30 AM - 4:00 PM ET
- Special OPEN candle handling
- Pre-market/Regular/After-hours sessions

**CRYPTO** - 24/7 cryptocurrency trading
- No market hours restrictions
- No special OPEN handling (continuous trading)
- Works with BTC, ETH, and other crypto pairs

**FOREX** - Foreign exchange markets
- Sunday 5 PM - Friday 5 PM ET
- Near 24-hour trading during weekdays
- Multiple session support (Asia/Europe/US)

**FUTURES** - Futures contracts
- Extended hours (Sunday 6 PM - Friday 5 PM ET)
- Electronic (Globex) and Regular Trading Hours
- Suitable for ES, NQ, commodities futures

**COMMODITIES** - Physical commodities
- Traditional market hours (varies by commodity)
- Special handling for different commodity types
- Works with gold, silver, oil futures

### 📊 Key Improvements
- ✅ **Multi-asset support** - No more hardcoded SPX files
- ✅ **Flexible market hours** - Adapts to different trading sessions  
- ✅ **24/7 market support** - Handles crypto and forex properly
- ✅ **Custom ratios** - Define your own Fibonacci levels
- ✅ **Yahoo Finance integration** - Fetch any ticker automatically
- ✅ **True Wilder's ATR** - Accurate ATR calculation maintained
""")
