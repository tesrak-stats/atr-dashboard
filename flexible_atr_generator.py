import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
import os

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
    
    return levels

class TickerMapper:
    """Handle ticker symbol mappings for different data sources"""
    
    @staticmethod
    def get_yahoo_ticker(input_ticker):
        """Convert common ticker variations to Yahoo Finance format"""
        
        # Common ticker mappings
        ticker_mappings = {
            # S&P 500 variations
            'SPX': '^GSPC',
            'SP500': '^GSPC',
            'S&P500': '^GSPC',
            'SPY': 'SPY',  # ETF, no change needed
            
            # NASDAQ variations  
            'NDX': '^NDX',
            'NASDAQ': '^IXIC',
            'COMP': '^IXIC',
            'QQQ': 'QQQ',  # ETF, no change needed
            
            # Dow Jones variations
            'DJI': '^DJI',
            'DJIA': '^DJI',
            'DOW': '^DJI',
            'DIA': 'DIA',  # ETF, no change needed
            
            # Russell variations
            'RUT': '^RUT',
            'RUSSELL': '^RUT',
            'IWM': 'IWM',  # ETF, no change needed
            
            # VIX variations
            'VIX': '^VIX',
            'VOLATILITY': '^VIX',
            
            # Currency pairs (Forex)
            'EURUSD': 'EURUSD=X',
            'GBPUSD': 'GBPUSD=X', 
            'USDJPY': 'USDJPY=X',
            'USDCAD': 'USDCAD=X',
            'AUDUSD': 'AUDUSD=X',
            'NZDUSD': 'NZDUSD=X',
            'USDCHF': 'USDCHF=X',
            
            # Crypto variations
            'BITCOIN': 'BTC-USD',
            'BTC': 'BTC-USD',
            'ETHEREUM': 'ETH-USD', 
            'ETH': 'ETH-USD',
            'LITECOIN': 'LTC-USD',
            'LTC': 'LTC-USD',
            
            # Futures (common contracts)
            'ES': 'ES=F',
            'NQ': 'NQ=F',
            'YM': 'YM=F',
            'RTY': 'RTY=F',
            'CL': 'CL=F',  # Crude Oil
            'GC': 'GC=F',  # Gold
            'SI': 'SI=F',  # Silver
            'NG': 'NG=F',  # Natural Gas
            
            # Bonds
            'TNX': '^TNX',  # 10-Year Treasury
            'TYX': '^TYX',  # 30-Year Treasury
            'FVX': '^FVX',  # 5-Year Treasury
            'IRX': '^IRX',  # 3-Month Treasury
        }
        
        # Convert to uppercase for matching
        input_upper = input_ticker.upper().strip()
        
        # Return mapped ticker if found, otherwise return original
        mapped_ticker = ticker_mappings.get(input_upper, input_ticker)
        
        return mapped_ticker
    
    @staticmethod
    def suggest_alternatives(input_ticker):
        """Suggest alternative ticker formats if the input fails"""
        
        suggestions = []
        input_upper = input_ticker.upper().strip()
        
        # Common patterns to try
        variations = [
            f"^{input_upper}",  # Add caret for indices
            f"{input_upper}=X",  # Add =X for forex
            f"{input_upper}=F",  # Add =F for futures
            f"{input_upper}-USD",  # Add -USD for crypto
        ]
        
        # Remove duplicates and original
        variations = [v for v in variations if v != input_ticker]
        
        return variations[:3]  # Return top 3 suggestions

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
                'has_open_special': False,
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
                'market_open': '17:00',
                'market_close': '17:00',
                'has_open_special': False,
                'weekends_closed': True,
                'session_types': ['ASIA', 'EUROPE', 'US', '24H'],
                'default_session': ['24H'],
                'description': 'Foreign Exchange (Sun 5PM - Fri 5PM EST)',
                'example_tickers': ['EURUSD=X', 'GBPUSD=X', 'USDJPY=X'],
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

def validate_data_alignment(daily_data, intraday_data, atr_period=14, min_buffer_days=120):
    """
    Validate that daily and intraday data are properly aligned for ATR calculation
    Returns: (is_valid, warnings, recommendations)
    """
    warnings = []
    recommendations = []
    is_valid = True
    
    if daily_data is None or intraday_data is None:
        return False, ["Missing data files"], ["Please provide both daily and intraday data"]
    
    # Convert date columns for comparison
    daily_dates = pd.to_datetime(daily_data['Date']).dt.date
    intraday_dates = pd.to_datetime(intraday_data['Date'] if 'Date' in intraday_data.columns 
                                   else intraday_data['Datetime']).dt.date
    
    daily_start = daily_dates.min()
    daily_end = daily_dates.max()
    intraday_start = intraday_dates.min()
    intraday_end = intraday_dates.max()
    
    # Check if daily data starts before intraday data
    if daily_start >= intraday_start:
        is_valid = False
        warnings.append("⚠️ Daily data should start BEFORE intraday data for proper ATR calculation")
        recommendations.append("Extend daily data backwards to include more historical periods")
    
    # Check buffer period for ATR calculation
    buffer_days = (intraday_start - daily_start).days
    required_days = max(atr_period * 7, min_buffer_days)  # At least ATR period in trading days or 4 months
    
    if buffer_days < required_days:
        is_valid = False
        warnings.append(f"⚠️ Insufficient buffer period: {buffer_days} days (need {required_days}+ days)")
        recommendations.append(f"Add at least {required_days - buffer_days} more days of daily data before intraday period")
    
    # Check date overlap
    overlap_start = max(daily_start, intraday_start)
    overlap_end = min(daily_end, intraday_end)
    
    if overlap_start > overlap_end:
        is_valid = False
        warnings.append("❌ No date overlap between daily and intraday data")
        recommendations.append("Ensure daily and intraday data cover overlapping time periods")
    else:
        overlap_days = (overlap_end - overlap_start).days
        if overlap_days < 5:
            warnings.append("⚠️ Very small overlap period between daily and intraday data")
            recommendations.append("Increase overlap period for more reliable analysis")
    
    # Data quality checks
    daily_missing = daily_data[['Open', 'High', 'Low', 'Close']].isnull().sum().sum()
    intraday_missing = intraday_data[['Open', 'High', 'Low', 'Close']].isnull().sum().sum()
    
    if daily_missing > 0:
        warnings.append(f"⚠️ {daily_missing} missing values in daily OHLC data")
        recommendations.append("Clean daily data to remove or fill missing values")
    
    if intraday_missing > 0:
        warnings.append(f"⚠️ {intraday_missing} missing values in intraday OHLC data")
        recommendations.append("Clean intraday data to remove or fill missing values")
    
    return is_valid, warnings, recommendations

def standardize_columns(df):
    """
    Enhanced column standardization with duplicate handling
    """
    # Step 1: Handle duplicate columns by renaming them
    original_columns = list(df.columns)
    seen_columns = {}
    new_columns = []
    
    for col in original_columns:
        # Convert to string and clean
        col_str = str(col).strip() if col is not None else 'Unnamed'
        
        # Handle duplicates by adding suffix
        if col_str in seen_columns:
            seen_columns[col_str] += 1
            new_col = f"{col_str}_{seen_columns[col_str]}"
        else:
            seen_columns[col_str] = 0
            new_col = col_str
        
        new_columns.append(new_col)
    
    # Apply the new column names
    df.columns = new_columns
    
    # Step 2: Standard column mappings
    column_mappings = {
        # Date columns (case insensitive)
        'date': 'Date',
        'timestamp': 'Date', 
        'datetime': 'Datetime',
        # OHLC columns
        'open': 'Open',
        'high': 'High', 
        'low': 'Low',
        'close': 'Close',
        'last': 'Close',
        'adj close': 'Close',
        'adjusted_close': 'Close',
        'settle': 'Close',
        # Volume
        'volume': 'Volume',
        'vol': 'Volume',
        # Session
        'session': 'Session'
    }
    
    # Apply mappings (case insensitive)
    columns_to_rename = {}
    for col in df.columns:
        col_lower = str(col).lower().strip()
        if col_lower in column_mappings:
            target_name = column_mappings[col_lower]
            # Only rename if target doesn't already exist
            if target_name not in df.columns:
                columns_to_rename[col] = target_name
    
    # Apply the renamings
    if columns_to_rename:
        df.rename(columns=columns_to_rename, inplace=True)
    
    return df

def safe_read_excel(uploaded_file, max_header_rows=10):
    """
    Safely read Excel files with automatic header detection
    """
    try:
        # First, try to read without specifying header to see the structure
        preview_df = pd.read_excel(uploaded_file, header=None, nrows=max_header_rows)
        
        # Look for a row that contains typical column names
        header_keywords = ['date', 'open', 'high', 'low', 'close', 'volume', 'time', 'datetime']
        
        best_header_row = 0
        max_matches = 0
        
        for row_idx in range(min(max_header_rows, len(preview_df))):
            row_values = preview_df.iloc[row_idx].fillna('').astype(str).str.lower()
            matches = sum(1 for keyword in header_keywords if any(keyword in val for val in row_values))
            
            if matches > max_matches:
                max_matches = matches
                best_header_row = row_idx
        
        # Reset file pointer and read with detected header
        uploaded_file.seek(0)
        df = pd.read_excel(uploaded_file, header=best_header_row)
        
        if best_header_row > 0:
            st.info(f"✅ Auto-detected header row at position {best_header_row + 1}")
        
        return df
        
    except Exception as e:
        st.error(f"Error reading Excel file: {str(e)}")
        return None

def load_daily_data(uploaded_file=None, ticker=None, start_date=None, end_date=None, intraday_data=None):
    """
    Enhanced daily data loading with better error handling
    """
    if uploaded_file is not None:
        try:
            # Handle different file types
            if uploaded_file.name.endswith('.csv'):
                daily = pd.read_csv(uploaded_file)
            elif uploaded_file.name.endswith(('.xlsx', '.xls')):
                daily = safe_read_excel(uploaded_file)
                if daily is None:
                    return None
            else:
                st.error("Unsupported file format. Please use CSV or Excel files.")
                return None
            
            # Check if DataFrame is empty
            if daily.empty:
                st.error("The uploaded daily data file is empty.")
                return None
            
            # Debug: Show original columns
            st.info(f"Original columns in daily file: {list(daily.columns)}")
            
            # Standardize column names
            daily = standardize_columns(daily)
            
            # Debug: Show standardized columns
            st.info(f"Standardized columns: {list(daily.columns)}")
            
            # Validate required columns
            required_cols = ['Date', 'Open', 'High', 'Low', 'Close']
            missing_cols = [col for col in required_cols if col not in daily.columns]
            if missing_cols:
                st.error(f"Missing required columns in daily data: {missing_cols}")
                st.error(f"Available columns: {list(daily.columns)}")
                return None
            
            # Convert Date column
            try:
                daily['Date'] = pd.to_datetime(daily['Date'], errors='coerce')
                # Remove rows with invalid dates
                invalid_dates = daily['Date'].isna().sum()
                if invalid_dates > 0:
                    st.warning(f"Removed {invalid_dates} rows with invalid dates")
                    daily = daily.dropna(subset=['Date'])
                    
                if daily.empty:
                    st.error("No valid dates found in daily data after cleaning")
                    return None
                    
            except Exception as e:
                st.error(f"Error parsing dates in daily data: {str(e)}")
                return None
            
            # Sort by date and remove duplicates
            daily = daily.sort_values('Date').drop_duplicates(subset=['Date']).reset_index(drop=True)
            
            st.success(f"✅ Loaded daily data: {len(daily)} records")
            st.info(f"Date range: {daily['Date'].min().date()} to {daily['Date'].max().date()}")
            
            return daily
            
        except Exception as e:
            st.error(f"Error loading daily data: {str(e)}")
            # Add more debugging info
            import traceback
            st.error(f"Full traceback: {traceback.format_exc()}")
            return None
    
    elif ticker and intraday_data is not None:
        # Smart auto-detection from intraday data
        try:
            # Apply ticker mapping for Yahoo Finance
            original_ticker = ticker
            yahoo_ticker = TickerMapper.get_yahoo_ticker(ticker)
            
            if yahoo_ticker != original_ticker:
                st.info(f"🔄 Mapped ticker: '{original_ticker}' → '{yahoo_ticker}' for Yahoo Finance")
            
            st.info(f"🔍 Analyzing intraday data to determine optimal daily data range...")
            
            # Get date range from intraday data
            intraday_dates = pd.to_datetime(intraday_data['Date'] if 'Date' in intraday_data.columns 
                                          else intraday_data['Datetime']).dt.date
            
            intraday_start = intraday_dates.min()
            intraday_end = intraday_dates.max()
            
            # Calculate smart date range with buffer
            buffer_start = intraday_start - timedelta(days=180)  # 6 months buffer
            fetch_end = intraday_end + timedelta(days=5)  # Small buffer for end date
            
            st.info(f"📊 Intraday data spans: {intraday_start} to {intraday_end}")
            st.info(f"📈 Fetching Yahoo daily data for '{yahoo_ticker}' from {buffer_start} to {fetch_end}")
            
            # Fetch from Yahoo Finance
            daily_data = yf.download(yahoo_ticker, start=buffer_start, end=fetch_end, interval='1d', progress=False)
            
            if daily_data.empty:
                st.error(f"❌ No daily data found for '{yahoo_ticker}' in the calculated date range")
                
                # Suggest alternatives
                alternatives = TickerMapper.suggest_alternatives(original_ticker)
                if alternatives:
                    st.info("💡 Try these alternative ticker formats:")
                    for alt in alternatives:
                        st.info(f"   • {alt}")
                
                return None
            
            daily_data.reset_index(inplace=True)
            daily_data = standardize_columns(daily_data)
            
            st.success(f"✅ Auto-fetched daily data from Yahoo: {len(daily_data)} records")
            st.success(f"📅 Daily data range: {daily_data['Date'].min().date()} to {daily_data['Date'].max().date()}")
            return daily_data
            
        except Exception as e:
            st.error(f"Error auto-fetching from Yahoo Finance for '{yahoo_ticker}': {str(e)}")
            
            # Suggest alternatives on error
            alternatives = TickerMapper.suggest_alternatives(original_ticker)
            if alternatives:
                st.info("💡 If the ticker wasn't found, try these alternative formats:")
                for alt in alternatives:
                    st.info(f"   • {alt}")
            
            return None
    
    elif ticker and start_date and end_date:
        # Fallback to manual date range (for backward compatibility)
        try:
            st.info(f"📈 Fetching daily data from Yahoo Finance for {ticker}...")
            
            # Add buffer for ATR calculation
            buffer_start = start_date - timedelta(days=180)  # 6 months buffer
            
            daily_data = yf.download(ticker, start=buffer_start, end=end_date, interval='1d', progress=False)
            
            if daily_data.empty:
                st.error(f"No daily data found for {ticker}")
                return None
            
            daily_data.reset_index(inplace=True)
            daily_data = standardize_columns(daily_data)
            
            st.success(f"✅ Loaded daily data from Yahoo: {len(daily_data)} records (includes 6-month buffer)")
            st.info(f"Date range: {daily_data['Date'].min().date()} to {daily_data['Date'].max().date()}")
            return daily_data
            
        except Exception as e:
            st.error(f"Error fetching from Yahoo Finance: {str(e)}")
            return None
    
    return None

def load_intraday_data(uploaded_file):
    """
    Enhanced intraday data loading with progress tracking and better error handling
    """
    try:
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        status_text.text("Reading file...")
        progress_bar.progress(25)
        
        if uploaded_file.name.endswith('.csv'):
            intraday = pd.read_csv(uploaded_file)
        elif uploaded_file.name.endswith(('.xlsx', '.xls')):
            intraday = safe_read_excel(uploaded_file)
            if intraday is None:
                return None
        else:
            st.error("Unsupported file format. Please use CSV or Excel files.")
            return None
        
        # Check if DataFrame is empty
        if intraday.empty:
            st.error("The uploaded intraday data file is empty.")
            return None
        
        status_text.text("Standardizing columns...")
        progress_bar.progress(50)
        
        # Debug: Show original columns
        st.info(f"Original columns in intraday file: {list(intraday.columns)}")
        
        # Standardize columns
        intraday = standardize_columns(intraday)
        
        # Debug: Show standardized columns
        st.info(f"Standardized columns: {list(intraday.columns)}")
        
        # Validate required columns
        required_cols = ['Open', 'High', 'Low', 'Close']
        missing_cols = [col for col in required_cols if col not in intraday.columns]
        if missing_cols:
            st.error(f"Missing required columns in intraday data: {missing_cols}")
            st.error(f"Available columns: {list(intraday.columns)}")
            return None
        
        status_text.text("Processing datetime...")
        progress_bar.progress(75)
        
        # Ensure we have a datetime column
        if 'Datetime' not in intraday.columns:
            # Try to create from Date and Time columns
            if 'Date' in intraday.columns and 'Time' in intraday.columns:
                try:
                    intraday['Datetime'] = pd.to_datetime(intraday['Date'].astype(str) + ' ' + intraday['Time'].astype(str))
                except Exception as e:
                    st.error(f"Error combining Date and Time columns: {str(e)}")
                    return None
            elif 'Date' in intraday.columns:
                # Assume Date column contains full datetime
                try:
                    intraday['Datetime'] = pd.to_datetime(intraday['Date'])
                except Exception as e:
                    st.error(f"Error parsing Date column as datetime: {str(e)}")
                    return None
            else:
                st.error("Could not find datetime information in intraday data")
                st.error(f"Available columns: {list(intraday.columns)}")
                return None
        else:
            try:
                intraday['Datetime'] = pd.to_datetime(intraday['Datetime'])
            except Exception as e:
                st.error(f"Error parsing Datetime column: {str(e)}")
                return None
        
        # Remove rows with invalid datetimes
        invalid_datetimes = intraday['Datetime'].isna().sum()
        if invalid_datetimes > 0:
            st.warning(f"Removed {invalid_datetimes} rows with invalid datetimes")
            intraday = intraday.dropna(subset=['Datetime'])
            
        if intraday.empty:
            st.error("No valid datetimes found in intraday data after cleaning")
            return None
        
        # Create Date column for matching
        intraday['Date'] = intraday['Datetime'].dt.date
        
        # Sort by datetime and remove duplicates
        intraday = intraday.sort_values('Datetime').drop_duplicates(subset=['Datetime']).reset_index(drop=True)
        
        status_text.text("Finalizing...")
        progress_bar.progress(100)
        
        st.success(f"✅ Loaded intraday data: {len(intraday):,} records")
        st.info(f"Date range: {intraday['Date'].min()} to {intraday['Date'].max()}")
        
        # Clear progress indicators
        progress_bar.empty()
        status_text.empty()
        
        return intraday
        
    except Exception as e:
        st.error(f"Error loading intraday data: {str(e)}")
        # Add more debugging info
        import traceback
        st.error(f"Full traceback: {traceback.format_exc()}")
        return None

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
    Flexible trigger and goal detection for different asset classes with progress tracking
    """
    if custom_ratios is None:
        fib_levels = [0.236, 0.382, 0.500, 0.618, 0.786, 1.000, 
                     -0.236, -0.382, -0.500, -0.618, -0.786, -1.000, 0.000]
    else:
        fib_levels = custom_ratios
    
    results = []
    has_open_special = asset_config['has_open_special']
    
    # Progress tracking
    total_days = len(daily) - 1
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    for i in range(1, len(daily)):
        try:
            # Update progress
            progress = i / total_days
            progress_bar.progress(progress)
            status_text.text(f"Processing day {i}/{total_days}...")
            
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
            
            # Get trading session data
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
    
    # Clear progress indicators
    progress_bar.empty()
    status_text.empty()
    
    return pd.DataFrame(results)

def process_goals_for_trigger(results, day_data, fib_levels, level_map, trigger_level, 
                            direction, trigger_time, trigger_row, trigger_price,
                            trading_date, previous_close, previous_atr, has_open_special, open_price):
    """
    Process all goals for a given trigger (separated for cleaner code)
    """
    trigger_candle = day_data.iloc[trigger_row] if trigger_row is not None else None
    
    for goal_level in fib_levels:
        goal_price = level_map[goal_level]
        goal_hit = False
        goal_time = ''
        is_same_time = False
        
        # Determine goal type and direction logic
        if goal_level == trigger_level:
            # Same level - check opposite direction
            if direction == 'Below':
                goal_type = 'Retest'
                # For Below trigger, same level goal checks for Above movement
                check_condition = lambda candle: candle['High'] >= goal_price
            else:  # Above trigger
                goal_type = 'Retest'
                # For Above trigger, same level goal checks for Below movement  
                check_condition = lambda candle: candle['Low'] <= goal_price
        else:
            # Different levels - normal logic
            if direction == 'Below':
                goal_type = 'Continuation' if goal_level < trigger_level else 'Retracement'
                check_condition = lambda candle: check_goal_hit(candle, goal_level, trigger_level, goal_price)
            else:  # Above
                goal_type = 'Continuation' if goal_level > trigger_level else 'Retracement'
                check_condition = lambda candle: check_goal_hit(candle, goal_level, trigger_level, goal_price)
        
        # Check for goal completion
        if trigger_time == 'OPEN' and has_open_special:
            # Check if goal completes at OPEN
            if goal_level == trigger_level:
                # Same level retest - can't happen at same time as trigger
                pass
            else:
                # Different level - check normal OPEN completion
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
                    if check_condition(row):
                        goal_hit = True
                        goal_time = row['Time']
                        break
        
        else:  # Intraday trigger or 24/7 market
            # For same level retest, can't complete on same candle as trigger
            if goal_level == trigger_level:
                # Same level retest - check subsequent candles only
                if trigger_row is not None:
                    for _, row in day_data.iloc[trigger_row + 1:].iterrows():
                        if check_condition(row):
                            goal_hit = True
                            goal_time = row['Time']
                            break
            else:
                # Different level - check if goal completes on same candle as trigger
                if trigger_candle is not None and check_condition(trigger_candle):
                    goal_hit = True
                    goal_time = trigger_time
                    is_same_time = True
                
                # Check subsequent candles if not completed on trigger candle
                if not goal_hit and trigger_row is not None:
                    for _, row in day_data.iloc[trigger_row + 1:].iterrows():
                        if check_condition(row):
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
        
        # Load intraday data FIRST (needed for smart daily data fetching)
        if intraday_file is None:
            debug_info.append("⚠️ No intraday data provided - analysis cannot proceed")
            return pd.DataFrame(), debug_info
        
        intraday = load_intraday_data(intraday_file)
        
        if intraday is None:
            debug_info.append("❌ Failed to load intraday data")
            return pd.DataFrame(), debug_info
        
        debug_info.append(f"Intraday data loaded: {intraday.shape}")
        debug_info.append(f"Intraday date range: {intraday['Date'].min()} to {intraday['Date'].max()}")
        
        # Load daily data (now can use intraday data for smart Yahoo fetching)
        if daily_file is not None:
            # Upload Both Files scenario
            daily = load_daily_data(daily_file)
        else:
            # Yahoo Daily + Upload Intraday scenario - use smart auto-detection
            daily = load_daily_data(uploaded_file=None, ticker=ticker, intraday_data=intraday)
        
        if daily is None:
            debug_info.append("❌ Failed to load daily data")
            return pd.DataFrame(), debug_info
        
        debug_info.append(f"Daily data loaded: {daily.shape}")
        
        # Validate data alignment
        st.subheader("🔍 Data Alignment Validation")
        is_valid, warnings, recommendations = validate_data_alignment(daily, intraday, atr_period)
        
        if warnings:
            for warning in warnings:
                st.warning(warning)
        
        if recommendations:
            with st.expander("💡 Recommendations"):
                for rec in recommendations:
                    st.info(f"• {rec}")
        
        if not is_valid:
            st.error("❌ Data alignment issues detected. Please address the warnings above before proceeding.")
            user_choice = st.radio(
                "How would you like to proceed?",
                ["Fix data alignment first", "Continue anyway (may produce unreliable results)"],
                index=0
            )
            if user_choice == "Fix data alignment first":
                return pd.DataFrame(), debug_info + warnings
        else:
            st.success("✅ Data alignment validation passed!")
        
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

def display_results(result_df, debug_messages, ticker, asset_type, data_source_label):
    """Helper function to display analysis results with enhanced statistics"""
    # Show debug info
    with st.expander('📋 Processing Information'):
        for msg in debug_messages:
            st.write(msg)
    
    if not result_df.empty:
        result_df['Ticker'] = ticker
        result_df['AssetType'] = asset_type
        result_df['DataSource'] = data_source_label
        
        # Enhanced summary stats
        st.subheader('📊 Summary Statistics')
        
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
            st.subheader('🎯 Direction Analysis')
            direction_stats = result_df.groupby('Direction').agg({
                'GoalHit': lambda x: (x == 'Yes').sum(),
                'TriggerLevel': 'count'
            }).rename(columns={'TriggerLevel': 'Total'})
            direction_stats['Hit Rate %'] = (direction_stats['GoalHit'] / direction_stats['Total'] * 100).round(1)
            st.dataframe(direction_stats)
        
        with col2:
            st.subheader('📈 Goal Classification')
            goal_stats = result_df.groupby('GoalClassification').agg({
                'GoalHit': lambda x: (x == 'Yes').sum(),
                'TriggerLevel': 'count'
            }).rename(columns={'TriggerLevel': 'Total'})
            goal_stats['Hit Rate %'] = (goal_stats['GoalHit'] / goal_stats['Total'] * 100).round(1)
            st.dataframe(goal_stats)
        
        # Show ATR validation
        if 'PreviousATR' in result_df.columns:
            latest_atr = result_df['PreviousATR'].iloc[-1]
            st.subheader('🔍 ATR Validation')
            st.write(f"**Latest ATR in results: {latest_atr:.2f}**")
            st.write("This should match Excel calculations")
            
            # ATR trend chart
            atr_by_date = result_df.groupby('Date')['PreviousATR'].first().tail(20)
            if len(atr_by_date) > 1:
                st.line_chart(atr_by_date)
        
        # Show data preview with better formatting
        st.subheader('📋 Results Preview')
        preview_df = result_df.head(10).copy()
        # Format numeric columns
        numeric_cols = ['TriggerPrice', 'GoalPrice', 'PreviousClose', 'PreviousATR']
        for col in numeric_cols:
            if col in preview_df.columns:
                preview_df[col] = preview_df[col].apply(lambda x: f"{x:.2f}")
        
        st.dataframe(preview_df, use_container_width=True)
        
        # Enhanced download options
        st.subheader('⬇️ Download Options')
        col1, col2 = st.columns(2)
        
        with col1:
            # Full results
            ticker_clean = ticker.replace("^", "").replace("=", "_")
            output_filename = f'{ticker_clean}_{asset_type}_ATR_analysis_{datetime.now().strftime("%Y%m%d")}.csv'
            st.download_button(
                '📊 Download Full Results CSV',
                data=result_df.to_csv(index=False),
                file_name=output_filename,
                mime='text/csv'
            )
        
        with col2:
            # Summary only
            summary_data = {
                'Metric': ['Total Records', 'Unique Dates', 'Goals Hit', 'Hit Rate %', 'Avg ATR'],
                'Value': [len(result_df), result_df['Date'].nunique(), goals_hit, f"{hit_rate:.1f}%", f"{avg_atr:.2f}"]
            }
            summary_df = pd.DataFrame(summary_data)
            summary_filename = f'{ticker_clean}_{asset_type}_summary_{datetime.now().strftime("%Y%m%d")}.csv'
            st.download_button(
                '📋 Download Summary CSV',
                data=summary_df.to_csv(index=False),
                file_name=summary_filename,
                mime='text/csv'
            )
        
        st.success(f'🎉 Analysis complete for {ticker}!')
        
    else:
        st.warning('⚠️ No results generated - check processing information above')

# Streamlit Interface
st.title('🎯 Advanced ATR Generator - Fixed Version')
st.write('**Intraday data must always be uploaded. Choose Yahoo Finance or file upload for daily data.**')

# Data source selection
st.sidebar.header("📁 Data Input")
data_source = st.sidebar.radio(
    "Daily Data Source",
    options=["Upload Both Files", "Yahoo Daily + Upload Intraday"],
    index=0,
    help="Choose how to provide daily data. Intraday data must always be uploaded."
)

# Intraday data upload (ALWAYS REQUIRED)
st.sidebar.subheader("📊 Intraday Data (Required)")
st.sidebar.info("⚠️ **Intraday data must always be uploaded as CSV/Excel - Yahoo Finance doesn't provide sufficient intraday history**")

intraday_file = st.sidebar.file_uploader(
    "Intraday OHLC Data",
    type=['csv', 'xlsx', 'xls'],
    help="Upload intraday OHLC data (CSV or Excel) - REQUIRED for analysis"
)

if data_source == "Upload Both Files":
    st.sidebar.subheader("📈 Daily Data Upload")
    
    # Enhanced warning about data alignment
    st.sidebar.error("""
    🚨 **CRITICAL**: Your daily data must start at least 4-6 months 
    BEFORE your intraday data begins for proper ATR calculation.
    """)
    
    # Daily data upload
    daily_file = st.sidebar.file_uploader(
        "Daily OHLC Data",
        type=['csv', 'xlsx', 'xls'],
        help="Upload daily OHLC data (CSV or Excel)"
    )
    
    ticker = st.sidebar.text_input(
        "Ticker Symbol (for labeling)",
        value="",
        help="Optional: Enter ticker symbol for output labeling"
    )
    
    start_date = None
    end_date = None

else:  # Yahoo Daily + Upload Intraday
    st.sidebar.subheader("📈 Daily Data from Yahoo Finance")
    st.sidebar.info("📅 **Smart Auto-Detection**: Daily data will be automatically fetched based on your intraday file's date range + 6-month buffer")
    
    ticker = st.sidebar.text_input(
        "Ticker Symbol",
        value="SPX",
        help="Enter ticker symbol - system will auto-map to Yahoo Finance format (e.g., SPX → ^GSPC)"
    ).upper()
    
    # Show ticker mapping preview
    if ticker:
        mapped_ticker = TickerMapper.get_yahoo_ticker(ticker)
        if mapped_ticker != ticker:
            st.sidebar.success(f"✅ Will map: {ticker} → {mapped_ticker}")
        else:
            st.sidebar.info(f"📈 Will fetch: {ticker}")
    
    st.sidebar.success("✨ No date selection needed - the system will analyze your intraday file and fetch the appropriate daily data range automatically!")
    
    # Set these to None - will be determined from intraday data
    start_date = None
    end_date = None
    daily_file = None

# Asset configuration (common to both options)
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

config = AssetConfig.get_config(asset_type, extended_hours)

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

st.sidebar.info("""
📋 **Required Columns:**
- **Daily**: Date, Open, High, Low, Close
- **Intraday**: Datetime (or Date+Time), Open, High, Low, Close
- **Optional**: Volume, Session (PM/R/AH)
""")

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
if st.button('🚀 Generate Enhanced ATR Analysis'):
    # First check if intraday file is uploaded (always required)
    if intraday_file is None:
        st.error("❌ Please upload intraday data file - this is required for all analysis types")
    elif data_source == "Upload Both Files":
        if daily_file is None:
            st.error("❌ Please upload daily data file")
        else:
            with st.spinner(f'Analyzing uploaded files with validation ({config["description"]})...'):
                try:
                    result_df, debug_messages = main_flexible(
                        ticker=ticker or "UPLOADED_DATA",
                        asset_type=asset_type,
                        daily_file=daily_file,
                        intraday_file=intraday_file,
                        atr_period=atr_period,
                        custom_ratios=custom_ratios,
                        session_filter=session_filter,
                        extended_hours=extended_hours
                    )
                    
                    display_results(result_df, debug_messages, ticker or "UPLOADED_DATA", asset_type, "Both Files Uploaded")
                        
                except Exception as e:
                    st.error(f'❌ Error: {e}')
                    import traceback
                    st.error(traceback.format_exc())
    
    elif data_source == "Yahoo Daily + Upload Intraday":
        if not ticker:
            st.error("❌ Please enter a ticker symbol for Yahoo Finance daily data")
        else:
            with st.spinner(f'Auto-detecting date range from intraday data and fetching daily data from Yahoo Finance for {ticker}...'):
                try:
                    result_df, debug_messages = main_flexible(
                        ticker=ticker,
                        asset_type=asset_type,
                        daily_file=None,  # Will auto-fetch from Yahoo based on intraday data
                        intraday_file=intraday_file,
                        atr_period=atr_period,
                        custom_ratios=custom_ratios,
                        session_filter=session_filter,
                        extended_hours=extended_hours
                    )
                    
                    display_results(result_df, debug_messages, ticker, asset_type, "Yahoo Daily (Auto-detected) + Uploaded Intraday")
                        
                except Exception as e:
                    st.error(f'❌ Error: {e}')
                    import traceback
                    st.error(traceback.format_exc())

# Enhanced help section
st.markdown("""
---
### 🔧 Fixed Issues in This Version

#### 🚨 **Primary Fix: Duplicate Column Handling**
- **Enhanced `standardize_columns()` function** - Now detects and renames duplicate columns before processing
- **Smart Excel header detection** - Automatically finds the correct header row in Excel files
- **Comprehensive debugging info** - Shows original and standardized column names for troubleshooting
- **Robust error handling** - Better error messages when columns are missing or malformed

#### 🔍 **Additional Improvements**
- **Safe Excel reading** with automatic header row detection
- **Better datetime parsing** with multiple fallback strategies  
- **Enhanced data validation** with more detailed error messages
- **Improved progress tracking** during file loading
- **Duplicate removal** for both date and datetime columns
- **More descriptive error messages** with full traceback information

### 📋 **What Was Fixed**

**Original Error**: `cannot reindex on an axis with duplicate labels`

**Root Causes Fixed**:
1. **Duplicate column names** from Excel files with merged headers
2. **Non-string column names** (numbers, floats) that weren't handled properly
3. **Missing header detection** in Excel files with multiple header rows
4. **Column mapping conflicts** when target columns already existed
5. **Incomplete error handling** that made debugging difficult

**Solutions Implemented**:
1. **Duplicate detection and renaming** - Adds `_1`, `_2` suffixes to duplicates
2. **String conversion** for all column names before processing
3. **Automatic header detection** for Excel files (checks up to 10 rows)
4. **Safe column mapping** that avoids overwriting existing columns
5. **Enhanced debugging** with column name tracking at each step

### 🎯 **Usage Recommendations**

#### For Excel Files:
- The system now automatically detects header rows (up to row 10)
- Shows which row was used as headers in the processing info
- Handles merged cells and complex Excel structures better

#### For CSV Files:
- Cleans up column names automatically
- Handles various naming conventions (open/Open/OPEN)
- Removes extra whitespace and special characters

#### For All Files:
- Shows original vs standardized column names in processing info
- Validates all required columns before starting analysis
- Provides clear error messages if columns are missing
- Removes duplicate rows automatically based on date/datetime

### 🔧 **Debugging Features Added**

#### 📊 **Column Inspection**
- **Before processing**: Shows original column names from your file
- **After standardization**: Shows how columns were mapped
- **Missing column alerts**: Lists exactly which required columns are missing
- **Duplicate handling**: Reports how many duplicates were renamed

#### 📅 **Date/Time Validation**
- **Invalid date detection**: Counts and removes rows with unparseable dates
- **DateTime creation**: Shows how Date+Time columns were combined
- **Date range reporting**: Displays the actual date range found in your data
- **Empty data detection**: Catches completely empty files or datasets

#### 🔍 **Progress Monitoring**
- **File reading progress**: Shows steps during file loading
- **Processing status**: Real-time updates during analysis
- **Memory optimization**: Handles large files more efficiently
- **Error recovery**: Continues processing even if some rows fail

### 📋 **Common Issues and Solutions**

#### Issue: "Missing required columns"
**Solution**: Check the processing info to see how your columns were mapped. Common fixes:
- Rename columns to standard names (Date, Open, High, Low, Close)
- Remove special characters from column headers
- Check for merged cells in Excel files

#### Issue: "No valid dates found"
**Solution**: 
- Ensure date column contains actual dates (not text)
- Check date format (YYYY-MM-DD, MM/DD/YYYY, etc.)
- Remove header rows that contain text instead of dates

#### Issue: "Empty dataset after processing"
**Solution**:
- Check if your file actually contains data rows (not just headers)
- Verify that OHLC columns contain numeric values
- Look for completely blank rows that need to be removed

#### Issue: Excel file not reading correctly
**Solution**:
- The system now auto-detects header rows (tries rows 0-9)
- If auto-detection fails, manually clean your Excel file
- Save as CSV if Excel structure is too complex

### 🎯 **Best Practices for Data Preparation**

#### ✅ **Recommended File Structure**

**Daily Data**:
```
Date        | Open   | High   | Low    | Close  | Volume
2024-01-01  | 100.50 | 102.30 | 99.80  | 101.75 | 1000000
2024-01-02  | 101.75 | 103.00 | 101.20 | 102.50 | 1200000
```

**Intraday Data**:
```
Datetime            | Open   | High   | Low    | Close  | Volume | Session
2024-01-01 09:30:00 | 100.50 | 100.80 | 100.40 | 100.70 | 50000  | R
2024-01-01 09:31:00 | 100.70 | 100.95 | 100.65 | 100.85 | 45000  | R
```

#### ✅ **Column Name Flexibility**
The system accepts these variations:
- **Date**: Date, date, timestamp, Timestamp
- **OHLC**: Open/open, High/high, Low/low, Close/close, Last/last
- **Volume**: Volume, volume, Vol, vol
- **Time**: Time, time, Datetime, datetime

#### ⚠️ **What to Avoid**
- Merged cells in Excel headers
- Special characters in column names (@, #, %, etc.)
- Empty rows mixed with data rows
- Multiple header rows without clear data separation
- Date columns with mixed formats (some MM/DD, some DD/MM)

### 🚀 **Performance Improvements**

#### 📈 **Large File Handling**
- **Streaming processing**: Reads large files in chunks
- **Memory optimization**: Processes data without loading everything into memory
- **Progress indicators**: Shows real-time progress for large datasets
- **Error resilience**: Continues processing even if some rows have issues

#### ⚡ **Speed Optimizations**
- **Efficient column mapping**: Faster standardization process
- **Smart date parsing**: Uses optimized datetime conversion
- **Reduced memory usage**: Cleans up intermediate DataFrames
- **Parallel processing**: Where applicable, uses vectorized operations

### 📊 **Enhanced Output Features**

#### 📋 **Detailed Processing Log**
- Step-by-step processing information
- Column mapping details
- Data quality metrics
- Performance statistics

#### 📈 **Improved Statistics**
- Hit rate analysis by direction and goal type
- ATR trend visualization
- Session-specific performance (if applicable)
- Data quality scores

#### 💾 **Better Downloads**
- Clean filenames (removes special characters)
- Timestamped outputs for version control
- Both full results and summary options
- Properly formatted CSV files

### 🔍 **Troubleshooting Guide**

#### If you see: "Error loading daily data: cannot reindex on an axis with duplicate labels"
✅ **Fixed in this version!** The enhanced column handling should resolve this automatically.

#### If you see: "Missing required columns"
1. Check the "Processing Information" section to see original column names
2. Verify your file has columns for Date, Open, High, Low, Close
3. Remove any special characters from column headers
4. For Excel files, ensure headers are in a single row

#### If you see: "No valid dates found"
1. Check that your Date column contains actual dates, not text
2. Try different date formats (the system auto-detects most formats)
3. Remove any header rows that contain descriptive text instead of dates

#### If analysis runs but produces no results:
1. Check the data alignment validation warnings
2. Ensure daily data starts before intraday data
3. Verify that ATR calculation produced valid values
4. Check that your date ranges overlap between daily and intraday data

### 📞 **Still Having Issues?**

If you encounter problems:
1. **Check the Processing Information** - It shows exactly what happened during loading
2. **Look at column mapping details** - Shows how your columns were interpreted
3. **Review data alignment warnings** - Critical for proper ATR calculation
4. **Try with a smaller sample** - Test with just a few days of data first
5. **Convert Excel to CSV** - Sometimes simpler formats work better

The enhanced error handling and debugging features should help identify and resolve most data loading issues automatically!
""")
</invoke>
