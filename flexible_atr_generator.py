import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta, date
import os
import json
import tempfile
import shutil

class DataProcessingCheckpoint:
    """Handle checkpoint/resume functionality for multi-file processing"""
    
    def __init__(self, session_id=None):
        # Use session-specific checkpoint file
        if session_id is None:
            session_id = st.session_state.get('session_id', self._generate_session_id())
            st.session_state['session_id'] = session_id
        
        self.checkpoint_file = f"checkpoint_{session_id}.json"
        self.temp_dir = f"temp_{session_id}"
        self.state = self.load_checkpoint()
    
    def _generate_session_id(self):
        """Generate unique session ID"""
        return datetime.now().strftime("%Y%m%d_%H%M%S")
    
    def load_checkpoint(self):
        """Load existing checkpoint or create new one"""
        if os.path.exists(self.checkpoint_file):
            try:
                with open(self.checkpoint_file, 'r') as f:
                    return json.load(f)
            except:
                pass  # If checkpoint is corrupted, start fresh
        
        return {
            'processed_files': [],
            'failed_files': [],
            'last_successful_file': None,
            'target_timeframe': None,
            'start_time': None,
            'total_files': 0,
            'temp_data_files': [],
            'session_id': st.session_state.get('session_id'),
            'processing_complete': False
        }
    
    def save_checkpoint(self):
        """Save current state to disk"""
        # Ensure temp directory exists
        if not os.path.exists(self.temp_dir):
            os.makedirs(self.temp_dir)
        
        try:
            with open(self.checkpoint_file, 'w') as f:
                json.dump(self.state, f, indent=2, default=str)
        except Exception as e:
            st.error(f"Failed to save checkpoint: {e}")
    
    def mark_file_processed(self, filename, temp_data_file=None, row_count=0):
        """Mark a file as successfully processed"""
        self.state['processed_files'].append({
            'filename': filename,
            'temp_file': temp_data_file,
            'row_count': row_count,
            'timestamp': datetime.now().isoformat()
        })
        self.state['last_successful_file'] = filename
        if temp_data_file:
            self.state['temp_data_files'].append(temp_data_file)
        self.save_checkpoint()
    
    def mark_file_failed(self, filename, error_msg):
        """Mark a file as failed with error details"""
        self.state['failed_files'].append({
            'filename': filename,
            'error': str(error_msg),
            'timestamp': datetime.now().isoformat()
        })
        self.save_checkpoint()
    
    def clear_checkpoint(self):
        """Clear checkpoint and cleanup temp files"""
        # Remove temp files
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir, ignore_errors=True)
        
        # Remove checkpoint file
        if os.path.exists(self.checkpoint_file):
            os.remove(self.checkpoint_file)
        
        # Reset state
        self.state = {
            'processed_files': [],
            'failed_files': [],
            'last_successful_file': None,
            'target_timeframe': None,
            'start_time': None,
            'total_files': 0,
            'temp_data_files': [],
            'session_id': st.session_state.get('session_id'),
            'processing_complete': False
        }
    
    def get_processed_filenames(self):
        """Get list of already processed filenames"""
        return [f['filename'] for f in self.state['processed_files']]
    
    def get_failed_filenames(self):
        """Get list of failed filenames"""
        return [f['filename'] for f in self.state['failed_files']]

def calculate_atr(df, period=14):
    """
    Calculate TRUE Wilder's ATR - VALIDATED IMPLEMENTATION
    Matches Excel formula exactly:
    1. Wait for 14 periods before starting ATR
    2. First ATR = simple average of first 14 TR values
    3. Subsequent ATR = (1/14) * current_TR + (13/14) * previous_ATR
    """
    df = df.copy()
    
    # Calculate True Range
    df['H-L'] = df['High'] - df['Low']
    df['H-PC'] = abs(df['High'] - df['Close'].shift(1))
    df['L-PC'] = abs(df['Low'] - df['Close'].shift(1))
    df['TR'] = df[['H-L', 'H-PC', 'L-PC']].max(axis=1)
    
    # Calculate TRUE Wilder's ATR (not pandas EMA!)
    atr_values = [None] * len(df)
    
    for i in range(len(df)):
        if i < period:
            # No ATR until we have enough data (like Excel)
            atr_values[i] = None
        elif i == period:
            # First ATR = simple average of first 14 TR values
            atr_values[i] = df['TR'].iloc[i-period+1:i+1].mean()
        else:
            # Subsequent ATR = (1/14) * current_TR + (13/14) * previous_ATR
            prev_atr = atr_values[i-1]
            current_tr = df['TR'].iloc[i]
            atr_values[i] = (1/period) * current_tr + ((period-1)/period) * prev_atr
    
    df['ATR'] = atr_values
    
    # Clean up temporary columns
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
            'STOCKS_24H': {
                'market_open': '00:00',
                'market_close': '23:59',
                'has_open_special': False,
                'weekends_closed': False,
                'session_types': ['24H'],
                'default_session': ['24H'],
                'description': 'US Stocks (24-Hour Data)',
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
            },
            'COMMODITIES': {
                'market_open': '09:30',
                'market_close': '16:30',
                'has_open_special': True,
                'weekends_closed': True,
                'session_types': ['R', 'AH'],
                'default_session': ['R'],
                'description': 'Commodities (varies by commodity)',
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
    
    # Debug: Show available columns
    st.info(f"Daily data columns: {list(daily_data.columns)}")
    st.info(f"Intraday data columns: {list(intraday_data.columns)}")
    
    # Convert date columns for comparison - handle different formats safely
    try:
        if 'Date' in daily_data.columns:
            daily_dates = pd.to_datetime(daily_data['Date'], errors='coerce')
            if hasattr(daily_dates.iloc[0], 'date'):
                daily_dates = daily_dates.dt.date
        else:
            # Check for alternative date column names
            date_cols = [col for col in daily_data.columns if 'date' in col.lower()]
            if date_cols:
                st.info(f"Using alternative date column: {date_cols[0]}")
                daily_dates = pd.to_datetime(daily_data[date_cols[0]], errors='coerce')
                if hasattr(daily_dates.iloc[0], 'date'):
                    daily_dates = daily_dates.dt.date
            else:
                warnings.append("Daily data missing Date column")
                return False, warnings, ["Ensure daily data has a 'Date' column"]
    except Exception as e:
        warnings.append(f"Error processing daily data dates: {str(e)}")
        return False, warnings, ["Check daily data date format"]
    
    try:
        if 'Date' in intraday_data.columns:
            intraday_dates = pd.to_datetime(intraday_data['Date'], errors='coerce')
            if hasattr(intraday_dates.iloc[0], 'date'):
                intraday_dates = intraday_dates.dt.date
        elif 'Datetime' in intraday_data.columns:
            intraday_dates = pd.to_datetime(intraday_data['Datetime'], errors='coerce')
            if hasattr(intraday_dates.iloc[0], 'date'):
                intraday_dates = intraday_dates.dt.date
        else:
            # Check for alternative date/datetime columns
            date_cols = [col for col in intraday_data.columns if 'date' in col.lower() or 'time' in col.lower()]
            if date_cols:
                st.info(f"Using alternative datetime column: {date_cols[0]}")
                intraday_dates = pd.to_datetime(intraday_data[date_cols[0]], errors='coerce')
                if hasattr(intraday_dates.iloc[0], 'date'):
                    intraday_dates = intraday_dates.dt.date
            else:
                warnings.append("Intraday data missing Date/Datetime column")
                return False, warnings, ["Ensure intraday data has a 'Date' or 'Datetime' column"]
    except Exception as e:
        warnings.append(f"Error processing intraday data dates: {str(e)}")
        return False, warnings, ["Check intraday data date format"]
    
    daily_start = daily_dates.min()
    daily_end = daily_dates.max()
    intraday_start = intraday_dates.min()
    intraday_end = intraday_dates.max()
    
    # Check if daily data starts before intraday data
    if daily_start >= intraday_start:
        is_valid = False
        warnings.append("Daily data should start BEFORE intraday data for proper ATR calculation")
        recommendations.append("Extend daily data backwards to include more historical periods")
    
    # Check buffer period for ATR calculation
    buffer_days = (intraday_start - daily_start).days
    required_days = max(atr_period * 7, min_buffer_days)  # At least ATR period in trading days or 4 months
    
    if buffer_days < required_days:
        is_valid = False
        warnings.append(f"Insufficient buffer period: {buffer_days} days (need {required_days}+ days)")
        recommendations.append(f"Add at least {required_days - buffer_days} more days of daily data before intraday period")
    
    # Check date overlap
    overlap_start = max(daily_start, intraday_start)
    overlap_end = min(daily_end, intraday_end)
    
    if overlap_start > overlap_end:
        is_valid = False
        warnings.append("No date overlap between daily and intraday data")
        recommendations.append("Ensure daily and intraday data cover overlapping time periods")
    else:
        overlap_days = (overlap_end - overlap_start).days
        if overlap_days < 5:
            warnings.append("Very small overlap period between daily and intraday data")
            recommendations.append("Increase overlap period for more reliable analysis")
    
    # Data quality checks - with better error handling
    try:
        # Check if OHLC columns exist in daily data
        required_daily_cols = ['Open', 'High', 'Low', 'Close']
        missing_daily_cols = [col for col in required_daily_cols if col not in daily_data.columns]
        
        if missing_daily_cols:
            warnings.append(f"Daily data missing OHLC columns: {missing_daily_cols}")
            recommendations.append("Ensure daily data has Open, High, Low, Close columns")
        else:
            # Only check for missing values if columns exist
            daily_missing = daily_data[required_daily_cols].isnull().sum().sum()
            if daily_missing > 0:
                warnings.append(f"{daily_missing} missing values in daily OHLC data")
                recommendations.append("Clean daily data to remove or fill missing values")
    except Exception as e:
        warnings.append(f"Error checking daily data quality: {str(e)}")
        recommendations.append("Check daily data format and column names")
    
    try:
        # Check if OHLC columns exist in intraday data
        required_intraday_cols = ['Open', 'High', 'Low', 'Close']
        missing_intraday_cols = [col for col in required_intraday_cols if col not in intraday_data.columns]
        
        if missing_intraday_cols:
            warnings.append(f"Intraday data missing OHLC columns: {missing_intraday_cols}")
            recommendations.append("Ensure intraday data has Open, High, Low, Close columns")
        else:
            # Only check for missing values if columns exist
            intraday_missing = intraday_data[required_intraday_cols].isnull().sum().sum()
            if intraday_missing > 0:
                warnings.append(f"{intraday_missing} missing values in intraday OHLC data")
                recommendations.append("Clean intraday data to remove or fill missing values")
    except Exception as e:
        warnings.append(f"Error checking intraday data quality: {str(e)}")
        recommendations.append("Check intraday data format and column names")
    
    return is_valid, warnings, recommendations

def load_daily_data(uploaded_file):
    """Load daily data from uploaded CSV file with robust encoding handling"""
    try:
        # Handle different file types
        if uploaded_file.name.endswith('.csv'):
            # Try multiple encodings for CSV files
            encodings_to_try = ['utf-8', 'latin-1', 'iso-8859-1', 'cp1252', 'utf-16']
            
            for encoding in encodings_to_try:
                try:
                    # Reset file pointer
                    uploaded_file.seek(0)
                    daily = pd.read_csv(uploaded_file, encoding=encoding)
                    st.info(f"Successfully loaded CSV with {encoding} encoding")
                    break
                except UnicodeDecodeError:
                    continue
                except Exception as e:
                    # If it's not an encoding error, break and handle it below
                    if "codec can't decode" not in str(e):
                        raise e
                    continue
            else:
                # If all encodings failed
                st.error("Could not read CSV file with any supported encoding")
                st.info("Try saving your CSV file as UTF-8 encoding")
                return None
                
        elif uploaded_file.name.endswith(('.xlsx', '.xls')):
            # Excel files don't have encoding issues
            daily = pd.read_excel(uploaded_file, header=0)
            # If that doesn't work, try other common header positions
            if 'Date' not in daily.columns and 'Close' not in daily.columns:
                for header_row in [1, 2, 3, 4, 5]:
                    try:
                        daily = pd.read_excel(uploaded_file, header=header_row)
                        if 'Date' in daily.columns or 'Close' in daily.columns:
                            st.info(f"Found headers at row {header_row + 1}")
                            break
                    except:
                        continue
        else:
            st.error("Unsupported file format. Please use CSV or Excel files.")
            return None
        
        # Standardize column names
        daily = standardize_columns(daily)
        
        # Validate required columns
        required_cols = ['Date', 'Open', 'High', 'Low', 'Close']
        missing_cols = [col for col in required_cols if col not in daily.columns]
        if missing_cols:
            st.error(f"Missing required columns in daily data: {missing_cols}")
            st.info(f"Available columns: {list(daily.columns)}")
            return None
        
        # Sort by date and ensure proper date format
        daily['Date'] = pd.to_datetime(daily['Date'], errors='coerce')
        
        # Check for invalid dates
        invalid_dates = daily['Date'].isna().sum()
        if invalid_dates > 0:
            st.warning(f"Found {invalid_dates} invalid dates - removing these rows")
            daily = daily.dropna(subset=['Date'])
        
        if daily.empty:
            st.error("No valid data remaining after date processing")
            return None
        
        daily = daily.sort_values('Date').reset_index(drop=True)
        
        st.success(f"Loaded daily data: {len(daily)} records")
        date_min = daily['Date'].min()
        date_max = daily['Date'].max()
        # Handle different date formats for display
        if hasattr(date_min, 'date'):
            date_min_str = date_min.date()
            date_max_str = date_max.date()
        else:
            date_min_str = date_min
            date_max_str = date_max
        st.info(f"Date range: {date_min_str} to {date_max_str}")
        
        return daily
        
    except Exception as e:
        st.error(f"Error loading daily data: {str(e)}")
        st.info("Try these solutions:")
        st.info("1. Save your CSV file with UTF-8 encoding")
        st.info("2. Open in Excel and re-save as CSV")
        st.info("3. Use Excel format (.xlsx) instead of CSV")
        return None

def load_intraday_data(uploaded_file):
    """Load intraday data from uploaded file with robust encoding handling"""
    try:
        # Handle case where uploaded_file might be a list (safety check)
        if isinstance(uploaded_file, list):
            if len(uploaded_file) > 0:
                uploaded_file = uploaded_file[0]
            else:
                st.error("No intraday files provided")
                return None
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        status_text.text("Reading file...")
        progress_bar.progress(25)
        
        if uploaded_file.name.endswith('.csv'):
            # Try multiple encodings for CSV files
            encodings_to_try = ['utf-8', 'latin-1', 'iso-8859-1', 'cp1252', 'utf-16']
            
            for encoding in encodings_to_try:
                try:
                    # Reset file pointer
                    uploaded_file.seek(0)
                    intraday = pd.read_csv(uploaded_file, encoding=encoding)
                    st.info(f"Successfully loaded CSV with {encoding} encoding")
                    break
                except UnicodeDecodeError:
                    continue
                except Exception as e:
                    # If it's not an encoding error, break and handle it below
                    if "codec can't decode" not in str(e):
                        raise e
                    continue
            else:
                # If all encodings failed
                st.error("Could not read CSV file with any supported encoding")
                st.info("Try saving your CSV file as UTF-8 encoding")
                return None
                
        elif uploaded_file.name.endswith(('.xlsx', '.xls')):
            intraday = pd.read_excel(uploaded_file, header=0)
        else:
            st.error("Unsupported file format. Please use CSV or Excel files.")
            return None
        
        status_text.text("Standardizing columns...")
        progress_bar.progress(50)
        
        # Standardize columns
        intraday = standardize_columns(intraday)
        
        # Validate required columns
        required_cols = ['Open', 'High', 'Low', 'Close']
        missing_cols = [col for col in required_cols if col not in intraday.columns]
        if missing_cols:
            st.error(f"Missing required columns in intraday data: {missing_cols}")
            st.info(f"Available columns: {list(intraday.columns)}")
            return None
        
        status_text.text("Processing datetime...")
        progress_bar.progress(75)
        
        # Ensure we have a datetime column
        if 'Datetime' not in intraday.columns:
            # Try to create from Date and Time columns
            if 'Date' in intraday.columns and 'Time' in intraday.columns:
                intraday['Datetime'] = pd.to_datetime(intraday['Date'].astype(str) + ' ' + intraday['Time'].astype(str), errors='coerce')
            elif 'Date' in intraday.columns:
                # Assume Date column contains full datetime
                intraday['Datetime'] = pd.to_datetime(intraday['Date'], errors='coerce')
            else:
                st.error("Could not find datetime information in intraday data")
                return None
        else:
            intraday['Datetime'] = pd.to_datetime(intraday['Datetime'], errors='coerce')
        
        # Check for invalid datetime entries
        invalid_datetime = intraday['Datetime'].isna().sum()
        if invalid_datetime > 0:
            st.warning(f"Found {invalid_datetime} invalid datetime entries - removing these rows")
            intraday = intraday.dropna(subset=['Datetime'])
        
        if intraday.empty:
            st.error("No valid data remaining after datetime processing")
            return None
        
        # Create Date column for matching
        intraday['Date'] = intraday['Datetime'].dt.date
        
        # Sort by datetime
        intraday = intraday.sort_values('Datetime').reset_index(drop=True)
        
        status_text.text("Finalizing...")
        progress_bar.progress(100)
        
        st.success(f"Loaded intraday data: {len(intraday):,} records")
        st.info(f"Date range: {intraday['Date'].min()} to {intraday['Date'].max()}")
        
        # Clear progress indicators
        progress_bar.empty()
        status_text.empty()
        
        return intraday
        
    except Exception as e:
        st.error(f"Error loading intraday data: {str(e)}")
        st.info("Try these solutions:")
        st.info("1. Save your CSV file with UTF-8 encoding")
        st.info("2. Open in Excel and re-save as CSV")
        st.info("3. Use Excel format (.xlsx) instead of CSV")
        return None

def standardize_columns(df):
    """Standardize column names across different data sources"""
    try:
        # Clean column names first - handle non-string column names
        clean_columns = []
        for col in df.columns:
            if isinstance(col, str):
                clean_columns.append(col.strip())
            else:
                # Convert non-strings (floats, numbers) to strings
                clean_columns.append(str(col).strip())
        
        df.columns = clean_columns
        
        # Common column mappings including single letters
        column_mappings = {
            # Date columns
            'date': 'Date',
            'timestamp': 'Date', 
            'datetime': 'Datetime',
            # OHLC columns - including single letter variations
            'open': 'Open',
            'o': 'Open',          # Single letter
            'high': 'High', 
            'h': 'High',          # Single letter
            'low': 'Low',
            'l': 'Low',           # Single letter
            'close': 'Close',
            'c': 'Close',         # Single letter
            'last': 'Close',  # Common in some data providers
            'adj close': 'Close',
            'adj_close': 'Close',
            'adjusted_close': 'Close',
            'settle': 'Close',  # Common in futures data
            # Volume
            'volume': 'Volume',
            'vol': 'Volume',
            'v': 'Volume',        # Single letter
            # Session
            'session': 'Session'
        }
        
        # Apply mappings (case insensitive)
        for old_name, new_name in column_mappings.items():
            for col in df.columns:
                if isinstance(col, str) and col.lower() == old_name:
                    df.rename(columns={col: new_name}, inplace=True)
                    break
        
        return df
        
    except Exception as e:
        st.error(f"Error in standardize_columns: {e}")
        st.info(f"DataFrame columns: {list(df.columns)}")
        st.info(f"DataFrame shape: {df.shape}")
        return df  # Return original df if standardization fails

def filter_by_session_and_hours(intraday_data, date, asset_config, session_filter=None):
    """Filter intraday data based on sessions and trading hours"""
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
    total_days = len(daily) - 1
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    for i in range(1, len(daily)):
        try:
            # Update progress
            progress = i / total_days
            progress_bar.progress(progress)
            status_text.text(f"Processing day {i}/{total_days}...")
            
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
    
    # Get previous day for ATR calculation
    debug_index = daily_debug.index[0]
    if debug_index == 0:
        st.error(f"Cannot debug first day - need previous day for ATR calculation")
        return
    
    previous_row = daily.iloc[debug_index - 1]
    current_row = daily.iloc[debug_index]
    
    previous_close = previous_row['Close']
    previous_atr = previous_row['ATR']
    
    if pd.isna(previous_atr):
        st.error(f"No valid ATR for previous day")
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
        st.metric("Previous Close", f"{previous_close:.2f}")
    with col2:
        st.metric("Previous ATR", f"{previous_atr:.2f}")
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

def main_csv_only(ticker, asset_type, daily_file, intraday_file, atr_period=14, 
                 custom_ratios=None, session_filter=None, extended_hours=False, 
                 debug_mode=False, debug_date=None):
    """Main function for CSV-only ATR analysis"""
    debug_info = []
    
    try:
        # Early debug mode check
        if debug_mode and debug_date:
            st.success(f"DEBUG MODE - Will process ONLY {debug_date}")
        else:
            st.info("FULL MODE - Will process all days")
        
        # Get asset configuration
        asset_config = AssetConfig.get_config(asset_type, extended_hours)
        debug_info.append(f"Asset Type: {asset_config['description']}")
        debug_info.append(f"Market Hours: {asset_config['market_open']} - {asset_config['market_close']}")
        debug_info.append(f"Extended Hours: {extended_hours}")
        
        # Load daily data
        daily = load_daily_data(daily_file)
        if daily is None:
            debug_info.append("Failed to load daily data")
            return pd.DataFrame(), debug_info
        
        debug_info.append(f"Daily data loaded: {daily.shape}")
        debug_info.append(f"Daily date range: {daily['Date'].min()} to {daily['Date'].max()}")
        
        # Load intraday data
        intraday = load_intraday_data(intraday_file)
        if intraday is None:
            debug_info.append("Failed to load intraday data")
            return pd.DataFrame(), debug_info
        
        debug_info.append(f"Intraday data loaded: {intraday.shape}")
        debug_info.append(f"Intraday date range: {intraday['Date'].min()} to {intraday['Date'].max()}")
        
        # Validate data alignment
        if not debug_mode:
            st.subheader("Data Alignment Validation")
            is_valid, warnings, recommendations = validate_data_alignment(daily, intraday, atr_period)
            
            if warnings:
                for warning in warnings:
                    st.warning(warning)
            
            if recommendations:
                with st.expander("Recommendations"):
                    for rec in recommendations:
                        st.info(f"• {rec}")
            
            if not is_valid:
                st.error("Data alignment issues detected. Please address the warnings above.")
                user_choice = st.radio(
                    "How would you like to proceed?",
                    ["Fix data alignment first", "Continue anyway (may produce unreliable results)"],
                    index=0
                )
                if user_choice == "Fix data alignment first":
                    return pd.DataFrame(), debug_info + warnings
            else:
                st.success("Data alignment validation passed!")
        
        # Calculate ATR using TRUE Wilder's method
        debug_info.append(f"Calculating ATR with TRUE Wilder's method, period {atr_period}...")
        daily = calculate_atr(daily, period=atr_period)
        
        # Validate ATR
        valid_atr = daily[daily['ATR'].notna()]
        if not valid_atr.empty:
            recent_atr = valid_atr['ATR'].tail(3).round(2).tolist()
            debug_info.append(f"ATR calculated successfully. Recent values: {recent_atr}")
        else:
            debug_info.append("No valid ATR values calculated")
            return pd.DataFrame(), debug_info
        
        # Check for session column
        if 'Session' in intraday.columns:
            unique_sessions = intraday['Session'].unique()
            debug_info.append(f"Session types found: {list(unique_sessions)}")
        
        # Quick Debug Mode
        if debug_mode and debug_date:
            st.info(f"Debug Mode Active - Analyzing single day: {debug_date}")
            debug_success = debug_single_day_analysis(daily, intraday, debug_date, custom_ratios)
            if debug_success:
                return pd.DataFrame(), debug_info + [f"Debug analysis completed for {debug_date}"]
            else:
                return pd.DataFrame(), debug_info + [f"Debug analysis failed for {debug_date}"]
        
        # Run full systematic analysis
        if not debug_mode:
            debug_info.append("Running SYSTEMATIC trigger and goal detection...")
            df = detect_triggers_and_goals_systematic(daily, intraday, custom_ratios)
            debug_info.append(f"Detection complete: {len(df)} trigger-goal combinations found")
            
            # Additional statistics
            if not df.empty:
                above_triggers = len(df[df['Direction'] == 'Above'])
                below_triggers = len(df[df['Direction'] == 'Below'])
                debug_info.append(f"Above triggers: {above_triggers}, Below triggers: {below_triggers}")
                
                goals_hit = len(df[df['GoalHit'] == 'Yes'])
                hit_rate = goals_hit / len(df) * 100 if len(df) > 0 else 0
                debug_info.append(f"Goals hit: {goals_hit}/{len(df)} ({hit_rate:.1f}%)")
                
                # Validation metrics
                same_time_count = len(df[df['SameTime'] == True])
                debug_info.append(f"Same-time scenarios found: {same_time_count}")
                
                open_triggers = len(df[df['TriggerTime'] == 'OPEN'])
                intraday_triggers = len(df[df['TriggerTime'] != 'OPEN'])
                debug_info.append(f"OPEN triggers: {open_triggers}, Intraday triggers: {intraday_triggers}")
            
            return df, debug_info
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
            st.write(f"**Latest ATR: {latest_atr:.2f}** (TRUE Wilder's method)")
            
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
st.title('🎯 CSV-Only ATR Analysis Generator')
st.write('**Clean, focused ATR analysis using the validated systematic trigger/goal detection logic**')
st.write('**Upload your daily and intraday CSV files to get started**')

# File upload section
st.header("📁 Data Upload")

col1, col2 = st.columns(2)

with col1:
    st.subheader("📈 Daily Data")
    daily_file = st.file_uploader(
        "Upload Daily OHLC Data",
        type=['csv', 'xlsx', 'xls'],
        help="CSV or Excel file with daily OHLC data",
        key="daily_upload"
    )
    
    if daily_file:
        st.success(f"✅ Daily file uploaded: {daily_file.name}")
        st.info("📊 Daily data should start at least 4-6 months before your intraday data for proper ATR calculation")

with col2:
    st.subheader("📊 Intraday Data")
    intraday_file = st.file_uploader(
        "Upload Intraday OHLC Data",
        type=['csv', 'xlsx', 'xls'],
        help="CSV or Excel file with intraday OHLC data",
        key="intraday_upload"
    )
    
    if intraday_file:
        st.success(f"✅ Intraday file uploaded: {intraday_file.name}")
        st.info("📊 Intraday data should be properly formatted with datetime information")

# Configuration section
if daily_file and intraday_file:
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
            options=['STOCKS', 'STOCKS_24H', 'CRYPTO', 'FOREX', 'FUTURES', 'COMMODITIES'],
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
            "ATR Period", 
            min_value=1, 
            max_value=50, 
            value=14,
            help="Period for ATR calculation (default: 14)"
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
    
    # Session filtering (if applicable)
    config = AssetConfig.get_config(asset_type, extended_hours)
    if len(config['session_types']) > 1:
        with st.expander("📊 Session Filtering"):
            session_filter = st.multiselect(
                "Filter by Sessions",
                options=config['session_types'],
                default=config['default_session'],
                help="Select trading sessions to include in analysis"
            )
    else:
        session_filter = None
    
    # Run analysis button
    st.markdown("---")
    
    if st.button('🚀 Generate ATR Analysis', type="primary", use_container_width=True):
        with st.spinner('Processing with SYSTEMATIC logic...'):
            try:
                result_df, debug_messages = main_csv_only(
                    ticker=ticker,
                    asset_type=asset_type,
                    daily_file=daily_file,
                    intraday_file=intraday_file,
                    atr_period=atr_period,
                    custom_ratios=custom_ratios,
                    session_filter=session_filter,
                    extended_hours=extended_hours,
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
    st.info("👆 **Please upload both daily and intraday CSV files to proceed**")
    
    # Show file format requirements
    with st.expander("📋 Required File Formats", expanded=True):
        st.markdown("""
        **📈 Daily Data Requirements:**
        - **Columns**: Date, Open, High, Low, Close
        - **Alternative formats**: o, h, l, c (single letters)
        - **Date format**: Any standard date format
        - **Coverage**: Should start 4-6 months before intraday data
        
        **📊 Intraday Data Requirements:**
        - **Columns**: Date/Datetime, Open, High, Low, Close
        - **Alternative formats**: o, h, l, c (single letters)
        - **Datetime**: Full datetime or separate Date + Time columns
        - **Timeframe**: Any intraday timeframe (1min, 5min, 10min, etc.)
        
        **✅ Supported Formats:**
        - CSV files (.csv)
        - Excel files (.xlsx, .xls)
        - Both long format (Open, High, Low, Close) and short format (o, h, l, c)
        """)
    
    # Show workflow
    with st.expander("🔧 Analysis Workflow", expanded=False):
        st.markdown("""
        **🎯 Step-by-Step Process:**
        
        1. **Upload Files** - Daily and intraday CSV/Excel files
        2. **Configure Settings** - Ticker, asset type, ATR period
        3. **Data Validation** - System checks alignment and quality
        4. **ATR Calculation** - TRUE Wilder's method (matches Excel)
        5. **Systematic Detection** - Trigger and goal analysis
        6. **Results Export** - Download full analysis or summary
        
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
    - TRUE Wilder's ATR calculation
    - Matches Excel formulas exactly
    - Same-candle completion rules
    - Cross-zero scenario handling
    """)

with col2:
    st.markdown("""
    **📊 Data Flexibility**
    - CSV and Excel support
    - Multiple column formats
    - o,h,l,c or Open,High,Low,Close
    - Automatic column detection
    - Date/datetime parsing
    """)

with col3:
    st.markdown("""
    **🔧 Analysis Features**
    - Multi-asset class support
    - Custom ratio definitions
    - Session filtering
    - Debug mode for single days
    - Comprehensive statistics
    """)

st.info("💡 **Tip**: Use the CSV Data Handler tool to prepare and clean your data files before analysis!")

st.markdown("""
---
### 🎯 About This Tool

This is a **clean, focused ATR analysis tool** that uses the validated systematic trigger/goal detection logic. 

**Key Improvements from the original:**
- ✅ **CSV-only input** - No complex public data source handling
- ✅ **Simplified interface** - Focus on the core analysis
- ✅ **Better reliability** - No external API dependencies  
- ✅ **Faster processing** - Streamlined data handling
- ✅ **Cleaner codebase** - Single responsibility principle

**Perfect workflow:**
1. **CSV Data Handler** → Process and prepare your data files
2. **This ATR Tool** → Run systematic trigger/goal analysis  
3. **Export Results** → Get clean CSV files for further analysis

This creates a clean, maintainable pipeline where each tool does one thing really well!
""")
