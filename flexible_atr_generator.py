import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta, date
import os
import json
import os
import tempfile
import shutil
from datetime import datetime
import gzip

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

class EnhancedDataLoader:
    """Enhanced data loader with multi-file and resampling support"""
    
    def __init__(self):
        self.checkpoint = DataProcessingCheckpoint()
    
    def resample_ohlc_data(self, df, timeframe='10T'):
        """Resample OHLC data to specified timeframe"""
        df = df.copy()
        
        # Ensure we have datetime column
        if 'Datetime' not in df.columns:
            if 'Date' in df.columns and 'Time' in df.columns:
                df['Datetime'] = pd.to_datetime(df['Date'].astype(str) + ' ' + df['Time'].astype(str))
            elif 'Date' in df.columns:
                df['Datetime'] = pd.to_datetime(df['Date'])
            else:
                raise ValueError("Could not find datetime information")
        else:
            df['Datetime'] = pd.to_datetime(df['Datetime'])
        
        # Set datetime as index for resampling
        df.set_index('Datetime', inplace=True)
        
        # Define aggregation rules
        agg_rules = {
            'Open': 'first',
            'High': 'max',
            'Low': 'min', 
            'Close': 'last'
        }
        
        # Add volume if present
        if 'Volume' in df.columns:
            agg_rules['Volume'] = 'sum'
        
        # Add session if present
        if 'Session' in df.columns:
            agg_rules['Session'] = 'first'
        
        # Resample
        resampled = df.resample(timeframe, closed='left', label='left').agg(agg_rules)
        
        # Remove rows with no data (NaN in OHLC)
        resampled = resampled.dropna(subset=['Open', 'High', 'Low', 'Close'])
        
        # Reset index to get Datetime back as column
        resampled = resampled.reset_index()
        
        # Create Date column for compatibility
        resampled['Date'] = resampled['Datetime'].dt.date
        
        return resampled
    
    def load_single_file_with_resampling(self, uploaded_file, target_timeframe='10T'):
        """Load and resample a single file"""
        try:
            # Determine file type and load
            if uploaded_file.name.endswith('.csv'):
                df = pd.read_csv(uploaded_file)
            elif uploaded_file.name.endswith(('.xlsx', '.xls')):
                df = pd.read_excel(uploaded_file, header=0)
            else:
                raise ValueError(f"Unsupported file format: {uploaded_file.name}")
            
            # Standardize columns
            df = standardize_columns(df)
            
            # Validate required columns
            required_cols = ['Open', 'High', 'Low', 'Close']
            missing_cols = [col for col in required_cols if col not in df.columns]
            if missing_cols:
                raise ValueError(f"Missing required columns: {missing_cols}")
            
            # Resample if needed
            if target_timeframe != '1T':  # Don't resample if target is 1 minute
                df_resampled = self.resample_ohlc_data(df, target_timeframe)
            else:
                # Still need to ensure proper datetime handling
                if 'Datetime' not in df.columns:
                    if 'Date' in df.columns and 'Time' in df.columns:
                        df['Datetime'] = pd.to_datetime(df['Date'].astype(str) + ' ' + df['Time'].astype(str))
                    elif 'Date' in df.columns:
                        df['Datetime'] = pd.to_datetime(df['Date'])
                    else:
                        raise ValueError("Could not find datetime information")
                
                df['Date'] = pd.to_datetime(df['Datetime']).dt.date
                df_resampled = df
            
            return df_resampled
            
        except Exception as e:
            raise Exception(f"Error processing {uploaded_file.name}: {str(e)}")
    
    def load_multiple_files_with_checkpoints(self, uploaded_files, target_timeframe='10T'):
        """Load multiple files with checkpoint support"""
        
        # Check for existing checkpoint
        if self.checkpoint.state['processed_files'] or self.checkpoint.state['failed_files']:
            st.warning("🔄 **Previous Processing Session Found**")
            
            col1, col2 = st.columns(2)
            with col1:
                if self.checkpoint.state['processed_files']:
                    st.success(f"✅ Previously processed: {len(self.checkpoint.state['processed_files'])} files")
                    total_rows = sum(f.get('row_count', 0) for f in self.checkpoint.state['processed_files'])
                    st.info(f"📊 Total processed rows: {total_rows:,}")
            
            with col2:
                if self.checkpoint.state['failed_files']:
                    st.error(f"❌ Previous failures: {len(self.checkpoint.state['failed_files'])} files")
                    with st.expander("Show Failed Files"):
                        for failed in self.checkpoint.state['failed_files']:
                            st.error(f"• **{failed['filename']}**")
                            st.write(f"  Error: {failed['error']}")
            
            # Recovery options
            recovery_choice = st.radio(
                "How would you like to proceed?",
                [
                    "Continue from checkpoint (recommended)",
                    "Start fresh (clear all progress)", 
                    "Skip failed files and continue"
                ],
                key="recovery_choice"
            )
            
            if recovery_choice == "Start fresh (clear all progress)":
                self.checkpoint.clear_checkpoint()
                st.success("✅ Checkpoint cleared. Please re-run to start fresh.")
                return None
            elif recovery_choice == "Skip failed files and continue":
                failed_names = self.checkpoint.get_failed_filenames()
                uploaded_files = [f for f in uploaded_files if f.name not in failed_names]
                st.info(f"📋 Skipping {len(failed_names)} failed files")
        
        # Initialize or update checkpoint
        self.checkpoint.state['total_files'] = len(uploaded_files)
        self.checkpoint.state['target_timeframe'] = target_timeframe
        if not self.checkpoint.state['start_time']:
            self.checkpoint.state['start_time'] = datetime.now().isoformat()
        self.checkpoint.save_checkpoint()
        
        # Progress tracking
        progress_bar = st.progress(0)
        status_text = st.empty()
        detail_text = st.empty()
        
        processed_dataframes = []
        processed_filenames = self.checkpoint.get_processed_filenames()
        
        # Process files
        for i, uploaded_file in enumerate(uploaded_files):
            
            # Skip already processed files
            if uploaded_file.name in processed_filenames:
                status_text.text(f"✅ Skipping {uploaded_file.name} (already processed)")
                
                # Load from temp file
                temp_filename = os.path.join(self.checkpoint.temp_dir, f"{uploaded_file.name}_{target_timeframe}.parquet")
                if os.path.exists(temp_filename):
                    try:
                        df_temp = pd.read_parquet(temp_filename)
                        processed_dataframes.append(df_temp)
                        detail_text.text(f"📂 Loaded {len(df_temp):,} rows from cache")
                    except Exception as e:
                        st.warning(f"⚠️ Could not load cached data for {uploaded_file.name}: {e}")
                
                progress_bar.progress((i + 1) / len(uploaded_files))
                continue
            
            try:
                # Update status
                status_text.text(f"🔄 Processing {uploaded_file.name} ({i+1}/{len(uploaded_files)})...")
                detail_text.text(f"📁 Loading and resampling to {target_timeframe}...")
                
                # Process the file
                df_processed = self.load_single_file_with_resampling(uploaded_file, target_timeframe)
                
                detail_text.text(f"📊 Processed {len(df_processed):,} rows")
                
                # Save to temp file
                temp_filename = os.path.join(self.checkpoint.temp_dir, f"{uploaded_file.name}_{target_timeframe}.parquet")
                os.makedirs(self.checkpoint.temp_dir, exist_ok=True)
                df_processed.to_parquet(temp_filename)
                
                # Add to collection
                processed_dataframes.append(df_processed)
                
                # Update checkpoint
                self.checkpoint.mark_file_processed(
                    uploaded_file.name, 
                    temp_filename, 
                    len(df_processed)
                )
                
                # Show success
                st.success(f"✅ {uploaded_file.name}: {len(df_processed):,} bars ({target_timeframe})")
                
            except Exception as e:
                # Handle error
                error_msg = str(e)
                st.error(f"❌ **Error with {uploaded_file.name}**")
                st.error(f"Details: {error_msg}")
                
                self.checkpoint.mark_file_failed(uploaded_file.name, error_msg)
                
                # Give user choice
                error_choice = st.radio(
                    f"How to handle error with {uploaded_file.name}?",
                    ["Skip this file and continue", "Stop processing here"],
                    key=f"error_choice_{i}_{uploaded_file.name}"
                )
                
                if error_choice == "Stop processing here":
                    st.error("🛑 **Processing Stopped**")
                    st.info("💾 Progress has been saved. You can resume by re-running this section.")
                    return None
                else:
                    st.info(f"⏭️ Skipping {uploaded_file.name} and continuing...")
            
            # Update progress
            progress_bar.progress((i + 1) / len(uploaded_files))
        
        # Combine all processed data
        if processed_dataframes:
            status_text.text("🔗 Combining all processed files...")
            detail_text.text("📊 Sorting by datetime...")
            
            # Concatenate and sort
            combined_df = pd.concat(processed_dataframes, ignore_index=True)
            combined_df = combined_df.sort_values('Datetime').reset_index(drop=True)
            
            # Success summary
            total_rows = len(combined_df)
            date_range = f"{combined_df['Date'].min()} to {combined_df['Date'].max()}"
            
            st.success("🎉 **Multi-File Processing Complete!**")
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("📁 Files Processed", len(processed_dataframes))
            with col2:
                st.metric("📊 Total Bars", f"{total_rows:,}")
            with col3:
                st.metric("📅 Date Range", date_range)
            
            # Mark processing as complete and cleanup
            self.checkpoint.state['processing_complete'] = True
            self.checkpoint.save_checkpoint()
            
            # Clear progress indicators
            progress_bar.empty()
            status_text.empty()
            detail_text.empty()
            
            return combined_df
        
        else:
            st.warning("⚠️ No files were successfully processed")
            return None

# Modified load_intraday_data function to support both single and multiple files
def load_intraday_data_enhanced(uploaded_files, target_timeframe='10T'):
    """
    Enhanced intraday data loading that supports both single and multiple files
    """
    if not uploaded_files:
        return None
    
    # Handle single file upload (backward compatibility)
    if not isinstance(uploaded_files, list):
        uploaded_files = [uploaded_files]
    
    # Single file - use existing logic
    if len(uploaded_files) == 1:
        return load_intraday_data(uploaded_files[0])
    
    # Multiple files - use enhanced loader
    else:
        enhanced_loader = EnhancedDataLoader()
        return enhanced_loader.load_multiple_files_with_checkpoints(uploaded_files, target_timeframe)

def calculate_atr(df, period=14):
    """
    Calculate TRUE Wilder's ATR - ACTUALLY VALIDATED THIS TIME!
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
    
    # Convert date columns for comparison - handle different formats safely
    if 'Date' in daily_data.columns:
        daily_dates = pd.to_datetime(daily_data['Date'])
        if hasattr(daily_dates.iloc[0], 'date'):
            daily_dates = daily_dates.dt.date
    else:
        st.error("Daily data missing Date column")
        return False, ["Daily data missing Date column"], []
    
    if 'Date' in intraday_data.columns:
        intraday_dates = pd.to_datetime(intraday_data['Date'])
        if hasattr(intraday_dates.iloc[0], 'date'):
            intraday_dates = intraday_dates.dt.date
    elif 'Datetime' in intraday_data.columns:
        intraday_dates = pd.to_datetime(intraday_data['Datetime'])
        if hasattr(intraday_dates.iloc[0], 'date'):
            intraday_dates = intraday_dates.dt.date
    else:
        st.error("Intraday data missing Date/Datetime column")
        return False, ["Intraday data missing Date/Datetime column"], []
    
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

def download_yahoo_data_chunked(ticker, start_date, end_date, chunk_years=3, max_retries=3):
    """
    Download Yahoo Finance data in chunks with retry logic
    """
    import time
    
    # Ensure we have date objects, not datetime objects
    if hasattr(start_date, 'date'):
        start_date = start_date.date()
    if hasattr(end_date, 'date'):
        end_date = end_date.date()
    
    all_data = []
    current_start = start_date
    
    # Calculate total timespan for progress tracking
    total_days = (end_date - start_date).days
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    chunk_count = 0
    
    while current_start < end_date:
        chunk_count += 1
        
        # Calculate chunk end date (don't exceed final end_date)
        chunk_end_date = current_start + timedelta(days=chunk_years * 365)
        chunk_end = min(chunk_end_date, end_date)
        
        # Update progress
        days_processed = (current_start - start_date).days
        progress = min(days_processed / total_days, 1.0)
        progress_bar.progress(progress)
        status_text.text(f"📊 Downloading chunk {chunk_count}: {current_start} to {chunk_end}")
        
        # Try to download this chunk with retries
        chunk_data = None
        for attempt in range(max_retries):
            try:
                st.info(f"🔄 Attempt {attempt + 1}/{max_retries} for chunk {chunk_count}")
                
                chunk_data = yf.download(
                    ticker, 
                    start=current_start, 
                    end=chunk_end + timedelta(days=1),  # Add 1 day to ensure end date is included
                    interval='1d', 
                    progress=False
                )
                
                if not chunk_data.empty:
                    st.success(f"✅ Chunk {chunk_count} downloaded: {len(chunk_data)} records")
                    break
                else:
                    st.warning(f"⚠️ Chunk {chunk_count} returned empty data")
                    
            except Exception as e:
                st.warning(f"⚠️ Chunk {chunk_count}, attempt {attempt + 1} failed: {str(e)}")
                if attempt < max_retries - 1:
                    wait_time = (attempt + 1) * 2  # Progressive backoff: 2s, 4s, 6s
                    st.info(f"⏳ Waiting {wait_time} seconds before retry...")
                    time.sleep(wait_time)
        
        # Check if chunk download succeeded
        if chunk_data is None or chunk_data.empty:
            st.error(f"❌ Failed to download chunk {chunk_count} after {max_retries} attempts")
            
            # Give user choice
            user_choice = st.radio(
                f"How to handle failed chunk {chunk_count} ({current_start} to {chunk_end})?",
                ["Skip this chunk and continue", "Stop download and use partial data", "Retry with smaller chunks"],
                key=f"chunk_error_{chunk_count}"
            )
            
            if user_choice == "Skip this chunk and continue":
                st.warning(f"⏭️ Skipping chunk {chunk_count}, continuing with next chunk...")
                current_start = chunk_end
                continue
            elif user_choice == "Stop download and use partial data":
                st.warning("🛑 Stopping download, will use partial data collected so far")
                break
            else:  # Retry with smaller chunks
                st.info("🔄 Retrying with smaller chunk size...")
                smaller_chunk_years = max(1, chunk_years // 2)
                return download_yahoo_data_chunked(ticker, start_date, end_date, smaller_chunk_years, max_retries)
        
        else:
            # Successful chunk - add to collection
            all_data.append(chunk_data)
        
        # Move to next chunk
        current_start = chunk_end
        
        # Small delay to be nice to Yahoo Finance
        time.sleep(0.5)
    
    # Clear progress indicators
    progress_bar.empty()
    status_text.empty()
    
    # Combine all chunks
    if all_data:
        st.info("🔗 Combining all downloaded chunks...")
        combined_data = pd.concat(all_data, ignore_index=False)
        
        # Remove any duplicate dates that might occur at chunk boundaries
        combined_data = combined_data[~combined_data.index.duplicated(keep='first')]
        
        # Sort by date
        combined_data = combined_data.sort_index()
        
        # CRITICAL: Reset index to convert DatetimeIndex to Date column
        combined_data.reset_index(inplace=True)
        
        # Ensure Date column exists
        if 'Date' not in combined_data.columns and len(combined_data.columns) > 0:
            # First column should be the date after reset_index
            combined_data.rename(columns={combined_data.columns[0]: 'Date'}, inplace=True)
        
        st.success(f"✅ Chunked download complete: {len(combined_data)} total records")
        if 'Date' in combined_data.columns:
            # Safe date display
            try:
                date_min = combined_data['Date'].min()
                date_max = combined_data['Date'].max()
                if hasattr(date_min, 'date'):
                    date_min = date_min.date()
                if hasattr(date_max, 'date'):
                    date_max = date_max.date()
                st.info(f"📅 Final date range: {date_min} to {date_max}")
            except Exception as e:
                st.info(f"📅 Final dataset ready with {len(combined_data)} records")
        
        return combined_data
    else:
        st.error("❌ No data was successfully downloaded")
        return pd.DataFrame()

def load_daily_data(uploaded_file=None, ticker=None, start_date=None, end_date=None, intraday_data=None):
    """
    Load daily data from uploaded file or Yahoo Finance
    If using Yahoo Finance, automatically detect date range from intraday data
    NOW WITH CHUNKED DOWNLOAD SUPPORT
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
            
            # Validate required columns
            required_cols = ['Date', 'Open', 'High', 'Low', 'Close']
            missing_cols = [col for col in required_cols if col not in daily.columns]
            if missing_cols:
                st.error(f"Missing required columns in daily data: {missing_cols}")
                return None
            
            # Sort by date and ensure proper date format
            daily['Date'] = pd.to_datetime(daily['Date'])
            daily = daily.sort_values('Date').reset_index(drop=True)
            
            st.success(f"✅ Loaded daily data: {len(daily)} records")
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
                                          else intraday_data['Datetime'])
            
            # Handle different date formats safely
            if hasattr(intraday_dates.iloc[0], 'date'):
                # It's already a datetime, extract date
                intraday_start = intraday_dates.dt.date.min()
                intraday_end = intraday_dates.dt.date.max()
            else:
                # It's already a date
                intraday_start = intraday_dates.min()
                intraday_end = intraday_dates.max()
                
            # Ensure we have date objects for timedelta operations
            # Convert to date objects safely
            try:
                if hasattr(intraday_start, 'date'):
                    intraday_start = intraday_start.date()
                if hasattr(intraday_end, 'date'):
                    intraday_end = intraday_end.date()
            except:
                # If conversion fails, try pandas conversion
                intraday_start = pd.to_datetime(intraday_start).date()
                intraday_end = pd.to_datetime(intraday_end).date()
            
            # Calculate smart date range with buffer
            buffer_start = intraday_start - timedelta(days=180)  # 6 months buffer
            fetch_end = intraday_end + timedelta(days=5)  # Small buffer for end date
            
            st.info(f"📊 Intraday data spans: {intraday_start} to {intraday_end}")
            st.info(f"📈 Fetching Yahoo daily data for '{yahoo_ticker}' from {buffer_start} to {fetch_end}")
            
            # Try chunked download first for large date ranges
            date_span_years = (fetch_end - buffer_start).days / 365.25
            
            if date_span_years > 5:  # Use chunked download for > 5 years
                st.info(f"📊 Large date range detected ({date_span_years:.1f} years). Using chunked download...")
                daily_data = download_yahoo_data_chunked(yahoo_ticker, buffer_start, fetch_end)
            else:
                # Try regular download for smaller ranges
                st.info(f"📊 Standard download for {date_span_years:.1f} years...")
                try:
                    daily_data = yf.download(yahoo_ticker, start=buffer_start, end=fetch_end, interval='1d', progress=False)
                except Exception as e:
                    st.warning(f"⚠️ Standard download failed: {e}")
                    st.info("🔄 Falling back to chunked download...")
                    daily_data = download_yahoo_data_chunked(yahoo_ticker, buffer_start, fetch_end)
            
            if daily_data.empty:
                st.error(f"❌ No daily data found for '{yahoo_ticker}' in the calculated date range")
                
                # Suggest alternatives
                alternatives = TickerMapper.suggest_alternatives(original_ticker)
                if alternatives:
                    st.info("💡 Try these alternative ticker formats:")
                    for alt in alternatives:
                        st.info(f"   • {alt}")
                
                return None
            
            # CRITICAL: Yahoo Finance returns data with DatetimeIndex, not 'Date' column
            daily_data.reset_index(inplace=True)
            
            # Ensure Date column exists and is properly formatted
            if 'Date' not in daily_data.columns:
                # Yahoo data usually has the date as index, now as first column after reset_index
                if daily_data.index.name == 'Date' or 'Date' in str(daily_data.columns[0]).lower():
                    # First column should be the date after reset_index
                    daily_data.rename(columns={daily_data.columns[0]: 'Date'}, inplace=True)
                else:
                    st.error("❌ Could not find Date column in Yahoo Finance data")
                    st.info(f"Available columns: {list(daily_data.columns)}")
                    return None
            
            # Standardize column names (after ensuring Date exists)
            debug_info.append(f"🔧 DEBUG: Columns before standardize: {list(daily_data.columns)}")
            daily_data = standardize_columns(daily_data)
            debug_info.append(f"🔧 DEBUG: Columns after standardize: {list(daily_data.columns)}")
            
            st.success(f"✅ Auto-fetched daily data from Yahoo: {len(daily_data)} records")
            
            # Safe date range display
            try:
                date_min = daily_data['Date'].min()
                date_max = daily_data['Date'].max()
                if hasattr(date_min, 'date'):
                    date_min = date_min.date()
                if hasattr(date_max, 'date'):
                    date_max = date_max.date()
                st.success(f"📅 Daily data range: {date_min} to {date_max}")
            except Exception as e:
                st.warning(f"⚠️ Date range display error: {e}")
                st.success(f"📅 Daily data ready: {len(daily_data)} records")
            
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
            
            # Try chunked download for large date ranges
            date_span_years = (end_date - buffer_start).days / 365.25
            
            if date_span_years > 5:  # Use chunked download for > 5 years
                st.info(f"📊 Large date range detected ({date_span_years:.1f} years). Using chunked download...")
                daily_data = download_yahoo_data_chunked(ticker, buffer_start, end_date)
            else:
                # Try regular download for smaller ranges
                try:
                    daily_data = yf.download(ticker, start=buffer_start, end=end_date, interval='1d', progress=False)
                except Exception as e:
                    st.warning(f"⚠️ Standard download failed: {e}")
                    st.info("🔄 Falling back to chunked download...")
                    daily_data = download_yahoo_data_chunked(ticker, buffer_start, end_date)
            
            if daily_data.empty:
                st.error(f"No daily data found for {ticker}")
                return None
            
            # CRITICAL: Yahoo Finance returns data with DatetimeIndex, not 'Date' column
            daily_data.reset_index(inplace=True)
            
            # Ensure Date column exists and is properly formatted
            if 'Date' not in daily_data.columns:
                # Yahoo data usually has the date as index, now as first column after reset_index
                if daily_data.index.name == 'Date' or 'Date' in str(daily_data.columns[0]).lower():
                    # First column should be the date after reset_index
                    daily_data.rename(columns={daily_data.columns[0]: 'Date'}, inplace=True)
                else:
                    st.error("❌ Could not find Date column in Yahoo Finance data")
                    st.info(f"Available columns: {list(daily_data.columns)}")
                    return None
            
            # Standardize column names (after ensuring Date exists)
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
    Load intraday data from uploaded file with progress tracking
    """
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
            intraday = pd.read_csv(uploaded_file)
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
            return None
        
        status_text.text("Processing datetime...")
        progress_bar.progress(75)
        
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
        
        # Sort by datetime
        intraday = intraday.sort_values('Datetime').reset_index(drop=True)
        
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
        return None

def standardize_columns(df):
    """
    Standardize column names across different data sources
    """
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
            'last': 'Close',  # Common in some data providers
            'adj close': 'Close',
            'adjusted_close': 'Close',
            'settle': 'Close',  # Common in futures data
            # Volume
            'volume': 'Volume',
            'vol': 'Volume',
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
        st.error(f"❌ Error in standardize_columns: {e}")
        st.info(f"DataFrame columns: {list(df.columns)}")
        st.info(f"DataFrame shape: {df.shape}")
        return df  # Return original df if standardization fails

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
                
                # Check intraday candles for below trigger
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
                        # CHANGE: Allow same level for retracement testing
                        # if goal_level == trigger_level:  # Skip same level
                        #     continue
                        
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
                        
                        # ===== CRITICAL RETRACEMENT LOGIC - DO NOT MODIFY WITHOUT UNDERSTANDING =====
                        # RETRACEMENT GOALS: Cannot complete on same candle as trigger because we cannot
                        # determine intra-candle sequence (did goal hit before or after trigger?)
                        # CONTINUATION GOALS: Can complete on same candle (price continues same direction)
                        # 
                        # OPEN TRIGGER = opening price of 0930 candle specifically
                        # 0930 TIME = high/low of 0930 candle (different from OPEN price)
                        # ============================================================================
                        
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
                            # ===== CRITICAL INTRADAY RETRACEMENT LOGIC =====
                            # For CONTINUATION goals: Can check same candle as trigger
                            # For RETRACEMENT goals: MUST skip same candle as trigger
                            # For RETEST goals: MUST skip same candle as trigger (like retracement)
                            # Reason: Unknown intra-candle sequence for retracements and retests
                            # ===============================================
                            
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
                
                # Check intraday candles for above trigger (only if not already triggered at OPEN)
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
                        # CHANGE: Allow same level for retracement testing
                        # if goal_level == trigger_level:  # Skip same level
                        #     continue
                        
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
                            # ===== CRITICAL INTRADAY RETRACEMENT LOGIC =====
                            # For CONTINUATION goals: Can check same candle as trigger  
                            # For RETRACEMENT goals: MUST skip same candle as trigger
                            # For RETEST goals: MUST skip same candle as trigger (like retracement)
                            # Reason: Unknown intra-candle sequence for retracements and retests
                            # ===============================================
                            
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
    
    # Add zero-fill records for hours with no triggers (for troubleshooting)
    df_results = pd.DataFrame(results)
    if not df_results.empty:
        # Get all unique dates and combinations
        unique_dates = df_results['Date'].unique()
        
        # Create zero-fill records for missing hour combinations
        zero_fill_records = []
        for date in unique_dates:
            date_data = df_results[df_results['Date'] == date]
            existing_combinations = set()
            
            for _, row in date_data.iterrows():
                combo_key = f"{row['Direction']}_{row['TriggerLevel']}_{row['GoalLevel']}"
                existing_combinations.add(combo_key)
            
            # Find combinations that should exist but don't have records
            for direction in ['Above', 'Below']:
                for trigger_level in fib_levels:
                    for goal_level in fib_levels:
                        if trigger_level == goal_level:
                            continue
                        
                        combo_key = f"{direction}_{trigger_level}_{goal_level}"
                        if combo_key not in existing_combinations:
                            # Add zero record for troubleshooting
                            zero_fill_records.append({
                                'Date': date,
                                'Direction': direction,
                                'TriggerLevel': trigger_level,
                                'TriggerTime': '',
                                'TriggerPrice': 0.0,
                                'GoalLevel': goal_level,
                                'GoalPrice': 0.0,
                                'GoalHit': 'No',
                                'GoalTime': '',
                                'GoalClassification': 'No Trigger',
                                'PreviousClose': 0.0,
                                'PreviousATR': 0.0,
                                'SameTime': False,
                                'RetestedTrigger': 'No'
                            })
        
        if zero_fill_records:
            zero_df = pd.DataFrame(zero_fill_records)
            df_results = pd.concat([df_results, zero_df], ignore_index=True)

    return df_results

# ==============================================================================================
# END OF CRITICAL SECTION
# ==============================================================================================

def main_flexible(ticker=None, asset_type='STOCKS', daily_file=None, intraday_file=None, 
                 start_date=None, end_date=None, atr_period=14, custom_ratios=None, 
                 session_filter=None, extended_hours=False, intraday_data=None):
    """
    Main function for flexible ATR analysis with file inputs
    NOW USING THE SYSTEMATIC TRIGGER AND GOAL DETECTION FROM run_generate.py
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
        if intraday_data is None:
            if not intraday_file:
                debug_info.append("⚠️ No intraday data provided - analysis cannot proceed")
                return pd.DataFrame(), debug_info
            
            # Debug information
            debug_info.append(f"🔧 DEBUG: intraday_file type: {type(intraday_file)}")
            if hasattr(intraday_file, 'name'):
                debug_info.append(f"🔧 DEBUG: intraday_file.name: {intraday_file.name}")
            
            # Handle single file from Streamlit uploader (not a list)
            intraday = load_intraday_data(intraday_file)
        else:
            intraday = intraday_data
        
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
        
        # Calculate ATR using TRUE Wilder's method
        debug_info.append(f"🧮 Calculating ATR with TRUE Wilder's method, period {atr_period}...")
        daily = calculate_atr(daily, period=atr_period)
        
        # Validate ATR
        valid_atr = daily[daily['ATR'].notna()]
        if not valid_atr.empty:
            recent_atr = valid_atr['ATR'].tail(3).round(2).tolist()
            debug_info.append(f"ATR calculated successfully using TRUE Wilder's method. Recent values: {recent_atr}")
        else:
            debug_info.append("⚠️ No valid ATR values calculated")
            return pd.DataFrame(), debug_info
        
        # Check for session column
        if 'Session' in intraday.columns:
            unique_sessions = intraday['Session'].unique()
            debug_info.append(f"Session types found: {list(unique_sessions)}")
        
        # Run analysis using the SYSTEMATIC logic from run_generate.py
        debug_info.append("🎯 Running SYSTEMATIC trigger and goal detection (from run_generate.py)...")
        df = detect_triggers_and_goals_systematic(daily, intraday, custom_ratios)
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
            
            # Validation from run_generate.py
            same_time_count = len(df[df['SameTime'] == True])
            debug_info.append(f"✅ Same-time scenarios found: {same_time_count}")
            
            same_time_hits = len(df[(df['SameTime'] == True) & (df['GoalHit'] == 'Yes')])
            debug_info.append(f"✅ Same-time hits: {same_time_hits}")
            
            open_goals = len(df[df['GoalTime'] == 'OPEN'])
            debug_info.append(f"✅ Records with GoalTime=OPEN: {open_goals}")
            
            # Check cross-zero scenarios (corrected logic)
            cross_zero_below_to_above = len(df[(df['Direction'] == 'Below') & (df['GoalLevel'] > df['TriggerLevel'])])
            cross_zero_above_to_below = len(df[(df['Direction'] == 'Above') & (df['GoalLevel'] < df['TriggerLevel'])])
            debug_info.append(f"✅ Cross-zero scenarios - Below triggers to above goals: {cross_zero_below_to_above}")
            debug_info.append(f"✅ Cross-zero scenarios - Above triggers to below goals: {cross_zero_above_to_below}")
            
            open_triggers = len(df[df['TriggerTime'] == 'OPEN'])
            intraday_triggers = len(df[df['TriggerTime'] != 'OPEN'])
            debug_info.append(f"✅ OPEN triggers: {open_triggers}, Intraday triggers: {intraday_triggers}")
        
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
            st.write("This should match Excel calculations (TRUE Wilder's method)")
            
            # ATR trend chart
            atr_by_date = result_df.groupby('Date')['PreviousATR'].first().tail(20)
            if len(atr_by_date) > 1:
                st.line_chart(atr_by_date)
        
        # Show systematic validation metrics
        st.subheader('🔍 Systematic Logic Validation')
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
            output_filename = f'{ticker_clean}_{asset_type}_ATR_analysis_SYSTEMATIC_{datetime.now().strftime("%Y%m%d")}.csv'
            st.download_button(
                '📊 Download Full Results CSV',
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
            summary_filename = f'{ticker_clean}_{asset_type}_summary_SYSTEMATIC_{datetime.now().strftime("%Y%m%d")}.csv'
            st.download_button(
                '📋 Download Summary CSV',
                data=summary_df.to_csv(index=False),
                file_name=summary_filename,
                mime='text/csv'
            )
        
        st.success(f'🎉 Analysis complete for {ticker} using SYSTEMATIC logic from run_generate.py!')
        
    else:
        st.warning('⚠️ No results generated - check processing information above')

# Streamlit Interface
st.title('🎯 Advanced ATR Generator - Enhanced with SYSTEMATIC Logic')
st.write('**Now includes the validated systematic trigger/goal detection logic from run_generate.py**')
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
    accept_multiple_files=False,
    help="Upload single file - the system will process it with the systematic logic"
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

# Generate button section
if st.button('🚀 Generate Enhanced ATR Analysis with SYSTEMATIC Logic'):
    # First check if intraday files are uploaded
    if intraday_file is None:
        st.error("❌ Please upload intraday data file")
    elif data_source == "Upload Both Files":
        if daily_file is None:
            st.error("❌ Please upload daily data file")
        else:
            with st.spinner('Processing with SYSTEMATIC logic from run_generate.py...'):
                try:
                    result_df, debug_messages = main_flexible(
                        ticker=ticker or "UPLOADED_DATA",
                        asset_type=asset_type,
                        daily_file=daily_file,
                        intraday_file=intraday_file,  # Pass single file directly
                        atr_period=atr_period,
                        custom_ratios=custom_ratios,
                        session_filter=session_filter,
                        extended_hours=extended_hours
                    )
                    
                    display_results(result_df, debug_messages, ticker or "UPLOADED_DATA", asset_type, "Upload Both Files")
                        
                except Exception as e:
                    st.error(f'❌ Error: {e}')
                    import traceback
                    st.error(traceback.format_exc())
    
    elif data_source == "Yahoo Daily + Upload Intraday":
        if not ticker:
            st.error("❌ Please enter a ticker symbol for Yahoo Finance daily data")
        else:
            with st.spinner(f'Auto-detecting date range and processing with SYSTEMATIC logic...'):
                try:
                    result_df, debug_messages = main_flexible(
                        ticker=ticker,
                        asset_type=asset_type,
                        daily_file=None,
                        intraday_file=intraday_file,  # Pass single file directly
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
### 🔧 MAJOR ENHANCEMENT: SYSTEMATIC Logic Integration

#### 🎯 **Critical Logic from run_generate.py Now Included**
This enhanced version now includes the **validated systematic trigger and goal detection logic** from your `run_generate.py` file, with all the explicit comments and warnings preserved:

**Key Features Merged:**
- ✅ **TRUE Wilder's ATR calculation** (not pandas EMA!)
- ✅ **PERFECT SYSTEMATIC approach** - each level checked in both directions
- ✅ **FIXED 0930 candle goal completion logic**
- ✅ **Critical retracement logic** with intra-candle sequence handling
- ✅ **Explicit comments preserved** about not modifying core sections
- ✅ **Zero-fill records** for troubleshooting
- ✅ **Enhanced validation metrics** from the original logic

#### 🚨 **Protected Logic Sections**
The core trigger and goal detection logic is now protected with clear comments:
```python
# ==============================================================================================
# CRITICAL SECTION: DO NOT MODIFY THE CORE TRIGGER AND GOAL DETECTION LOGIC
# This section contains the validated systematic logic from run_generate.py
# ==============================================================================================
```

#### 📊 **Systematic Approach Preserved**
1. **Each trigger level checked in BOTH directions:**
   - LOW <= trigger (Below direction) → check all 12 goals
   - HIGH >= trigger (Above direction) → check all 12 goals

2. **Goal completion logic:**
   - Above goals: check HIGH >= goal
   - Below goals: check LOW <= goal

3. **CRITICAL retracement handling:**
   - CONTINUATION goals: Can complete on same candle as trigger
   - RETRACEMENT goals: MUST skip same candle as trigger
   - Reason: Unknown intra-candle sequence for retracements

4. **OPEN vs 0930 distinction:**
   - OPEN TRIGGER = opening price of 0930 candle specifically
   - 0930 TIME = high/low of 0930 candle (different from OPEN price)

#### 🔍 **Enhanced Validation Metrics**
The system now provides all the validation metrics from `run_generate.py`:
- Same-time scenarios found
- Same-time hits
- Records with GoalTime=OPEN
- Cross-zero scenarios (Below triggers to above goals, Above triggers to below goals)
- OPEN triggers vs Intraday triggers
- Zero-level trigger analysis

#### 📈 **Flexible Data Sources + Systematic Logic**
You now get the best of both worlds:
- **Flexible data input options** (Yahoo Finance auto-detection, file uploads, multi-asset support)
- **Systematic trigger/goal detection** (the exact logic that was working correctly)
- **Enhanced progress tracking** and validation
- **TRUE Wilder's ATR calculation** that matches Excel

#### ⚠️ **What Was Preserved**
All the critical comments and logic from `run_generate.py`:
- "DO NOT MODIFY WITHOUT UNDERSTANDING" sections
- Explicit retracement logic explanations
- OPEN vs intraday trigger handling
- Same-candle completion rules
- Zero-fill record generation for troubleshooting

#### 🎯 **Result Validation**
The output now includes systematic validation that matches `run_generate.py`:
- ATR values should match Excel calculations
- Same-time scenario counts
- Cross-zero scenario analysis
- OPEN trigger validation
- Goal classification breakdown (Continuation vs Retracement)

This merger ensures you get the **proven systematic logic** from `run_generate.py` while keeping all the **enhanced flexibility and features** from the newer flexible ATR generator.
""")
