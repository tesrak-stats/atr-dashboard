import streamlit as st
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta, date, time
import time as time_module
import os
import tempfile
import zipfile
from io import BytesIO

class TickerMapper:
    """Handle ticker symbol mappings for different data sources"""
    
    @staticmethod
    def get_public_ticker(input_ticker):
        """Convert common ticker variations to public data source format"""
        
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

class CSVProcessor:
    """Handle CSV file processing and combination"""
    
    @staticmethod
    def detect_ticker_from_filename(filename):
        """Try to detect ticker from filename"""
        # Common patterns in filenames
        filename_upper = filename.upper()
        
        # Remove common file extensions and patterns
        clean_name = filename_upper.replace('.CSV', '').replace('.XLSX', '').replace('.XLS', '')
        
        # Try to extract ticker patterns
        import re
        
        # Pattern 1: Ticker at start (e.g., "SPX_data.csv", "AAPL_1min.csv")
        match = re.match(r'^([A-Z^=\-]{2,6})', clean_name)
        if match:
            return match.group(1)
        
        # Pattern 2: Ticker in middle (e.g., "data_SPX_2024.csv")
        match = re.search(r'_([A-Z^=\-]{2,6})_', clean_name)
        if match:
            return match.group(1)
        
        return None
    
    @staticmethod
    def detect_ticker_from_content(df):
        """Try to detect ticker from DataFrame content"""
        # Look for ticker/symbol columns
        ticker_columns = ['ticker', 'symbol', 'instrument', 'asset']
        
        for col in df.columns:
            if col.lower() in ticker_columns:
                # Get the most common value
                ticker_value = df[col].mode()
                if not ticker_value.empty:
                    return str(ticker_value.iloc[0]).upper()
        
        return None
    
    @staticmethod
    def standardize_columns(df):
        """Standardize column names across different CSV formats"""
        # Create a copy to avoid modifying original
        df = df.copy()
        
        # Clean column names
        df.columns = [str(col).strip() for col in df.columns]
        
        # Common column mappings
        column_mappings = {
            # Date/Time columns
            'date': 'Date',
            'time': 'Time',
            'datetime': 'Datetime',
            'timestamp': 'Datetime',
            'date_time': 'Datetime',
            
            # OHLC columns
            'open': 'Open',
            'high': 'High',
            'low': 'Low',
            'close': 'Close',
            'last': 'Close',
            'settle': 'Close',
            'adj_close': 'Close',
            'adjusted_close': 'Close',
            
            # Volume
            'volume': 'Volume',
            'vol': 'Volume',
            'size': 'Volume',
            
            # Other
            'symbol': 'Ticker',
            'instrument': 'Ticker',
            'asset': 'Ticker'
        }
        
        # Apply mappings (case insensitive)
        for old_name, new_name in column_mappings.items():
            for col in df.columns:
                if col.lower() == old_name:
                    df.rename(columns={col: new_name}, inplace=True)
                    break
        
        return df
    
    @staticmethod
    def create_datetime_column(df):
        """Create a proper Datetime column from available date/time info"""
        if 'Datetime' in df.columns:
            df['Datetime'] = pd.to_datetime(df['Datetime'])
            return df
        
        if 'Date' in df.columns and 'Time' in df.columns:
            # Combine Date and Time
            df['Datetime'] = pd.to_datetime(df['Date'].astype(str) + ' ' + df['Time'].astype(str))
        elif 'Date' in df.columns:
            # Use Date as Datetime
            df['Datetime'] = pd.to_datetime(df['Date'])
        else:
            raise ValueError("Could not find date/time information in CSV")
        
        return df
    
    @staticmethod
    def resample_ohlc_data(df, target_timeframe, custom_start_time=None, custom_end_time=None):
        """Resample OHLC data to target timeframe with optional custom time filtering"""
        df = df.copy()
        
        # Ensure we have a Datetime column
        df = CSVProcessor.create_datetime_column(df)
        
        # Apply custom time filtering if specified
        if custom_start_time and custom_end_time:
            df['Time_obj'] = df['Datetime'].dt.time
            start_time = pd.to_datetime(custom_start_time, format='%H:%M').time()
            end_time = pd.to_datetime(custom_end_time, format='%H:%M').time()
            
            # Filter by time range
            time_mask = (df['Time_obj'] >= start_time) & (df['Time_obj'] <= end_time)
            df = df[time_mask]
            df.drop('Time_obj', axis=1, inplace=True)
            
            if df.empty:
                raise ValueError(f"No data found in time range {custom_start_time} to {custom_end_time}")
        
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
        
        # Add other columns if present
        for col in df.columns:
            if col not in agg_rules and col not in ['Open', 'High', 'Low', 'Close', 'Volume']:
                agg_rules[col] = 'first'  # Keep first value for other columns
        
        # Handle different timeframe formats
        if target_timeframe.upper() == 'WEEKLY':
            resampled = df.resample('W', closed='left', label='left').agg(agg_rules)
        elif target_timeframe.upper() == 'MONTHLY':
            resampled = df.resample('M', closed='left', label='left').agg(agg_rules)
        elif target_timeframe.upper() == 'QUARTERLY':
            resampled = df.resample('Q', closed='left', label='left').agg(agg_rules)
        else:
            # Standard minute-based resampling (e.g., '10T', '30T', '1H')
            resampled = df.resample(target_timeframe, closed='left', label='left').agg(agg_rules)
        
        # Remove rows with no data
        resampled = resampled.dropna(subset=['Open', 'High', 'Low', 'Close'])
        
        # Reset index to get Datetime back as column
        resampled = resampled.reset_index()
        
        # Create Date column
        resampled['Date'] = resampled['Datetime'].dt.date
        
        return resampled
    
    @staticmethod
    def process_multiple_csvs(uploaded_files, target_timeframe, custom_start_time=None, custom_end_time=None):
        """Process multiple CSV files and combine them"""
        all_dataframes = []
        detected_tickers = set()
        file_info = []
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        for i, uploaded_file in enumerate(uploaded_files):
            try:
                status_text.text(f"Processing {uploaded_file.name} ({i+1}/{len(uploaded_files)})...")
                
                # Load the CSV
                if uploaded_file.name.endswith('.csv'):
                    df = pd.read_csv(uploaded_file)
                elif uploaded_file.name.endswith(('.xlsx', '.xls')):
                    df = pd.read_excel(uploaded_file)
                else:
                    st.warning(f"Skipping {uploaded_file.name} - unsupported format")
                    continue
                
                # Standardize columns
                df = CSVProcessor.standardize_columns(df)
                
                # Detect ticker
                ticker_from_filename = CSVProcessor.detect_ticker_from_filename(uploaded_file.name)
                ticker_from_content = CSVProcessor.detect_ticker_from_content(df)
                
                detected_ticker = ticker_from_content or ticker_from_filename or "UNKNOWN"
                detected_tickers.add(detected_ticker)
                
                # Validate required columns
                required_cols = ['Open', 'High', 'Low', 'Close']
                missing_cols = [col for col in required_cols if col not in df.columns]
                if missing_cols:
                    st.error(f"❌ {uploaded_file.name} missing columns: {missing_cols}")
                    continue
                
                # Resample data
                df_resampled = CSVProcessor.resample_ohlc_data(
                    df, target_timeframe, custom_start_time, custom_end_time
                )
                
                # Add source info
                df_resampled['Source_File'] = uploaded_file.name
                df_resampled['Detected_Ticker'] = detected_ticker
                
                all_dataframes.append(df_resampled)
                
                file_info.append({
                    'filename': uploaded_file.name,
                    'original_rows': len(df),
                    'resampled_rows': len(df_resampled),
                    'detected_ticker': detected_ticker,
                    'date_range': f"{df_resampled['Date'].min()} to {df_resampled['Date'].max()}"
                })
                
                st.success(f"✅ {uploaded_file.name}: {len(df)} → {len(df_resampled)} rows ({detected_ticker})")
                
            except Exception as e:
                st.error(f"❌ Error processing {uploaded_file.name}: {str(e)}")
                continue
            
            progress_bar.progress((i + 1) / len(uploaded_files))
        
        # Clear progress indicators
        progress_bar.empty()
        status_text.empty()
        
        # Check for ticker consistency
        if len(detected_tickers) > 1:
            st.warning(f"⚠️ **Multiple tickers detected:** {', '.join(detected_tickers)}")
            st.warning("This may indicate mixed data from different instruments!")
            
            # Let user decide how to proceed
            ticker_choice = st.radio(
                "How would you like to handle multiple tickers?",
                ["Continue anyway (combine all data)", "Cancel and review files"],
                key="ticker_choice"
            )
            
            if ticker_choice == "Cancel and review files":
                return None, file_info
        
        # Combine all dataframes
        if all_dataframes:
            combined_df = pd.concat(all_dataframes, ignore_index=True)
            combined_df = combined_df.sort_values('Datetime').reset_index(drop=True)
            
            # Remove source columns from final output (keep for debugging)
            output_df = combined_df.drop(['Source_File', 'Detected_Ticker'], axis=1, errors='ignore')
            
            return output_df, file_info
        else:
            return None, file_info
    """Handle ticker symbol mappings for different data sources"""
    
    @staticmethod
    def get_public_ticker(input_ticker):
        """Convert common ticker variations to public data source format"""
        
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

def clean_column_names(df):
    """Clean and standardize column names"""
    # Handle MultiIndex columns (common with yfinance)
    if isinstance(df.columns, pd.MultiIndex):
        st.info("🔧 Detected MultiIndex columns, flattening...")
        # Flatten MultiIndex columns - keep only the first level (OHLC names)
        df.columns = df.columns.get_level_values(0)
    
    # Ensure Date column exists after reset_index
    if 'Date' not in df.columns:
        # Check for alternative column names
        potential_date_cols = [col for col in df.columns if 'date' in str(col).lower()]
        if potential_date_cols:
            df.rename(columns={potential_date_cols[0]: 'Date'}, inplace=True)
            st.info(f"✅ Renamed '{potential_date_cols[0]}' to 'Date'")
        elif len(df.columns) > 0:
            # First column is usually the date after reset_index
            df.rename(columns={df.columns[0]: 'Date'}, inplace=True)
            st.info(f"✅ Renamed first column to 'Date'")
    
    return df

def download_data_chunked(ticker, start_date, end_date, chunk_years=3, max_retries=3):
    """Download data in chunks with retry logic"""
    
    # Ensure we have date objects
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
        
        # Calculate chunk end date
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
                chunk_data = yf.download(
                    ticker, 
                    start=current_start, 
                    end=chunk_end + timedelta(days=1),
                    interval='1d', 
                    progress=False
                )
                
                if not chunk_data.empty:
                    st.success(f"✅ Chunk {chunk_count}: {len(chunk_data)} records")
                    break
                else:
                    st.warning(f"⚠️ Chunk {chunk_count} returned empty data")
                    
            except Exception as e:
                st.warning(f"⚠️ Chunk {chunk_count}, attempt {attempt + 1} failed: {str(e)}")
                if attempt < max_retries - 1:
                    wait_time = (attempt + 1) * 2
                    time.sleep(wait_time)
        
        # Handle failed chunk
        if chunk_data is None or chunk_data.empty:
            st.error(f"❌ Failed to download chunk {chunk_count} after {max_retries} attempts")
            
            user_choice = st.radio(
                f"How to handle failed chunk {chunk_count}?",
                ["Skip this chunk and continue", "Stop download", "Retry with smaller chunks"],
                key=f"chunk_error_{chunk_count}"
            )
            
            if user_choice == "Skip this chunk and continue":
                st.warning(f"⏭️ Skipping chunk {chunk_count}")
                current_start = chunk_end
                continue
            elif user_choice == "Stop download":
                st.warning("🛑 Stopping download")
                break
            else:  # Retry with smaller chunks
                st.info("🔄 Retrying with smaller chunks...")
                smaller_chunk_years = max(1, chunk_years // 2)
                return download_data_chunked(ticker, start_date, end_date, smaller_chunk_years, max_retries)
        else:
            all_data.append(chunk_data)
        
        current_start = chunk_end
        time.sleep(0.5)  # Be respectful to the API
    
    # Clear progress indicators
    progress_bar.empty()
    status_text.empty()
    
    # Combine all chunks
    if all_data:
        st.info("🔗 Combining all chunks...")
        combined_data = pd.concat(all_data, ignore_index=False)
        combined_data = combined_data[~combined_data.index.duplicated(keep='first')]
        combined_data = combined_data.sort_index()
        
        # Reset index and clean columns
        combined_data.reset_index(inplace=True)
        combined_data = clean_column_names(combined_data)
        
        st.success(f"✅ Download complete: {len(combined_data)} total records")
        return combined_data
    else:
        st.error("❌ No data was successfully downloaded")
        return pd.DataFrame()

def download_single_request(ticker, start_date, end_date):
    """Download data in a single request"""
    try:
        data = yf.download(ticker, start=start_date, end=end_date, interval='1d', progress=False)
        
        if data.empty:
            return pd.DataFrame()
        
        # Reset index and clean columns
        data.reset_index(inplace=True)
        data = clean_column_names(data)
        
        return data
        
    except Exception as e:
        st.error(f"Download failed: {str(e)}")
        return pd.DataFrame()

def validate_data(df, data_type="data"):
    """Validate downloaded data"""
    if df.empty:
        st.error(f"❌ No {data_type} was downloaded")
        return False
    
    # Check for required columns
    required_cols = ['Date', 'Open', 'High', 'Low', 'Close']
    missing_cols = [col for col in required_cols if col not in df.columns]
    
    if missing_cols:
        st.error(f"❌ Missing required columns in {data_type}: {missing_cols}")
        st.info(f"Available columns: {list(df.columns)}")
        return False
    
    # Check for data quality
    null_counts = df[required_cols].isnull().sum()
    if null_counts.sum() > 0:
        st.warning(f"⚠️ Found missing values in {data_type}:")
        for col, count in null_counts.items():
            if count > 0:
                st.warning(f"  {col}: {count} missing values")
    
    return True

def generate_filename(ticker, data_type, start_date, end_date):
    """Generate a descriptive filename"""
    ticker_clean = ticker.replace("^", "").replace("=", "_").replace("-", "_")
    start_str = start_date.strftime("%Y%m%d") if hasattr(start_date, 'strftime') else str(start_date).replace("-", "")
    end_str = end_date.strftime("%Y%m%d") if hasattr(end_date, 'strftime') else str(end_date).replace("-", "")
    
    return f"{ticker_clean}_{data_type}_{start_str}_to_{end_str}.csv"

# Streamlit Interface
st.title('📊 CSV Data Handler - Public Source to CSV')
st.write('**Download financial data from public sources and export as clean CSV files**')
st.write('**Perfect for preparing data for the ATR Analysis tool**')

# Sidebar configuration
st.sidebar.header("🎯 Data Configuration")

# Ticker input
ticker = st.sidebar.text_input(
    "Ticker Symbol",
    value="SPX",
    help="Enter ticker symbol (e.g., SPX, AAPL, BTC)"
).upper()

# Show ticker mapping
if ticker:
    mapped_ticker = TickerMapper.get_public_ticker(ticker)
    if mapped_ticker != ticker:
        st.sidebar.success(f"✅ Will map: {ticker} → {mapped_ticker}")
    else:
        st.sidebar.info(f"📈 Will fetch: {ticker}")

# Date range
st.sidebar.subheader("📅 Date Range")

date_mode = st.sidebar.radio(
    "Date Selection Mode",
    ["Smart ATR Range", "Custom Range"],
    help="Smart mode adds buffer for ATR calculation"
)

if date_mode == "Smart ATR Range":
    # Simple date range with automatic buffer
    intraday_start = st.sidebar.date_input(
        "Intraday Analysis Start Date",
        value=date(2024, 1, 1),
        help="When you want your intraday analysis to begin"
    )
    
    intraday_end = st.sidebar.date_input(
        "Intraday Analysis End Date", 
        value=date.today(),
        help="When you want your intraday analysis to end"
    )
    
    # Auto-calculate buffer
    buffer_months = st.sidebar.slider("Buffer Months for Daily Data", 3, 12, 6)
    daily_start = intraday_start - timedelta(days=buffer_months * 30)
    daily_end = intraday_end + timedelta(days=5)
    
    st.sidebar.info(f"📊 Daily data will span: {daily_start} to {daily_end}")
    st.sidebar.info(f"📈 Buffer: {buffer_months} months before intraday start")

else:
    # Manual date range
    daily_start = st.sidebar.date_input("Daily Data Start Date", value=date(2023, 1, 1))
    daily_end = st.sidebar.date_input("Daily Data End Date", value=date.today())
    intraday_start = daily_start
    intraday_end = daily_end

# Download options
st.sidebar.subheader("⚙️ Download Options")

download_mode = st.sidebar.radio(
    "Download Method",
    ["Single Request", "Chunked Download"],
    help="Use chunked for large date ranges (>5 years)"
)

if download_mode == "Chunked Download":
    chunk_size = st.sidebar.slider("Chunk Size (Years)", 1, 5, 3)

# Main interface
col1, col2 = st.columns(2)

with col1:
    st.subheader("📈 Daily Data Download")
    
    if st.button("🚀 Download Daily Data", type="primary"):
        if not ticker:
            st.error("❌ Please enter a ticker symbol")
        else:
            mapped_ticker = TickerMapper.get_public_ticker(ticker)
            
            with st.spinner(f'Downloading daily data for {mapped_ticker}...'):
                # Calculate date span
                date_span_years = (daily_end - daily_start).days / 365.25
                
                if download_mode == "Chunked Download" or date_span_years > 5:
                    daily_data = download_data_chunked(mapped_ticker, daily_start, daily_end, chunk_size)
                else:
                    daily_data = download_single_request(mapped_ticker, daily_start, daily_end)
                
                if validate_data(daily_data, "daily data"):
                    st.success(f"✅ Downloaded {len(daily_data)} daily records")
                    
                    # Show preview
                    st.subheader("📋 Data Preview")
                    st.dataframe(daily_data.head(), use_container_width=True)
                    
                    # Show summary
                    col1a, col2a, col3a = st.columns(3)
                    with col1a:
                        st.metric("Records", len(daily_data))
                    with col2a:
                        st.metric("Date Range", f"{daily_data['Date'].min().date()} to {daily_data['Date'].max().date()}")
                    with col3a:
                        st.metric("Columns", len(daily_data.columns))
                    
                    # Download button
                    filename = generate_filename(ticker, "daily", daily_start, daily_end)
                    st.download_button(
                        "📥 Download Daily CSV",
                        data=daily_data.to_csv(index=False),
                        file_name=filename,
                        mime="text/csv"
                    )
                else:
                    # Suggest alternatives
                    alternatives = TickerMapper.suggest_alternatives(ticker)
                    if alternatives:
                        st.info("💡 Try these alternative formats:")
                        for alt in alternatives:
                            st.info(f"   • {alt}")

with col2:
    st.subheader("📊 Intraday Data Download")
    st.info("⚠️ **Note:** Most public sources don't provide sufficient intraday history. This is mainly for short-term data or testing.")
    
    intraday_interval = st.selectbox(
        "Intraday Interval",
        ["1m", "2m", "5m", "15m", "30m", "60m", "90m"],
        index=4,  # Default to 30m
        help="Shorter intervals have more limited history"
    )
    
    if st.button("🚀 Download Intraday Data"):
        if not ticker:
            st.error("❌ Please enter a ticker symbol")
        else:
            mapped_ticker = TickerMapper.get_public_ticker(ticker)
            
            # Limit intraday range (most sources only give ~60 days of intraday)
            max_intraday_days = 60
            if (intraday_end - intraday_start).days > max_intraday_days:
                st.warning(f"⚠️ Intraday range limited to {max_intraday_days} days due to data source restrictions")
                intraday_start_limited = intraday_end - timedelta(days=max_intraday_days)
            else:
                intraday_start_limited = intraday_start
            
            with st.spinner(f'Downloading intraday data for {mapped_ticker}...'):
                try:
                    intraday_data = yf.download(
                        mapped_ticker, 
                        start=intraday_start_limited, 
                        end=intraday_end,
                        interval=intraday_interval,
                        progress=False
                    )
                    
                    if not intraday_data.empty:
                        # Reset index and clean columns
                        intraday_data.reset_index(inplace=True)
                        intraday_data = clean_column_names(intraday_data)
                        
                        # Create Datetime column for intraday data
                        if 'Datetime' not in intraday_data.columns and 'Date' in intraday_data.columns:
                            intraday_data['Datetime'] = intraday_data['Date']
                        
                        if validate_data(intraday_data, "intraday data"):
                            st.success(f"✅ Downloaded {len(intraday_data)} intraday records")
                            
                            # Show preview
                            st.subheader("📋 Data Preview")
                            st.dataframe(intraday_data.head(), use_container_width=True)
                            
                            # Show summary
                            col1b, col2b = st.columns(2)
                            with col1b:
                                st.metric("Records", len(intraday_data))
                            with col2b:
                                st.metric("Interval", intraday_interval)
                            
                            # Download button
                            filename = generate_filename(ticker, f"intraday_{intraday_interval}", intraday_start_limited, intraday_end)
                            st.download_button(
                                "📥 Download Intraday CSV",
                                data=intraday_data.to_csv(index=False),
                                file_name=filename,
                                mime="text/csv"
                            )
                    else:
                        st.error("❌ No intraday data available for this ticker/range")
                        
                except Exception as e:
                    st.error(f"❌ Intraday download failed: {str(e)}")

# Help section
st.markdown("---")
st.subheader("📚 Usage Guide")

col1, col2 = st.columns(2)

with col1:
    st.markdown("""
    **🎯 For ATR Analysis:**
    1. Set your desired intraday analysis period
    2. Use "Smart ATR Range" with 6+ month buffer
    3. Download daily data first
    4. Download intraday data (if available)
    5. Use both CSVs in the ATR Analysis tool
    
    **📊 Recommended Settings:**
    - Buffer: 6-12 months for ATR calculation
    - Download mode: Chunked for large ranges
    - Daily data: Always available
    - Intraday: Limited to ~60 days from most sources
    """)

with col2:
    st.markdown("""
    **💡 Tips:**
    - **SPX** maps to **^GSPC** automatically
    - **Large date ranges** use chunked download
    - **Intraday data** is limited by public sources
    - **For extensive intraday data**, use broker exports
    - **CSV files** are ready for ATR Analysis tool
    
    **🔧 Supported Assets:**
    - Stocks (AAPL, GOOGL, SPY)
    - Indices (SPX, NDX, DJI) 
    - Crypto (BTC, ETH)
    - Forex (EURUSD, GBPUSD)
    - Futures (ES, NQ, CL)
    """)

st.info("💾 **Next Step:** Use the downloaded CSV files in the ATR Analysis tool for systematic trigger/goal detection!")
