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

# Streamlit Interface
st.title('📊 Enhanced CSV Data Handler')
st.write('**Combine multiple CSV files and resample to any timeframe you need**')

# Mode selection
mode = st.selectbox(
    "🎯 Choose Processing Mode",
    ["📁 Multi-CSV Processor", "📈 Public Data Download", "🔧 Single File Resampler"],
    help="Select what you want to do"
)

# ========================================================================================
# MULTI-CSV PROCESSOR (Main Feature)
# ========================================================================================
if mode == "📁 Multi-CSV Processor":
    st.header("📁 Multi-CSV Processor")
    st.write("**Upload multiple CSV files and combine them into one unified dataset**")
    
    # File upload - Make this prominent
    st.subheader("📤 File Upload")
    uploaded_files = st.file_uploader(
        "Choose Multiple CSV Files",
        type=['csv', 'xlsx', 'xls'],
        accept_multiple_files=True,
        help="Select multiple CSV/Excel files to combine and process",
        key="multi_csv_uploader"
    )
    
    # Show upload status
    if uploaded_files:
        st.success(f"✅ **{len(uploaded_files)} files uploaded successfully!**")
        
        # Show file list
        with st.expander("📋 Uploaded Files", expanded=True):
            for i, file in enumerate(uploaded_files, 1):
                st.write(f"{i}. **{file.name}** ({file.size:,} bytes)")
        
        st.markdown("---")
        
        # Configuration options
        st.subheader("⚙️ Processing Configuration")
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**🎯 Output Configuration**")
            
            # Target timeframe
            timeframe_type = st.radio(
                "Timeframe Type",
                ["Minutes/Hours", "Daily Aggregations"],
                help="Choose between minute-based or daily-based aggregations",
                key="timeframe_type_multi"
            )
            
            if timeframe_type == "Minutes/Hours":
                target_timeframe = st.selectbox(
                    "Target Timeframe",
                    ["1T", "2T", "5T", "10T", "15T", "30T", "1H", "2H", "4H"],
                    index=3,  # Default to 10T
                    help="T = minutes, H = hours"
                )
            else:
                target_timeframe = st.selectbox(
                    "Target Timeframe", 
                    ["WEEKLY", "MONTHLY", "QUARTERLY"],
                    help="Aggregate daily data into longer periods"
                )
        
        with col2:
            st.markdown("**⏰ Custom Time Filtering**")
            
            use_custom_time = st.checkbox(
                "Apply Custom Time Filter",
                help="Filter data to specific hours (e.g., market hours only)",
                key="use_custom_time_multi"
            )
            
            if use_custom_time:
                custom_start = st.time_input(
                    "Start Time",
                    value=time(9, 30),
                    help="Include data from this time onward",
                    key="custom_start_multi"
                )
                
                custom_end = st.time_input(
                    "End Time", 
                    value=time(16, 0),
                    help="Include data up to this time",
                    key="custom_end_multi"
                )
                
                custom_start_str = custom_start.strftime("%H:%M")
                custom_end_str = custom_end.strftime("%H:%M")
                
                st.info(f"📅 Will filter data to **{custom_start_str} - {custom_end_str}**")
            else:
                custom_start_str = None
                custom_end_str = None
        
        st.markdown("---")
        
        # Process button - Make this prominent
        if st.button("🚀 **Process Multiple CSVs**", type="primary", use_container_width=True):
            with st.spinner("Processing multiple CSV files..."):
                combined_data, file_info = CSVProcessor.process_multiple_csvs(
                    uploaded_files, 
                    target_timeframe,
                    custom_start_str,
                    custom_end_str
                )
                
                if combined_data is not None:
                    st.balloons()  # Celebration animation
                    st.success(f"🎉 **Successfully combined {len(uploaded_files)} files!**")
                    
                    # Show file processing summary
                    st.subheader("📋 Processing Summary")
                    summary_df = pd.DataFrame(file_info)
                    st.dataframe(summary_df, use_container_width=True)
                    
                    # Show combined data preview
                    st.subheader("📊 Combined Data Preview")
                    st.dataframe(combined_data.head(10), use_container_width=True)
                    
                    # Show summary metrics
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("Total Records", f"{len(combined_data):,}")
                    with col2:
                        st.metric("Date Range", f"{combined_data['Date'].min()} to {combined_data['Date'].max()}")
                    with col3:
                        st.metric("Timeframe", target_timeframe)
                    with col4:
                        if use_custom_time:
                            st.metric("Time Filter", f"{custom_start_str}-{custom_end_str}")
                        else:
                            st.metric("Time Filter", "None")
                    
                    # Download combined file - Make this prominent
                    st.markdown("---")
                    st.subheader("📥 Download Results")
                    
                    combined_filename = f"Combined_{target_timeframe}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
                    
                    st.download_button(
                        "📥 **Download Combined CSV**",
                        data=combined_data.to_csv(index=False),
                        file_name=combined_filename,
                        mime="text/csv",
                        key="download_combined",
                        use_container_width=True,
                        type="primary"
                    )
                    
                    st.success(f"✅ Ready to download: **{combined_filename}**")
                    
                else:
                    st.error("❌ Failed to process CSV files. Please check the file processing summary above.")
    
    else:
        # Show helpful instructions when no files are uploaded
        st.info("👆 **Please upload multiple CSV files to get started**")
        
        # Show example of what files should look like
        with st.expander("📋 Expected File Format", expanded=False):
            st.markdown("""
            **Your CSV files should contain these columns:**
            - **Date** (or Datetime, Time)
            - **Open**
            - **High** 
            - **Low**
            - **Close**
            - **Volume** (optional)
            
            **Example filenames that work well:**
            - `SPX_20240101.csv`
            - `AAPL_1min_data.csv`
            - `ES_intraday.csv`
            - `data_2024_01_01.csv`
            
            **The system will:**
            - ✅ Auto-detect ticker symbols from filenames
            - ✅ Warn if mixed tickers are found
            - ✅ Standardize column names automatically
            - ✅ Handle various date/time formats
            """)
        
        # Show sample workflow
        with st.expander("🔧 Sample Workflow", expanded=False):
            st.markdown("""
            **Example: Combine 25 daily 1-minute files into 10-minute bars**
            
            1. Upload 25 CSV files (e.g., from broker exports)
            2. Set timeframe to **10T** (10 minutes)
            3. Apply time filter **9:30 - 16:00** (market hours)
            4. Click **Process Multiple CSVs**
            5. Download single combined file ready for ATR analysis
            
            **Result:** 25 files → 1 clean file with 10-minute bars
            """)

# ========================================================================================
# PUBLIC DATA DOWNLOAD
# ========================================================================================
elif mode == "📈 Public Data Download":
    st.header("📈 Public Data Download")
    st.write("Download financial data from public sources and export as CSV")
    
    # Configuration in sidebar
    with st.sidebar:
        st.header("🎯 Download Configuration")
        
        # Ticker input
        ticker = st.text_input(
            "Ticker Symbol",
            value="SPX",
            help="Enter ticker symbol (e.g., SPX, AAPL, BTC)"
        ).upper()
        
        # Show ticker mapping
        if ticker:
            mapped_ticker = TickerMapper.get_public_ticker(ticker)
            if mapped_ticker != ticker:
                st.success(f"✅ Will map: {ticker} → {mapped_ticker}")
            else:
                st.info(f"📈 Will fetch: {ticker}")
        
        # Date range
        st.subheader("📅 Date Range")
        
        date_mode = st.radio(
            "Date Selection Mode",
            ["Smart ATR Range", "Custom Range"],
            help="Smart mode adds buffer for ATR calculation"
        )
        
        if date_mode == "Smart ATR Range":
            # Simple date range with automatic buffer
            intraday_start = st.date_input(
                "Intraday Analysis Start Date",
                value=date(2024, 1, 1),
                help="When you want your intraday analysis to begin"
            )
            
            intraday_end = st.date_input(
                "Intraday Analysis End Date", 
                value=date.today(),
                help="When you want your intraday analysis to end"
            )
            
            # Auto-calculate buffer
            buffer_months = st.slider("Buffer Months for Daily Data", 3, 12, 6)
            daily_start = intraday_start - timedelta(days=buffer_months * 30)
            daily_end = intraday_end + timedelta(days=5)
            
            st.info(f"📊 Daily data will span: {daily_start} to {daily_end}")
            st.info(f"📈 Buffer: {buffer_months} months before intraday start")
        
        else:
            # Manual date range
            daily_start = st.date_input("Daily Data Start Date", value=date(2023, 1, 1))
            daily_end = st.date_input("Daily Data End Date", value=date.today())
    
    st.info("⚠️ **Note:** Public sources have limitations. For extensive historical intraday data, use the Multi-CSV Processor with broker files.")
    
    if st.button("🚀 Download Daily Data", type="primary"):
        if not ticker:
            st.error("❌ Please enter a ticker symbol")
        else:
            mapped_ticker = TickerMapper.get_public_ticker(ticker)
            
            with st.spinner(f'Downloading daily data for {mapped_ticker}...'):
                try:
                    daily_data = yf.download(mapped_ticker, start=daily_start, end=daily_end, interval='1d', progress=False)
                    
                    if not daily_data.empty:
                        # Reset index and clean columns
                        daily_data.reset_index(inplace=True)
                        
                        # Handle MultiIndex columns
                        if isinstance(daily_data.columns, pd.MultiIndex):
                            daily_data.columns = daily_data.columns.get_level_values(0)
                        
                        # Ensure Date column
                        if 'Date' not in daily_data.columns and len(daily_data.columns) > 0:
                            daily_data.rename(columns={daily_data.columns[0]: 'Date'}, inplace=True)
                        
                        st.success(f"✅ Downloaded {len(daily_data)} daily records")
                        
                        # Show preview
                        st.subheader("📋 Data Preview")
                        st.dataframe(daily_data.head(), use_container_width=True)
                        
                        # Download button
                        filename = f"{ticker}_daily_{daily_start.strftime('%Y%m%d')}_to_{daily_end.strftime('%Y%m%d')}.csv"
                        st.download_button(
                            "📥 Download Daily CSV",
                            data=daily_data.to_csv(index=False),
                            file_name=filename,
                            mime="text/csv"
                        )
                    else:
                        st.error("❌ No data available for this ticker/range")
                        
                        # Suggest alternatives
                        alternatives = TickerMapper.suggest_alternatives(ticker)
                        if alternatives:
                            st.info("💡 Try these alternative formats:")
                            for alt in alternatives:
                                st.info(f"   • {alt}")
                                
                except Exception as e:
                    st.error(f"❌ Download failed: {str(e)}")

# ========================================================================================
# SINGLE FILE RESAMPLER
# ========================================================================================
elif mode == "🔧 Single File Resampler":
    st.header("🔧 Single File Resampler")
    st.write("**Upload a single CSV and resample it to different timeframes**")
    
    # Single file upload
    single_file = st.file_uploader(
        "Upload Single CSV File",
        type=['csv', 'xlsx', 'xls'], 
        help="Upload one CSV/Excel file to resample"
    )
    
    if single_file:
        st.success(f"✅ File uploaded: {single_file.name}")
        
        # Load and preview the file
        try:
            if single_file.name.endswith('.csv'):
                df = pd.read_csv(single_file)
            else:
                df = pd.read_excel(single_file)
            
            df = CSVProcessor.standardize_columns(df)
            
            st.subheader("📋 Original Data Preview")
            st.dataframe(df.head(), use_container_width=True)
            st.info(f"Original data: {len(df)} records")
            
            # Detected ticker
            detected_ticker = CSVProcessor.detect_ticker_from_content(df) or CSVProcessor.detect_ticker_from_filename(single_file.name)
            if detected_ticker:
                st.info(f"🏷️ Detected ticker: **{detected_ticker}**")
            
            # Resampling options
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("🎯 Resampling Options")
                
                # Standard timeframes
                timeframe_category = st.selectbox(
                    "Timeframe Category",
                    ["Minutes", "Hours", "Daily Aggregations"]
                )
                
                if timeframe_category == "Minutes":
                    resample_timeframe = st.selectbox(
                        "Target Timeframe",
                        ["1T", "2T", "5T", "10T", "15T", "30T"],
                        index=3
                    )
                elif timeframe_category == "Hours":
                    resample_timeframe = st.selectbox(
                        "Target Timeframe",
                        ["1H", "2H", "3H", "4H", "6H", "8H", "12H"]
                    )
                else:  # Daily Aggregations
                    resample_timeframe = st.selectbox(
                        "Target Timeframe",
                        ["WEEKLY", "MONTHLY", "QUARTERLY"]
                    )
            
            with col2:
                st.subheader("⚙️ Time Filtering")
                
                # Time filtering
                apply_time_filter = st.checkbox("Apply Time Filter")
                
                if apply_time_filter:
                    filter_start = st.time_input("Filter Start Time", value=time(9, 30))
                    filter_end = st.time_input("Filter End Time", value=time(16, 0))
                    
                    filter_start_str = filter_start.strftime("%H:%M")
                    filter_end_str = filter_end.strftime("%H:%M")
                    
                    st.info(f"📅 Time filter: {filter_start_str} - {filter_end_str}")
                else:
                    filter_start_str = None
                    filter_end_str = None
            
            # Process button
            if st.button("🔄 Resample Data", type="primary"):
                try:
                    with st.spinner("Resampling data..."):
                        resampled_data = CSVProcessor.resample_ohlc_data(
                            df, resample_timeframe, filter_start_str, filter_end_str
                        )
                        
                        st.success(f"✅ Resampled: {len(df)} → {len(resampled_data)} records")
                        
                        # Show resampled preview
                        st.subheader("📊 Resampled Data Preview")
                        st.dataframe(resampled_data.head(), use_container_width=True)
                        
                        # Summary metrics
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Original Records", len(df))
                        with col2:
                            st.metric("Resampled Records", len(resampled_data))
                        with col3:
                            compression_ratio = (1 - len(resampled_data) / len(df)) * 100
                            st.metric("Compression", f"{compression_ratio:.1f}%")
                        
                        # Download resampled file
                        base_name = single_file.name.split('.')[0]
                        resampled_filename = f"{base_name}_resampled_{resample_timeframe}.csv"
                        
                        st.download_button(
                            "📥 Download Resampled CSV",
                            data=resampled_data.to_csv(index=False),
                            file_name=resampled_filename,
                            mime="text/csv"
                        )
                        
                except Exception as e:
                    st.error(f"❌ Resampling failed: {str(e)}")
            
        except Exception as e:
            st.error(f"❌ Error loading file: {str(e)}")

# Help section
st.markdown("---")
st.subheader("📚 Usage Guide")

st.markdown("""
**🎯 Multi-CSV Processor** (Recommended)
- Perfect for combining broker data files
- Upload 25+ 1-minute CSV files → Get 1 combined 10-minute file
- Smart ticker detection and validation
- Custom time filtering for market hours

**📈 Public Data Download**
- Download from public sources (limited intraday history)
- Good for daily data with ATR buffers
- Auto-maps common tickers (SPX → ^GSPC)

**🔧 Single File Resampler**
- Transform one file to different timeframes
- Convert 1-minute → 10-minute, daily → weekly, etc.
- Apply custom time filters

💾 **Next Step:** Use processed files in the ATR Analysis tool!
""")
