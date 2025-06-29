import streamlit as st
import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
import io
# Output interval selection (common for both data sources)
output_intervals = {
    "1 minute": 1,
    "2 minutes": 2,
    "5 minutes": 5,
    "10 minutes": 10,
    "15 minutes": 15,
    "30 minutes": 30,
    "1 hour": 60,
    "2 hours": 120,
    "4 hours": 240,
    "1 day": 1440,
    "1 week": 10080,
    "1 month": 43200,
    "1 quarter": 129600
}

output_interval = st.sidebar.selectbox(
    "Output Candle Interval",
    options=list(output_intervals.keys()),
    index=9,  # Default to 1 day
    help="Select the desired output interval for processed data"
)# Stock Market Data Fetcher - Streamlit App
# Requirements: streamlit, yfinance, pandas
# Deploy to: Streamlit Cloud via GitHub



# Page configuration
st.set_page_config(
    page_title="Stock Market Data Fetcher",
    page_icon="📈",
    layout="wide"
)

st.title("📈 Stock Market Data Fetcher")
st.markdown("Fetch stock data from Yahoo Finance and convert to custom candle intervals")

# Initialize session state
if 'data' not in st.session_state:
    st.session_state.data = None
if 'processed_data' not in st.session_state:
    st.session_state.processed_data = None

# Sidebar for inputs
st.sidebar.header("📊 Data Parameters")

# Data source selection
data_source = st.sidebar.radio(
    "Data Source",
    options=["Yahoo Finance", "Upload CSV"],
    index=0,
    help="Choose between fetching from Yahoo Finance or uploading your own CSV"
)

if data_source == "Yahoo Finance":
    # Ticker input
    ticker = st.sidebar.text_input(
        "Stock Ticker Symbol",
        value="AAPL",
        help="Enter stock symbol (e.g., AAPL, GOOGL, TSLA)"
    ).upper()

    # Date inputs
    col1, col2 = st.sidebar.columns(2)
    with col1:
        start_date = st.date_input(
            "Start Date",
            value=datetime.now() - timedelta(days=30),
            help="Select start date for data fetch"
        )
    with col2:
        end_date = st.date_input(
            "End Date",
            value=datetime.now(),
            help="Select end date for data fetch"
        )

    # Candle type selection
    candle_intervals = {
        "1 minute": "1m",
        "2 minutes": "2m", 
        "5 minutes": "5m",
        "15 minutes": "15m",
        "30 minutes": "30m",
        "60 minutes": "60m",
        "90 minutes": "90m",
        "1 hour": "1h",
        "1 day": "1d",
        "5 days": "5d",
        "1 week": "1wk",
        "1 month": "1mo",
        "3 months": "3mo"
    }

    input_interval = st.sidebar.selectbox(
        "Input Candle Interval",
        options=list(candle_intervals.keys()),
        index=2,  # Default to 5 minutes
        help="Select the interval for fetching data from Yahoo Finance"
    )

    # Pre/Post market data option
    include_prepost = st.sidebar.checkbox(
        "Include Pre/Post Market Data",
        value=False,
        help="Check to include pre-market and after-hours trading data"
    )

else:  # CSV Upload
    # File upload
    uploaded_file = st.sidebar.file_uploader(
        "Upload CSV File",
        type=['csv'],
        help="Upload a CSV file with OHLCV data"
    )
    
    # CSV format information
    st.sidebar.info("""
    📋 **CSV Format Required:**
    - DateTime column (any name)
    - Open, High, Low, Close columns
    - Volume column (optional)
    - Headers required in first row
    """)
    
    # Input timeframe for CSV data
    csv_intervals = {
        "1 minute": 1,
        "2 minutes": 2,
        "5 minutes": 5,
        "10 minutes": 10,
        "15 minutes": 15,
        "30 minutes": 30,
        "1 hour": 60,
        "2 hours": 120,
        "4 hours": 240,
        "1 day": 1440,
        "1 week": 10080,
        "1 month": 43200,
        "1 quarter": 129600
    }
    
    csv_input_interval = st.sidebar.selectbox(
        "CSV Data Timeframe",
        options=list(csv_intervals.keys()),
        index=2,  # Default to 5 minutes
        help="What timeframe is your CSV data in?"
    )

# Function to load and process CSV data
def load_csv_data(uploaded_file):
    """Load and process uploaded CSV data"""
    try:
        # Read CSV
        df = pd.read_csv(uploaded_file)
        
        # Display column names for user reference
        st.sidebar.write("**CSV Columns Found:**")
        st.sidebar.write(list(df.columns))
        
        # Try to identify datetime column
        datetime_cols = []
        for col in df.columns:
            if any(word in col.lower() for word in ['date', 'time', 'timestamp', 'datetime']):
                datetime_cols.append(col)
        
        if datetime_cols:
            datetime_col = st.sidebar.selectbox(
                "Select DateTime Column",
                options=datetime_cols,
                help="Choose the column containing date/time data"
            )
        else:
            datetime_col = st.sidebar.selectbox(
                "Select DateTime Column",
                options=df.columns,
                help="Choose the column containing date/time data"
            )
        
        # Try to identify OHLCV columns
        ohlcv_mapping = {}
        
        for target in ['Open', 'High', 'Low', 'Close', 'Volume']:
            potential_cols = [col for col in df.columns if target.lower() in col.lower()]
            if potential_cols:
                default_col = potential_cols[0]
            else:
                default_col = df.columns[0] if len(df.columns) > 0 else None
            
            if target == 'Volume':
                # Volume is optional
                volume_cols = ['None'] + list(df.columns)
                selected_col = st.sidebar.selectbox(
                    f"Select {target} Column (Optional)",
                    options=volume_cols,
                    index=volume_cols.index(default_col) if default_col in volume_cols else 0,
                    help=f"Choose the column for {target} data, or 'None' if not available"
                )
                if selected_col != 'None':
                    ohlcv_mapping[target] = selected_col
            else:
                # OHLC are required
                selected_col = st.sidebar.selectbox(
                    f"Select {target} Column",
                    options=df.columns,
                    index=list(df.columns).index(default_col) if default_col in df.columns else 0,
                    help=f"Choose the column for {target} data"
                )
                ohlcv_mapping[target] = selected_col
        
        return df, datetime_col, ohlcv_mapping
        
    except Exception as e:
        st.error(f"Error reading CSV: {str(e)}")
        return None, None, None

# Function to process CSV data into standard format
def process_csv_data(df, datetime_col, ohlcv_mapping):
    """Convert CSV data to standard OHLCV format"""
    try:
        # Create a copy of the data
        processed_df = df.copy()
        
        # Convert datetime column
        processed_df['DateTime'] = pd.to_datetime(processed_df[datetime_col])
        processed_df.set_index('DateTime', inplace=True)
        
        # Rename columns to standard format
        rename_dict = {}
        for standard_name, csv_col in ohlcv_mapping.items():
            rename_dict[csv_col] = standard_name
        
        processed_df = processed_df.rename(columns=rename_dict)
        
        # Select only OHLCV columns
        required_cols = ['Open', 'High', 'Low', 'Close']
        optional_cols = ['Volume'] if 'Volume' in ohlcv_mapping else []
        
        final_cols = required_cols + optional_cols
        processed_df = processed_df[final_cols]
        
        # Convert to numeric
        processed_df = processed_df.apply(pd.to_numeric, errors='coerce')
        
        # Remove any rows with NaN in OHLC
        processed_df = processed_df.dropna(subset=required_cols)
        
        return processed_df
        
    except Exception as e:
        st.error(f"Error processing CSV data: {str(e)}")
        return None
def fetch_stock_data(symbol, start, end, interval, include_extended_hours):
    """Fetch stock data from Yahoo Finance"""
    try:
        ticker_obj = yf.Ticker(symbol)
        data = ticker_obj.history(
            start=start,
            end=end,
            interval=candle_intervals[interval],
            auto_adjust=True,
            prepost=include_extended_hours
        )
        
        if data.empty:
            st.error(f"No data found for ticker {symbol}")
            return None
            
        return data
    except Exception as e:
        st.error(f"Error fetching data: {str(e)}")
        return None

# Function to resample data to custom intervals
def resample_data(data, target_minutes):
    """Resample OHLCV data to target interval in minutes"""
    if data is None or data.empty:
        return None
        
    try:
        # Create resampling rule
        if target_minutes < 60:
            rule = f"{target_minutes}T"  # T for minutes
        elif target_minutes == 60:
            rule = "1H"  # H for hours
        elif target_minutes < 1440:
            rule = f"{target_minutes//60}H"  # Hours
        elif target_minutes == 1440:
            rule = "1D"  # Daily
        elif target_minutes == 10080:
            rule = "1W"  # Weekly
        elif target_minutes == 43200:
            rule = "1M"  # Monthly (approximate)
        elif target_minutes == 129600:
            rule = "1Q"  # Quarterly
        else:
            rule = f"{target_minutes//1440}D"  # Multiple days
        
        # Resample the data
        resampled = data.resample(rule).agg({
            'Open': 'first',
            'High': 'max',
            'Low': 'min',
            'Close': 'last',
            'Volume': 'sum'
        }).dropna()
        
        return resampled
    except Exception as e:
        st.error(f"Error resampling data: {str(e)}")
        return None

# Function to convert DataFrame to CSV
def convert_df_to_csv(df):
    """Convert DataFrame to CSV format for download"""
    return df.to_csv().encode('utf-8')

# Data loading section
if data_source == "Yahoo Finance":
    # Fetch Data Button
    if st.sidebar.button("🔄 Fetch Data", type="primary"):
        with st.spinner(f"Fetching {input_interval} data for {ticker}..."):
            st.session_state.data = fetch_stock_data(
                ticker, 
                start_date, 
                end_date, 
                input_interval,
                include_prepost
            )
            st.session_state.data_source_info = f"{ticker} - {input_interval}"
            # Reset processed data when new data is fetched
            st.session_state.processed_data = None

else:  # CSV Upload
    if uploaded_file is not None:
        # Load and configure CSV
        if st.sidebar.button("📁 Load CSV Configuration", type="primary"):
            df, datetime_col, ohlcv_mapping = load_csv_data(uploaded_file)
            if df is not None:
                st.session_state.csv_config = {
                    'df': df,
                    'datetime_col': datetime_col,
                    'ohlcv_mapping': ohlcv_mapping
                }
        
        # Process CSV data if configuration exists
        if hasattr(st.session_state, 'csv_config') and st.sidebar.button("🔄 Process CSV Data", type="secondary"):
            with st.spinner("Processing CSV data..."):
                config = st.session_state.csv_config
                st.session_state.data = process_csv_data(
                    config['df'], 
                    config['datetime_col'], 
                    config['ohlcv_mapping']
                )
                if st.session_state.data is not None:
                    st.session_state.data_source_info = f"CSV Upload - {csv_input_interval}"
                # Reset processed data when new data is loaded
                st.session_state.processed_data = None
    else:
        st.sidebar.warning("Please upload a CSV file first")

# Process Data Button (common for both sources)
if st.sidebar.button("⚙️ Process Data", type="secondary"):
    if st.session_state.data is not None:
        with st.spinner(f"Converting to {output_interval} candles..."):
            # Determine input interval in minutes for CSV data
            if data_source == "CSV Upload":
                input_minutes = csv_intervals[csv_input_interval]
            else:
                # For Yahoo Finance, we'll use the current interval
                input_minutes = None  # Will be handled by existing logic
                
            st.session_state.processed_data = resample_data(
                st.session_state.data,
                output_intervals[output_interval]
            )
    else:
        st.warning("Please load data first!")

# Main content area
if st.session_state.data is not None:
    data_info = st.session_state.get('data_source_info', 'Unknown source')
    st.success(f"✅ Successfully loaded {len(st.session_state.data)} candles from {data_info}")
    
    # Display raw data info
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Records", len(st.session_state.data))
    with col2:
        st.metric("Date Range", f"{st.session_state.data.index[0].strftime('%Y-%m-%d')} to {st.session_state.data.index[-1].strftime('%Y-%m-%d')}")
    with col3:
        if data_source == "Yahoo Finance":
            st.metric("Interval", input_interval)
        else:
            st.metric("Interval", csv_input_interval)
    
    # Show raw data preview
    with st.expander("📋 Raw Data Preview"):
        st.dataframe(st.session_state.data.head(10))
    
    # Display processed data if available
    if st.session_state.processed_data is not None:
        st.success(f"✅ Successfully processed to {len(st.session_state.processed_data)} {output_interval} candles")
        
        # Display processed data info
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Processed Records", len(st.session_state.processed_data))
        with col2:
            st.metric("Compression Ratio", f"{len(st.session_state.data)}/{len(st.session_state.processed_data)}")
        with col3:
            st.metric("Output Interval", output_interval)
        
        # Show processed data preview
        with st.expander("📊 Processed Data Preview"):
            st.dataframe(st.session_state.processed_data.head(10))
        
        # Download section
        st.header("💾 Download Data")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Raw Data")
            csv_raw = convert_df_to_csv(st.session_state.data)
            if data_source == "Yahoo Finance":
                filename_raw = f"{ticker}_{input_interval}_{start_date}_to_{end_date}_raw.csv"
            else:
                filename_raw = f"uploaded_data_{csv_input_interval}_raw.csv"
            st.download_button(
                label="📥 Download Raw CSV",
                data=csv_raw,
                file_name=filename_raw,
                mime="text/csv"
            )
        
        with col2:
            st.subheader("Processed Data")
            csv_processed = convert_df_to_csv(st.session_state.processed_data)
            if data_source == "Yahoo Finance":
                filename_processed = f"{ticker}_{output_interval}_{start_date}_to_{end_date}_processed.csv"
            else:
                filename_processed = f"uploaded_data_{output_interval}_processed.csv"
            st.download_button(
                label="📥 Download Processed CSV",
                data=csv_processed,
                file_name=filename_processed,
                mime="text/csv"
            )
        
        # Data visualization
        st.header("📈 Data Visualization")
        
        tab1, tab2 = st.tabs(["Raw Data Chart", "Processed Data Chart"])
        
        with tab1:
            st.line_chart(st.session_state.data['Close'])
        
        with tab2:
            st.line_chart(st.session_state.processed_data['Close'])

else:
    if data_source == "Yahoo Finance":
        st.info("👆 Enter your parameters in the sidebar and click 'Fetch Data' to get started!")
    else:
        st.info("👆 Upload a CSV file and configure the columns to get started!")
    
    # Instructions
    st.header("📖 How to Use")
    
    if data_source == "Yahoo Finance":
        st.markdown("""
        **Yahoo Finance Mode:**
        1. **Enter Ticker Symbol**: Input the stock symbol (e.g., AAPL, GOOGL, TSLA)
        2. **Select Date Range**: Choose start and end dates for data fetching
        3. **Choose Input Interval**: Select the interval for fetching data from Yahoo Finance
        4. **Choose Output Interval**: Select the desired interval for processed data
        5. **Fetch Data**: Click the 'Fetch Data' button to retrieve data
        6. **Process Data**: Click 'Process Data' to convert to your desired interval
        7. **Download**: Use the download buttons to get CSV files
        """)
    else:
        st.markdown("""
        **CSV Upload Mode:**
        1. **Upload CSV**: Select a CSV file with your OHLCV data
        2. **Configure Columns**: Click 'Load CSV Configuration' and map your columns
        3. **Set Input Timeframe**: Tell the app what interval your CSV data represents
        4. **Process CSV**: Click 'Process CSV Data' to load your data
        5. **Choose Output Interval**: Select your desired output timeframe
        6. **Process Data**: Click 'Process Data' to convert intervals
        7. **Download**: Export your processed data
        
        **CSV Format Requirements:**
        - Must have DateTime column (any format pandas can parse)
        - Must have Open, High, Low, Close columns
        - Volume column is optional
        - Headers required in first row
        - Common formats: 'YYYY-MM-DD HH:MM:SS', 'MM/DD/YYYY HH:MM', etc.
        """)
    
    st.header("⚠️ Important Notes")
    st.markdown("""
    - **Yahoo Finance**: Limited historical intraday data (usually 60 days for minute data)
    - **CSV Upload**: No data limitations - process any historical data you have
    - **Output Interval**: Should typically be larger than input interval for proper aggregation
    - **Market Hours**: Yahoo data is typically available only during market hours
    - **CSV DateTime**: Must be in a format that pandas can automatically parse
    """)

# Footer
st.markdown("---")
st.markdown("Built with ❤️ using Streamlit and yfinance")
