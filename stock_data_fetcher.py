# Stock Market Data Fetcher - Streamlit App
# Requirements: streamlit, yfinance, pandas
# Deploy to: Streamlit Cloud via GitHub

import streamlit as st
import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
import io

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

# Output interval selection
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
    "1 day": 1440
}

output_interval = st.sidebar.selectbox(
    "Output Candle Interval",
    options=list(output_intervals.keys()),
    index=3,  # Default to 10 minutes
    help="Select the desired output interval for processed data"
)

# Function to fetch data
def fetch_stock_data(symbol, start, end, interval):
    """Fetch stock data from Yahoo Finance"""
    try:
        ticker_obj = yf.Ticker(symbol)
        data = ticker_obj.history(
            start=start,
            end=end,
            interval=candle_intervals[interval],
            auto_adjust=True,
            prepost=True
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
        else:
            rule = f"{target_minutes//1440}D"  # Days
        
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

# Fetch Data Button
if st.sidebar.button("🔄 Fetch Data", type="primary"):
    with st.spinner(f"Fetching {input_interval} data for {ticker}..."):
        st.session_state.data = fetch_stock_data(
            ticker, 
            start_date, 
            end_date, 
            input_interval
        )
        # Reset processed data when new data is fetched
        st.session_state.processed_data = None

# Process Data Button
if st.sidebar.button("⚙️ Process Data", type="secondary"):
    if st.session_state.data is not None:
        with st.spinner(f"Converting to {output_interval} candles..."):
            st.session_state.processed_data = resample_data(
                st.session_state.data,
                output_intervals[output_interval]
            )
    else:
        st.warning("Please fetch data first!")

# Main content area
if st.session_state.data is not None:
    st.success(f"✅ Successfully fetched {len(st.session_state.data)} {input_interval} candles for {ticker}")
    
    # Display raw data info
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Records", len(st.session_state.data))
    with col2:
        st.metric("Date Range", f"{st.session_state.data.index[0].strftime('%Y-%m-%d')} to {st.session_state.data.index[-1].strftime('%Y-%m-%d')}")
    with col3:
        st.metric("Interval", input_interval)
    
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
            st.download_button(
                label="📥 Download Raw CSV",
                data=csv_raw,
                file_name=f"{ticker}_{input_interval}_{start_date}_to_{end_date}_raw.csv",
                mime="text/csv"
            )
        
        with col2:
            st.subheader("Processed Data")
            csv_processed = convert_df_to_csv(st.session_state.processed_data)
            st.download_button(
                label="📥 Download Processed CSV",
                data=csv_processed,
                file_name=f"{ticker}_{output_interval}_{start_date}_to_{end_date}_processed.csv",
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
    st.info("👆 Enter your parameters in the sidebar and click 'Fetch Data' to get started!")
    
    # Instructions
    st.header("📖 How to Use")
    st.markdown("""
    1. **Enter Ticker Symbol**: Input the stock symbol (e.g., AAPL, GOOGL, TSLA)
    2. **Select Date Range**: Choose start and end dates for data fetching
    3. **Choose Input Interval**: Select the interval for fetching data from Yahoo Finance
    4. **Choose Output Interval**: Select the desired interval for processed data
    5. **Fetch Data**: Click the 'Fetch Data' button to retrieve data
    6. **Process Data**: Click 'Process Data' to convert to your desired interval
    7. **Download**: Use the download buttons to get CSV files
    
    **Note**: The output interval should typically be larger than the input interval for proper aggregation.
    """)
    
    st.header("⚠️ Important Notes")
    st.markdown("""
    - **Data Availability**: Yahoo Finance has limitations on historical intraday data (usually 60 days for minute data)
    - **Market Hours**: Data is typically available only during market hours
    - **Weekends/Holidays**: No data is available for non-trading days
    - **Rate Limits**: Yahoo Finance may throttle requests if too many are made quickly
    """)

# Footer
st.markdown("---")
st.markdown("Built with ❤️ using Streamlit and yfinance")
