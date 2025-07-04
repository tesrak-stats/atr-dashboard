import streamlit as st
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta, date, time
import time as time_module
import os
import tempfile
import zipfile
from io import BytesIO

# Session start times by asset type for 8-hour rolling analysis
SESSION_STARTS = {
    'STOCKS': 9,      # 9:30 AM market open
    'FUTURES': 18,    # 6:00 PM ES session start  
    'CRYPTO': 20,     # 8:00 PM daily reset (TradingView style)
    'FOREX': 17       # 5:00 PM forex week start
}

def calculate_atr(df, period=14):
    """
    Calculate TRUE Wilder's ATR for any timeframe
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

def combine_timeframes_with_atr(daily_file, intraday_file, atr_period=14, align_method='date_match', asset_type='STOCKS'):
    """
    Combine daily and intraday data with ATR calculation
    SIMPLIFIED - Only outputs single ATR column with current ATR
    """
    results = []
    
    try:
        # Handle different input types (file uploads vs session state data)
        if isinstance(daily_file, pd.DataFrame):
            # Data from session state
            daily_df = daily_file.copy()
        else:
            # Load daily data from file with robust reader
            daily_df = CSVProcessor.robust_csv_reader(daily_file, daily_file.name if hasattr(daily_file, 'name') else "daily_file")
        
        if isinstance(intraday_file, pd.DataFrame):
            # Data from session state
            intraday_df = intraday_file.copy()
        else:
            # Load intraday data from file with robust reader
            intraday_df = CSVProcessor.robust_csv_reader(intraday_file, intraday_file.name if hasattr(intraday_file, 'name') else "intraday_file")
        
        # Validate that we actually loaded data
        if daily_df.empty:
            st.error("❌ Daily file appears to be empty or unreadable")
            return None
        
        if intraday_df.empty:
            st.error("❌ Intraday file appears to be empty or unreadable")
            return None
        
        st.info(f"📊 Loaded daily data: {daily_df.shape[0]} rows, {daily_df.shape[1]} columns")
        st.info(f"📊 Loaded intraday data: {intraday_df.shape[0]} rows, {intraday_df.shape[1]} columns")
        
        # Show column names for debugging
        st.info(f"📋 Daily columns: {list(daily_df.columns)}")
        st.info(f"📋 Intraday columns: {list(intraday_df.columns)}")
        
        # Standardize columns
        daily_df = CSVProcessor.standardize_columns(daily_df)
        intraday_df = CSVProcessor.standardize_columns(intraday_df)
        
        st.info(f"📋 Standardized daily columns: {list(daily_df.columns)}")
        st.info(f"📋 Standardized intraday columns: {list(intraday_df.columns)}")
        
        # Store data in session state for download buttons
        st.session_state['debug_raw_daily'] = daily_df.copy()
        st.session_state['debug_raw_intraday'] = intraday_df.copy()
        
        # Validate required columns
        required_cols = ['Date', 'Open', 'High', 'Low', 'Close']
        
        daily_missing = [col for col in required_cols if col not in daily_df.columns]
        intraday_missing = [col for col in required_cols if col not in intraday_df.columns]
        
        if daily_missing:
            st.error(f"❌ Daily data missing columns: {daily_missing}")
            return None
        
        if intraday_missing:
            st.error(f"❌ Intraday data missing columns: {intraday_missing}")
            return None
        
        # Clean and validate OHLC data
        st.info("🧹 Cleaning and validating OHLC data...")
        
        # Function to clean OHLC data
        def clean_ohlc_data(df, data_type="data"):
            original_count = len(df)
            removed_rows = []  # Track what gets removed
            
            # Convert OHLC columns to numeric, forcing errors to NaN
            ohlc_cols = ['Open', 'High', 'Low', 'Close']
            for col in ohlc_cols:
                df[col] = pd.to_numeric(df[col], errors='coerce')
            
            # Check for obvious corporate action indicators before removing
            corporate_action_indicators = []
            for idx, row in df.iterrows():
                if any(pd.isna(row[col]) for col in ohlc_cols):
                    # Check if original data had corporate action indicators
                    original_row = daily_df.iloc[idx] if data_type == "Daily data" else intraday_df.iloc[idx]
                    for col in ohlc_cols:
                        original_val = str(original_row[col]).upper()
                        if any(indicator in original_val for indicator in ['SPLIT', 'DIV', 'DIVIDEND', 'CORP', 'ACTION', 'HALT', 'SUSPEND']):
                            corporate_action_indicators.append(f"Row {idx}: {original_val}")
                            removed_rows.append({
                                'row': idx,
                                'date': row.get('Date', 'Unknown'),
                                'reason': 'Corporate Action',
                                'details': original_val
                            })
            
            # Remove rows where any OHLC value is NaN or invalid
            df_clean = df.dropna(subset=ohlc_cols)
            
            # Track NaN removals
            nan_removed = original_count - len(df_clean)
            for i in range(nan_removed):
                removed_rows.append({
                    'row': 'Multiple',
                    'date': 'Various',
                    'reason': 'NaN/Invalid Values',
                    'details': 'Non-numeric OHLC data'
                })
            
            # Advanced validation: Check for potential stock splits
            if len(df_clean) > 1:
                # Calculate day-to-day price changes
                df_clean = df_clean.sort_values('Date').reset_index(drop=True)
                
                # Look for extreme price jumps that might indicate unadjusted splits
                prev_close = df_clean['Close'].shift(1)
                next_open = df_clean['Open']
                
                # Calculate overnight gaps
                overnight_change = (next_open - prev_close) / prev_close
                
                # Flag potential splits (>40% overnight change)
                potential_splits = overnight_change.abs() > 0.4
                
                if potential_splits.any():
                    split_dates = df_clean[potential_splits]['Date'].tolist()
                    st.error(f"🚨 **POTENTIAL STOCK SPLITS DETECTED** in {data_type}:")
                    st.error(f"📅 **Split dates**: {split_dates}")
                    st.error(f"💹 **Overnight changes**: {overnight_change[potential_splits].round(3).tolist()}")
                    st.error("⚠️ **CRITICAL**: Your data may NOT be split-adjusted!")
                    st.error("💡 **Recommendation**: Use split-adjusted data from your broker or data provider")
                    
                    # Track split indicators
                    for i, date in enumerate(split_dates):
                        removed_rows.append({
                            'row': 'Split Detection',
                            'date': date,
                            'reason': 'Potential Stock Split',
                            'details': f"Overnight change: {overnight_change[potential_splits].iloc[i]:.3f}"
                        })
                    
                    # Check if it looks like a 2:1 split pattern
                    split_ratios = []
                    for i in potential_splits[potential_splits].index:
                        if i > 0:
                            ratio = prev_close.iloc[i] / next_open.iloc[i]
                            split_ratios.append(f"{ratio:.2f}:1")
                    
                    if split_ratios:
                        st.error(f"🔍 **Estimated split ratios**: {split_ratios}")
            
            # Standard OHLC validation
            invalid_mask = ~(
                (df_clean['High'] >= df_clean['Low']) &
                (df_clean['Open'] >= df_clean['Low']) &
                (df_clean['Open'] <= df_clean['High']) &
                (df_clean['Close'] >= df_clean['Low']) &
                (df_clean['Close'] <= df_clean['High']) &
                (df_clean['High'] > 0) &  # Prices should be positive
                (df_clean['Low'] > 0)
            )
            
            # Track invalid OHLC removals
            invalid_rows = df_clean[invalid_mask]
            for idx, row in invalid_rows.iterrows():
                removed_rows.append({
                    'row': idx,
                    'date': row.get('Date', 'Unknown'),
                    'reason': 'Invalid OHLC Logic',
                    'details': f"O:{row['Open']:.2f} H:{row['High']:.2f} L:{row['Low']:.2f} C:{row['Close']:.2f}"
                })
            
            df_clean = df_clean[~invalid_mask]
            
            cleaned_count = len(df_clean)
            removed_count = original_count - cleaned_count
            
            # Store removal details in session state for dropdown
            removal_key = f"removed_data_{data_type.replace(' ', '_').lower()}"
            st.session_state[removal_key] = removed_rows
            
            # Report what was removed
            if removed_count > 0:
                st.warning(f"🧹 {data_type}: Removed {removed_count} invalid OHLC rows")
                
                # Show expandable removal details
                with st.expander(f"🔍 **View Removed Data Details** ({removed_count} rows)", expanded=False):
                    if removed_rows:
                        removal_df = pd.DataFrame(removed_rows)
                        st.dataframe(removal_df, use_container_width=True)
                        
                        # Show summary by reason
                        reason_counts = removal_df['reason'].value_counts()
                        st.write("**Removal Summary by Reason:**")
                        for reason, count in reason_counts.items():
                            st.write(f"   • **{reason}**: {count} rows")
                    else:
                        st.write("No detailed removal information available")
                
                if corporate_action_indicators:
                    st.warning("📋 **Corporate action indicators found:**")
                    for indicator in corporate_action_indicators[:5]:  # Show first 5
                        st.warning(f"   • {indicator}")
                    if len(corporate_action_indicators) > 5:
                        st.warning(f"   • ... and {len(corporate_action_indicators) - 5} more")
                
                st.info(f"✅ {data_type}: {cleaned_count} valid OHLC rows remaining")
            else:
                st.success(f"✅ {data_type}: All {cleaned_count} rows have valid OHLC data")
            
            return df_clean.reset_index(drop=True)
        
        # Clean both datasets
        daily_df = clean_ohlc_data(daily_df, "Daily data")
        intraday_df = clean_ohlc_data(intraday_df, "Intraday data")
        
        # Check if we still have data after cleaning
        if daily_df.empty:
            st.error("❌ No valid daily OHLC data remaining after cleaning")
            return None
        
        if intraday_df.empty:
            st.error("❌ No valid intraday OHLC data remaining after cleaning")
            return None
        
        # Process dates with futures-aware logic
        def assign_trading_date(datetime_val, asset_type):
            """
            Assign proper trading date based on asset type
            For futures: 18:00 Monday = Tuesday session
            For stocks: Use calendar date
            """
            if asset_type == 'FUTURES':
                # For futures, session starts at 18:00 (6 PM)
                # If time is 18:00 or later, it belongs to next calendar day's session
                if datetime_val.hour >= 18:
                    return (datetime_val + timedelta(days=1)).date()
                else:
                    return datetime_val.date()
            else:
                # For stocks and other assets, use calendar date
                return datetime_val.date()
        
        # Process daily data dates
        daily_df['Date'] = pd.to_datetime(daily_df['Date'])
        if asset_type == 'FUTURES':
            daily_df['Date'] = daily_df['Date'].apply(lambda x: assign_trading_date(x, asset_type))
        else:
            daily_df['Date'] = daily_df['Date'].dt.date
        
        # Handle intraday datetime with futures-aware date assignment
        if 'Datetime' not in intraday_df.columns:
            if 'Date' in intraday_df.columns and 'Time' in intraday_df.columns:
                intraday_df['Datetime'] = pd.to_datetime(intraday_df['Date'].astype(str) + ' ' + intraday_df['Time'].astype(str))
            else:
                intraday_df['Datetime'] = pd.to_datetime(intraday_df['Date'])
        else:
            intraday_df['Datetime'] = pd.to_datetime(intraday_df['Datetime'])
        
        # Assign proper trading dates for intraday data
        intraday_df['Date'] = intraday_df['Datetime'].apply(lambda x: assign_trading_date(x, asset_type))
        
        # Show futures date assignment info
        if asset_type == 'FUTURES':
            st.info("🕐 **Futures Date Assignment**: Times 18:00+ assigned to next day's session")
            st.info("Example: Monday 18:00 → Tuesday session, Monday 17:00 → Monday session")
        
        # Sort data and handle duplicates
        daily_df = daily_df.sort_values('Date').reset_index(drop=True)
        intraday_df = intraday_df.sort_values('Datetime').reset_index(drop=True)
        
        # Check for and handle duplicate dates in daily data
        duplicate_dates = daily_df['Date'].duplicated().sum()
        if duplicate_dates > 0:
            st.warning(f"⚠️ Found {duplicate_dates} duplicate dates in daily data. Using first occurrence of each date.")
            daily_df = daily_df.drop_duplicates(subset=['Date'], keep='first')
        
        # FIXED: Correct day counting
        unique_daily_dates = daily_df['Date'].nunique()
        unique_intraday_dates = intraday_df['Date'].nunique()
        
        st.info(f"📅 **CORRECTED Day Count**: Daily data has {unique_daily_dates} unique days")
        st.info(f"📅 **CORRECTED Day Count**: Intraday data has {unique_intraday_dates} unique days")
        
        # Calculate ATR on daily data
        st.info("📊 Calculating ATR on daily data...")
        st.info(f"Daily data shape before ATR: {daily_df.shape}")
        st.info(f"Daily data columns: {list(daily_df.columns)}")
        st.info(f"Daily data sample:\n{daily_df.head()}")
        
        daily_with_atr = calculate_atr(daily_df, period=atr_period)
        
        st.info(f"Daily data shape after ATR: {daily_with_atr.shape}")
        st.info(f"ATR column sample: {daily_with_atr['ATR'].head(20).tolist()}")
        
        # Store processed data in session state
        st.session_state['debug_daily_with_atr'] = daily_with_atr.copy()
        
        # Check for data validation info and propagate warnings
        if hasattr(daily_df, 'attrs') and 'completeness' in daily_df.attrs:
            completeness = daily_df.attrs['completeness']
            if completeness < 95:
                st.warning(f"⚠️ **Data Quality Alert**: Base timeframe has {completeness:.1f}% completeness")
                st.warning(f"Original request: {daily_df.attrs['requested_start']} to {daily_df.attrs['requested_end']}")
                st.warning(f"Actual data: {daily_df.attrs['actual_start']} to {daily_df.attrs['actual_end']}")
                st.warning("🚨 **ATR calculation may be based on insufficient historical data**")
        
        # Validate ATR calculation
        valid_atr = daily_with_atr[daily_with_atr['ATR'].notna()]
        if valid_atr.empty:
            st.error("❌ Failed to calculate ATR - check daily data quality")
            st.error(f"All ATR values are NaN. Daily data needs numeric OHLC columns.")
            st.error(f"Daily data types: {daily_df.dtypes}")
            return None
        
        st.success(f"✅ ATR calculated successfully: {len(valid_atr)} valid ATR values")
        
        # ATR quality check with realistic period requirements
        optimal_periods = max(84, atr_period * 4)  # Use 84 or 4x ATR period, whichever is higher
        
        if len(valid_atr) >= optimal_periods:
            st.success(f"✅ **Excellent ATR Quality**: {len(valid_atr)} periods (optimal: {optimal_periods}+)")
        elif len(valid_atr) >= atr_period * 4:
            st.info(f"✅ **Good ATR Quality**: {len(valid_atr)} periods (minimum: {atr_period * 4})")
        elif len(valid_atr) >= atr_period * 2:
            st.warning(f"⚠️ **Marginal ATR Quality**: {len(valid_atr)} periods (recommended: {optimal_periods})")
        else:
            st.error(f"❌ **Poor ATR Quality**: {len(valid_atr)} periods (need: {optimal_periods}+)")
        
        # Store valid ATR data
        st.session_state['debug_valid_atr'] = valid_atr.copy()
        
        # Data alignment info
        daily_start = daily_df['Date'].min()
        daily_end = daily_df['Date'].max()
        intraday_start = intraday_df['Date'].min()
        intraday_end = intraday_df['Date'].max()
        
        st.info(f"📅 Daily data: {daily_start} to {daily_end}")
        st.info(f"📅 Intraday data: {intraday_start} to {intraday_end}")
        
        # Check alignment
        if daily_start >= intraday_start:
            st.warning("⚠️ Daily data should ideally start before intraday data for proper ATR calculation")
        
        # Combine data based on alignment method
        if align_method == 'date_match':
            st.info("🔄 Combining data using date matching...")
            
            # SIMPLIFIED: Only create current ATR lookup (no dual ATR)
            st.info("🔧 Creating ATR lookup dictionary...")
            atr_lookup = {}
            
            # Create current ATR lookup - this is the "currently used" ATR
            for i, row in daily_with_atr.iterrows():
                atr_lookup[row['Date']] = row['ATR']
            
            st.info(f"📊 ATR lookup created with {len(atr_lookup)} entries")
            
            # Debug the lookup process
            sample_intraday_dates = intraday_df['Date'].head(5).tolist()
            st.info(f"🔍 Sample intraday dates: {sample_intraday_dates}")
            
            # Check ATR around the intraday start date
            intraday_start_date = intraday_df['Date'].min()
            st.info(f"🔍 Intraday starts on: {intraday_start_date}")
            
            # Find daily data around that date
            daily_around_start = daily_with_atr[
                (daily_with_atr['Date'] >= intraday_start_date - timedelta(days=5)) &
                (daily_with_atr['Date'] <= intraday_start_date + timedelta(days=5))
            ][['Date', 'ATR']].head(10)
            
            st.info(f"🔍 Daily ATR around intraday start:\n{daily_around_start}")
            
            sample_lookups = []
            for date in sample_intraday_dates:
                atr_val = atr_lookup.get(date, 'NOT_FOUND')
                sample_lookups.append(f"{date}: {atr_val}")
            st.info(f"🔍 Sample ATR lookups: {sample_lookups}")
            
            # SIMPLIFIED: Add only single ATR column to intraday data
            st.info("📊 Mapping ATR values to intraday data...")
            intraday_df['ATR'] = intraday_df['Date'].map(atr_lookup)
            
            # Show ATR calculation info
            st.info(f"✅ Single ATR column calculated:")
            st.info("📊 **ATR**: Currently used ATR (from last complete base time period)")
            
            # Check how many matches we got
            matched_atr = intraday_df['ATR'].notna().sum()
            total_intraday = len(intraday_df)
            st.info(f"✅ ATR mapping result: {matched_atr}/{total_intraday} intraday records got ATR values")
            
            # Store final mapped data
            st.session_state['debug_intraday_with_atr'] = intraday_df.copy()
            
            # Filter to only intraday records with ATR - KEEP ALL COLUMNS
            combined_df = intraday_df[intraday_df['ATR'].notna()].copy()
            
            if combined_df.empty:
                st.error("❌ No date overlap between daily and intraday data")
                st.error(f"Daily range: {daily_start} to {daily_end}")
                st.error(f"Intraday range: {intraday_start} to {intraday_end}")
                
                # Debug the actual overlap
                overlap_start = max(daily_start, intraday_start)
                overlap_end = min(daily_end, intraday_end)
                st.error(f"Expected overlap: {overlap_start} to {overlap_end}")
                
                return None
            
            st.success(f"✅ Combined data: {len(combined_df):,} intraday records with ATR")
            
            return combined_df
        
        else:
            st.error("❌ Invalid alignment method")
            return None
            
    except Exception as e:
        st.error(f"❌ Error combining timeframes: {str(e)}")
        import traceback
        st.error(traceback.format_exc())
        return None

class TickerMapper:
    """Handle ticker symbol mappings for different data sources"""
    
    @staticmethod
    def create_custom_candles(df, custom_periods):
        """Create custom candles based on defined time periods"""
        df = df.copy()
        
        # Ensure we have a Datetime column
        df = CSVProcessor.create_datetime_column(df)
        
        # Group by date
        df['Date_only'] = df['Datetime'].dt.date
        daily_groups = df.groupby('Date_only')
        
        custom_candles = []
        
        for date, day_data in daily_groups:
            for period_idx, period in enumerate(custom_periods):
                period_name = period['name']
                start_time = pd.to_datetime(period['start'], format='%H:%M').time()
                end_time = pd.to_datetime(period['end'], format='%H:%M').time()
                
                # Filter data for this time period
                day_data['Time_obj'] = day_data['Datetime'].dt.time
                period_mask = (day_data['Time_obj'] >= start_time) & (day_data['Time_obj'] <= end_time)
                period_data = day_data[period_mask]
                
                if not period_data.empty:
                    # Create OHLC candle for this period
                    candle = {
                        'Date': date,
                        'Datetime': pd.Timestamp.combine(date, start_time),
                        'Period_Name': period_name,
                        'Period_Start': period['start'],
                        'Period_End': period['end'],
                        'Open': period_data['Open'].iloc[0],
                        'High': period_data['High'].max(),
                        'Low': period_data['Low'].min(),
                        'Close': period_data['Close'].iloc[-1],
                    }
                    
                    # Add volume if present
                    if 'Volume' in period_data.columns:
                        candle['Volume'] = period_data['Volume'].sum()
                    
                    custom_candles.append(candle)
        
        if custom_candles:
            result_df = pd.DataFrame(custom_candles)
            return result_df.sort_values(['Date', 'Period_Start']).reset_index(drop=True)
        else:
            return pd.DataFrame()
    
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
    def robust_csv_reader(file_input, filename="file"):
        """
        Robust CSV reader that handles various delimiter and encoding issues
        Tries multiple approaches to successfully read the file
        """
        # Common delimiters to try
        delimiters = [',', ';', '\t', '|']
        
        # Common encodings to try
        encodings = ['utf-8', 'latin1', 'cp1252', 'iso-8859-1']
        
        # Try different combinations
        for encoding in encodings:
            for delimiter in delimiters:
                try:
                    # Try with header=None first (for unlabeled data)
                    df = pd.read_csv(file_input, delimiter=delimiter, encoding=encoding, header=None)
                    
                    # Check if we got multiple columns
                    if df.shape[1] > 1:
                        st.info(f"✅ **File read successfully**: {filename}")
                        st.info(f"📊 **Format detected**: {df.shape[1]} columns, delimiter='{delimiter}', encoding='{encoding}'")
                        
                        # If we have many columns, let's see if first row looks like headers
                        if df.shape[1] >= 4:
                            first_row = df.iloc[0].astype(str)
                            
                            # Check if first row contains header-like text
                            header_indicators = ['date', 'time', 'open', 'high', 'low', 'close', 'volume', 'datetime', 'timestamp']
                            looks_like_header = any(
                                any(indicator in str(cell).lower() for indicator in header_indicators)
                                for cell in first_row
                            )
                            
                            if looks_like_header:
                                st.info("🔍 **First row appears to be headers** - re-reading with header=0")
                                # Re-read with headers
                                file_input.seek(0)
                                df = pd.read_csv(file_input, delimiter=delimiter, encoding=encoding, header=0)
                            else:
                                st.info("🔍 **No headers detected** - treating first row as data")
                                
                        return df
                    
                except Exception as e:
                    # Try next combination
                    continue
                finally:
                    # Reset file pointer for next attempt
                    if hasattr(file_input, 'seek'):
                        file_input.seek(0)
        
        # If all else fails, try pandas' built-in delimiter detection
        try:
            file_input.seek(0)
            df = pd.read_csv(file_input, delimiter=None, engine='python', header=None)
            if df.shape[1] > 1:
                st.warning(f"⚠️ **Fallback successful**: {filename} read with pandas auto-detection")
                return df
        except:
            pass
        
        # Final fallback - try reading as single column and check content
        try:
            file_input.seek(0)
            df = pd.read_csv(file_input, header=None)
            
            if df.shape[1] == 1:
                # Check if single column contains comma-separated data
                sample_row = str(df.iloc[0, 0])
                if ',' in sample_row and len(sample_row.split(',')) >= 4:
                    st.info(f"🔍 **Single column with comma-separated data detected**: {filename}")
                    st.info(f"📋 **Sample**: {sample_row[:100]}...")
                    
                    # Split the single column into multiple columns
                    split_data = []
                    for idx, row in df.iterrows():
                        row_data = str(row.iloc[0]).split(',')
                        split_data.append(row_data)
                    
                    # Create new DataFrame with split data
                    df_split = pd.DataFrame(split_data)
                    st.success(f"✅ **Column splitting successful**: {df_split.shape[1]} columns created")
                    
                    return df_split
        except:
            pass
        
        # If nothing worked, raise an error
        raise ValueError(f"Could not read {filename} with any delimiter/encoding combination")

    @staticmethod
    def smart_column_detection(df):
        """
        Smart detection for unlabeled columns
        Only activates when proper headers are missing
        Assumes: First column = Date/Datetime, then O, H, L, C, [Volume]
        """
        original_columns = list(df.columns)
        
        # Check if we already have proper OHLC headers
        ohlc_cols = ['Open', 'High', 'Low', 'Close']
        date_cols = ['Date', 'Datetime']
        
        has_ohlc = all(col in df.columns for col in ohlc_cols)
        has_date = any(col in df.columns for col in date_cols)
        
        # Skip smart detection if we already have proper headers
        if has_ohlc and has_date:
            st.success("✅ **Proper column headers detected** - using existing headers")
            st.info(f"📋 **Found columns**: {', '.join([col for col in df.columns if col in ohlc_cols + date_cols])}")
            return df
        
        # Only proceed with smart detection if headers are missing
        if has_ohlc:
            st.info("✅ **OHLC headers found** - no smart detection needed")
            return df
            
        st.info("🔍 **No proper headers found** - activating smart column detection")
        
        # Check if we have enough columns for OHLC data
        if len(df.columns) < 5:  # Need at least Date + OHLC
            st.info("⚠️ **Insufficient columns** for OHLC data - skipping smart detection")
            return df
            
        # Try to detect if this looks like OHLC data
        numeric_cols = []
        for col in df.columns[1:]:  # Skip first column (assumed date)
            try:
                # Try to convert to numeric
                pd.to_numeric(df[col], errors='raise')
                numeric_cols.append(col)
            except:
                pass
        
        # Need at least 4 numeric columns for OHLC
        if len(numeric_cols) < 4:
            st.info("⚠️ **Insufficient numeric columns** for OHLC data - skipping smart detection")
            return df
            
        # Check if the numeric data looks like OHLC (High >= Low, etc.)
        try:
            sample_data = df[numeric_cols[:4]].head(10)
            sample_numeric = sample_data.apply(pd.to_numeric, errors='coerce')
            
            # Basic OHLC validation on sample
            if len(sample_numeric.columns) >= 4:
                # Assume order is O, H, L, C
                o_col, h_col, l_col, c_col = sample_numeric.columns[:4]
                
                # Check if High >= Low in most cases
                high_low_valid = (sample_numeric[h_col] >= sample_numeric[l_col]).sum() >= len(sample_numeric) * 0.8
                
                if high_low_valid:
                    st.warning("🔍 **Smart Column Detection Activated**")
                    st.warning(f"⚠️ **ASSUMED COLUMN MAPPING** - Please verify:")
                    st.warning(f"   • **{original_columns[0]}** → Date/Datetime")
                    st.warning(f"   • **{original_columns[1]}** → Open")
                    st.warning(f"   • **{original_columns[2]}** → High") 
                    st.warning(f"   • **{original_columns[3]}** → Low")
                    st.warning(f"   • **{original_columns[4]}** → Close")
                    
                    if len(original_columns) > 5:
                        st.warning(f"   • **{original_columns[5]}** → Volume")
                    
                    st.warning("🚨 **IMPORTANT**: Review your data to confirm this mapping is correct!")
                    
                    # Apply the mapping
                    new_columns = {}
                    new_columns[original_columns[0]] = 'Date'  # First column becomes Date
                    new_columns[original_columns[1]] = 'Open'
                    new_columns[original_columns[2]] = 'High'  
                    new_columns[original_columns[3]] = 'Low'
                    new_columns[original_columns[4]] = 'Close'
                    
                    if len(original_columns) > 5:
                        new_columns[original_columns[5]] = 'Volume'
                    
                    df = df.rename(columns=new_columns)
                    
                    st.success("✅ **Smart mapping applied** - Processing will continue with assumed column structure")
                    
        except Exception as e:
            # If smart detection fails, just return original
            st.info("⚠️ **Smart detection failed** - using original column names")
            pass
            
        return df

    @staticmethod
    def detect_and_split_datetime(df):
        """
        Detect datetime columns and split them into Date and Time if needed
        Handles various datetime formats and column names
        """
        # Common datetime column names to check
        datetime_candidates = ['datetime', 'timestamp', 'date_time', 'date time', 'dateTime', 'date/time', 'dt']
        
        for col in df.columns:
            if col.lower() in datetime_candidates:
                try:
                    # Try to parse as datetime
                    parsed_datetime = pd.to_datetime(df[col])
                    
                    # Check if this column has time information
                    has_time_info = (parsed_datetime.dt.hour != 0).any() or (parsed_datetime.dt.minute != 0).any()
                    
                    if has_time_info:
                        st.info(f"🔄 **Auto-detected**: '{col}' contains datetime information - splitting into Date and Time")
                        
                        # Create separate Date and Time columns
                        df['Date'] = parsed_datetime.dt.date
                        df['Time'] = parsed_datetime.dt.time
                        df['Datetime'] = parsed_datetime
                        
                        # Remove original column if it's not already standardized
                        if col not in ['Date', 'Time', 'Datetime']:
                            df = df.drop(columns=[col])
                        
                        return df
                        
                except Exception:
                    continue
        
        # Check if Date column might contain datetime info
        if 'Date' in df.columns and 'Time' not in df.columns:
            try:
                # Sample a few values to check format
                sample_values = df['Date'].head(10).astype(str)
                
                # Look for time patterns in the date column
                has_time_pattern = any(
                    ':' in str(val) and len(str(val)) > 10 
                    for val in sample_values
                )
                
                if has_time_pattern:
                    parsed_datetime = pd.to_datetime(df['Date'])
                    
                    # Check if parsed values actually have time info
                    has_time_info = (parsed_datetime.dt.hour != 0).any() or (parsed_datetime.dt.minute != 0).any()
                    
                    if has_time_info:
                        st.info("🔄 **Auto-detected**: Date column contains time information - splitting into Date and Time")
                        
                        # Split into separate columns
                        df['Time'] = parsed_datetime.dt.time
                        df['Date'] = parsed_datetime.dt.date
                        df['Datetime'] = parsed_datetime
                        
                        return df
                        
            except Exception:
                pass
        
        return df

    @staticmethod
    def standardize_columns(df):
        """Standardize column names across different CSV formats"""
        # Create a copy to avoid modifying original
        df = df.copy()
        
        # Clean column names
        df.columns = [str(col).strip() for col in df.columns]
        
        # First, try smart column detection for unlabeled data
        df = CSVProcessor.smart_column_detection(df)
        
        # Then, try to detect and split datetime columns
        df = CSVProcessor.detect_and_split_datetime(df)
        
        # Common column mappings
        column_mappings = {
            # Date/Time columns
            'date': 'Date',
            'time': 'Time',
            'datetime': 'Datetime',
            'timestamp': 'Datetime',
            'date_time': 'Datetime',
            'date time': 'Datetime',
            'dateTime': 'Datetime',
            'date/time': 'Datetime',
            'dt': 'Datetime',
            
            # OHLC columns - including single letter variations
            'open': 'Open',
            'o': 'Open',          # Single letter
            'high': 'High',
            'h': 'High',          # Single letter
            'low': 'Low',
            'l': 'Low',           # Single letter
            'close': 'Close',
            'c': 'Close',         # Single letter
            'last': 'Close',
            'settle': 'Close',
            'adj_close': 'Close',
            'adjusted_close': 'Close',
            
            # Volume variations
            'volume': 'Volume',
            'vol': 'Volume',
            'v': 'Volume',        # Single letter
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
            
            # Extract separate Date and Time columns if they don't exist
            if 'Date' not in df.columns:
                df['Date'] = df['Datetime'].dt.date
            if 'Time' not in df.columns:
                df['Time'] = df['Datetime'].dt.time
            
            return df
        
        if 'Date' in df.columns and 'Time' in df.columns:
            # Combine Date and Time
            df['Datetime'] = pd.to_datetime(df['Date'].astype(str) + ' ' + df['Time'].astype(str))
        elif 'Date' in df.columns:
            # Check if Date column contains time information
            try:
                # Try to parse as datetime
                parsed_datetime = pd.to_datetime(df['Date'])
                
                # Check if any parsed values have time information (not just midnight)
                has_time_info = (parsed_datetime.dt.hour != 0).any() or (parsed_datetime.dt.minute != 0).any()
                
                if has_time_info:
                    # Date column contains datetime info
                    df['Datetime'] = parsed_datetime
                    
                    # Extract separate Date and Time columns
                    df['Date'] = df['Datetime'].dt.date
                    df['Time'] = df['Datetime'].dt.time
                    
                    st.info("🔄 **Auto-detected**: Date column contains time information - extracted Date and Time columns")
                else:
                    # Date column is date-only
                    df['Datetime'] = pd.to_datetime(df['Date'])
                    
            except Exception:
                # Fallback: treat as date-only
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
            
            # Check if session crosses midnight (start >= end)
            crosses_midnight = start_time >= end_time
            
            if crosses_midnight:
                # Session crosses midnight (e.g., 18:00 to 17:00 next day)
                # Include times >= start_time OR <= end_time
                time_mask = (df['Time_obj'] >= start_time) | (df['Time_obj'] <= end_time)
                st.info(f"🕐 **Midnight-crossing session**: {custom_start_time} today → {custom_end_time} next day")
            else:
                # Normal session within same day
                time_mask = (df['Time_obj'] >= start_time) & (df['Time_obj'] <= end_time)
                st.info(f"📅 **Same-day session**: {custom_start_time} - {custom_end_time}")
            
            df = df[time_mask]
            df.drop('Time_obj', axis=1, inplace=True)
            
            if df.empty:
                raise ValueError(f"No data found in time range {custom_start_time} to {custom_end_time}")
            
            st.success(f"✅ **Time filtering applied**: {len(df)} records in session")
            
            # For midnight-crossing sessions, need special handling for daily aggregation
            if crosses_midnight and target_timeframe.upper() == '1D':
                st.info("🔄 **Special daily aggregation** for midnight-crossing session")
                
                # Create custom session groups for proper daily aggregation
                df = df.copy()
                df['Session_Date'] = df['Datetime'].apply(lambda x: 
                    x.date() if x.time() >= start_time else (x - timedelta(days=1)).date()
                )
                
                # Group by session date instead of calendar date
                session_groups = df.groupby('Session_Date')
                
                daily_candles = []
                for session_date, session_data in session_groups:
                    if len(session_data) > 0:
                        candle = {
                            'Datetime': pd.Timestamp.combine(session_date, start_time),
                            'Date': session_date,
                            'Open': session_data['Open'].iloc[0],
                            'High': session_data['High'].max(),
                            'Low': session_data['Low'].min(),
                            'Close': session_data['Close'].iloc[-1],
                        }
                        
                        # Add volume if present
                        if 'Volume' in session_data.columns:
                            candle['Volume'] = session_data['Volume'].sum()
                        
                        # Add other columns if present
                        for col in session_data.columns:
                            if col not in ['Datetime', 'Date', 'Open', 'High', 'Low', 'Close', 'Volume', 'Time_obj', 'Session_Date']:
                                candle[col] = session_data[col].iloc[0]
                        
                        daily_candles.append(candle)
                
                # Create result DataFrame
                resampled = pd.DataFrame(daily_candles)
                resampled = resampled.sort_values('Date').reset_index(drop=True)
                
                st.success(f"✅ **Session-based daily candles**: {len(resampled)} daily sessions created")
                return resampled
        
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
        elif target_timeframe.upper() == '1D':
            resampled = df.resample('D', closed='left', label='left').agg(agg_rules)
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
    def detect_date_gaps(df, max_gap_days=7):
        """
        Detect large gaps in date continuity that might indicate missing data
        """
        if 'Date' not in df.columns or len(df) < 2:
            return
            
        # Get unique dates and sort them
        unique_dates = pd.to_datetime(df['Date']).dt.date.unique()
        unique_dates = sorted(unique_dates)
        
        # Find gaps between consecutive dates
        gaps = []
        for i in range(1, len(unique_dates)):
            current_date = unique_dates[i]
            prev_date = unique_dates[i-1]
            
            gap_days = (current_date - prev_date).days
            
            # Flag gaps larger than max_gap_days
            if gap_days > max_gap_days:
                gaps.append({
                    'start_date': prev_date,
                    'end_date': current_date,
                    'gap_days': gap_days
                })
        
        # Report findings
        if gaps:
            st.warning(f"📅 **Date Gap Analysis**: Found {len(gaps)} large gaps (>{max_gap_days} days)")
            
            # Show significant gaps
            for gap in gaps[:5]:  # Show first 5 gaps
                st.warning(f"   • **{gap['gap_days']} day gap**: {gap['start_date']} → {gap['end_date']}")
            
            if len(gaps) > 5:
                st.warning(f"   • ... and {len(gaps) - 5} more gaps")
            
            # Calculate total missing days
            total_missing = sum(gap['gap_days'] - 1 for gap in gaps)  # -1 because 1 day gap is normal
            st.warning(f"📊 **Estimated missing trading days**: ~{total_missing}")
            
            # Show data completeness estimate
            total_span = (unique_dates[-1] - unique_dates[0]).days
            completeness = ((total_span - total_missing) / total_span) * 100 if total_span > 0 else 100
            
            if completeness < 90:
                st.error(f"⚠️ **Data completeness estimate**: {completeness:.1f}% - Consider getting more complete data")
            elif completeness < 95:
                st.warning(f"⚠️ **Data completeness estimate**: {completeness:.1f}% - Some gaps present")
            else:
                st.info(f"✅ **Data completeness estimate**: {completeness:.1f}% - Good continuity")
                
        else:
            st.success("✅ **Date Gap Analysis**: No significant gaps detected - good data continuity")

    @staticmethod
    def process_multiple_csvs(uploaded_files, processing_config):
        """Process multiple CSV files and combine them"""
        all_dataframes = []
        detected_tickers = set()
        file_info = []
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        for i, uploaded_file in enumerate(uploaded_files):
            try:
                status_text.text(f"Processing {uploaded_file.name} ({i+1}/{len(uploaded_files)})...")
                
                # Load the file with robust CSV reader
                df = CSVProcessor.robust_csv_reader(uploaded_file, uploaded_file.name)
                
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
                
                # Process based on configuration
                if processing_config['processing_type'] == 'standard_resample':
                    # Standard resampling
                    df_processed = CSVProcessor.resample_ohlc_data(
                        df, 
                        processing_config['target_timeframe'],
                        processing_config.get('filter_start'),
                        processing_config.get('filter_end')
                    )
                    rows_description = f"{len(df)} → {len(df_processed)} rows ({processing_config['target_timeframe']})"
                    
                elif processing_config['processing_type'] == 'custom_candles':
                    # Custom candle creation
                    df_processed = TickerMapper.create_custom_candles(
                        df,
                        processing_config['custom_periods']
                    )
                    periods_count = len(processing_config['custom_periods'])
                    rth_status = " (RTH filtered)" if processing_config.get('rth_filter', True) else " (all hours)"
                    rows_description = f"{len(df)} → {len(df_processed)} custom candles ({periods_count} periods/day{rth_status})"
                
                # Add source info
                df_processed['Source_File'] = uploaded_file.name
                df_processed['Detected_Ticker'] = detected_ticker
                
                all_dataframes.append(df_processed)
                
                file_info.append({
                    'filename': uploaded_file.name,
                    'original_rows': len(df),
                    'processed_rows': len(df_processed),
                    'detected_ticker': detected_ticker,
                    'processing_type': processing_config['processing_type'],
                    'date_range': f"{df_processed['Date'].min()} to {df_processed['Date'].max()}" if not df_processed.empty else "No data"
                })
                
                st.success(f"✅ {uploaded_file.name}: {rows_description} ({detected_ticker})")
                
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
            
            # CRITICAL: Handle overlapping data
            st.info("🔍 Checking for overlapping data...")
            
            # Count records before deduplication
            records_before = len(combined_df)
            
            # Remove duplicates based on Datetime (keep first occurrence)
            combined_df = combined_df.drop_duplicates(subset=['Datetime'], keep='first')
            
            # Count records after deduplication
            records_after = len(combined_df)
            duplicates_removed = records_before - records_after
            
            if duplicates_removed > 0:
                st.warning(f"⚠️ **Overlapping Data Detected**: Removed {duplicates_removed:,} duplicate records")
                st.info("📋 **Resolution**: Kept first occurrence of each datetime (earliest file processed)")
            else:
                st.success("✅ **No Overlapping Data**: All records are unique")
            
            # Sort by datetime after deduplication
            combined_df = combined_df.sort_values(['Date', 'Datetime']).reset_index(drop=True)
            
            # Run date gap analysis
            st.subheader("📅 Date Gap Analysis")
            CSVProcessor.detect_date_gaps(combined_df)
            
            # Remove source columns from final output (keep for debugging)
            output_df = combined_df.drop(['Source_File', 'Detected_Ticker'], axis=1, errors='ignore')
            
            return output_df, file_info
        else:
            return None, file_info

# FIXED: Custom Candle Generator for Single File Resampler
class CustomCandleGenerator:
    """Generate custom candles with flexible time periods"""
    
    @staticmethod
    def create_custom_candles_advanced(df, custom_periods):
        """
        Advanced custom candle creation with full flexibility
        Can create any number of candles per day with any time ranges
        """
        df = df.copy()
        
        # Ensure we have a Datetime column
        df = CSVProcessor.create_datetime_column(df)
        
        # Group by date
        df['Date_only'] = df['Datetime'].dt.date
        daily_groups = df.groupby('Date_only')
        
        custom_candles = []
        
        st.info(f"🕯️ **Creating {len(custom_periods)} custom candles per day**")
        
        for date, day_data in daily_groups:
            for period_idx, period in enumerate(custom_periods):
                period_name = period['name']
                start_time = pd.to_datetime(period['start'], format='%H:%M').time()
                end_time = pd.to_datetime(period['end'], format='%H:%M').time()
                
                # Filter data for this time period
                day_data = day_data.copy()
                day_data['Time_obj'] = day_data['Datetime'].dt.time
                
                # Handle periods that might cross midnight
                if start_time <= end_time:
                    # Normal period within same day
                    period_mask = (day_data['Time_obj'] >= start_time) & (day_data['Time_obj'] <= end_time)
                else:
                    # Period crosses midnight (e.g., 22:00 to 06:00)
                    period_mask = (day_data['Time_obj'] >= start_time) | (day_data['Time_obj'] <= end_time)
                
                period_data = day_data[period_mask]
                
                if not period_data.empty:
                    # Sort by time to ensure proper OHLC
                    period_data = period_data.sort_values('Datetime')
                    
                    # Create OHLC candle for this period
                    candle = {
                        'Date': date,
                        'Datetime': pd.Timestamp.combine(date, start_time),
                        'Period_Name': period_name,
                        'Period_Start': period['start'],
                        'Period_End': period['end'],
                        'Open': period_data['Open'].iloc[0],
                        'High': period_data['High'].max(),
                        'Low': period_data['Low'].min(),
                        'Close': period_data['Close'].iloc[-1],
                        'Records_Used': len(period_data)  # Debug info
                    }
                    
                    # Add volume if present
                    if 'Volume' in period_data.columns:
                        candle['Volume'] = period_data['Volume'].sum()
                    
                    custom_candles.append(candle)
        
        if custom_candles:
            result_df = pd.DataFrame(custom_candles)
            result_df = result_df.sort_values(['Date', 'Period_Start']).reset_index(drop=True)
            
            # Show summary
            total_days = result_df['Date'].nunique()
            candles_per_day = len(custom_periods)
            st.success(f"✅ **Custom Candles Created**: {len(result_df)} candles from {total_days} days")
            st.info(f"📊 **Pattern**: {candles_per_day} candles per day × {total_days} days")
            
            return result_df
        else:
            st.error("❌ No custom candles created - check time periods and data availability")
            return pd.DataFrame()

# Streamlit Interface
st.title('📊 Enhanced CSV Data Handler')
st.write('**Combine multiple CSV files and resample to any timeframe you need**')

# FIXED: Persistent session state for holding data
def initialize_session_state():
    """Initialize session state variables if they don't exist"""
    if 'last_processed_data' not in st.session_state:
        st.session_state['last_processed_data'] = None
    if 'last_processed_filename' not in st.session_state:
        st.session_state['last_processed_filename'] = None
    if 'last_processed_summary' not in st.session_state:
        st.session_state['last_processed_summary'] = None

initialize_session_state()

# Sidebar for held data workspace
with st.sidebar:
    st.header("💾 Data Workspace")
    
    # Check for held data
    has_base_data = 'atr_combiner_base_data' in st.session_state
    has_analysis_data = 'atr_combiner_analysis_data' in st.session_state
    
    if has_base_data or has_analysis_data:
        st.success("📊 **Held Data Available**")
        
        if has_base_data:
            base_filename = st.session_state.get('atr_combiner_base_filename', 'Base Data')
            st.info(f"📈 **Base Timeframe**: {base_filename}")
            
            if st.button("🗑️ Clear Base Data", key="sidebar_clear_base"):
                del st.session_state['atr_combiner_base_data']
                if 'atr_combiner_base_filename' in st.session_state:
                    del st.session_state['atr_combiner_base_filename']
                st.rerun()
        
        if has_analysis_data:
            analysis_filename = st.session_state.get('atr_combiner_analysis_filename', 'Analysis Data')
            st.info(f"📊 **Analysis Timeframe**: {analysis_filename}")
            
            if st.button("🗑️ Clear Analysis Data", key="sidebar_clear_analysis"):
                del st.session_state['atr_combiner_analysis_data']
                if 'atr_combiner_analysis_filename' in st.session_state:
                    del st.session_state['atr_combiner_analysis_filename']
                st.rerun()
        
        st.markdown("---")
        
        # Clear all button
        if st.button("🗑️ **Clear All Held Data**", key="sidebar_clear_all"):
            keys_to_clear = ['atr_combiner_base_data', 'atr_combiner_base_filename', 
                           'atr_combiner_analysis_data', 'atr_combiner_analysis_filename']
            for key in keys_to_clear:
                if key in st.session_state:
                    del st.session_state[key]
            st.rerun()
        
        # Quick navigation to ATR Combiner
        st.markdown("### 🚀 Ready to Combine?")
        st.info("💡 Switch to **Multi-Timeframe ATR Combiner** mode to use held data")
        
    else:
        st.info("💡 **No Data Held**")
        st.write("Process data in any mode and use the 'Hold as Input' buttons to store data for ATR combining.")
        
    st.markdown("---")
    st.markdown("### 🔧 Current Mode")
    mode_display = {
        "📁 Multi-CSV Processor": "📁 Multi-CSV Processor",
        "📈 Public Data Download": "📈 Public Data Download", 
        "🔧 Single File Resampler": "🔧 Single File Resampler",
        "🎯 Multi-Timeframe ATR Combiner": "🎯 ATR Combiner"
    }

# Mode selection
mode = st.selectbox(
    "🎯 Choose Processing Mode",
    ["📁 Multi-CSV Processor", "📈 Public Data Download", "🔧 Single File Resampler", "🎯 Multi-Timeframe ATR Combiner"],
    help="Select what you want to do"
)

# Update sidebar with current mode
current_mode = mode_display.get(mode, mode)
with st.sidebar:
    st.info(f"**{current_mode}**")

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
        type=['csv', 'txt', 'xlsx', 'xls'],
        accept_multiple_files=True,
        help="Select multiple CSV/Excel/TXT files to combine and process",
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
            st.markdown("**Processing Method**")
            
            processing_method = st.radio(
                "How to process the data?",
                ["Standard Resampling", "Custom Candle Periods"],
                help="Choose between standard timeframe resampling or custom time-based candles",
                key="processing_method_multi"
            )
            
            if processing_method == "Standard Resampling":
                # RTH Only filter - checked by default for ATR compatibility
                rth_only = st.checkbox(
                    "Regular Trading Hours Only (9:30 AM - 4:00 PM)",
                    value=True,
                    help="Filter to regular trading hours only - recommended for ATR analysis compatibility",
                    key="rth_only_standard"
                )
                
                if rth_only:
                    custom_start_str = "09:30"
                    custom_end_str = "16:00"
                    st.info("📅 **RTH Filter Active**: 09:30 AM - 4:00 PM (compatible with ATR generator)")
                else:
                    # Manual time filtering if RTH is unchecked
                    use_custom_time = st.checkbox(
                        "Apply Custom Time Filter",
                        help="Set your own time range",
                        key="use_custom_time_standard"
                    )
                    
                    if use_custom_time:
                        custom_start = st.time_input(
                            "Start Time",
                            value=time(9, 30),
                            help="Include data from this time onward",
                            key="custom_start_standard"
                        )
                        
                        custom_end = st.time_input(
                            "End Time", 
                            value=time(16, 0),
                            help="Include data up to this time",
                            key="custom_end_standard"
                        )
                        
                        custom_start_str = custom_start.strftime("%H:%M")
                        custom_end_str = custom_end.strftime("%H:%M")
                        
                        st.info(f"📅 Custom time filter: **{custom_start_str} - {custom_end_str}**")
                    else:
                        custom_start_str = None
                        custom_end_str = None
                        st.warning("⚠️ **No time filtering** - extended hours data may cause issues with ATR generator")
                
                # Set processing config
                processing_config = {
                    'processing_type': 'standard_resample',
                    'target_timeframe': target_timeframe,
                    'filter_start': custom_start_str,
                    'filter_end': custom_end_str
                }
            
            else:
                # Custom candle periods
                st.info("💡 **Create custom candles from time periods**")
                st.write("Each time period becomes one OHLC candle per day")
                
                # RTH Only filter for custom candles too
                rth_only_custom = st.checkbox(
                    "Apply RTH Filter to Custom Candles",
                    value=True,
                    help="Only use data from regular trading hours (9:30-16:00) for custom candle creation",
                    key="rth_only_custom"
                )
                
                # Number of periods per day
                num_periods = st.number_input(
                    "Periods per Day",
                    min_value=1,
                    max_value=8,
                    value=2,
                    help="How many custom candles per trading day"
                )
                
                custom_periods = []
                for i in range(num_periods):
                    st.markdown(f"**Period {i+1}:**")
                    
                    col_a, col_b, col_c = st.columns([1, 1, 1])
                    
                    with col_a:
                        period_name = st.text_input(
                            "Name",
                            value=f"Period_{i+1}",
                            key=f"period_name_{i}",
                            help="Name for this time period"
                        )
                    
                    with col_b:
                        # Default times within RTH
                        default_start_hour = 9 + i * 3 if 9 + i * 3 < 16 else 9 + (i % 2) * 3
                        period_start = st.time_input(
                            "Start",
                            value=time(default_start_hour, 30 if i == 0 else 0),  # First period starts at 9:30
                            key=f"period_start_{i}"
                        )
                    
                    with col_c:
                        default_end_hour = 12 + i * 3 if 12 + i * 3 <= 16 else 12 + (i % 2) * 3
                        period_end = st.time_input(
                            "End",
                            value=time(default_end_hour, 0),
                            key=f"period_end_{i}"
                        )
                    
                    # Validate period is within RTH if RTH filter is enabled
                    if rth_only_custom:
                        start_time = period_start.strftime("%H:%M")
                        end_time = period_end.strftime("%H:%M")
                        
                        if start_time < "09:30" or end_time > "16:00":
                            st.warning(f"⚠️ Period {i+1} extends outside RTH (9:30-16:00)")
                    
                    custom_periods.append({
                        'name': period_name,
                        'start': period_start.strftime("%H:%M"),
                        'end': period_end.strftime("%H:%M")
                    })
                
                # Show period summary
                st.markdown("**📋 Configured Periods:**")
                for period in custom_periods:
                    st.write(f"   • **{period['name']}**: {period['start']} - {period['end']}")
                
                # Show RTH filter status
                if rth_only_custom:
                    st.info("✅ **RTH Filter**: Only data from 9:30-16:00 will be used for candle creation")
                else:
                    st.warning("⚠️ **No RTH Filter**: Extended hours data will be included (may cause ATR generator issues)")
                
                # Example output description
                st.info("📊 **Example Output**: Day 1 → 2 candles, Day 2 → 2 candles, etc.")
                
                # Set processing config
                processing_config = {
                    'processing_type': 'custom_candles',
                    'custom_periods': custom_periods,
                    'rth_filter': rth_only_custom
                }
        
        st.markdown("---")
        
        # Process button - Make this prominent
        if st.button("🚀 **Process Multiple CSVs**", type="primary", use_container_width=True):
            with st.spinner("Processing multiple CSV files..."):
                combined_data, file_info = CSVProcessor.process_multiple_csvs(
                    uploaded_files, 
                    processing_config
                )
                
                if combined_data is not None:
                    st.balloons()  # Celebration animation
                    st.success(f"🎉 **Successfully processed {len(uploaded_files)} files!**")
                    
                    # Generate filename
                    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                    if processing_config['processing_type'] == 'standard_resample':
                        combined_filename = f"Combined_{processing_config['target_timeframe']}_{timestamp}.csv"
                    else:
                        combined_filename = f"Combined_CustomCandles_{len(processing_config['custom_periods'])}periods_{timestamp}.csv"
                    
                    # Store the processed data in session state IMMEDIATELY
                    st.session_state['last_processed_data'] = combined_data.copy()
                    st.session_state['last_processed_filename'] = combined_filename
                    
                    # Show file processing summary
                    st.subheader("📋 Processing Summary")
                    summary_df = pd.DataFrame(file_info)
                    st.session_state['last_processed_summary'] = summary_df.copy()
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
                        if processing_config['processing_type'] == 'standard_resample':
                            st.metric("Timeframe", processing_config['target_timeframe'])
                        else:
                            st.metric("Periods/Day", len(processing_config['custom_periods']))
                    with col4:
                        # FIXED: Correct day counting
                        unique_days = combined_data['Date'].nunique()
                        st.metric("Unique Days", f"{unique_days:,}")
                    
                    # Download and workflow options
                    st.markdown("---")
                    st.subheader("📥 Next Steps")
                    
                    # Create columns for better layout
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.markdown("### 💾 Download Options")
                        
                        # Primary download button
                        st.download_button(
                            "📥 **Download Combined CSV**",
                            data=combined_data.to_csv(index=False),
                            file_name=combined_filename,
                            mime="text/csv",
                            key="download_combined",
                            use_container_width=True,
                            type="primary"
                        )
                        
                        # Additional download options
                        st.download_button(
                            "📋 Download Processing Summary",
                            data=summary_df.to_csv(index=False),
                            file_name=f"processing_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                            mime="text/csv",
                            key="download_summary",
                            use_container_width=True
                        )
                    
                    with col2:
                        st.markdown("### 🔄 Continue Processing")
                        
                        # Hold for ATR Combiner - Always available
                        st.markdown("**Use in ATR Combiner:**")
                        
                        col_a, col_b = st.columns(2)
                        with col_a:
                            # FIXED: Hold buttons that don't break download
                            if st.button("📊 Hold as Base\n(ATR Source)", key="use_as_base", use_container_width=True):
                                st.session_state['atr_combiner_base_data'] = combined_data.copy()
                                st.session_state['atr_combiner_base_filename'] = combined_filename
                                st.success("✅ Saved as Base!")
                                st.info("💡 Switch to **ATR Combiner** mode")
                        
                        with col_b:
                            if st.button("📈 Hold as Analysis\n(Intraday)", key="use_as_analysis", use_container_width=True):
                                st.session_state['atr_combiner_analysis_data'] = combined_data.copy()
                                st.session_state['atr_combiner_analysis_filename'] = combined_filename
                                st.success("✅ Saved as Analysis!")
                                st.info("💡 Switch to **ATR Combiner** mode")
                        
                        # Show current hold status
                        if 'atr_combiner_base_data' in st.session_state:
                            st.info("📊 **Base data held** in workspace")
                        if 'atr_combiner_analysis_data' in st.session_state:
                            st.info("📈 **Analysis data held** in workspace")
                    
                    # Show final data characteristics
                    st.markdown("---")
                    st.subheader("📊 Final Dataset Characteristics")
                    
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        date_span = (combined_data['Date'].max() - combined_data['Date'].min()).days
                        st.metric("📅 Date Span", f"{date_span} days")
                    
                    with col2:
                        if processing_config['processing_type'] == 'standard_resample':
                            st.metric("⏱️ Timeframe", processing_config['target_timeframe'])
                        else:
                            st.metric("🕐 Periods/Day", len(processing_config['custom_periods']))
                    
                    with col3:
                        unique_days = combined_data['Date'].nunique()
                        avg_daily_records = len(combined_data) / max(1, unique_days)
                        st.metric("📊 Avg Records/Day", f"{avg_daily_records:.1f}")
                    
                    # Show what's ready for ATR analysis
                    st.markdown("### 🎯 ATR Analysis Ready")
                    st.info("""
                    **Your processed data is now ready for ATR analysis:**
                    - ✅ Clean OHLC data with validation
                    - ✅ Proper datetime formatting 
                    - ✅ Consistent timeframe structure
                    - ✅ Duplicate removal and gap analysis
                    - ✅ Compatible with ATR Level Analyzer
                    """)
                    
                    # Show sample of custom candle output if applicable
                    if processing_config['processing_type'] == 'custom_candles':
                        st.markdown("---")
                        st.subheader("🔍 Custom Candle Details")
                        
                        # Show how many candles per day
                        if 'Period_Name' in combined_data.columns:
                            sample_date = combined_data['Date'].iloc[0]
                            day_sample = combined_data[combined_data['Date'] == sample_date]
                            
                            st.info(f"📊 **Example for {sample_date}**: {len(day_sample)} custom candles created")
                            st.dataframe(day_sample[['Period_Name', 'Period_Start', 'Period_End', 'Open', 'High', 'Low', 'Close']], use_container_width=True)
                    
                else:
                    st.error("❌ Failed to process CSV files. Please check the file processing summary above.")

# FIXED: Show persistent actions for last processed data
if st.session_state.get('last_processed_data') is not None:
    st.markdown("---")
    st.subheader("🔄 **Continue with Last Processed Data**")
    
    last_data = st.session_state['last_processed_data']
    last_filename = st.session_state['last_processed_filename']
    
    st.info(f"📊 **Available**: {last_filename} ({len(last_data):,} records)")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        # Persistent download button
        st.download_button(
            "📥 **Download Again**",
            data=last_data.to_csv(index=False),
            file_name=last_filename,
            mime="text/csv",
            key="download_persistent",
            use_container_width=True
        )
    
    with col2:
        # Persistent hold as base
        if st.button("📊 **Hold as Base**", key="hold_base_persistent", use_container_width=True):
            st.session_state['atr_combiner_base_data'] = last_data.copy()
            st.session_state['atr_combiner_base_filename'] = last_filename
            st.success("✅ Held as Base!")
            st.rerun()
    
    with col3:
        # Persistent hold as analysis
        if st.button("📈 **Hold as Analysis**", key="hold_analysis_persistent", use_container_width=True):
            st.session_state['atr_combiner_analysis_data'] = last_data.copy()
            st.session_state['atr_combiner_analysis_filename'] = last_filename
            st.success("✅ Held as Analysis!")
            st.rerun()

else:
        # Show helpful instructions when no files are uploaded
        st.info("👆 **Please upload multiple CSV files to get started**")
        
        # Show example of what files should look like
        with st.expander("📋 Expected File Format", expanded=False):
            st.markdown("""
            **Your CSV files should contain these columns (any format):**
            
            **Standard Format:**
            - **Date** (or Datetime, Time)
            - **Open**, **High**, **Low**, **Close**
            - **Volume** (optional)
            
            **Short Format (also supported):**
            - **Date** (or Datetime, Time)  
            - **o**, **h**, **l**, **c** (lowercase single letters)
            - **v** (volume - optional)
            
            **Unlabeled Format (NEW - Smart Detection):**
            - **Column 1**: Date/Datetime (any format)
            - **Column 2**: Open price
            - **Column 3**: High price
            - **Column 4**: Low price
            - **Column 5**: Close price
            - **Column 6**: Volume (optional)
            
            **Mixed Format Examples:**
            - `Date, o, h, l, c, v`
            - `datetime, Open, High, Low, Close, Volume`
            - `date, time, O, H, L, C`
            - `9/23/2012 20:35, 4100, 4110, 4095, 4105, 1000` (unlabeled)
            
            **Example filenames that work well:**
            - `SPX_20240101.csv`
            - `AAPL_1min_data.csv`
            - `ES_intraday.csv`
            - `data_2024_01_01.csv`
            - `unlabeled_ohlc_data.csv`
            
            **The system will:**
            - ✅ Auto-detect ticker symbols from filenames
            - ✅ Handle both long (Open, High, Low, Close) and short (o, h, l, c) formats
            - ✅ **NEW**: Smart detect unlabeled columns and assume Date + OHLC order
            - ✅ Warn if mixed tickers are found
            - ✅ Standardize all column names automatically
            - ✅ Handle various date/time formats
            - ✅ **NEW**: Show warnings when assumptions are made
            """)

        
        # Show sample workflow
        with st.expander("🔧 Sample Workflows", expanded=False):
            st.markdown("""
            **🎯 Standard Resampling Workflow:**
            1. Upload 25 daily 1-minute CSV files
            2. Choose "Standard Resampling" 
            3. Set timeframe to **10T** (10 minutes)
            4. Apply time filter **9:30 - 16:00** (market hours)
            5. Get single combined file with 10-minute bars
            
            **Custom Candle Periods Workflow:**
            1. Upload multiple CSV files with intraday data
            2. Choose "Custom Candle Periods"
            3. Define periods: **Morning (9:00-12:00)**, **Afternoon (12:00-16:00)**
            4. Each day creates 2 custom OHLC candles
            5. Perfect for session-based analysis
            
            **Custom Candle Output Example:**
            ```
            Date        Period_Name  Period_Start  Period_End  Open   High   Low    Close
            2024-01-01  Morning      09:00        12:00       4100   4150   4090   4140
            2024-01-01  Afternoon    12:00        16:00       4140   4180   4130   4175
            2024-01-02  Morning      09:00        12:00       4175   4200   4160   4190
            2024-01-02  Afternoon    12:00        16:00       4190   4210   4180   4205
            ```
            """)


# ========================================================================================
# PUBLIC DATA DOWNLOAD
# ========================================================================================
if mode == "📈 Public Data Download":
    st.header("📈 Public Data Download")
    st.write("Download financial data from public sources and export as CSV")
    
    st.info("⚠️ **Note:** Public sources have limitations. For extensive historical intraday data, use the Multi-CSV Processor with broker files.")
    
    # Configuration in main frame
    st.subheader("🎯 Download Configuration")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**📈 Ticker & Data Source**")
        
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
    
    with col2:
        st.markdown("**📅 Date Range Configuration**")
        
        # Check if we have held data to suggest smart dates
        held_base_data = st.session_state.get('atr_combiner_base_data')
        held_analysis_data = st.session_state.get('atr_combiner_analysis_data')
        
        suggested_start = None
        suggested_end = None
        suggestion_context = ""
        
        if held_base_data is not None:
            # We have held base data - suggest dates that complement it
            held_start = held_base_data['Date'].min()
            held_end = held_base_data['Date'].max()
            
            # Convert to proper date format for comparison
            if hasattr(held_start, 'date'):
                held_start = held_start.date()
            if hasattr(held_end, 'date'):
                held_end = held_end.date()
            
            # Suggest extending the range
            suggested_start = held_start - timedelta(days=365)  # 1 year before
            suggested_end = held_end + timedelta(days=30)  # 30 days after
            suggestion_context = f"📊 **Smart suggestion based on held base data** ({held_start} to {held_end})"
            
            st.info(f"🔍 **Detected held base data**: {held_start} to {held_end}")
            st.info("💡 **Suggested range**: Extended to provide ATR buffer and overlap")
            
        elif held_analysis_data is not None:
            # We have held analysis data - suggest dates that provide good ATR coverage
            held_start = held_analysis_data['Date'].min()
            held_end = held_analysis_data['Date'].max()
            
            # Convert to proper date format
            if hasattr(held_start, 'date'):
                held_start = held_start.date()
            if hasattr(held_end, 'date'):
                held_end = held_end.date()
            
            # For base data to support analysis, suggest earlier start
            suggested_start = held_start - timedelta(days=180)  # 6 months before for ATR
            suggested_end = held_end + timedelta(days=5)  # Few days after
            suggestion_context = f"📈 **Smart suggestion based on held analysis data** ({held_start} to {held_end})"
            
            st.info(f"🔍 **Detected held analysis data**: {held_start} to {held_end}")
            st.info("💡 **Suggested range**: Extended back 6 months to provide ATR calculation buffer")
    
    # Date range selection - full width
    st.subheader("📅 Date Range Selection")
    
    date_mode = st.radio(
        "Date Selection Mode",
        ["Smart ATR Range", "Custom Range", "Suggested Range"] if suggested_start else ["Smart ATR Range", "Custom Range"],
        help="Smart mode adds buffer for ATR calculation, Suggested uses held data context",
        horizontal=True
    )
    
    if date_mode == "Suggested Range" and suggested_start:
        st.success(suggestion_context)
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Use suggested dates as defaults but allow modification
            daily_start = st.date_input(
                "Daily Data Start Date",
                value=suggested_start,
                min_value=date(1850, 1, 1),
                max_value=date.today(),
                help="Suggested based on your held data - extends back to provide ATR buffer"
            )
        
        with col2:
            daily_end = st.date_input(
                "Daily Data End Date", 
                value=suggested_end,
                min_value=date(1850, 1, 1),
                max_value=date.today(),
                help="Suggested to complement your held data"
            )
        
        # Show the logic
        with st.expander("🎯 **Suggestion Logic**", expanded=False):
            if held_base_data is not None:
                st.info(f"   • Held base data: {held_base_data['Date'].min().date() if hasattr(held_base_data['Date'].min(), 'date') else held_base_data['Date'].min()} to {held_base_data['Date'].max().date() if hasattr(held_base_data['Date'].max(), 'date') else held_base_data['Date'].max()}")
                st.info(f"   • Suggested: Extend 1 year back, 30 days forward")
                st.info(f"   • Purpose: Provide overlap and additional data coverage")
            elif held_analysis_data is not None:
                st.info(f"   • Held intraday data: {held_analysis_data['Date'].min().date() if hasattr(held_analysis_data['Date'].min(), 'date') else held_analysis_data['Date'].min()} to {held_analysis_data['Date'].max().date() if hasattr(held_analysis_data['Date'].max(), 'date') else held_analysis_data['Date'].max()}")
                st.info(f"   • Suggested: 6 months back for ATR buffer")
                st.info(f"   • Purpose: Provide sufficient history for ATR calculation")
        
    elif date_mode == "Smart ATR Range":
        col1, col2 = st.columns(2)
        
        with col1:
            # Simple date range with automatic buffer
            analysis_start = st.date_input(
                "Analysis Period Start Date",
                value=suggested_start if suggested_start else date(2024, 1, 1),
                min_value=date(1850, 1, 1),
                max_value=date.today(),
                help="When you want your analysis period to begin"
            )
            
            analysis_end = st.date_input(
                "Analysis Period End Date", 
                value=suggested_end if suggested_end else date.today(),
                min_value=date(1850, 1, 1),
                max_value=date.today(),
                help="When you want your analysis period to end"
            )
        
        with col2:
            # Auto-calculate buffer with extended range for larger timeframes
            buffer_months = st.slider(
                "Buffer for ATR Calculation", 
                4, 300, 12,  # 4 months to 25 years, default 1 year
                help="Historical data buffer based on target ATR timeframe"
            )
            data_start = analysis_start - timedelta(days=buffer_months * 30)
            data_end = analysis_end + timedelta(days=5)
            
            buffer_years = buffer_months / 12
            st.info(f"📊 Data will span: {data_start} to {data_end}")
            st.info(f"📈 Buffer: {buffer_months} months ({buffer_years:.1f} years)")
            
            # Show ATR calculation guidance based on 84-period rule
            if buffer_months >= 84:  # 7 years for monthly ATR
                st.success("✅ **Excellent** for monthly ATR calculations (7+ years)")
            elif buffer_months >= 20:  # ~1.6 years for weekly ATR  
                st.success("✅ **Good** for weekly ATR calculations (1.6+ years)")
            elif buffer_months >= 4:  # 4 months for daily ATR
                st.success("✅ **Adequate** for daily ATR calculations (4+ months)")
            else:
                st.error("❌ **Insufficient** - Less than 4 months not recommended for any ATR calculation")
                
            # Educational guidance
            st.info("🎓 **ATR Buffer Requirements (84-period rule):**")
            st.info("   • **Daily ATR**: 4+ months (84 days minimum)")
            st.info("   • **Weekly ATR**: 20+ months (84 weeks ≈ 1.6 years)")
            st.info("   • **Monthly ATR**: 84+ months (7 years)")  
            st.info("   • **Quarterly ATR**: 252+ months (21 years)")
            
            if buffer_months >= 84:
                st.info("🎯 **Your selection supports all ATR timeframes**")
            elif buffer_months >= 20:
                st.info("🎯 **Your selection supports daily & weekly ATR**")
            elif buffer_months >= 4:
                st.info("🎯 **Your selection supports daily ATR only**")
    
    else:
        # Manual date range
        col1, col2 = st.columns(2)
        
        with col1:
            data_start = st.date_input(
                "Data Start Date", 
                value=suggested_start if suggested_start else date(2023, 1, 1),
                min_value=date(1850, 1, 1),
                max_value=date.today(),
                help="Start date for data download"
            )
        
        with col2:
            data_end = st.date_input(
                "Data End Date", 
                value=suggested_end if suggested_end else date.today(),
                min_value=date(1850, 1, 1),
                max_value=date.today(),
                help="End date for data download"
            )
    
    if st.button("🚀 Download Data", type="primary"):
        if not ticker:
            st.error("❌ Please enter a ticker symbol")
        else:
            mapped_ticker = TickerMapper.get_public_ticker(ticker)
            
            with st.spinner(f'Downloading data for {mapped_ticker}...'):
                try:
                    downloaded_data = yf.download(mapped_ticker, start=data_start, end=data_end, interval='1d', progress=False)
                    
                    if not downloaded_data.empty:
                        # Reset index and clean columns
                        downloaded_data.reset_index(inplace=True)
                        
                        # Handle MultiIndex columns
                        if isinstance(downloaded_data.columns, pd.MultiIndex):
                            downloaded_data.columns = downloaded_data.columns.get_level_values(0)
                        
                        # Ensure Date column
                        if 'Date' not in downloaded_data.columns and len(downloaded_data.columns) > 0:
                            downloaded_data.rename(columns={downloaded_data.columns[0]: 'Date'}, inplace=True)
                        
                        # Validate date completeness
                        downloaded_data['Date'] = pd.to_datetime(downloaded_data['Date'])
                        actual_start = downloaded_data['Date'].min().date()
                        actual_end = downloaded_data['Date'].max().date()
                        
                        st.success(f"✅ Downloaded {len(downloaded_data)} records")
                        st.info(f"📅 **Requested range**: {data_start} to {data_end}")
                        st.info(f"📅 **Actual range**: {actual_start} to {actual_end}")
                        
                        # Check for date gaps
                        if actual_start > data_start:
                            missing_days = (actual_start - data_start).days
                            st.warning(f"⚠️ **Missing early data**: {missing_days} days missing from start of requested range")
                            st.warning(f"Data starts {actual_start} instead of {data_start}")
                        
                        if actual_end < data_end:
                            missing_days = (data_end - actual_end).days
                            st.warning(f"⚠️ **Missing recent data**: {missing_days} days missing from end of requested range")
                            st.warning(f"Data ends {actual_end} instead of {data_end}")
                        
                        # Check for weekends/holidays vs actual gaps
                        requested_trading_days = pd.bdate_range(start=data_start, end=data_end)
                        actual_trading_days = pd.to_datetime(downloaded_data['Date']).dt.date
                        
                        # Calculate trading day completeness (more accurate)
                        trading_day_completeness = (len(actual_trading_days) / len(requested_trading_days)) * 100 if len(requested_trading_days) > 0 else 100
                        
                        # Calculate calendar day completeness (original method)
                        requested_calendar_days = (data_end - data_start).days
                        actual_calendar_days = len(downloaded_data)
                        calendar_completeness = (actual_calendar_days / requested_calendar_days) * 100 if requested_calendar_days > 0 else 100
                        
                        st.info(f"📊 **Trading days analysis**: {len(actual_trading_days)} of {len(requested_trading_days)} trading days ({trading_day_completeness:.1f}%)")
                        st.info(f"📅 **Calendar days analysis**: {actual_calendar_days} of {requested_calendar_days} calendar days ({calendar_completeness:.1f}%)")
                        
                        # Show the difference
                        if abs(trading_day_completeness - calendar_completeness) > 10:
                            st.info("💡 **Note**: Large difference between trading day vs calendar day completeness is normal (weekends/holidays)")
                        
                        # Use trading day completeness for quality assessment
                        completeness = trading_day_completeness
                        
                        # More nuanced quality assessment
                        if completeness >= 95:
                            st.success(f"✅ **Excellent trading day completeness**: {completeness:.1f}%")
                        elif completeness >= 85:
                            st.info(f"✅ **Good trading day completeness**: {completeness:.1f}% - Some holidays/market closures missing")
                        elif completeness >= 70:
                            st.warning(f"⚠️ **Moderate trading day completeness**: {completeness:.1f}% - May include extended market closures")
                        else:
                            st.error(f"❌ **Low trading day completeness**: {completeness:.1f}% - Significant data gaps present")
                            st.error("🚨 **CRITICAL**: This data may be insufficient for reliable analysis!")
                            st.error("**Recommendation**: Check ticker symbol, adjust date range, or use alternative data source")
                        
                        # Show what might be missing
                        if completeness < 95:
                            missing_trading_days = len(requested_trading_days) - len(actual_trading_days)
                            st.info(f"📊 **Missing trading days**: ~{missing_trading_days} days")
                            
                            # Common reasons for missing data
                            st.info("**Common reasons for missing trading days:**")
                            st.info("   • Market holidays (Christmas, New Year, etc.)")
                            st.info("   • Extended market closures (9/11, extreme weather)")
                            st.info("   • Data source limitations for older dates")
                            st.info("   • Instrument-specific trading schedules")
                            
                            if ticker.upper() in ['SPX', '^GSPC', 'SPY']:
                                st.info("   • S&P 500 index: Limited historical data availability")
                            elif ticker.upper() in ['BTC', 'ETH', 'BTC-USD', 'ETH-USD']:
                                st.info("   • Crypto: 24/7 trading, gaps likely indicate data source issues")
                            elif '=F' in mapped_ticker:
                                st.info("   • Futures: May have different trading schedules or contract rollovers")
                        
                        # Add data quality metrics to the dataframe for later reference
                        downloaded_data.attrs['requested_start'] = data_start
                        downloaded_data.attrs['requested_end'] = data_end
                        downloaded_data.attrs['actual_start'] = actual_start
                        downloaded_data.attrs['actual_end'] = actual_end
                        downloaded_data.attrs['completeness'] = completeness
                        downloaded_data.attrs['data_source'] = f"Yahoo Finance ({mapped_ticker})"
                        
                        # Show preview
                        st.subheader("📋 Data Preview")
                        st.dataframe(downloaded_data.head(), use_container_width=True)
                        
                        # Download button
                        filename = f"{ticker}_data_{data_start.strftime('%Y%m%d')}_to_{data_end.strftime('%Y%m%d')}.csv"
                        st.download_button(
                            "📥 Download CSV",
                            data=downloaded_data.to_csv(index=False),
                            file_name=filename,
                            mime="text/csv"
                        )
                        
                        # Option to use in Multi-Timeframe ATR Combiner
                        st.markdown("### 🔄 Or Use in Multi-Timeframe ATR Combiner")
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            if st.button("📊 Use as Base Timeframe (ATR Source)", key="yahoo_use_as_base"):
                                st.session_state['atr_combiner_base_data'] = downloaded_data
                                st.session_state['atr_combiner_base_filename'] = filename
                                st.success("✅ Data saved as Base Timeframe for ATR Combiner!")
                                st.info("💡 Now switch to 'Multi-Timeframe ATR Combiner' mode to use this data.")
                        
                        with col2:
                            if st.button("📈 Use as Analysis Timeframe", key="yahoo_use_as_analysis"):
                                st.session_state['atr_combiner_analysis_data'] = downloaded_data
                                st.session_state['atr_combiner_analysis_filename'] = filename
                                st.success("✅ Data saved as Analysis Timeframe for ATR Combiner!")
                                st.info("💡 Now switch to 'Multi-Timeframe ATR Combiner' mode to use this data.")
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
# SINGLE FILE RESAMPLER (FIXED - Real Custom Candle Generator)
# ========================================================================================
elif mode == "🔧 Single File Resampler":
    st.header("🔧 Single File Resampler")
    st.write("**Upload a single CSV and resample it to different timeframes**")
    
    # Single file upload
    single_file = st.file_uploader(
        "Upload Single CSV File",
        type=['csv', 'txt', 'xlsx', 'xls'], 
        help="Upload one CSV/Excel/TXT file to resample"
    )
    
    if single_file:
        st.success(f"✅ File uploaded: {single_file.name}")
        
        # Load and preview the file
        try:
            df = CSVProcessor.robust_csv_reader(single_file, single_file.name)
            
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
                
                # Processing method selection
                processing_method = st.radio(
                    "Processing Method",
                    ["Standard Resampling", "Custom Candle Generator"],
                    help="Choose between standard timeframe resampling or custom candle periods",
                    key="processing_method_single"
                )
                
                if processing_method == "Standard Resampling":
                    # Standard timeframes
                    timeframe_category = st.selectbox(
                        "Timeframe Category",
                        ["Minutes", "Hours", "Daily/Weekly"]
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
                    else:  # Daily/Weekly
                        resample_timeframe = st.selectbox(
                            "Target Timeframe",
                            ["1D", "WEEKLY", "MONTHLY", "QUARTERLY"],
                            help="1D = Daily aggregation from intraday data"
                        )
                
                else:  # Custom Candle Generator
                    st.info("🕯️ **Custom Candle Generator**")
                    st.write("Create multiple custom candles per day with flexible time periods")
                    
                    # Number of candles per day
                    num_candles = st.number_input(
                        "Candles per Day",
                        min_value=1,
                        max_value=10,
                        value=2,
                        help="How many custom candles to create per trading day"
                    )
                    
                    # Preset templates
                    preset_templates = {
                        "Morning/Afternoon": [
                            {"name": "Morning", "start": "09:30", "end": "12:00"},
                            {"name": "Afternoon", "start": "12:00", "end": "16:00"}
                        ],
                        "3-Period Day": [
                            {"name": "Morning", "start": "09:00", "end": "11:00"},
                            {"name": "Midday", "start": "11:00", "end": "14:00"},
                            {"name": "Afternoon", "start": "14:00", "end": "16:00"}
                        ],
                        "4-Period Day": [
                            {"name": "Opening", "start": "09:30", "end": "11:00"},
                            {"name": "Morning", "start": "11:00", "end": "12:30"},
                            {"name": "Afternoon", "start": "12:30", "end": "14:30"},
                            {"name": "Closing", "start": "14:30", "end": "16:00"}
                        ],
                        "Custom": []
                    }
                    
                    template_choice = st.selectbox(
                        "Template",
                        list(preset_templates.keys()),
                        help="Choose a preset template or create custom periods"
                    )
                    
                    if template_choice != "Custom" and len(preset_templates[template_choice]) == num_candles:
                        st.success(f"✅ **Using {template_choice} template**")
                        custom_periods = preset_templates[template_choice].copy()
                        
                        # Show template preview
                        st.markdown("**📋 Template Preview:**")
                        for i, period in enumerate(custom_periods):
                            st.write(f"   {i+1}. **{period['name']}**: {period['start']} - {period['end']}")
                        
                        # Allow modification
                        if st.checkbox("Modify Template", help="Customize the template periods"):
                            for i in range(num_candles):
                                st.markdown(f"**Candle {i+1}:**")
                                col_a, col_b, col_c = st.columns([1, 1, 1])
                                
                                with col_a:
                                    custom_periods[i]['name'] = st.text_input(
                                        "Name",
                                        value=custom_periods[i]['name'],
                                        key=f"template_name_{i}"
                                    )
                                
                                with col_b:
                                    start_time = st.time_input(
                                        "Start",
                                        value=pd.to_datetime(custom_periods[i]['start'], format='%H:%M').time(),
                                        key=f"template_start_{i}"
                                    )
                                    custom_periods[i]['start'] = start_time.strftime("%H:%M")
                                
                                with col_c:
                                    end_time = st.time_input(
                                        "End",
                                        value=pd.to_datetime(custom_periods[i]['end'], format='%H:%M').time(),
                                        key=f"template_end_{i}"
                                    )
                                    custom_periods[i]['end'] = end_time.strftime("%H:%M")
                    
                    else:
                        # Custom configuration
                        st.info("🔧 **Custom Configuration**")
                        custom_periods = []
                        
                        for i in range(num_candles):
                            st.markdown(f"**Candle {i+1}:**")
                            col_a, col_b, col_c = st.columns([1, 1, 1])
                            
                            with col_a:
                                period_name = st.text_input(
                                    "Name",
                                    value=f"Candle_{i+1}",
                                    key=f"custom_name_{i}",
                                    help="Name for this candle period"
                                )
                            
                            with col_b:
                                # Smart defaults based on candle number
                                if i == 0:
                                    default_start = time(9, 30)
                                else:
                                    # Calculate based on previous candle
                                    hours_per_candle = 6.5 / num_candles  # 6.5 hours in trading day
                                    start_hour = int(9.5 + (i * hours_per_candle))
                                    start_min = int(((9.5 + (i * hours_per_candle)) % 1) * 60)
                                    default_start = time(start_hour, start_min)
                                
                                period_start = st.time_input(
                                    "Start",
                                    value=default_start,
                                    key=f"custom_start_{i}"
                                )
                            
                            with col_c:
                                # Smart defaults for end time
                                if i == num_candles - 1:  # Last candle
                                    default_end = time(16, 0)
                                else:
                                    hours_per_candle = 6.5 / num_candles
                                    end_hour = int(9.5 + ((i + 1) * hours_per_candle))
                                    end_min = int(((9.5 + ((i + 1) * hours_per_candle)) % 1) * 60)
                                    default_end = time(end_hour, end_min)
                                
                                period_end = st.time_input(
                                    "End",
                                    value=default_end,
                                    key=f"custom_end_{i}"
                                )
                            
                            custom_periods.append({
                                'name': period_name,
                                'start': period_start.strftime("%H:%M"),
                                'end': period_end.strftime("%H:%M")
                            })
                    
                    # Show final configuration
                    st.markdown("**📋 Final Custom Candle Configuration:**")
                    for i, period in enumerate(custom_periods):
                        st.write(f"   {i+1}. **{period['name']}**: {period['start']} - {period['end']}")
                    
                    # Validation
                    overlaps = []
                    for i in range(len(custom_periods)):
                        for j in range(i+1, len(custom_periods)):
                            start1 = pd.to_datetime(custom_periods[i]['start'], format='%H:%M').time()
                            end1 = pd.to_datetime(custom_periods[i]['end'], format='%H:%M').time()
                            start2 = pd.to_datetime(custom_periods[j]['start'], format='%H:%M').time()
                            end2 = pd.to_datetime(custom_periods[j]['end'], format='%H:%M').time()
                            
                            # Check for overlap
                            if (start1 <= start2 < end1) or (start1 < end2 <= end1) or (start2 <= start1 < end2):
                                overlaps.append(f"{custom_periods[i]['name']} overlaps with {custom_periods[j]['name']}")
                    
                    if overlaps:
                        st.warning("⚠️ **Period Overlaps Detected:**")
                        for overlap in overlaps:
                            st.warning(f"   • {overlap}")
                        st.info("💡 **Note**: Overlaps are allowed but may result in duplicated data")
                    else:
                        st.success("✅ **No overlaps detected** - clean period separation")
            
            with col2:
                st.subheader("⚙️ Time Filtering")
                
                # Time filtering options
                time_filter_mode = st.selectbox(
                    "Time Filter Mode",
                    ["No Filter (24 Hours)", "Regular Trading Hours (RTH)", "Custom Session", "Custom Time Range"],
                    help="Choose how to filter data by time before resampling"
                )
                
                if time_filter_mode == "Regular Trading Hours (RTH)":
                    filter_start_str = "09:30"
                    filter_end_str = "16:00"
                    st.info(f"📅 RTH Filter: {filter_start_str} - {filter_end_str}")
                    
                elif time_filter_mode == "Custom Session":
                    st.markdown("**🎯 Preset Sessions:**")
                    session_preset = st.selectbox(
                        "Choose Session Type",
                        ["ES Futures (18:00-17:00)", "Crypto UTC Reset (00:00-23:59)", "Forex London (08:00-17:00)", "Custom"],
                        help="Common session boundaries for different instruments"
                    )
                    
                    if session_preset == "ES Futures (18:00-17:00)":
                        filter_start_str = "18:00"
                        filter_end_str = "17:00"
                        st.success("🎯 **ES Futures Session**: 18:00 today → 17:00 next day")
                        st.info("Creates proper futures daily sessions (crosses midnight)")
                        
                    elif session_preset == "Crypto UTC Reset (00:00-23:59)":
                        filter_start_str = "00:00"
                        filter_end_str = "23:59"
                        st.success("🎯 **Crypto UTC Reset**: Midnight to midnight")
                        st.info("Standard crypto daily candles (UTC timezone)")
                        
                    elif session_preset == "Forex London (08:00-17:00)":
                        filter_start_str = "08:00"
                        filter_end_str = "17:00"
                        st.success("🎯 **Forex London Session**: 08:00 - 17:00")
                        st.info("London market hours (no midnight crossing)")
                        
                    else:  # Custom
                        col_a, col_b = st.columns(2)
                        with col_a:
                            custom_start = st.time_input(
                                "Session Start Time",
                                value=time(18, 0),
                                help="When your custom session starts"
                            )
                        with col_b:
                            custom_end = st.time_input(
                                "Session End Time", 
                                value=time(17, 0),
                                help="When your custom session ends (can be next day)"
                            )
                        
                        filter_start_str = custom_start.strftime("%H:%M")
                        filter_end_str = custom_end.strftime("%H:%M")
                        
                        # Check if it crosses midnight
                        crosses_midnight = custom_start >= custom_end
                        if crosses_midnight:
                            st.info(f"🕐 **Custom Session**: {filter_start_str} today → {filter_end_str} next day")
                            st.warning("⚠️ **Crosses midnight** - session spans two calendar days")
                        else:
                            st.info(f"🕐 **Custom Session**: {filter_start_str} - {filter_end_str} (same day)")
                            
                elif time_filter_mode == "Custom Time Range":
                    st.markdown("**🕐 Custom Time Range:**")
                    col_a, col_b = st.columns(2)
                    
                    with col_a:
                        filter_start = st.time_input("Start Time", value=time(9, 30))
                    with col_b:
                        filter_end = st.time_input("End Time", value=time(16, 0))
                    
                    filter_start_str = filter_start.strftime("%H:%M")
                    filter_end_str = filter_end.strftime("%H:%M")
                    
                    st.info(f"📅 Custom filter: {filter_start_str} - {filter_end_str}")
                    
                else:  # No Filter
                    filter_start_str = None
                    filter_end_str = None
                    st.info("📅 No time filtering - using all 24 hours")
            
            # Process button
            if st.button("🔄 Process Data", type="primary"):
                try:
                    with st.spinner("Processing data..."):
                        if processing_method == "Standard Resampling":
                            # Standard resampling
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
                            resampled_filename = f"{base_name}_resampled_{resample_timeframe}_{datetime.now().strftime('%H%M%S')}.csv"
                            
                            # Store in session state
                            st.session_state['last_processed_data'] = resampled_data.copy()
                            st.session_state['last_processed_filename'] = resampled_filename
                            
                        else:  # Custom Candle Generator
                            # Use the advanced custom candle generator
                            custom_candle_data = CustomCandleGenerator.create_custom_candles_advanced(
                                df, custom_periods
                            )
                            
                            if not custom_candle_data.empty:
                                st.success(f"✅ Custom Candles Created: {len(df)} → {len(custom_candle_data)} custom candles")
                                
                                # Show custom candle preview
                                st.subheader("🕯️ Custom Candle Data Preview")
                                st.dataframe(custom_candle_data.head(), use_container_width=True)
                                
                                # Summary metrics
                                col1, col2, col3, col4 = st.columns(4)
                                with col1:
                                    st.metric("Original Records", len(df))
                                with col2:
                                    st.metric("Custom Candles", len(custom_candle_data))
                                with col3:
                                    unique_days = custom_candle_data['Date'].nunique()
                                    st.metric("Trading Days", unique_days)
                                with col4:
                                    candles_per_day = len(custom_candle_data) / max(1, unique_days)
                                    st.metric("Candles/Day", f"{candles_per_day:.1f}")
                                
                                # Show sample day breakdown
                                st.subheader("📋 Sample Day Breakdown")
                                sample_date = custom_candle_data['Date'].iloc[0]
                                day_sample = custom_candle_data[custom_candle_data['Date'] == sample_date]
                                
                                st.info(f"📅 **Sample Date**: {sample_date} → {len(day_sample)} candles created")
                                st.dataframe(day_sample[['Period_Name', 'Period_Start', 'Period_End', 'Open', 'High', 'Low', 'Close', 'Records_Used']], use_container_width=True)
                                
                                # Download custom candle file
                                base_name = single_file.name.split('.')[0]
                                custom_filename = f"{base_name}_custom_{len(custom_periods)}candles_{datetime.now().strftime('%H%M%S')}.csv"
                                
                                # Store in session state
                                st.session_state['last_processed_data'] = custom_candle_data.copy()
                                st.session_state['last_processed_filename'] = custom_filename
                                
                                resampled_data = custom_candle_data  # For consistency in download section
                                resampled_filename = custom_filename
                            else:
                                st.error("❌ Failed to create custom candles")
                                return None
                        
                        # Download section
                        st.markdown("---")
                        st.subheader("📥 Download & Next Steps")
                        
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.markdown("### 💾 Download")
                            
                            # Primary download button
                            st.download_button(
                                "📥 **Download Processed CSV**",
                                data=resampled_data.to_csv(index=False),
                                file_name=resampled_filename,
                                mime="text/csv",
                                key="download_resampled",
                                use_container_width=True,
                                type="primary"
                            )
                        
                        with col2:
                            st.markdown("### 🔄 Use in ATR Combiner")
                            
                            col_a, col_b = st.columns(2)
                            
                            with col_a:
                                if st.button("📊 Hold as Base\n(ATR Source)", key="resample_use_as_base", use_container_width=True):
                                    st.session_state['atr_combiner_base_data'] = resampled_data.copy()
                                    st.session_state['atr_combiner_base_filename'] = resampled_filename
                                    st.success("✅ Saved as Base!")
                                    st.info("💡 Switch to **ATR Combiner** mode")
                            
                            with col_b:
                                if st.button("📈 Hold as Analysis\n(Intraday)", key="resample_use_as_analysis", use_container_width=True):
                                    st.session_state['atr_combiner_analysis_data'] = resampled_data.copy()
                                    st.session_state['atr_combiner_analysis_filename'] = resampled_filename
                                    st.success("✅ Saved as Analysis!")
                                    st.info("💡 Switch to **ATR Combiner** mode")
                        
                        # Show what's ready for next steps
                        st.markdown("---")
                        st.subheader("🎯 What's Next?")
                        
                        if processing_method == "Standard Resampling":
                            st.info(f"""
                            **Standard Resampling Complete!**
                            
                            ✅ **Processed**: {len(df):,} records → {len(resampled_data):,} {resample_timeframe} candles
                            ✅ **Time Filter**: {filter_start_str or 'None'} to {filter_end_str or 'None'}
                            ✅ **Ready for**: ATR analysis, trading system backtesting, or further processing
                            
                            **Perfect for**: Standard timeframe analysis with consistent intervals
                            """)
                        else:
                            st.info(f"""
                            **Custom Candle Generation Complete!**
                            
                            ✅ **Created**: {len(custom_periods)} custom candles per day
                            ✅ **Total Days**: {custom_candle_data['Date'].nunique()} trading days
                            ✅ **Total Candles**: {len(custom_candle_data)} custom periods
                            ✅ **Ready for**: Session-based analysis, custom timeframe backtesting
                            
                            **Perfect for**: Session-based analysis, custom trading periods, or non-standard timeframes
                            """)
                        
                except Exception as e:
                    st.error(f"❌ Processing failed: {str(e)}")
                    import traceback
                    st.error(traceback.format_exc())
            
        except Exception as e:
            st.error(f"❌ Error loading file: {str(e)}")

# FIXED: Show persistent actions for last processed data from Single File Resampler
if st.session_state.get('last_processed_data') is not None and mode == "🔧 Single File Resampler":
    st.markdown("---")
    st.subheader("🔄 **Continue with Last Processed Data**")
    
    last_data = st.session_state['last_processed_data']
    last_filename = st.session_state['last_processed_filename']
    
    st.info(f"📊 **Available**: {last_filename} ({len(last_data):,} records)")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        # Persistent download button
        st.download_button(
            "📥 **Download Again**",
            data=last_data.to_csv(index=False),
            file_name=last_filename,
            mime="text/csv",
            key="download_persistent_single",
            use_container_width=True
        )
    
    with col2:
        # Persistent hold as base
        if st.button("📊 **Hold as Base**", key="hold_base_persistent_single", use_container_width=True):
            st.session_state['atr_combiner_base_data'] = last_data.copy()
            st.session_state['atr_combiner_base_filename'] = last_filename
            st.success("✅ Held as Base!")
            st.rerun()
    
    with col3:
        # Persistent hold as analysis
        if st.button("📈 **Hold as Analysis**", key="hold_analysis_persistent_single", use_container_width=True):
            st.session_state['atr_combiner_analysis_data'] = last_data.copy()
            st.session_state['atr_combiner_analysis_filename'] = last_filename
            st.success("✅ Held as Analysis!")
            st.rerun()

# ========================================================================================
# MULTI-TIMEFRAME ATR COMBINER (SIMPLIFIED - Single ATR Column)
# ========================================================================================
elif mode == "🎯 Multi-Timeframe ATR Combiner":
    st.header("🎯 Multi-Timeframe ATR Combiner")
    st.write("**Combine any two timeframes with ATR calculation for systematic analysis**")
    
    # SIMPLIFIED: Information about the tool
    st.info("""
    🎯 **Purpose**: Prepare ATR-ready files for systematic analysis
    
    **What this does:**
    - Calculates TRUE Wilder's ATR on your chosen base timeframe (any timeframe)
    - Combines with your analysis timeframe for trigger/goal analysis
    - Outputs single file with **single ATR column** ready for ATR Level Analyzer
    
    **FLEXIBLE EXAMPLES:**
    - **Monthly ATR + Daily Analysis**: Monthly bars for ATR → applied to daily bars
    - **Daily ATR + 10-minute Analysis**: Daily bars for ATR → applied to 10-minute bars  
    - **4-Hour ATR + 1-minute Analysis**: 4-hour bars for ATR → applied to 1-minute bars
    - **Weekly ATR + 1-hour Analysis**: Weekly bars for ATR → applied to hourly bars
    
    **SIMPLIFIED OUTPUT**: Only one ATR column containing the currently used ATR value
    """)
    
    # Configuration columns
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📊 Base Timeframe (ATR Source)")
        st.write("**This timeframe will be used for ATR calculation**")
        st.write("*Can be any timeframe: 1-minute, hourly, daily, weekly, monthly, etc.*")
        
        # Check if we have saved data from other modes
        if 'atr_combiner_base_data' in st.session_state:
            st.success(f"✅ **Using held data**: {st.session_state.get('atr_combiner_base_filename', 'Processed Data')}")
            
            # Show basic info about held data
            base_data = st.session_state['atr_combiner_base_data']
            unique_days = base_data['Date'].nunique()
            st.info(f"📊 **Held base data**: {len(base_data):,} records | {unique_days} unique periods | {base_data['Date'].min()} to {base_data['Date'].max()}")
            
            if st.button("🗑️ Clear Held Base Data", key="clear_base_main"):
                del st.session_state['atr_combiner_base_data']
                if 'atr_combiner_base_filename' in st.session_state:
                    del st.session_state['atr_combiner_base_filename']
                st.rerun()
            
            base_file = None  # Use session state data
        else:
            base_file = st.file_uploader(
                "Upload Base Timeframe Data",
                type=['csv', 'txt', 'xlsx', 'xls'],
                help="Upload the timeframe you want to calculate ATR on (e.g., daily, weekly, monthly, 4H, etc.)",
                key="base_timeframe"
            )
        
        if base_file or 'atr_combiner_base_data' in st.session_state:
            if base_file:
                st.success(f"✅ Base file: {base_file.name}")
            
            # ATR period selection
            atr_period = st.number_input(
                "ATR Period",
                min_value=1,
                max_value=100,
                value=14,
                help="Number of periods for ATR calculation (e.g., 14 periods of your base timeframe)"
            )
            
            st.info(f"📊 Will calculate {atr_period}-period ATR on base timeframe")
    
    with col2:
        st.subheader("📈 Analysis Timeframe")
        st.write("**This timeframe will be used for trigger/goal analysis**")
        st.write("*Can be any timeframe: typically shorter than base timeframe*")
        
        # Check if we have saved data from other modes
        if 'atr_combiner_analysis_data' in st.session_state:
            st.success(f"✅ **Using held data**: {st.session_state.get('atr_combiner_analysis_filename', 'Processed Data')}")
            
            # Show basic info about held data
            analysis_data = st.session_state['atr_combiner_analysis_data']
            unique_days = analysis_data['Date'].nunique()
            st.info(f"📊 **Held analysis data**: {len(analysis_data):,} records | {unique_days} unique periods | {analysis_data['Date'].min()} to {analysis_data['Date'].max()}")
            
            if st.button("🗑️ Clear Held Analysis Data", key="clear_analysis_main"):
                del st.session_state['atr_combiner_analysis_data']
                if 'atr_combiner_analysis_filename' in st.session_state:
                    del st.session_state['atr_combiner_analysis_filename']
                st.rerun()
            
            analysis_file = None  # Use session state data
        else:
            analysis_file = st.file_uploader(
                "Upload Analysis Timeframe Data",
                type=['csv', 'xlsx', 'xls', 'txt'],
                help="Upload the timeframe you want to analyze (e.g., 1-minute, 10-minute, hourly, daily, etc.)",
                key="analysis_timeframe"
            )
        
        if analysis_file or 'atr_combiner_analysis_data' in st.session_state:
            if analysis_file:
                st.success(f"✅ Analysis file: {analysis_file.name}")
            
            # Asset type for futures date handling
            asset_type = st.selectbox(
                "Asset Type",
                ["STOCKS", "FUTURES", "CRYPTO", "FOREX", "COMMODITIES"],
                help="Select asset type for proper date/session handling"
            )
            
            if asset_type == 'FUTURES':
                st.info("🕐 **Futures Mode**: Will handle 18:00+ times as next day's session")
            
            # Alignment method
            align_method = st.selectbox(
                "Alignment Method",
                ["date_match"],
                help="How to align different timeframes"
            )
            
            st.info("📅 Will match ATR values by date")
    
    # Processing section
    if (base_file or 'atr_combiner_base_data' in st.session_state) and (analysis_file or 'atr_combiner_analysis_data' in st.session_state):
        st.markdown("---")
        st.subheader("⚙️ Processing Configuration")
        
        # Show file details
        col1, col2 = st.columns(2)
        with col1:
            if base_file:
                st.write(f"**Base File**: {base_file.name}")
            else:
                st.write(f"**Base Data**: {st.session_state.get('atr_combiner_base_filename', 'Saved Data')}")
            st.write(f"**ATR Period**: {atr_period}")
        with col2:
            if analysis_file:
                st.write(f"**Analysis File**: {analysis_file.name}")
            else:
                st.write(f"**Analysis Data**: {st.session_state.get('atr_combiner_analysis_filename', 'Saved Data')}")
            st.write(f"**Alignment**: {align_method}")
        
        # Data preview option
        show_preview = st.checkbox("Show Data Preview", value=False)
        
        if show_preview:
            st.subheader("📋 Data Preview")
            
            # Preview base file
            try:
                if base_file:
                    if base_file.name.endswith('.csv'):
                        base_preview = pd.read_csv(base_file).head()
                    else:
                        base_preview = pd.read_excel(base_file).head()
                else:
                    base_preview = st.session_state['atr_combiner_base_data'].head()
                
                st.write("**Base Timeframe Preview:**")
                st.dataframe(base_preview, use_container_width=True)
                
            except Exception as e:
                st.error(f"Error previewing base data: {str(e)}")
            
            # Preview analysis file
            try:
                if analysis_file:
                    if analysis_file.name.endswith('.csv'):
                        analysis_preview = pd.read_csv(analysis_file).head()
                    else:
                        analysis_preview = pd.read_excel(analysis_file).head()
                else:
                    analysis_preview = st.session_state['atr_combiner_analysis_data'].head()
                
                st.write("**Analysis Timeframe Preview:**")
                st.dataframe(analysis_preview, use_container_width=True)
                
            except Exception as e:
                st.error(f"Error previewing analysis data: {str(e)}")
        
        # Process button
        if st.button("🚀 **Combine Timeframes with ATR**", type="primary", use_container_width=True):
            with st.spinner("Processing multi-timeframe ATR combination..."):
                # Use session state data if available, otherwise use uploaded files
                daily_file_to_use = base_file if base_file else st.session_state['atr_combiner_base_data']
                intraday_file_to_use = analysis_file if analysis_file else st.session_state['atr_combiner_analysis_data']
                
                combined_data = combine_timeframes_with_atr(
                    daily_file_to_use, 
                    intraday_file_to_use, 
                    atr_period=atr_period,
                    align_method=align_method,
                    asset_type=asset_type
                )
                
                if combined_data is not None:
                    st.balloons()
                    st.success("🎉 **Multi-timeframe ATR combination complete!**")
                    
                    # Show results summary
                    st.subheader("📊 Results Summary")
                    
                    # Metrics
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("Total Records", f"{len(combined_data):,}")
                    with col2:
                        st.metric("Date Range", f"{combined_data['Date'].min()} to {combined_data['Date'].max()}")
                    with col3:
                        valid_atr = combined_data['ATR'].notna().sum()
                        st.metric("Valid ATR Values", f"{valid_atr:,}")
                    with col4:
                        atr_coverage = (valid_atr / len(combined_data)) * 100
                        st.metric("ATR Coverage", f"{atr_coverage:.1f}%")
                    
                    # SIMPLIFIED: Show single ATR column
                    st.subheader("📊 Single ATR Column")
                    st.info("✅ **Simplified Output**: Only one ATR column containing the currently used ATR value")
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.write("**ATR Column (Currently Used ATR)**")
                        recent_atr = combined_data['ATR'].tail(10)
                        st.dataframe(recent_atr.round(2))
                    
                    with col2:
                        # ATR Statistics
                        atr_stats = combined_data['ATR'].describe()
                        st.write("**ATR Distribution:**")
                        st.dataframe(atr_stats.round(2))
                    
                    # Data preview
                    st.subheader("📋 Combined Data Preview")
                    st.dataframe(combined_data.head(10), use_container_width=True)
                    
                    # Show column explanations
                    st.subheader("📋 Column Explanations")
                    col_explanations = {
                        'Datetime': 'Analysis timeframe timestamp',
                        'Date': 'Date for matching timeframes',
                        'Open/High/Low/Close': 'Analysis timeframe OHLC data',
                        'ATR': 'Currently used ATR (calculated from base timeframe)',
                        'Volume': 'Volume (if available in original data)'
                    }
                    
                    for col, desc in col_explanations.items():
                        if any(col_key in combined_data.columns for col_key in col.split('/')):
                            st.write(f"**{col}**: {desc}")
                    
                    # Download section
                    st.markdown("---")
                    st.subheader("📥 Download ATR-Ready File")
                    
                    # Generate filename
                    if base_file:
                        base_name = base_file.name.split('.')[0]
                    else:
                        base_name = "HeldBase"
                    
                    if analysis_file:
                        analysis_name = analysis_file.name.split('.')[0]
                    else:
                        analysis_name = "HeldAnalysis"
                    
                    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                    
                    output_filename = f"ATRReady_{base_name}_{analysis_name}_{atr_period}ATR_{timestamp}.csv"
                    
                    st.download_button(
                        "📥 **Download ATR-Ready CSV**",
                        data=combined_data.to_csv(index=False),
                        file_name=output_filename,
                        mime="text/csv",
                        key="download_atr_ready",
                        use_container_width=True,
                        type="primary"
                    )
                    
                    st.success(f"✅ Ready for ATR Level Analyzer: **{output_filename}**")
                    
                    # Next steps
                    st.markdown("---")
                    st.subheader("🎯 Next Steps")
                    st.info("""
                    🚀 **Ready for Analysis!**
                    
                    **Your file now contains:**
                    - ✅ **ATR** column with currently used ATR value from base timeframe
                    - ✅ Both timeframes aligned perfectly
                    - ✅ Clean, validated OHLC data from analysis timeframe
                    - ✅ Single file, ready for systematic analysis
                    
                    **Works with any timeframe combination:**
                    - Monthly ATR applied to daily analysis bars
                    - Daily ATR applied to 10-minute analysis bars
                    - 4-Hour ATR applied to 1-minute analysis bars
                    - Weekly ATR applied to hourly analysis bars
                    - Or any combination you chose!
                    
                    **Next:**
                    1. **Download** the ATR-ready file above
                    2. **Open** the ATR Level Analyzer tool
                    3. **Upload** this single file
                    4. **Get** systematic trigger/goal analysis results
                    
                    💡 **Ultimate flexibility - any timeframe combination works!**
                    """)
                    
                else:
                    st.error("❌ Failed to combine timeframes. Check the processing information above.")
    
    else:
        # Show instructions when files aren't uploaded
        st.info("👆 **Please upload both base and analysis timeframe files to get started**")
        
        # Show workflow explanation
        with st.expander("🔧 Multi-Timeframe ATR Workflow", expanded=True):
            st.markdown("""
            **🎯 What is Multi-Timeframe ATR Analysis?**
            
            This combines the power of longer-term ATR calculation with shorter-term analysis. **Any timeframe combination works:**
            
            **Example 1: Monthly ATR + Daily Analysis**
            - **Base**: Monthly OHLC data (for 14-month ATR calculation)
            - **Analysis**: Daily data (for trigger/goal detection)
            - **Result**: Each daily bar has the current monthly ATR value
            
            **Example 2: Daily ATR + 10-Minute Analysis**
            - **Base**: Daily OHLC data (for 14-day ATR calculation)
            - **Analysis**: 10-minute data (for trigger/goal detection)
            - **Result**: Each 10-minute bar has the current daily ATR value
            
            **Example 3: 4-Hour ATR + 1-Minute Analysis**
            - **Base**: 4-hour OHLC data (for 14-period ATR calculation)
            - **Analysis**: 1-minute data (for trigger/goal detection)
            - **Result**: Each 1-minute bar has the current 4-hour ATR value
            
            **Example 4: Weekly ATR + 1-Hour Analysis**
            - **Base**: Weekly OHLC data (for 14-week ATR calculation)
            - **Analysis**: 1-hour data (for trigger/goal detection)  
            - **Result**: Each 1-hour bar has the current weekly ATR value
            
            **🔧 Process:**
            1. **Upload base timeframe** (any timeframe for ATR calculation)
            2. **Upload analysis timeframe** (any timeframe for analysis)
            3. **Set ATR period** (e.g., 14 for 14-period ATR)
            4. **Combine** - system aligns data by date
            5. **Download** single ATR-ready file
            
            **💡 Why This Approach?**
            - **Separation of concerns**: ATR calculation vs analysis
            - **Ultimate flexibility**: Any timeframe combination
            - **Accuracy**: Proper ATR calculation on intended timeframe
            - **Efficiency**: Calculate once, analyze multiple times
            
            **🎯 SIMPLIFIED Output Format:**
            ```
            Datetime           Date        Open   High   Low    Close  ATR   Volume
            2024-01-01 09:30   2024-01-01  4100   4110   4095   4105   45.2  1000
            2024-01-01 09:40   2024-01-01  4105   4115   4100   4110   45.2  1500
            2024-01-01 09:50   2024-01-01  4110   4120   4105   4115   45.2  1200
            ```
            
            Each analysis timeframe bar includes:
            - Its own OHLC data  
            - **ATR**: Currently used ATR value (from base timeframe)
            - Volume (if available)
            - Clean, ready-to-analyze format
            
            **🚀 Timeframe Combinations:**
            - **Ultra-long term**: Yearly ATR + Monthly analysis
            - **Long term**: Monthly ATR + Daily analysis
            - **Medium term**: Daily ATR + Hourly analysis
            - **Short term**: 4-Hour ATR + 10-minute analysis
            - **Scalping**: 1-Hour ATR + 1-minute analysis
            - **Or any combination you need!**
            """)
        
        # Show supported file formats
        with st.expander("📁 Supported File Formats", expanded=False):
            st.markdown("""
            **✅ File Types Supported:**
            - **CSV** (.csv) - Most common format
            - **TXT** (.txt) - Tab/comma delimited text files
            - **Excel** (.xlsx, .xls) - Spreadsheet formats
            
            **📊 Required Columns (Both Files):**
            - **Date** (or Datetime) - Date/time information
            - **Open** - Opening price
            - **High** - High price  
            - **Low** - Low price
            - **Close** - Closing price
            
            **🔧 Column Name Flexibility:**
            - **Long form**: Date, Open, High, Low, Close
            - **Short form**: Date, o, h, l, c
            - **Mixed**: Any combination of the above
            
            **📅 Date/Time Format Support:**
            - **Separate columns**: Date + Time columns
            - **Combined datetime**: 2024-01-01 09:30:00
            - **Date only**: 2024-01-01, 01/01/2024
            - **Auto-detection**: System detects and splits datetime columns
            - **Multiple formats**: timestamp, datetime, date_time, etc.
            
            **🔄 Auto-Processing:**
            - Detects datetime columns automatically
            - Splits combined datetime into Date and Time
            - Handles various column names (timestamp, datetime, date_time)
            - Preserves original Datetime column for analysis
            
            **💡 Pro Tips:**
            - Use consistent date formats between files
            - Ensure base timeframe has enough history for ATR calculation
            - Analysis timeframe should overlap with base timeframe dates
            """)

# Help section
st.markdown("---")
st.subheader("📚 Usage Guide")

st.markdown("""
**🎯 Multi-CSV Processor** (Recommended)
- Perfect for combining broker data files
- Upload 25+ 1-minute CSV files → Get 1 combined 10-minute file
- Smart ticker detection and validation
- Custom time filtering for market hours

**🎯 Multi-Timeframe ATR Combiner** ⭐ (ULTIMATE FLEXIBILITY!)
- Combine ANY two timeframes with ATR calculation
- Examples: Monthly ATR + Daily analysis, Daily ATR + 10-minute analysis, 4H ATR + 1-minute analysis
- **NEW**: Single ATR column output (simplified workflow)
- **ATR** column contains currently used ATR value from your chosen base timeframe
- No more complex dual-column configurations!

**📈 Public Data Download**
- Download from public sources (limited intraday history)
- Good for daily data with ATR buffers
- Auto-maps common tickers (SPX → ^GSPC)

**🔧 Single File Resampler** (ENHANCED!)
- Transform one file to different timeframes
- **NEW**: Real Custom Candle Generator
- Create any number of candles per day with flexible time periods
- Perfect for session-based analysis

💾 **Next Step:** Use processed files in the ATR Analysis tool!

---

## 🎯 Ready for ATR Level Analysis?

Once you have your ATR-ready files, proceed to systematic trigger/goal analysis:

### 🔗 [**ATR Level Analyzer**](https://atr-dashboard-ekuggfmlyg4gmtw85ksacm.streamlit.app/)

**What it does:**
- ✅ **Single file input** - Upload your ATR-ready CSV
- ✅ **Single ATR column** - Simplified analysis with one ATR value
- ✅ **Systematic analysis** - Trigger/goal detection using pre-calculated ATR
- ✅ **Professional results** - Export-ready analysis data
- ✅ **No file juggling** - Pure analysis, no data preparation

**Perfect workflow:**
1. **Process your data here** → Get ATR-ready file with single ATR column
2. **Upload to ATR Level Analyzer** → Works with any timeframe combination
3. **Get systematic results** → Professional analysis output  
4. **Download results** → Ready for trading or further analysis

**Flexible Examples:**
- **Long-term**: Monthly ATR applied to daily analysis
- **Medium-term**: Daily ATR applied to hourly analysis  
- **Short-term**: 4-Hour ATR applied to 10-minute analysis
- **Scalping**: 1-Hour ATR applied to 1-minute analysis

🚀 **[Launch ATR Level Analyzer →](https://atr-dashboard-ekuggfmlyg4gmtw85ksacm.streamlit.app/)**
""")
