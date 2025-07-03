import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os

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

def validate_atr_ready_file(df):
    """
    Validate that the uploaded file is ATR-ready
    Returns: (is_valid, warnings, recommendations)
    """
    warnings = []
    recommendations = []
    is_valid = True
    
    # Check required columns
    required_columns = ['Date', 'Datetime', 'Open', 'High', 'Low', 'Close', 'ATR']
    missing_columns = [col for col in required_columns if col not in df.columns]
    
    if missing_columns:
        is_valid = False
        warnings.append(f"❌ Missing required columns: {missing_columns}")
        recommendations.append("Use Multi-Timeframe ATR Combiner to generate properly formatted file")
    
    # Check for ATR values
    if 'ATR' in df.columns:
        valid_atr = df['ATR'].notna().sum()
        total_records = len(df)
        atr_coverage = (valid_atr / total_records) * 100 if total_records > 0 else 0
        
        if atr_coverage < 50:
            is_valid = False
            warnings.append(f"❌ Insufficient ATR coverage: {atr_coverage:.1f}% ({valid_atr}/{total_records})")
            recommendations.append("Ensure base timeframe has sufficient historical data for ATR calculation")
        elif atr_coverage < 80:
            warnings.append(f"⚠️ Low ATR coverage: {atr_coverage:.1f}% ({valid_atr}/{total_records})")
            recommendations.append("Consider extending historical data range for better ATR coverage")
    
    # Check date range
    if 'Date' in df.columns:
        date_range = pd.to_datetime(df['Date']).dt.date
        start_date = date_range.min()
        end_date = date_range.max()
        days_span = (end_date - start_date).days
        
        if days_span < 30:
            warnings.append(f"⚠️ Short analysis period: {days_span} days")
            recommendations.append("Consider longer time period for more robust analysis")
    
    # Check for Previous_ATR (commonly used for analysis)
    if 'Previous_ATR' not in df.columns:
        warnings.append("⚠️ No Previous_ATR column found")
        recommendations.append("Previous day ATR is commonly used for systematic analysis")
    
    return is_valid, warnings, recommendations

def load_atr_ready_data(uploaded_file):
    """
    Load and validate ATR-ready data file
    """
    try:
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        status_text.text("Reading ATR-ready file...")
        progress_bar.progress(25)
        
        if uploaded_file.name.endswith('.csv'):
            df = pd.read_csv(uploaded_file)
        elif uploaded_file.name.endswith(('.xlsx', '.xls')):
            df = pd.read_excel(uploaded_file)
        else:
            st.error("Unsupported file format. Please use CSV or Excel files.")
            return None
        
        status_text.text("Validating ATR-ready format...")
        progress_bar.progress(50)
        
        # Validate ATR-ready format
        is_valid, warnings, recommendations = validate_atr_ready_file(df)
        
        if warnings:
            st.subheader("⚠️ File Validation Results")
            for warning in warnings:
                st.warning(warning)
        
        if recommendations:
            st.subheader("💡 Recommendations")
            for rec in recommendations:
                st.info(f"• {rec}")
        
        if not is_valid:
            st.error("❌ File is not in proper ATR-ready format")
            return None
        
        status_text.text("Processing datetime columns...")
        progress_bar.progress(75)
        
        # Process datetime columns
        df['Datetime'] = pd.to_datetime(df['Datetime'])
        df['Date'] = pd.to_datetime(df['Date']).dt.date
        
        # Sort by datetime
        df = df.sort_values('Datetime').reset_index(drop=True)
        
        status_text.text("Validation complete...")
        progress_bar.progress(100)
        
        st.success(f"✅ ATR-ready file loaded successfully: {len(df):,} records")
        
        # Show file summary
        valid_atr = df['ATR'].notna().sum()
        atr_coverage = (valid_atr / len(df)) * 100
        st.info(f"📊 ATR Coverage: {atr_coverage:.1f}% ({valid_atr:,}/{len(df):,} records)")
        st.info(f"📅 Date Range: {df['Date'].min()} to {df['Date'].max()}")
        
        # Clear progress indicators
        progress_bar.empty()
        status_text.empty()
        
        return df
        
    except Exception as e:
        st.error(f"Error loading ATR-ready file: {str(e)}")
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

def detect_triggers_and_goals_systematic(df, asset_config, custom_ratios=None, session_filter=None):
    """
    Systematic trigger and goal detection using pre-calculated ATR values
    """
    if custom_ratios is None:
        fib_levels = [0.236, 0.382, 0.500, 0.618, 0.786, 1.000, 
                     -0.236, -0.382, -0.500, -0.618, -0.786, -1.000, 0.000]
    else:
        fib_levels = custom_ratios
    
    results = []
    has_open_special = asset_config['has_open_special']
    
    # Get unique dates with ATR values
    atr_dates = df[df['ATR'].notna()]['Date'].unique()
    
    # Progress tracking
    total_days = len(atr_dates)
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    for i, trading_date in enumerate(atr_dates):
        try:
            # Update progress
            progress = i / total_days
            progress_bar.progress(progress)
            status_text.text(f"Processing day {i+1}/{total_days}: {trading_date}")
            
            # Get day's data
            day_data = filter_by_session_and_hours(df, trading_date, asset_config, session_filter)
            
            if day_data.empty:
                continue
            
            # Get ATR values for this day
            day_atr = day_data['ATR'].iloc[0]  # Should be same for all records on same day
            
            # Skip if no valid ATR
            if pd.isna(day_atr):
                continue
            
            # Use Previous_ATR if available, otherwise current ATR
            if 'Previous_ATR' in day_data.columns and not pd.isna(day_data['Previous_ATR'].iloc[0]):
                analysis_atr = day_data['Previous_ATR'].iloc[0]
            else:
                analysis_atr = day_atr
            
            # Get previous close (use first record's close as reference)
            if 'Daily_Close' in day_data.columns:
                previous_close = day_data['Daily_Close'].iloc[0]
            else:
                # Fallback: use previous day's close from the data
                prev_day_data = df[df['Date'] < trading_date].tail(1)
                if not prev_day_data.empty:
                    previous_close = prev_day_data['Close'].iloc[0]
                else:
                    continue
            
            # Generate levels
            level_map = generate_atr_levels(previous_close, analysis_atr, custom_ratios)
            
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
                        trading_date, previous_close, analysis_atr, has_open_special, open_price
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
                        trading_date, previous_close, analysis_atr, has_open_special, open_price
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

def display_results(result_df, ticker, asset_type, file_info):
    """Helper function to display analysis results with enhanced statistics"""
    
    if not result_df.empty:
        result_df['Ticker'] = ticker
        result_df['AssetType'] = asset_type
        result_df['DataSource'] = file_info
        
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
        st.warning('⚠️ No results generated - check file format and data quality')

# Streamlit Interface
st.title('🎯 ATR Level Analyzer')
st.write('**Pure systematic trigger/goal analysis using pre-calculated ATR values**')

# File input section
st.header("📁 ATR-Ready Data Input")

st.info("""
🎯 **This tool requires ATR-ready files with pre-calculated ATR values.**

**Required columns**: Date, Datetime, Open, High, Low, Close, ATR
**Optional columns**: Previous_ATR, Daily_Open, Daily_High, Daily_Low, Daily_Close
""")

# Link to CSV Data Handler
st.markdown("""
---
### 📊 Need to prepare your data first?

**[🔗 Use Multi-Timeframe ATR Combiner](https://atr-dashboard-bpovcyydv44p7vrdqteryw.streamlit.app/)**

The Multi-Timeframe ATR Combiner will:
- ✅ Calculate TRUE Wilder's ATR on your base timeframe
- ✅ Combine with your analysis timeframe  
- ✅ Generate properly formatted ATR-ready files
- ✅ Handle data validation and cleaning

**Perfect for**: Daily ATR + 10-minute analysis, Weekly ATR + 1-hour analysis, etc.

---
""")

# File upload
atr_file = st.file_uploader(
    "Upload ATR-Ready File",
    type=['csv', 'xlsx', 'xls'],
    help="Upload a file that already contains ATR values calculated by the Multi-Timeframe ATR Combiner"
)

if atr_file:
    # Load and validate the file
    atr_data = load_atr_ready_data(atr_file)
    
    if atr_data is not None:
        # Asset configuration
        st.header("🏷️ Analysis Configuration")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Ticker input
            ticker = st.text_input(
                "Ticker Symbol (for labeling)",
                value="",
                help="Enter ticker symbol for output file labeling"
            )
            
            # Asset type selection
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
        
        with col2:
            # Get asset configuration
            config = AssetConfig.get_config(asset_type, extended_hours)
            
            st.info(f"**Asset Configuration:**\n{config['description']}")
            
            # Session filtering
            if len(config['session_types']) > 1:
                session_filter = st.multiselect(
                    "Filter by Sessions",
                    options=config['session_types'],
                    default=config['default_session'],
                    help="Select trading sessions to include in analysis"
                )
            else:
                session_filter = None
        
        # Advanced settings
        with st.expander("⚙️ Advanced Analysis Settings"):
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
        
        # Analysis button
        if st.button('🚀 Run ATR Level Analysis', type="primary"):
            if not ticker:
                ticker = "ANALYSIS"
            
            with st.spinner(f'Running systematic ATR analysis for {ticker}...'):
                try:
                    result_df = detect_triggers_and_goals_systematic(
                        atr_data,
                        config,
                        custom_ratios=custom_ratios,
                        session_filter=session_filter
                    )
                    
                    # Display results
                    display_results(
                        result_df, 
                        ticker, 
                        asset_type, 
                        f"ATR-Ready File: {atr_file.name}"
                    )
                        
                except Exception as e:
                    st.error(f'❌ Analysis failed: {e}')
                    import traceback
                    st.error(traceback.format_exc())

# Help section
st.markdown("""
---
### 📚 ATR Level Analyzer Guide

**🎯 What This Tool Does:**
- **Pure Analysis**: No data preparation - only systematic trigger/goal detection
- **Pre-calculated ATR**: Uses ATR values from properly formatted files
- **Professional Results**: Export-ready analysis with comprehensive statistics
- **Asset Flexibility**: Supports stocks, crypto, forex, futures, and commodities

**📋 Required File Format:**
Your file must contain these columns (created by Multi-Timeframe ATR Combiner):
- **Date**: Date for each record
- **Datetime**: Full timestamp for intraday data
- **Open, High, Low, Close**: Price data for analysis timeframe
- **ATR**: Pre-calculated ATR values from base timeframe

**🔧 Recommended Workflow:**
1. **Prepare Data**: Use Multi-Timeframe ATR Combiner to create ATR-ready files
2. **Upload Here**: Single file upload with all required data
3. **Configure Analysis**: Set asset type and advanced options
4. **Run Analysis**: Get systematic trigger/goal detection results
5. **Download Results**: Export professional analysis data

**💡 Key Benefits:**
- **No ATR Calculation**: All ATR values pre-calculated correctly
- **No File Juggling**: Single file input for complete analysis
- **Clean Architecture**: Data prep and analysis properly separated
- **Systematic Approach**: Consistent trigger/goal detection methodology
- **Professional Output**: Publication-ready results and statistics

**🚀 [Get Started with Data Preparation →](https://atr-dashboard-bpovcyydv44p7vrdqteryw.streamlit.app/)**
""")
