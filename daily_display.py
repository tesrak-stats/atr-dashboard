import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import json
import os
import glob
from datetime import datetime, time
try:
    from zoneinfo import ZoneInfo  # Python 3.9+
    USE_ZONEINFO = True
except ImportError:
    import pytz  # Fallback for older Python versions
    USE_ZONEINFO = False
from daily_atr_updater import calculate_atr_levels, TICKER_CONFIG
from shared_chart_functions import (
    create_zonebaseline_heatmap,
    create_statecheck_matrix,
    get_rolling_8_periods,
    get_total_probability,
    create_rolling_matrix
)
st.set_page_config(layout="wide")

# Add this section right after the imports and before ticker configuration

# --- DATA DISCOVERY SYSTEM ---
def discover_available_tickers():
    """Discover all tickers that have data files"""
    # Look for all relevant CSV files
    session_files = glob.glob("atr_summary_*_SESSION_*.csv")
    rolling_files = glob.glob("atr_summary_*_ROLLING_*.csv") 
    statecheck_files = glob.glob("statecheck_detailed_*.csv")
    zonebaseline_detailed_files = glob.glob("zonebaseline_detailed_*.csv")
    zonebaseline_hourly_files = glob.glob("zonebaseline_hourly_*.csv")
    
    all_files = session_files + rolling_files + statecheck_files + zonebaseline_hourly_files + zonebaseline_detailed_files
    
    # Extract ticker symbols from filenames
    tickers = set()
    for file in all_files:
        # Extract ticker from patterns like "atr_summary_SPX_SESSION_*.csv"
        if "atr_summary_" in file:
            parts = file.split("_")
            if len(parts) >= 3:
                ticker = parts[2]  # SPX, ES, AAPL, etc.
                tickers.add(ticker)
        elif "statecheck_detailed_" in file:
            parts = file.split("_")
            if len(parts) >= 3:
                ticker = parts[2]  # SPX, ES, AAPL, etc.
                tickers.add(ticker)
    
    return sorted(list(tickers))

def discover_available_analyses(selected_ticker):
    """Discover which analysis types are available for the selected ticker"""
    KNOWN_ANALYSIS_PATTERNS = {
        "SESSION": "Session",
        "ROLLING": "Rolling", 
        "statecheck": "StateCheck",
        "ZONEBASELINE": "ZoneBaseline"
    }
    
    # Find all files for this ticker
    ticker_files = []
    ticker_files.extend(glob.glob(f"atr_summary_{selected_ticker}_*.csv"))
    ticker_files.extend(glob.glob(f"statecheck_detailed_{selected_ticker}_*.csv"))
    ticker_files.extend(glob.glob(f"zonebaseline_*_{selected_ticker}_*.csv"))  # FIX: Add zonebaseline pattern
    
    available_analyses = []
    for pattern, display_name in KNOWN_ANALYSIS_PATTERNS.items():
        if any(pattern.upper() in f.upper() for f in ticker_files):
            available_analyses.append(display_name)
    
    return available_analyses

def discover_available_trigger_times(selected_ticker, default_analysis="SESSION"):
    """Discover available trigger times by examining actual data"""
    # Priority order for finding trigger times
    analysis_priority = ["SESSION", "STATECHECK", "ZONEBASELINE", "ROLLING"]
    
    try:
        # Try each analysis type in priority order
        for analysis in analysis_priority:
            if analysis == "SESSION":
                files = glob.glob(f"atr_summary_{selected_ticker}_SESSION_*.csv")
            elif analysis == "STATECHECK":
                files = glob.glob(f"statecheck_detailed_{selected_ticker}_*.csv")
            elif analysis == "ZONEBASELINE":
                files = glob.glob(f"atr_summary_{selected_ticker}_ZONEBASELINE_*.csv")
            elif analysis == "ROLLING":
                files = glob.glob(f"atr_summary_{selected_ticker}_ROLLING_*.csv")
            else:
                continue
                
            if files:
                df = pd.read_csv(max(files))
                
                # Extract unique trigger times from TriggerTime column
                if 'TriggerTime' in df.columns:
                    trigger_times = sorted(df['TriggerTime'].unique())
                    
                    # Convert to proper format and add OPEN
                    formatted_times = ["OPEN"]
                    for t in trigger_times:
                        if pd.notna(t):
                            if isinstance(t, (int, float)):
                                time_int = int(t)
                                if 0 <= time_int <= 2359:
                                    formatted_times.append(f"{time_int:04d}")
                            else:
                                formatted_times.append(str(t))
                    
                    return sorted(list(set(formatted_times)))
        
        # Fallback to standard stock hours if no data found
        return ["OPEN", "0900", "1000", "1100", "1200", "1300", "1400", "1500"]
        
    except Exception as e:
        # Ultimate fallback
        return ["OPEN", "0900", "1000", "1100", "1200", "1300", "1400", "1500"]

def discover_available_fib_levels(selected_ticker, analysis_type, trigger_conditions=None):
    """Discover available fibonacci levels from actual data"""
    # Priority order for finding fib levels
    analysis_priority = ["SESSION", "STATECHECK", "ZONEBASELINE", "ROLLING"]
    
    try:
        # If specific analysis type requested, try that first
        files_to_try = []
        if analysis_type == "Session":
            files_to_try.append(("SESSION", glob.glob(f"atr_summary_{selected_ticker}_SESSION_*.csv")))
        elif analysis_type == "Rolling":
            files_to_try.append(("ROLLING", glob.glob(f"atr_summary_{selected_ticker}_ROLLING_*.csv")))
        elif analysis_type == "StateCheck":
            files_to_try.append(("STATECHECK", glob.glob(f"statecheck_detailed_{selected_ticker}_*.csv")))
        elif analysis_type == "ZoneBaseline":
            files_to_try.append(("ZONEBASELINE", glob.glob(f"atr_summary_{selected_ticker}_ZONEBASELINE_*.csv")))
        
        # Then try priority order for any we haven't tried
        for analysis in analysis_priority:
            if analysis == "SESSION" and analysis_type != "Session":
                files_to_try.append(("SESSION", glob.glob(f"atr_summary_{selected_ticker}_SESSION_*.csv")))
            elif analysis == "STATECHECK" and analysis_type != "StateCheck":
                files_to_try.append(("STATECHECK", glob.glob(f"statecheck_detailed_{selected_ticker}_*.csv")))
            elif analysis == "ZONEBASELINE" and analysis_type != "ZoneBaseline":
                files_to_try.append(("ZONEBASELINE", glob.glob(f"atr_summary_{selected_ticker}_ZONEBASELINE_*.csv")))
            elif analysis == "ROLLING" and analysis_type != "Rolling":
                files_to_try.append(("ROLLING", glob.glob(f"atr_summary_{selected_ticker}_ROLLING_*.csv")))
        
        # Try each file set
        for analysis_name, files in files_to_try:
            if files:
                df = pd.read_csv(max(files))
                
                # Extract unique fibonacci levels
                if 'GoalLevel' in df.columns:
                    discovered_levels = sorted(df['GoalLevel'].unique())
                    
                    # Validate that we have a reasonable number of levels
                    if 5 <= len(discovered_levels) <= 20:  # Reasonable range
                        return discovered_levels
        
        # Fallback to standard fibonacci levels
        return [1.0, 0.786, 0.618, 0.5, 0.382, 0.236, 0.0, -0.236, -0.382, -0.5, -0.618, -0.786, -1.0]
        
    except Exception as e:
        # Ultimate fallback
        return [1.0, 0.786, 0.618, 0.5, 0.382, 0.236, 0.0, -0.236, -0.382, -0.5, -0.618, -0.786, -1.0]

# --- MAIN DATA DISCOVERY IMPLEMENTATION ---

# Discover available tickers
available_tickers = discover_available_tickers()

if not available_tickers:
    st.error("❌ No data files found! Please ensure CSV files are in the correct directory.")
    st.info("📝 Expected file patterns: atr_summary_TICKER_ANALYSIS_*.csv or statecheck_detailed_TICKER_*.csv")
    st.stop()

# Create ticker options with discovery info
ticker_options = []
ticker_mapping = {}

# Add section headers and tickers
ticker_options.append("--- 📊 Available Tickers ---")

for ticker in available_tickers:
    # Get available analyses for this ticker to show status
    available_analyses = discover_available_analyses(ticker)
    
    # Create display name with analysis count
    analyses_count = len(available_analyses)
    analyses_preview = ", ".join(available_analyses[:2])
    if len(available_analyses) > 2:
        analyses_preview += "..."
    
    ticker_display = f"{ticker} ({analyses_count} analyses: {analyses_preview})"
    
    ticker_options.append(ticker_display)
    ticker_mapping[ticker_display] = ticker

# Ticker selector with discovery
selected_ticker_display = st.selectbox("Select Ticker", ticker_options, index=1)

# Validate ticker selection
if not selected_ticker_display.startswith("---"):
    if selected_ticker_display in ticker_mapping:
        selected_ticker = ticker_mapping[selected_ticker_display]
        
        # Discover available analyses for selected ticker
        available_analyses = discover_available_analyses(selected_ticker)
        
        if not available_analyses:
            st.error(f"❌ No supported analysis types found for {selected_ticker}")
            st.stop()
        
        st.success(f"✅ Selected: {selected_ticker} | Available: {', '.join(available_analyses)}")
    else:
        st.error("❌ Invalid ticker selection")
        st.stop()
else:
    st.error("Please select a valid ticker (not a header)")
    st.stop()

# Quick patch - create ticker_config from selected ticker
ticker_config = {
    selected_ticker: {
        "display_name": selected_ticker,
        "ticker_symbol": selected_ticker,
        "summary_file": ""  # Not used anymore since each analysis loads its own files
    }
}

# Discover trigger times for selected ticker
available_trigger_times = discover_available_trigger_times(selected_ticker)

# Discover fibonacci levels for initial analysis type
initial_fib_levels = discover_available_fib_levels(selected_ticker, available_analyses[0] if available_analyses else "Session")

# --- App Header ---
st.title("📈 Daily ATR Analysis")
st.caption("🔧 App Version: v4.0.0 - Full Data Discovery System")

# --- Analysis Type Selection (Dynamic) ---
st.subheader("📊 Analysis Type")

# Only show available analysis types for selected ticker
analysis_type = st.selectbox(
    "Select Analysis Type",
    available_analyses,
    index=0,
    help="Only analyses with available data for the selected ticker are shown"
)

# Update fibonacci levels based on selected analysis type
fib_levels = discover_available_fib_levels(selected_ticker, analysis_type)

# --- Analysis Parameters (Dynamic) ---
st.subheader("⚙️ Analysis Parameters")

if analysis_type == "ZoneBaseline":
    st.info("📍 **ZoneBaseline**: Static probability heatmap showing zone occupancy throughout the session")
    st.caption("No additional parameters needed - shows probability of being in each zone at each time")
    
elif analysis_type == "StateCheck":
    st.info("🔄 **StateCheck**: Zone transition probabilities from trigger zone over time")
    
    # Zone definitions (keep existing)
    zone_definitions = [
        "Zone 1: Above +1.0", "Zone 2: +0.786 to +1.0", "Zone 3: +0.618 to +0.786",
        "Zone 4: +0.5 to +0.618", "Zone 5: +0.382 to +0.5", "Zone 6: +0.236 to +0.382",
        "Zone 7: 0.0 to +0.236", "Zone 8: -0.236 to 0.0", "Zone 9: -0.382 to -0.236",
        "Zone 10: -0.5 to -0.382", "Zone 11: -0.618 to -0.5", "Zone 12: -0.786 to -0.618",
        "Zone 13: -1.0 to -0.786", "Zone 14: Below -1.0"
    ]
    
    col1, col2 = st.columns(2)
    with col1:
        trigger_zone = st.selectbox("Trigger Zone", zone_definitions, index=6)
    with col2:
        # Use discovered trigger times but convert to specific StateCheck format
        statecheck_times = [t for t in available_trigger_times if t != "OPEN"][:10]  # Limit for UI
        trigger_time = st.selectbox("Trigger Time", statecheck_times, index=0)
    
elif analysis_type == "Rolling":
    st.info("⏰ **Rolling**: 8-period rolling window analysis from trigger time")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        price_direction = st.selectbox("Price Location", ["Above", "Below"], index=0, key="rolling_direction")
    with col2:
        # Use discovered fibonacci levels
        trigger_level = st.selectbox("Trigger Level", fib_levels, index=len(fib_levels)//2, key="rolling_level")
    with col3:
        # Use discovered trigger times
        trigger_time = st.selectbox("Trigger Time", available_trigger_times, index=0, key="rolling_time")
    
    # Show rolling window preview (if we can calculate it)
    if len(available_trigger_times) > 8:
        st.caption(f"🔄 Rolling window: 8 periods starting from {trigger_time}")
    
else:  # Session
    st.info("📈 **Session**: Full session analysis with trigger conditions")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        price_direction = st.selectbox("Price Location", ["Above", "Below"], index=0)
    with col2:
        # Use discovered fibonacci levels
        trigger_level = st.selectbox("Trigger Level", fib_levels, index=len(fib_levels)//2)
    with col3:
        # Use discovered trigger times
        trigger_time = st.selectbox("Trigger Time", available_trigger_times, index=0)

# Show analysis summary
st.divider()
if analysis_type == "ZoneBaseline":
    st.write(f"**Analysis**: {analysis_type} | **Ticker**: {selected_ticker}")
elif analysis_type == "StateCheck":
    st.write(f"**Analysis**: {analysis_type} | **From**: {trigger_zone} at {trigger_time} | **Ticker**: {selected_ticker}")
else:
    st.write(f"**Analysis**: {analysis_type} | **Trigger**: {price_direction} {trigger_level} at {trigger_time} | **Ticker**: {selected_ticker}")

# Display discovery info
with st.expander("🔍 Data Discovery Info"):
    st.write(f"**Discovered Tickers**: {', '.join(available_tickers)}")
    st.write(f"**Available Analyses for {selected_ticker}**: {', '.join(available_analyses)}")
    st.write(f"**Available Trigger Times**: {len(available_trigger_times)} times ({available_trigger_times[0]} to {available_trigger_times[-1]})")
    st.write(f"**Available Fib Levels**: {len(fib_levels)} levels ({min(fib_levels):+.3f} to {max(fib_levels):+.3f})")

# --- Continue with existing analysis logic ---
# The rest of your analysis code continues unchanged here...
    
# Show analysis summary
st.divider()
if analysis_type == "ZoneBaseline":
    st.write(f"**Analysis**: {analysis_type}")
elif analysis_type == "StateCheck":
    st.write(f"**Analysis**: {analysis_type} | **From**: {trigger_zone} at {trigger_time}")
else:
    st.write(f"**Analysis**: {analysis_type} | **Trigger**: {price_direction} {trigger_level} at {trigger_time}")

# --- Utility Functions ---
def aggregate_statecheck_to_hourly(detailed_df):
    """Aggregate 10-minute StateCheck data to hourly buckets"""
    hourly_data = detailed_df.copy()
    
    # Bucket goal times to hours
    hourly_data['GoalTime'] = hourly_data['GoalTime'].apply(lambda x: int(int(x) / 100) * 100)
    hourly_data['GoalTime'] = hourly_data['GoalTime'].astype(str).str.zfill(4)
    
    # Group and sum the RAW COUNTS, not percentages
    aggregated = hourly_data.groupby(['TriggerZone', 'TriggerTime', 'GoalLevel', 'GoalTime']).agg({
        'NumHits': 'sum',        # Add up all the hits
        'NumTriggers': 'sum',  # Triggers should be same for all
        # Remove this line: 'PctCompletion': 'sum'  # DON'T sum percentages!
    }).reset_index()
    
    # Calculate percentage AFTER aggregation
    aggregated['PctCompletion'] = (aggregated['NumHits'] / aggregated['NumTriggers'] * 100).round(1)
    
    return aggregated


    
def get_current_market_time():
    """Get current Eastern Time and determine market time slot"""
    if USE_ZONEINFO:
        et = ZoneInfo('US/Eastern')
        current_et = datetime.now(et)
    else:
        et = pytz.timezone('US/Eastern')
        current_et = datetime.now(et)
    current_time = current_et.time()
    
    time_slots = [
        (time(9, 30), "OPEN"), (time(9, 0), "0900"), (time(10, 0), "1000"),
        (time(11, 0), "1100"), (time(12, 0), "1200"), (time(13, 0), "1300"),
        (time(14, 0), "1400"), (time(15, 0), "1500"), (time(16, 0), "CLOSE")
    ]
    
    time_slots.sort(key=lambda x: x[0])
    current_slot = "PREMARKET"
    for slot_time, slot_name in time_slots:
        if current_time >= slot_time:
            current_slot = slot_name
        else:
            break
    
    if current_time >= time(16, 0):
        current_slot = "AFTERHOURS"
    
    return current_et, current_slot

def calculate_remaining_probability(total_pct, completed_hourly_pcts, current_time_slot, time_order):
    """Calculate remaining probability based on current time"""
    if current_time_slot in ["PREMARKET", "AFTERHOURS", "CLOSE"]:
        return total_pct, "N/A"
    
    try:
        current_index = time_order.index(current_time_slot)
    except ValueError:
        return total_pct, "Current"
    
    completed_probability = 0
    for i, time_slot in enumerate(time_order):
        if i < current_index and time_slot in completed_hourly_pcts:
            completed_probability += completed_hourly_pcts[time_slot]
    
    remaining_pct = max(0, total_pct - completed_probability)
    return remaining_pct, current_time_slot

def get_atr_levels_for_ticker(ticker_key):
    """Get ATR levels for specific ticker from multi-ticker JSON file"""
    try:
        json_file = "atr_levels.json"
        
        if os.path.exists(json_file):
            with open(json_file, 'r') as f:
                saved_data = json.load(f)
                
            if "tickers" in saved_data and ticker_key in saved_data["tickers"]:
                ticker_data = saved_data["tickers"][ticker_key]
                if ticker_data.get("status") == "success":
                    return ticker_data
            elif ticker_key == "SPX" and saved_data.get("status") == "success":
                return saved_data
        
        # Calculate fresh if no saved data
        ticker_symbol = ticker_config[ticker_key]["ticker_symbol"]
        levels_data = calculate_atr_levels(ticker=ticker_symbol)
        
        if levels_data.get("status") == "success":
            # Save to multi-ticker format
            if os.path.exists(json_file):
                try:
                    with open(json_file, 'r') as f:
                        existing_data = json.load(f)
                except:
                    existing_data = {}
                
                if "tickers" not in existing_data:
                    existing_data = {"last_updated": datetime.now().isoformat(), "tickers": {}}
                
                existing_data["tickers"][ticker_key] = levels_data
                existing_data["last_updated"] = datetime.now().isoformat()
            else:
                existing_data = {
                    "last_updated": datetime.now().isoformat(),
                    "tickers": {ticker_key: levels_data}
                }
            
            with open(json_file, 'w') as f:
                json.dump(existing_data, f, indent=2)
        
        return levels_data
        
    except Exception as e:
        st.error(f"Error getting ATR levels for {ticker_key}: {str(e)}")
        return {"status": "error", "error": str(e)}
    """Get ATR levels for specific ticker from multi-ticker JSON file"""
    try:
        json_file = "atr_levels.json"
        
        if os.path.exists(json_file):
            with open(json_file, 'r') as f:
                saved_data = json.load(f)
                
            if "tickers" in saved_data and ticker_key in saved_data["tickers"]:
                ticker_data = saved_data["tickers"][ticker_key]
                if ticker_data.get("status") == "success":
                    return ticker_data
            elif ticker_key == "SPX" and saved_data.get("status") == "success":
                return saved_data
        
        # Calculate fresh if no saved data
        ticker_symbol = ticker_config[ticker_key]["ticker_symbol"]
        levels_data = calculate_atr_levels(ticker=ticker_symbol)
        
        if levels_data.get("status") == "success":
            # Save to multi-ticker format
            if os.path.exists(json_file):
                try:
                    with open(json_file, 'r') as f:
                        existing_data = json.load(f)
                except:
                    existing_data = {}
                
                if "tickers" not in existing_data:
                    existing_data = {"last_updated": datetime.now().isoformat(), "tickers": {}}
                
                existing_data["tickers"][ticker_key] = levels_data
                existing_data["last_updated"] = datetime.now().isoformat()
            else:
                existing_data = {
                    "last_updated": datetime.now().isoformat(),
                    "tickers": {ticker_key: levels_data}
                }
            
            with open(json_file, 'w') as f:
                json.dump(existing_data, f, indent=2)
        
        return levels_data
        
    except Exception as e:
        st.error(f"Error getting ATR levels for {ticker_key}: {str(e)}")
        return {"status": "error", "error": str(e)}

# --- Load Data ---
current_et_time, current_market_slot = get_current_market_time()



# Load ATR levels
atr_data = get_atr_levels_for_ticker(selected_ticker)
if atr_data.get("status") == "success":
    price_levels_dict = atr_data.get("levels", {})
else:
    price_levels_dict = {}
    st.error(f"❌ Could not load ATR levels for {selected_ticker}: {atr_data.get('error', 'Unknown error')}")

# --- What's This? Section ---
with st.expander("❓ What's This? - How to Use This Chart"):
    st.markdown("""
    **This chart shows the probability of reaching price levels based on historical data from 2,720 trading days.**
    
    📊 **How to Read:**
    - **Rows (Fib Levels):** Target price levels based on ATR (Average True Range)
    - **Columns (Times):** Hours during the trading day when the target was reached
    - **Percentages:** Historical success rate - how often price reached that level by that time
    - **Colors:** Match the horizontal line colors for easy reference
    
    🎯 **How to Use:**
    1. **Select Ticker:** Choose which instrument to analyze
    2. **Select Analysis Type:** Choose Session, StateCheck, Rolling, or ZoneBaseline
    3. **Set Parameters:** Configure trigger conditions based on analysis type
    4. **Read Results:** See probability of reaching levels throughout the day
    """)

# --- Chart Building ---
st.subheader(f"📈 Probability of Reaching Price Levels (%) - {ticker_config[selected_ticker]['display_name']}")
st.caption("Historical success rates based on 2,720 trading days of data")

# Define styling
fib_levels = [1.0, 0.786, 0.618, 0.5, 0.382, 0.236, 0.0, -0.236, -0.382, -0.5, -0.618, -0.786, -1.0]
fibo_styles = {
    1.0: ("lightgray", 3, 16), 0.786: ("lightgray", 1, 12), 0.618: ("lightgray", 2, 14),
    0.5: ("lightgray", 1, 12), 0.382: ("lightgray", 1, 12), 0.236: ("cyan", 2, 14),
    0.0: ("lightgray", 1, 12), -0.236: ("yellow", 2, 14), -0.382: ("lightgray", 1, 12),
    -0.5: ("lightgray", 1, 12), -0.618: ("lightgray", 2, 14), -0.786: ("lightgray", 1, 12),
    -1.0: ("lightgray", 3, 16)
}

# Mobile/Desktop view preference
if 'expanded_view_pref' not in st.session_state:
    st.session_state.expanded_view_pref = False

col1_ui, col2_ui = st.columns([3, 1])
with col1_ui:
    show_expanded_view = st.checkbox("🖥️ Show Full Matrix (All Times & Levels)", 
                                   value=st.session_state.expanded_view_pref)
with col2_ui:
    make_default = st.checkbox("💾 Make Default for Session", value=False)

if make_default:
    st.session_state.expanded_view_pref = show_expanded_view
    #st.success("✅ Session default updated!")

# --- Analysis-Specific Processing ---
if analysis_type == "StateCheck":
    try:
        statecheck_file = f"statecheck_detailed_{selected_ticker}_20250710_063704.csv"
        statecheck_df = pd.read_csv(statecheck_file)
       
        #st.success(f"✅ Loaded StateCheck data: {len(statecheck_df)} records")
        
        # Zone mapping
        zone_mapping = {
            "Zone 1: Above +1.0": "above_1.0", "Zone 2: +0.786 to +1.0": "0.786_to_1.0", 
            "Zone 3: +0.618 to +0.786": "0.618_to_0.786", "Zone 4: +0.5 to +0.618": "0.5_to_0.618",
            "Zone 5: +0.382 to +0.5": "0.382_to_0.5", "Zone 6: +0.236 to +0.382": "0.236_to_0.382",
            "Zone 7: 0.0 to +0.236": "0.0_to_0.236", "Zone 8: -0.236 to 0.0": "-0.236_to_0.0",
            "Zone 9: -0.382 to -0.236": "-0.382_to_-0.236", "Zone 10: -0.5 to -0.382": "-0.5_to_-0.382",
            "Zone 11: -0.618 to -0.5": "-0.618_to_-0.5", "Zone 12: -0.786 to -0.618": "-0.786_to_-0.618",
            "Zone 13: -1.0 to -0.786": "-1.0_to_-0.786", "Zone 14: Below -1.0": "below_-1.0"
        }
        
        trigger_zone_key = zone_mapping[trigger_zone]
        trigger_time_int = int(trigger_time)
        
        # Filter StateCheck data
        statecheck_filtered = statecheck_df[
            (statecheck_df["TriggerZone"] == trigger_zone_key) &
            (statecheck_df["TriggerTime"] == trigger_time_int)
        ].copy()
        
        
        if len(statecheck_filtered) == 0:
            st.warning(f"No StateCheck data found for {trigger_zone} at {trigger_time}")
            st.info("Try a different trigger zone or time combination.")
            st.stop()
        
        # Adapt StateCheck data to Session format
        adapted_data = statecheck_filtered.copy()
        column_mapping = {
            "GoalZone": "GoalLevel", "GoalTime": "GoalTime", "TransitionCount": "NumHits",
            "TotalTriggerOccurrences": "NumTriggers", "TransitionPercentage": "PctCompletion"
        }
        
        for old_col, new_col in column_mapping.items():
            if old_col in adapted_data.columns:
                adapted_data = adapted_data.rename(columns={old_col: new_col})
        
        # Convert goal zone strings to fibonacci levels
        goal_zone_to_fib = {
            "above_1.0": 1.0, "0.786_to_1.0": 0.786, "0.618_to_0.786": 0.618,  # CHANGED: 1.1 → 1.0
            "0.5_to_0.618": 0.5, "0.382_to_0.5": 0.382, "0.236_to_0.382": 0.236,
            "0.0_to_0.236": 0.0, "-0.236_to_0.0": -0.236, "-0.382_to_-0.236": -0.382,
            "-0.5_to_-0.382": -0.5, "-0.618_to_-0.5": -0.618, "-0.786_to_-0.618": -0.786,
            "-1.0_to_-0.786": -1.0, "below_-1.0": -1.15  # CHANGED: -1.1 → -1.0
        }

           

        if 'GoalLevel' in adapted_data.columns:
            adapted_data['GoalLevel'] = adapted_data['GoalLevel'].map(goal_zone_to_fib)
           
        if 'GoalTime' in adapted_data.columns:
            adapted_data['GoalTime'] = adapted_data['GoalTime'].astype(str).str.zfill(4)
        
        # Set data for chart building
        filtered = adapted_data
             
        available_levels = sorted(filtered['GoalLevel'].unique())

        display_fib_levels = sorted(available_levels)  # Keep ALL levels for chart data
        # Apply hourly bucketing based on expanded view setting
        if not show_expanded_view:
    # Aggregate to hourly buckets for mobile-friendly view
            filtered = aggregate_statecheck_to_hourly(filtered)
           
            
            display_columns = ["0900", "1000", "1100", "1200", "1300", "1400", "1500"]
            #st.info(f"📊 **StateCheck Chart**: {len(display_columns)} hourly periods × {len(display_fib_levels)} levels")
        else:

            # Use detailed 10-minute data
            available_times = sorted(filtered['GoalTime'].unique())
            display_columns = available_times
            #st.info(f"📊 **StateCheck Chart**: {len(display_columns)} time periods × {len(display_fib_levels)} levels | Scroll horizontally to see all data")
        
            
        time_order = display_columns.copy()
            
        
        #st.success(f"✅ StateCheck data adapted: {len(filtered)} records")
        
        # Create compatibility variables
        price_direction = f"Zone Transitions from {trigger_zone}"
        font_size_multiplier = 1.2

      
        
        # BUILD THE CHART using shared function
        fig, chart_use_container_width = create_statecheck_matrix(
            filtered_data=filtered,
            display_fib_levels=display_fib_levels,
            display_columns=display_columns,
            time_order=time_order,
            trigger_zone=trigger_zone,
            price_direction=price_direction,
            ticker_name=ticker_config[selected_ticker]['display_name'],
            text_offset=0.03,
            font_size_multiplier=font_size_multiplier,
            price_levels_dict=price_levels_dict
        )
        
        # Dynamic info message
        #if chart_use_container_width:
            #st.info(f"📊 **StateCheck Chart**: {len(display_columns)} time periods × {len(display_fib_levels)} levels")
        #else:
            #st.info(f"📊 **StateCheck Chart**: {len(display_columns)} time periods × {len(display_fib_levels)} levels | Scroll horizontally to see all data")
        
        # Display the chart
        st.markdown('<div style="width: 3200px; overflow-x: auto;">', unsafe_allow_html=True)
        st.plotly_chart(fig, use_container_width=False)
        st.markdown('</div>', unsafe_allow_html=True)       
        
        # Add color legend
        st.markdown("""
        **🎨 StateCheck Color Legend:**
        - 🟢 **Bright Green** (≥50%): Very High Probability
        - 🌟 **Light Green** (30-49%): High Probability  
        - 🟡 **Yellow** (20-29%): Medium-High Probability
        - 🟠 **Orange** (10-19%): Medium Probability
        - 🔶 **Light Red** (5-9%): Low Probability
        - ⚫ **Gray** (<5%): Very Low Probability
        """)
        
        # Skip remaining chart building
        st.stop()
        
    except Exception as e:
        st.error(f"Error loading StateCheck data: {str(e)}")
        st.stop()

elif analysis_type == "Rolling":
    
    # Load rolling data first - find most recent file
    try:
        rolling_files = glob.glob(f"atr_summary_{selected_ticker}_ROLLING_*.csv")
        if rolling_files:
            rolling_file = max(rolling_files)  # Most recent by filename
            df_rolling = pd.read_csv(rolling_file)
        else:
            st.error(f"❌ No rolling data files found for {selected_ticker}")
            st.stop()
    except Exception as e:
        st.error(f"❌ Error loading rolling data: {str(e)}")
        st.stop()
    
    # Filter rolling data
    filtered_rolling = df_rolling[
        (df_rolling["Direction"] == price_direction) &
        (df_rolling["TriggerLevel"] == trigger_level) &
        (df_rolling["TriggerTime"] == trigger_time)
    ].copy()
    
    if len(filtered_rolling) == 0:
        st.warning(f"No rolling data found for {price_direction} {trigger_level} at {trigger_time}")
        st.stop()
    
    # SIMPLE APPROACH: Use meaningful periods in natural rolling progression
    available_times_original = []  # For data lookup (exact CSV format)
    available_times_display = []   # For chart display (cleaned up)
    original_to_display = {}       # Mapping between formats
    
    # Extract periods with actual hits (no duplicate detection)
    if 'GoalTime' in filtered_rolling.columns:
        unique_goal_times = sorted(filtered_rolling['GoalTime'].unique())
        
        # Debug: Show all periods before filtering
        st.info(f"🔧 Debug: ALL periods in filtered data (before hit filtering): {unique_goal_times}")
        
        for gt in unique_goal_times:
            if pd.notna(gt):
                # Check if this time period has any actual hits
                period_data = filtered_rolling[filtered_rolling['GoalTime'] == gt]
                total_hits = period_data['NumHits'].sum()
                
                # Only include periods that have some actual activity
                if total_hits > 0:
                    if isinstance(gt, (int, float)):
                        time_int = int(gt)
                        if time_int == 900:
                            original_format = str(time_int)
                            display_format = "0900"
                        elif time_int < 1000 and time_int >= 0:
                            original_format = str(time_int)
                            display_format = f"{time_int:04d}"
                        elif time_int >= 1000:
                            original_format = str(time_int)
                            display_format = str(time_int)
                        else:
                            # Day/week numbers: 1, 2, 3, etc.
                            original_format = str(time_int)
                            display_format = f"P{time_int}"
                    else:
                        # String format - keep original exactly, clean up display
                        original_format = str(gt)  # Keep exact: "Next Day 0900"
                        
                        # Clean up for display: "Next Day 0900" → "0900"
                        if "Next Day" in str(gt):
                            display_format = str(gt).replace("Next Day", "").strip()
                        elif "NEXT DAY" in str(gt):
                            display_format = str(gt).replace("NEXT DAY", "").strip()
                        else:
                            display_format = str(gt).strip()
                    
                    available_times_original.append(original_format)
                    available_times_display.append(display_format)
                    original_to_display[original_format] = display_format
    
    # Fallback if no periods found
    if not available_times_original:
        available_times_original = ["0900", "1000", "1100", "1200", "1300", "1400", "1500"]
        available_times_display = ["0900", "1000", "1100", "1200", "1300", "1400", "1500"]
        original_to_display = {k: k for k in available_times_original}
        st.warning("⚠️ Could not detect time periods from data, using standard trading hours")
    
    # Create rolling window starting from trigger time (using ORIGINAL format for logic)
    if trigger_time == "OPEN":
        trigger_time_for_rolling = available_times_original[0] if available_times_original else "0900"
    else:
        # Find trigger time in original format
        trigger_candidates = [trigger_time]
        if trigger_time.isdigit():
            trigger_candidates.extend([
                trigger_time.zfill(4),
                str(int(trigger_time)) if trigger_time != str(int(trigger_time)) else None
            ])
        
        trigger_time_for_rolling = None
        for candidate in trigger_candidates:
            if candidate and candidate in available_times_original:
                trigger_time_for_rolling = candidate
                break
        
        if not trigger_time_for_rolling:
            trigger_time_for_rolling = available_times_original[0]
    
    # Find trigger time index
    if trigger_time_for_rolling in available_times_original:
        trigger_index = available_times_original.index(trigger_time_for_rolling)
    else:
        trigger_index = 0
    
    # Generate rolling sequence - natural progression with no duplicate detection
    rolling_hours_original = []
    rolling_hours_display = []
    
    # Try for up to 8 periods, but use whatever meaningful periods exist
    max_attempts = min(12, len(available_times_original) * 2)  # Allow some wrapping
    target_periods = 8
    
    for i in range(max_attempts):
        period_index = (trigger_index + i) % len(available_times_original)
        original_time = available_times_original[period_index]
        display_time = original_to_display[original_time]
        
        rolling_hours_original.append(original_time)
        rolling_hours_display.append(display_time)
        
        # Stop when we have enough periods or have gone through all available periods once
        if len(rolling_hours_original) >= min(target_periods, len(available_times_original)):
            break
    
    # Display configuration
    if show_expanded_view:
        display_columns = rolling_hours_display + ["TOTAL"]
        display_fib_levels = fib_levels
    else:    
        # For mobile, show first 4 hours of rolling window + TOTAL
        display_columns = rolling_hours_display[:4] + ["TOTAL"]
        
        # Adjust fib levels around trigger level
        trigger_index_fib = fib_levels.index(trigger_level)
        start_fib = max(0, trigger_index_fib - 3)
        end_fib = min(len(fib_levels), trigger_index_fib + 4)
        display_fib_levels = fib_levels[start_fib:end_fib]
    
    # Debug info - show results
    st.info(f"🔧 Debug: Found {len(available_times_original)} meaningful periods")
    st.info(f"🔧 Debug: Original format: {available_times_original}")
    st.info(f"🔧 Debug: Display format: {available_times_display}")
    st.info(f"🔧 Debug: Rolling sequence (display): {' → '.join(rolling_hours_display)}")
    
    # Show which periods have hits
    hit_info = []
    for orig in rolling_hours_original:
        # Handle different formats for lookup
        period_data = filtered_rolling[filtered_rolling['GoalTime'].astype(str) == str(orig)]
        if len(period_data) == 0:
            # Try original format matching
            period_data = filtered_rolling[filtered_rolling['GoalTime'] == orig]
        hits = period_data['NumHits'].sum() if len(period_data) > 0 else 0
        hit_info.append(f"{orig}({hits})")
    st.info(f"🔧 Debug: Hit breakdown: {' | '.join(hit_info)}")
    
    # Build chart using the fixed rolling matrix function
    fig, use_container_width = create_rolling_matrix(
        filtered_data=filtered_rolling,
        display_fib_levels=display_fib_levels,
        display_columns=display_columns,
        rolling_hours_original=rolling_hours_original,
        rolling_hours_display=rolling_hours_display,
        price_direction=price_direction,
        trigger_level=trigger_level,
        trigger_time=trigger_time,
        ticker_name=ticker_config[selected_ticker]['display_name'],
        show_expanded_view=show_expanded_view,
        price_levels_dict=price_levels_dict
    )
    
    # Display the chart
    st.plotly_chart(fig, use_container_width=use_container_width)
    
    # Chart Information Footer
    if atr_data.get("status") == "success":
        data_age = atr_data.get('data_age_days', 0)
        age_warning = f" (⚠️ {data_age} days old)" if data_age > 0 else ""
        st.caption(f"📊 ATR levels from {atr_data.get('reference_date', 'unknown')} | Close: {atr_data.get('reference_close', 'N/A')} | ATR: {atr_data.get('reference_atr', 'N/A')}{age_warning}")
    
    # Legend
    st.caption("📋 **Rolling Analysis Key:** ⚠️ = Less than 30 historical triggers (lower confidence) | Rolling window shows natural progression through periods with historical activity")
    
    # End Rolling analysis
    st.stop()
    
else:  # Session
    # Load session data using glob pattern
    try:
        session_files = glob.glob(f"atr_summary_{selected_ticker}_SESSION_*.csv")
        if session_files:
            session_file = max(session_files)  # Most recent by filename
            df = pd.read_csv(session_file)
        else:
            st.error(f"❌ No session data files found for {selected_ticker}")
            st.stop()
    except Exception as e:
        st.error(f"❌ Error loading session data for {selected_ticker}: {str(e)}")
        st.stop()

    # Session data filtering
    filtered = df[
        (df["Direction"] == price_direction) &
        (df["TriggerLevel"] == trigger_level) &
        (df["TriggerTime"] == trigger_time)
    ].copy()
    
    if len(filtered) == 0:
        st.warning(f"No session data found for {price_direction} {trigger_level} at {trigger_time}")
        st.stop()
    
    # EXTRACT MEANINGFUL PERIODS FROM DATA (like Rolling analysis does)
    available_times_original = []  # For data lookup (exact CSV format)
    available_times_display = []   # For chart display (cleaned up)
    original_to_display = {}       # Mapping between formats
    
    # Extract periods with actual hits from GoalTime column
    if 'GoalTime' in filtered.columns:
        unique_goal_times = sorted(filtered['GoalTime'].unique())
        
        for gt in unique_goal_times:
            if pd.notna(gt):
                # Check if this time period has any actual hits
                period_data = filtered[filtered['GoalTime'] == gt]
                total_hits = period_data['NumHits'].sum()
                
                # Only include periods that have some actual activity
                if total_hits > 0:
                    if isinstance(gt, (int, float)):
                        time_int = int(gt)
                        if time_int == 900:
                            original_format = str(time_int)
                            display_format = "0900"
                        elif time_int < 1000 and time_int >= 0:
                            original_format = str(time_int)
                            display_format = f"{time_int:04d}"
                        elif time_int >= 1000:
                            original_format = str(time_int)
                            display_format = str(time_int)
                        else:
                            # Day/week numbers: 1, 2, 3, etc.
                            original_format = str(time_int)
                            display_format = f"P{time_int}"
                    else:
                        # String format - keep original exactly, clean up display
                        original_format = str(gt)  # Keep exact: "Next Day 0900"
                        
                        # Clean up for display: "Next Day 0900" → "0900"
                        if "Next Day" in str(gt):
                            display_format = str(gt).replace("Next Day", "").strip()
                        elif "NEXT DAY" in str(gt):
                            display_format = str(gt).replace("NEXT DAY", "").strip()
                        else:
                            display_format = str(gt).strip()
                    
                    available_times_original.append(original_format)
                    available_times_display.append(display_format)
                    original_to_display[original_format] = display_format
    
    # Add OPEN if trigger time is OPEN (Session analysis needs this)
    if trigger_time == "OPEN":
        # Insert OPEN at the beginning
        available_times_original.insert(0, "OPEN")
        available_times_display.insert(0, "OPEN")
        original_to_display["OPEN"] = "OPEN"
    
    # Fallback if no periods found
    if len(available_times_original) <= 1:  # Just OPEN or nothing
        available_times_original = ["OPEN", "0900", "1000", "1100", "1200", "1300", "1400", "1500"]
        available_times_display = ["OPEN", "0900", "1000", "1100", "1200", "1300", "1400", "1500"]
        original_to_display = {k: k for k in available_times_original}
        st.warning("⚠️ Could not detect time periods from data, using standard trading hours")
    
    # DETECT IF THIS IS FUTURES DATA (many periods) vs STOCK DATA (few periods)
    num_meaningful_periods = len([t for t in available_times_original if t != "OPEN"])
    is_futures_data = num_meaningful_periods > 15  # Futures will have 20+ periods, stocks have ~7
    
    if is_futures_data:
        st.info(f"🔄 **Futures data detected** ({num_meaningful_periods} periods). Session navigation will be available.")
        # TODO: Add session navigation logic here
        # For now, show all periods (will be compressed but functional)
        
    # Build display columns 
    if show_expanded_view:
        display_columns = available_times_display + ["TOTAL", "REMAINING"]
        display_fib_levels = fib_levels
        chart_height = 700
        chart_width = 1800 if not is_futures_data else 2400  # Wider for futures
        font_size_multiplier = 1.0
        use_container_width = False
    else:    
        # For mobile view - limit to reasonable number of columns
        if is_futures_data:
            # Show just NY session for now (0800-1700 ET)
            ny_start = None
            ny_periods = []
            
            # Find periods in NY session range (0800-1700)
            for i, time_str in enumerate(available_times_display):
                if time_str != "OPEN":
                    try:
                        hour = int(time_str.replace(":", ""))
                        if 800 <= hour <= 1700:
                            ny_periods.append(time_str)
                    except:
                        pass
            
            # Limit to first 6 periods for readability
            display_columns = ny_periods[:6] + ["TOTAL", "REMAINING"] if ny_periods else available_times_display[:6] + ["TOTAL", "REMAINING"]
            st.info(f"📱 **Mobile view**: Showing NY session hours only. Use expanded view or session navigation for full data.")
        else:
            # Stock data - use data-driven logic for ALL triggers
            if trigger_time == "OPEN":
                # Use discovered periods instead of hard-coded times
                display_columns = ["OPEN"] + available_times_display[:3] + ["TOTAL", "REMAINING"]
            else:
                # Find trigger time and show context around it
                if trigger_time in available_times_display:
                    trigger_idx = available_times_display.index(trigger_time)
                    start_idx = max(0, trigger_idx - 2)
                    end_idx = min(len(available_times_display), trigger_idx + 4)
                    context_periods = available_times_display[start_idx:end_idx]
                    display_columns = context_periods + ["TOTAL", "REMAINING"]
                else:
                    display_columns = available_times_display[:6] + ["TOTAL", "REMAINING"]
        
        # Adjust fib levels around trigger level
        trigger_index = fib_levels.index(trigger_level)
        start_fib = max(0, trigger_index - 3)
        end_fib = min(len(fib_levels), trigger_index + 4)
        display_fib_levels = fib_levels[start_fib:end_fib]
        
        chart_height = 400
        chart_width = 700
        font_size_multiplier = 1.0
        use_container_width = False
    
    time_order = display_columns.copy()
    
    # --- Create lookup dictionary from pre-calculated data (UPDATED to use original format) ---
    data_lookup = {}
    for _, row in filtered.iterrows():
        goal_time = row["GoalTime"]
        if pd.notna(goal_time):
            # Use original format for lookup keys
            if isinstance(goal_time, (int, float)):
                goal_time_str = str(int(goal_time))
            else:
                goal_time_str = str(goal_time)
        else:
            goal_time_str = "Unknown"
        
        key = (float(row["GoalLevel"]), goal_time_str)
        data_lookup[key] = {
            "hits": row["NumHits"],
            "triggers": row["NumTriggers"], 
            "pct": row["PctCompletion"]
        }
    
    # Calculate totals and remaining (existing logic continues unchanged)
    goal_totals = {}
    goal_remaining = {}
    if len(filtered) > 0:
        goal_summary = filtered.groupby('GoalLevel').agg({
            'NumHits': 'sum',
            'NumTriggers': 'first'
        }).reset_index()
        
        standard_time_order = available_times_display.copy()  # Use discovered periods instead of hard-coded
        
        for _, row in goal_summary.iterrows():
            goal_level = row['GoalLevel']
            total_hits = row['NumHits']
            total_triggers = row['NumTriggers']
            total_pct = (total_hits / total_triggers * 100) if total_triggers > 0 else 0
            goal_totals[goal_level] = {
                "hits": total_hits,
                "triggers": total_triggers,
                "pct": total_pct
            }
            
            # Calculate remaining probability
            hourly_pcts = {}
            for time_slot in standard_time_order:
                if time_slot in original_to_display.values():  # Check display format
                    # Find original format
                    original_time_slot = None
                    for orig, disp in original_to_display.items():
                        if disp == time_slot:
                            original_time_slot = orig
                            break
                    
                    if original_time_slot:
                        key = (goal_level, original_time_slot)
                        if key in data_lookup:
                            hourly_pcts[time_slot] = data_lookup[key]["pct"]
            
            remaining_pct, current_slot_info = calculate_remaining_probability(
                total_pct, hourly_pcts, current_market_slot, standard_time_order
            )
            
            goal_remaining[goal_level] = {
                "pct": remaining_pct,
                "current_slot": current_slot_info,
                "total_pct": total_pct
            }
    
    # Get OPEN trigger data for tooltip (existing logic continues)
    open_trigger_data = {}
    if trigger_time == "OPEN" and len(filtered) > 0:
        open_triggers = filtered['NumTriggers'].iloc[0]
        
        for _, row in filtered.iterrows():
            goal_level = row['GoalLevel']
            if 'OpenCompletions' in row:
                open_completions = row['OpenCompletions']
            else:
                open_completions = "N/A"
            
            open_trigger_data[goal_level] = {
                "triggers": open_triggers,
                "completions": open_completions
            }
    
    # Continue with existing Session chart building logic...
    # (The chart building code from line ~800 onwards remains the same)
    
    # --- Build Session chart ---
    fig = go.Figure()
    text_offset = 0.03
    
    # Add "Fib Level" title above left axis
    fig.add_annotation(
        text="Fib Level",
        x=-0.05,
        y=max(display_fib_levels) + 0.15,
        xref="paper",
        yref="y",
        showarrow=False,
        font=dict(color="gray", size=12 * font_size_multiplier),
        xanchor="center",
        yanchor="bottom"
    )
    
    # Add "Price Level" title above right side
    fig.add_annotation(
        text="Price Level", 
        x=1.08,
        y=max(display_fib_levels) + 0.15,
        xref="paper", 
        yref="y",
        showarrow=False,
        font=dict(color="gray", size=12 * font_size_multiplier),
        xanchor="center",
        yanchor="bottom"
    )
    
    # --- Price labels as annotations ---
    if price_levels_dict:
        for level in display_fib_levels:
            level_key = f"{level:+.3f}"
            price_val = price_levels_dict.get(level_key, 0)
            
            fig.add_annotation(
                text=f"{price_val:.2f}",
                x=1.08,
                y=level + text_offset,
                xref="paper",
                yref="y",
                showarrow=False,
                font=dict(color="white", size=14 * font_size_multiplier),
                xanchor="left",
                yanchor="middle"
            )
    
    # --- Matrix cells ---
    for level in display_fib_levels:
        for t in time_order:
            if t not in display_columns:
                continue
                
            if t == "OPEN":
                if trigger_time == "OPEN" and level in open_trigger_data:
                    triggers = open_trigger_data[level]["triggers"]
                    completions = open_trigger_data[level]["completions"]
                    hover = f"OPEN Triggers: {triggers}, Goal {level} Completed at OPEN: {completions}"
                    
                    fig.add_trace(go.Scatter(
                        x=[t], y=[level + text_offset],
                        mode="text", text=[""],
                        hovertext=[hover], hoverinfo="text",
                        textfont=dict(color="white", size=13),
                        showlegend=False
                    ))
                else:
                    fig.add_trace(go.Scatter(
                        x=[t], y=[level + text_offset],
                        mode="text", text=[""],
                        hoverinfo="skip",
                        textfont=dict(color="white", size=13),
                        showlegend=False
                    ))
                continue
            
            if t == "TOTAL":
                if level in goal_totals:
                    total_data = goal_totals[level]
                    pct = total_data["pct"]
                    hits = total_data["hits"]
                    triggers = total_data["triggers"]
                    
                    line_color, line_width, font_size = fibo_styles.get(level, ("lightgray", 1, 12))
                    font_size = 12 * font_size_multiplier
                    
                    warn = " ⚠️" if triggers < 30 else ""
                    display_text = f"{pct:.1f}%"
                    hover = f"Total: {pct:.1f}% ({hits}/{triggers}){warn}"
                    
                    fig.add_trace(go.Scatter(
                        x=[t], y=[level + text_offset],
                        mode="text", text=[display_text],
                        hovertext=[hover], hoverinfo="text",
                        textfont=dict(color=line_color, size=font_size),
                        showlegend=False
                    ))
                else:
                    fig.add_trace(go.Scatter(
                        x=[t], y=[level + text_offset],
                        mode="text", text=[""],
                        hoverinfo="skip",
                        textfont=dict(color="white", size=12),
                        showlegend=False
                    ))
                continue
            
            if t == "REMAINING":
                if level in goal_remaining:
                    remaining_data = goal_remaining[level]
                    remaining_pct = remaining_data["pct"]
                    total_pct = remaining_data["total_pct"]
                    current_slot = remaining_data["current_slot"]
                    
                    line_color, line_width, font_size = fibo_styles.get(level, ("lightgray", 1, 12))
                    font_size = 12 * font_size_multiplier
                    
                    if current_slot == "N/A":
                        display_text = "N/A"
                        hover = "Market closed or no data"
                        text_color = "gray"
                    else:
                        display_text = f"{remaining_pct:.1f}%"
                        completed_pct = total_pct - remaining_pct
                        hover = f"Remaining: {remaining_pct:.1f}% (Total: {total_pct:.1f}%, Completed: {completed_pct:.1f}%) | Current: {current_slot}"
                        
                        # Color code based on remaining probability
                        if remaining_pct > 15:
                            text_color = "lime"
                        elif remaining_pct > 5:
                            text_color = "orange"
                        else:
                            text_color = "red"
                    
                    fig.add_trace(go.Scatter(
                        x=[t], y=[level + text_offset],
                        mode="text", text=[display_text],
                        hovertext=[hover], hoverinfo="text",
                        textfont=dict(color=text_color, size=font_size),
                        showlegend=False
                    ))
                else:
                    fig.add_trace(go.Scatter(
                        x=[t], y=[level + text_offset],
                        mode="text", text=[""],
                        hoverinfo="skip",
                        textfont=dict(color="white", size=12),
                        showlegend=False
                    ))
                continue
            
            # Regular time columns
            key = (float(level), t)
            if key in data_lookup:
                data = data_lookup[key]
                pct = data["pct"]
                hits = data["hits"]
                total = data["triggers"]
                
                # Check if times are before trigger time
                if trigger_time == "OPEN":
                    is_before_trigger = False
                elif trigger_time in time_order and t in time_order:
                    is_before_trigger = time_order.index(t) < time_order.index(trigger_time)
                else:
                    is_before_trigger = False
                
                if is_before_trigger:
                    display_text = ""
                    hover = "Before trigger time"
                    text_color = "gray"
                else:
                    warn = " ⚠️" if total < 30 else ""
                    display_text = f"{pct:.1f}%"
                    hover = f"{pct:.1f}% ({hits}/{total}){warn}"
                    
                    # Color coding based on fibonacci level
                    line_color, line_width, font_size = fibo_styles.get(level, ("white", 1, 12))
                    text_color = line_color
                
                font_size = 12 * font_size_multiplier
                
                fig.add_trace(go.Scatter(
                    x=[t], y=[level + text_offset],
                    mode="text", text=[display_text],
                    hovertext=[hover], hoverinfo="text",
                    textfont=dict(color=text_color, size=font_size),
                    showlegend=False
                ))
    
    # --- Anchor invisible point for OPEN ---
    fig.add_trace(go.Scatter(
        x=["OPEN"], y=[0.0],
        mode="markers",
        marker=dict(opacity=0),
        showlegend=False,
        hoverinfo="skip"
    ))
    
    # --- Horizontal lines for Fibonacci levels ---
    for level in display_fib_levels:
        if level in fibo_styles:
            color, width, font_size = fibo_styles[level]
            fig.add_shape(
                type="line", x0=0, x1=1, xref="paper", y0=level, y1=level, yref="y",
                line=dict(color=color, width=width), layer="below"
            )
    
    # --- Session trigger level highlighting ---
    if trigger_level in display_fib_levels:
        trigger_index = display_fib_levels.index(trigger_level)
        
        # Green shading above trigger level
        if trigger_index > 0:
            next_level_up = display_fib_levels[trigger_index - 1]
            fig.add_shape(
                type="rect",
                x0=0, x1=1, xref="paper",
                y0=trigger_level, y1=next_level_up, yref="y",
                fillcolor="rgba(0, 255, 0, 0.1)",
                line=dict(width=0),
                layer="below"
            )
        
        # Yellow shading below trigger level
        if trigger_index < len(display_fib_levels) - 1:
            next_level_down = display_fib_levels[trigger_index + 1]
            fig.add_shape(
                type="rect",
                x0=0, x1=1, xref="paper",
                y0=next_level_down, y1=trigger_level, yref="y",
                fillcolor="rgba(255, 255, 0, 0.1)",
                line=dict(width=0),
                layer="below"
            )
    
    # --- Session chart layout ---
    fig.update_layout(
        title=f"{ticker_config[selected_ticker]['display_name']} | {price_direction}",
        xaxis=dict(
            title="Projected Completion Time (Eastern Time)",
            categoryorder="array",
            categoryarray=display_columns,
            tickmode="array",
            tickvals=display_columns,
            ticktext=display_columns,
            tickfont=dict(color="white", size=12),
            fixedrange=False if not show_expanded_view else True
        ),
        yaxis=dict(
            title="",
            categoryorder="array",
            categoryarray=display_fib_levels,
            tickmode="array",
            tickvals=display_fib_levels,
            ticktext=[f"{lvl:+.3f}" for lvl in display_fib_levels],
            tickfont=dict(color="white", size=12 * font_size_multiplier),
            side="left",
            fixedrange=False if not show_expanded_view else True
        ),
        plot_bgcolor="black",
        paper_bgcolor="black",
        font=dict(color="white", size=12 * font_size_multiplier),
        height=chart_height,
        width=chart_width,
        margin=dict(l=80, r=150, t=60, b=60)
    )
    
    # Display the chart
    st.plotly_chart(fig, use_container_width=use_container_width)
    
    # --- Chart Information Footer ---
    if atr_data.get("status") == "success":
        data_age = atr_data.get('data_age_days', 0)
        age_warning = f" (⚠️ {data_age} days old)" if data_age > 0 else ""
        st.caption(f"📊 ATR levels from {atr_data.get('reference_date', 'unknown')} | Close: {atr_data.get('reference_close', 'N/A')} | ATR: {atr_data.get('reference_atr', 'N/A')}{age_warning}")
    
    # Legend
    st.caption("📋 **Chart Key:** ⚠️ = Less than 30 historical triggers (lower confidence) | **Remaining Colors:** 🟢 >15% | 🟠 5-15% | 🔴 <5% | Percentages show probability of reaching target level by specified time")

# --- Footer ---
if current_market_slot in ["PREMARKET", "AFTERHOURS"]:
    time_color = "🔴"
elif current_market_slot == "CLOSE":
    time_color = "⚫"
else:
    time_color = "🟢"

st.info(f"{time_color} **Current ET:** {current_et_time.strftime('%I:%M %p')} | **Market Slot:** {current_market_slot}")

# --- Multi-ticker status info ---
if st.checkbox("📊 Show Multi-Ticker Status"):
    try:
        with open("atr_levels.json", 'r') as f:
            all_data = json.load(f)
            
        if "tickers" in all_data:
            st.subheader("📈 All Ticker Status")
            status_data = []
            for ticker_key, ticker_data in all_data["tickers"].items():
                status_data.append({
                    "Ticker": ticker_key,
                    "Status": "✅" if ticker_data.get("status") == "success" else "❌",
                    "Close": f"${ticker_data.get('reference_close', 'N/A'):.2f}" if ticker_data.get('reference_close') else "N/A",
                    "ATR": f"${ticker_data.get('reference_atr', 'N/A'):.2f}" if ticker_data.get('reference_atr') else "N/A",
                    "Date": ticker_data.get('reference_date', 'N/A'),
                    "Age (days)": ticker_data.get('data_age_days', 'N/A')
                })
            
            status_df = pd.DataFrame(status_data)
            st.dataframe(status_df, use_container_width=True)
            st.caption(f"Last updated: {all_data.get('last_updated', 'Unknown')}")
        else:
            st.info("Legacy single-ticker format detected")
            
    except Exception as e:
        st.error(f"Could not load ticker status: {str(e)}")
