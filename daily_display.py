import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import json
import os
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
    get_rolling_8_hours,
    get_total_probability
)

# Temporary fix - add this after imports
# Create flattened list with group separators (show available tickers only)
ticker_options = []
ticker_mapping = {}  # Map display names to actual ticker symbols

# Define ticker groups
ticker_groups = {
    "📊 Indices": [
        "SPX - S&P 500",
        "NDX - Nasdaq 100", 
        "RUT - Russell 2000"
    ],
    "📈 Stocks": [
        "NVDA - NVIDIA",
        "AAPL - Apple",
        "GOOGL - Google",
        "TSLA - Tesla"
    ],
    "🏢 Sectors": [
        "XLF - Financial Select",
        "XLE - Energy Select", 
        "XLK - Technology Select",
        "XLV - Health Care Select"
    ],
    "🔮 Futures": [
        "ES - E-mini S&P 500",
        "NQ - E-mini Nasdaq",
        "YM - E-mini Dow",
        "RTY - E-mini Russell"
    ],
    "💰 Forex": [
        "EURUSD - Euro/US Dollar",
        "GBPUSD - British Pound/US Dollar",
        "USDJPY - US Dollar/Japanese Yen"
    ],
    "₿ Crypto": [
        "BTCUSD - Bitcoin",
        "ETHUSD - Ethereum"
    ]
}

# Check availability and build options list
for group_name, tickers in ticker_groups.items():
    # Always add group header
    ticker_options.append(f"--- {group_name} ---")
    
    group_has_available = False
    for ticker_display in tickers:
        ticker_symbol = ticker_display.split(" - ")[0]
        
        # Check for session data file
        summary_file = f"atr_summary_{ticker_symbol}_SESSION_{datetime.now().strftime('%Y%m%d')}_*.csv"
        
        # For now, just check if SPX file exists (since that's what we have)
        if ticker_symbol == "SPX":
            summary_file = "atr_summary_SPX_SESSION_20250710_064535.csv"
            if os.path.exists(summary_file):
                ticker_options.append(ticker_display)
                ticker_mapping[ticker_display] = ticker_symbol
                group_has_available = True
        # Add other tickers as their data becomes available
        # elif ticker_symbol == "ES" and os.path.exists(f"atr_summary_ES_SESSION_*.csv"):
        #     ticker_options.append(ticker_display)
        #     ticker_mapping[ticker_display] = ticker_symbol
        #     group_has_available = True
    
    # If no available tickers in this group, show placeholder
    if not group_has_available:
        ticker_options.append("--- (Coming Soon) ---")

# Ticker selector with availability check
if len([opt for opt in ticker_options if not opt.startswith("---")]) == 0:
    st.error("❌ No ticker data files found!")
    st.info("📝 Have a ticker request? Check back soon!")
    st.stop()

selected_ticker_display = st.selectbox("Select Ticker", ticker_options, index=1)  # Skip first separator

# Get actual ticker symbol (handle "Coming Soon" selections)
if not selected_ticker_display.startswith("---"):
    if selected_ticker_display in ticker_mapping:
        selected_ticker = ticker_mapping[selected_ticker_display]
        
        # Determine instrument category
        instrument_category = None
        for group_name, tickers in ticker_groups.items():
            if selected_ticker_display in tickers:
                instrument_category = group_name.split(" ", 1)[1]  # Remove emoji
                break
        
        st.success(f"✅ Selected: {selected_ticker_display} ({instrument_category})")
    else:
        st.error("❌ This ticker is not yet available")
        st.info("📝 Check back soon or visit our ticker request page!")
        st.stop()
else:
    st.error("Please select a valid ticker (not a header)")
    st.stop()

# Create ticker config dynamically
ticker_config = {
    selected_ticker: {
        "summary_file": f"atr_summary_{selected_ticker}_SESSION_20250710_064535.csv",
        "display_name": selected_ticker_display.split(" - ")[1],
        "ticker_symbol": selected_ticker
    }
}
    
# --- Grouped Ticker Selection ---
st.title("📈 Daily ATR Analysis")
st.caption("🔧 App Version: v3.0.0 - Multi-Instrument Daily Viewer")

# Create grouped ticker options
ticker_groups = {
    "📊 Indices": [
        "SPX - S&P 500",
        "NDX - Nasdaq 100", 
        "RUT - Russell 2000"
    ],
    "📈 Stocks": [
        "NVDA - NVIDIA",
        "AAPL - Apple",
        "GOOGL - Google",
        "TSLA - Tesla"
    ],
    "🏢 Sectors": [
        "XLF - Financial Select",
        "XLE - Energy Select", 
        "XLK - Technology Select",
        "XLV - Health Care Select"
    ],
    "🔮 Futures": [
        "ES - E-mini S&P 500",
        "NQ - E-mini Nasdaq",
        "YM - E-mini Dow",
        "RTY - E-mini Russell"
    ],
    "💰 Forex": [
        "EURUSD - Euro/US Dollar",
        "GBPUSD - British Pound/US Dollar",
        "USDJPY - US Dollar/Japanese Yen"
    ],
    "₿ Crypto": [
        "BTCUSD - Bitcoin",
        "ETHUSD - Ethereum"
    ]
}
# --- Analysis Type Selection ---
st.subheader("📊 Analysis Type")
analysis_type = st.selectbox(
    "Select Analysis Type",
    ["Session", "Rolling", "StateCheck", "ZoneBaseline"],
    index=0,
    help="Session: Full session analysis | Rolling: 8-hour window | StateCheck: Zone transitions | ZoneBaseline: Static zone probability"
)

# --- Conditional Controls Based on Analysis Type ---
st.subheader("⚙️ Analysis Parameters")

if analysis_type == "ZoneBaseline":
    st.info("📍 **ZoneBaseline**: Static probability heatmap showing zone occupancy throughout the session")
    st.caption("No additional parameters needed - shows probability of being in each zone at each time")
    
elif analysis_type == "StateCheck":
    st.info("🔄 **StateCheck**: Zone transition probabilities from trigger zone over time")
    
    # Zone definitions
    zone_definitions = [
    "Zone 1: Above +1.0",
    "Zone 2: +0.786 to +1.0", 
    "Zone 3: +0.618 to +0.786",
    "Zone 4: +0.5 to +0.618",      # <-- Added 0.5 level
    "Zone 5: +0.382 to +0.5",      # <-- Added 0.5 level
    "Zone 6: +0.236 to +0.382",
    "Zone 7: 0.0 to +0.236",
    "Zone 8: -0.236 to 0.0",
    "Zone 9: -0.382 to -0.236",
    "Zone 10: -0.5 to -0.382",     # <-- Added 0.5 level
    "Zone 11: -0.618 to -0.5",     # <-- Added 0.5 level
    "Zone 12: -0.786 to -0.618",
    "Zone 13: -1.0 to -0.786",
    "Zone 14: Below -1.0"
]
    
    col1, col2 = st.columns(2)
    with col1:
        trigger_zone = st.selectbox("Trigger Zone", zone_definitions, index=6)  # Default to Zone 7
    with col2:
        # Remove OPEN for StateCheck - use only regular hours (full trading day)
        trigger_time = st.selectbox("Trigger Time", [
            "0930", "0940", "0950", "1000", "1010", "1020", "1030", "1040", "1050", 
            "1100", "1110", "1120", "1130", "1140", "1150", "1200", "1210", "1220", 
            "1230", "1240", "1250", "1300", "1310", "1320", "1330", "1340", "1350", 
            "1400", "1410", "1420", "1430", "1440", "1450", "1500", "1510", "1520", 
            "1530", "1540", "1550"
        ], index=0)
    
elif analysis_type == "Rolling":
    st.info("⏰ **Rolling**: 8-hour rolling window analysis from trigger time")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        price_direction = st.selectbox("Price Location", ["Above", "Below"], index=0)
    with col2:
        # Use fib_levels from existing code
        fib_levels = [1.0, 0.786, 0.618, 0.5, 0.382, 0.236, 0.0, -0.236, -0.382, -0.5, -0.618, -0.786, -1.0]
        trigger_level = st.selectbox("Trigger Level", fib_levels, index=6)  # Default to 0.0
    with col3:
        trigger_time = st.selectbox("Trigger Time", ["OPEN", "0900", "1000", "1100", "1200", "1300", "1400", "1500"], index=0)
    
    # Show rolling window preview
    rolling_hours = get_rolling_8_hours(trigger_time)
    st.caption(f"🔄 Rolling window: {' → '.join(rolling_hours[:4])} → {' → '.join(rolling_hours[4:])}")
    
else:  # Session
    st.info("📈 **Session**: Full session analysis with trigger conditions")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        price_direction = st.selectbox("Price Location", ["Above", "Below"], index=0)
    with col2:
        # Use fib_levels from existing code
        fib_levels = [1.0, 0.786, 0.618, 0.5, 0.382, 0.236, 0.0, -0.236, -0.382, -0.5, -0.618, -0.786, -1.0]
        trigger_level = st.selectbox("Trigger Level", fib_levels, index=6)  # Default to 0.0
    with col3:
        trigger_time = st.selectbox("Trigger Time", ["OPEN", "0900", "1000", "1100", "1200", "1300", "1400", "1500"], index=0)

# Show analysis summary
st.divider()
if analysis_type == "ZoneBaseline":
    st.write(f"**Analysis**: {analysis_type}")
elif analysis_type == "StateCheck":
    st.write(f"**Analysis**: {analysis_type} | **From**: {trigger_zone} at {trigger_time}")
else:  # Session or Rolling
    st.write(f"**Analysis**: {analysis_type} | **Trigger**: {price_direction} {trigger_level} at {trigger_time}")
def get_current_market_time():
    """Get current Eastern Time and determine market time slot"""
    if USE_ZONEINFO:
        et = ZoneInfo('US/Eastern')
        current_et = datetime.now(et)
    else:
        et = pytz.timezone('US/Eastern')
        current_et = datetime.now(et)
    current_time = current_et.time()
    
    # Define market time slots
    time_slots = [
        (time(9, 30), "OPEN"),
        (time(9, 0), "0900"),
        (time(10, 0), "1000"),
        (time(11, 0), "1100"),
        (time(12, 0), "1200"),
        (time(13, 0), "1300"),
        (time(14, 0), "1400"),
        (time(15, 0), "1500"),
        (time(16, 0), "CLOSE")
    ]
    
    # Sort by time to find current slot
    time_slots.sort(key=lambda x: x[0])
    
    current_slot = "PREMARKET"
    for slot_time, slot_name in time_slots:
        if current_time >= slot_time:
            current_slot = slot_name
        else:
            break
    
    # If after 4 PM, consider it AFTERHOURS
    if current_time >= time(16, 0):
        current_slot = "AFTERHOURS"
    
    return current_et, current_slot

def calculate_remaining_probability(total_pct, completed_hourly_pcts, current_time_slot, time_order):
    """
    Calculate remaining probability based on current time
    total_pct: Total probability for the day
    completed_hourly_pcts: Dictionary of {time_slot: percentage} for completed hours
    current_time_slot: Current market time slot
    time_order: List of time slots in order
    """
    if current_time_slot in ["PREMARKET", "AFTERHOURS", "CLOSE"]:
        return total_pct, "N/A"
    
    # Find current position in time order
    try:
        current_index = time_order.index(current_time_slot)
    except ValueError:
        return total_pct, "Current"
    
    # Sum up probabilities for completed time slots
    completed_probability = 0
    for i, time_slot in enumerate(time_order):
        if i < current_index and time_slot in completed_hourly_pcts:
            completed_probability += completed_hourly_pcts[time_slot]
    
    remaining_pct = max(0, total_pct - completed_probability)
    return remaining_pct, current_time_slot

def get_atr_levels_for_ticker(ticker_key):
    """
    Get ATR levels for specific ticker from multi-ticker JSON file
    Returns the levels data or empty dict if error
    """
    try:
        json_file = "atr_levels.json"
        
        # Try to load from saved JSON file first (if exists and recent)
        if os.path.exists(json_file):
            with open(json_file, 'r') as f:
                saved_data = json.load(f)
                
                # Check if we have multi-ticker format
                if "tickers" in saved_data and ticker_key in saved_data["tickers"]:
                    ticker_data = saved_data["tickers"][ticker_key]
                    if ticker_data.get("status") == "success":
                        return ticker_data
                # Legacy single-ticker format fallback
                elif ticker_key == "SPX" and saved_data.get("status") == "success":
                    return saved_data
        
        # If no saved file or error, calculate fresh for this ticker
        ticker_symbol = ticker_config[ticker_key]["ticker_symbol"]
        levels_data = calculate_atr_levels(ticker=ticker_symbol)
        
        if levels_data.get("status") == "success":
            # If we have an existing multi-ticker file, update it
            if os.path.exists(json_file):
                try:
                    with open(json_file, 'r') as f:
                        existing_data = json.load(f)
                    
                    if "tickers" not in existing_data:
                        existing_data = {
                            "last_updated": datetime.now().isoformat(),
                            "tickers": {}
                        }
                    
                    existing_data["tickers"][ticker_key] = levels_data
                    existing_data["last_updated"] = datetime.now().isoformat()
                    
                    with open(json_file, 'w') as f:
                        json.dump(existing_data, f, indent=2)
                        
                except Exception:
                    # If updating fails, create new multi-ticker file
                    new_data = {
                        "last_updated": datetime.now().isoformat(),
                        "tickers": {ticker_key: levels_data}
                    }
                    with open(json_file, 'w') as f:
                        json.dump(new_data, f, indent=2)
            else:
                # Create new multi-ticker file
                new_data = {
                    "last_updated": datetime.now().isoformat(),
                    "tickers": {ticker_key: levels_data}
                }
                with open(json_file, 'w') as f:
                    json.dump(new_data, f, indent=2)
        
        return levels_data
        
    except Exception as e:
        st.error(f"Error getting ATR levels for {ticker_key}: {str(e)}")
        return {"status": "error", "error": str(e)}

# --- Get Current Market Time ---
current_et_time, current_market_slot = get_current_market_time()

# --- Page Layout with Ticker Selector ---
col_title1, col_title2 = st.columns([4, 1])
with col_title1:
    st.title("📈 ATR Levels Roadmap")
    st.caption("🔧 App Version: v2.5.0 - Multi-Ticker Support") # VERSION BUMP


# --- Load data based on selected ticker ---
try:
    df = pd.read_csv(ticker_config[selected_ticker]["summary_file"])
    st.success(f"✅ Loaded data for {ticker_config[selected_ticker]['display_name']}")
except FileNotFoundError:
    st.error(f"❌ Data file not found for {selected_ticker}: {ticker_config[selected_ticker]['summary_file']}")
    st.info("💡 You need to create summary CSV files for each ticker you want to support")
    st.stop()
except Exception as e:
    st.error(f"❌ Error loading data for {selected_ticker}: {str(e)}")
    st.stop()

# --- Load current ATR-based price levels ---
atr_data = get_atr_levels_for_ticker(selected_ticker)

if atr_data.get("status") == "success":
    atr_price_levels = atr_data
    price_levels_dict = atr_data.get("levels", {})
else:
    atr_price_levels = {}
    price_levels_dict = {}
    st.error(f"❌ Could not load ATR levels for {selected_ticker}: {atr_data.get('error', 'Unknown error')}")

# --- What's This? Section ---
with st.expander("❓ What's This? - How to Use This Chart"):
    unique_days = 2720
    day_text = f"{unique_days:,} trading days"
    
    st.markdown(f"""
    **This chart shows the probability of reaching price levels based on historical data from {day_text}.**
    
    📊 **How to Read:**
    - **Rows (Fib Levels):** Target price levels based on ATR (Average True Range)
    - **Columns (Times):** Hours during the trading day when the target was reached
    - **Percentages:** Historical success rate - how often price reached that level by that time
    - **Colors:** Match the horizontal line colors for easy reference
    - **Remaining:** Shows probability left for the day based on current market time
    
    🎯 **How to Use:**
    1. **Select Ticker:** Choose which instrument to analyze
    2. **Select Price Location:** Above or Below Trigger Level
    3. **Pick Trigger Level:** The level that has been traded at for the first time today
    4. **Choose Trigger Time:** When the trigger level was hit
    5. **Read Results:** See probability of reaching other levels throughout the day
    6. **Check Remaining:** See how much probability is left based on current time
    
    💡 **Example:** If price goes Above 0.0 at OPEN, there's a X% chance it reaches +0.618 by 1000 hours, with Y% remaining probability after current time.
    """)

# --- Chart Title and Labels ---
unique_days = 2720
day_text = f"{unique_days:,} trading days"

st.subheader(f"📈 Probability of Reaching Price Levels (%) - {ticker_config[selected_ticker]['display_name']}")
st.caption(f"Historical success rates based on {day_text} of data")

# --- Define fib_levels and styling early ---
fib_levels = [1.0, 0.786, 0.618, 0.5, 0.382, 0.236, 0.0,
              -0.236, -0.382, -0.5, -0.618, -0.786, -1.0]

fibo_styles = {
    1.0: ("lightgray", 3, 16),
    0.786: ("lightgray", 1, 12),
    0.618: ("lightgray", 2, 14),
    0.5: ("lightgray", 1, 12),
    0.382: ("lightgray", 1, 12),
    0.236: ("cyan", 2, 14),
    0.0: ("lightgray", 1, 12),
    -0.236: ("yellow", 2, 14),
    -0.382: ("lightgray", 1, 12),
    -0.5: ("lightgray", 1, 12),
    -0.618: ("lightgray", 2, 14),
    -0.786: ("lightgray", 1, 12),
    -1.0: ("lightgray", 3, 16),
}

# --- Handle URL Parameters from Parent Website ---
query_params = st.query_params if hasattr(st, 'query_params') else {}

url_view_pref = None
if 'view' in query_params:
    url_view_pref = query_params['view'].lower()
elif 'mobile' in query_params:
    mobile_param = query_params['mobile'].lower()
    url_view_pref = 'mobile' if mobile_param == 'true' else 'desktop'

# --- Mobile-First Design with Session-Based User Preference ---
if 'expanded_view_pref' not in st.session_state:
    if url_view_pref == 'desktop':
        st.session_state.expanded_view_pref = True
    elif url_view_pref == 'mobile':
        st.session_state.expanded_view_pref = False
    else:
        st.session_state.expanded_view_pref = False

# UI Controls with preference management
col1_ui, col2_ui = st.columns([3, 1])
with col1_ui:
    show_expanded_view = st.checkbox("🖥️ Show Full Matrix (All Times & Levels)", 
                                   value=st.session_state.expanded_view_pref, 
                                   key="expanded_toggle")
with col2_ui:
    make_default = st.checkbox("💾 Make Default for Session", 
                              value=False,
                              help="Remember this view choice until you close your browser",
                              key="make_default_toggle")

# Update session preference when toggle is checked
if make_default:
    st.session_state.expanded_view_pref = show_expanded_view
    st.success("✅ Session default updated!")

if show_expanded_view != st.session_state.expanded_view_pref:
    st.session_state.expanded_view_pref = show_expanded_view

# --- Display configuration ---
if analysis_type == "Session":
    if show_expanded_view:
        display_columns = ["OPEN", "0900", "1000", "1100", "1200", "1300", "1400", "1500", "TOTAL", "REMAINING"]
        display_fib_levels = fib_levels
        chart_height = 700
        chart_width = 1800
        font_size_multiplier = 1.0
        use_container_width = False
    else:    
        current_hour_index = ["OPEN", "0900", "1000", "1100", "1200", "1300", "1400", "1500"].index(trigger_time)
    
        if trigger_time == "OPEN":
        # For OPEN trigger: 0900, 1000, 1100, TOTAL, REMAINING
            display_columns = ["0900", "1000", "1100", "TOTAL", "REMAINING"]
        else:
        # For other triggers: trigger + 2 more hours + TOTAL + REMAINING
            end_index = min(current_hour_index + 3, 7)
            time_columns = ["OPEN", "0900", "1000", "1100", "1200", "1300", "1400", "1500"][current_hour_index:end_index + 1]
            display_columns = time_columns + ["TOTAL", "REMAINING"]
    
        trigger_index = fib_levels.index(trigger_level)
        start_fib = max(0, trigger_index - 3)
        end_fib = min(len(fib_levels), trigger_index + 4)
        display_fib_levels = fib_levels[start_fib:end_fib]
    
    # Mobile focused view - optimize for readability
        chart_height = 400
        chart_width = 700
        font_size_multiplier = 1.0
        use_container_width = False

# Create time_order to match display_columns exactly
    time_order = display_columns.copy()

# --- Debug trigger level data ---
   # if st.checkbox("🔍 Debug Mode - Show Data Structure"):
    #   st.write("**Selected Ticker Configuration:**")
     #  st.json(ticker_config[selected_ticker])
    
      # st.write("**ATR Data Status:**")
       #st.json({
        #    "status": atr_data.get("status"),
         #   "ticker": atr_data.get("ticker"),
          #  "reference_date": atr_data.get("reference_date"),
           # "data_age_days": atr_data.get("data_age_days")
        #})
    
       #st.write("**Filtered Data for Current Selection:**")
       #st.dataframe(filtered.head(10))
    
     #  st.write("**Available Goal Levels in Data:**")
      # available_goals = sorted(filtered['GoalLevel'].unique())
       #st.write(available_goals)
    
       #st.write("**Trigger Level Being Searched:**")
       #st.write(f"Trigger Level: {trigger_level} (type: {type(trigger_level)})")

# STEP 1: Replace lines 562-570 (the data filtering section) with this:
# Add this debug section RIGHT AFTER the analysis_type selection
# This will help us see what's happening before StateCheck logic runs

#st.write("🔍 **Debug Info:**")
#st.write(f"Selected Analysis Type: {analysis_type}")
#st.write(f"Selected Ticker: {selected_ticker}")

# Add early debug for StateCheck file
#if analysis_type == "StateCheck":
 #   st.write("**StateCheck Debug - Before Processing:**")
  #  statecheck_file = f"statecheck_detailed_{selected_ticker}_20250710_063704.csv"
   # st.write(f"Looking for file: {statecheck_file}")
    #st.write(f"File exists: {os.path.exists(statecheck_file)}")
    
    #if os.path.exists(statecheck_file):
     #   try:
            # Just peek at the file without processing
     #       test_df = pd.read_csv(statecheck_file)
      #      st.write(f"File loaded successfully: {len(test_df)} rows")
       #     st.write("**Columns in StateCheck file:**")
        #    st.write(list(test_df.columns))
         #   st.write("**First few rows:**")
          #  st.dataframe(test_df.head())
            
            # Show available trigger zones and times
           # if 'TriggerZone' in test_df.columns:
            #    st.write("**Available TriggerZone values:**")
             #   st.write(sorted(test_df['TriggerZone'].unique()))
            #if 'TriggerTime' in test_df.columns:
             #   st.write("**Available TriggerTime values:**")
              #  st.write(sorted(test_df['TriggerTime'].unique()))
                
        #except Exception as e:
         #   st.error(f"Error reading StateCheck file: {str(e)}")
    #else:
     #   st.error(f"StateCheck file not found: {statecheck_file}")
        # Show what files are available
      #  st.write("**Available files in directory:**")
       # available_files = [f for f in os.listdir('.') if f.endswith('.csv')]
        #st.write(available_files)

# Continue with the rest of your conditional logic...
# --- Conditional Data Processing Based on Analysis Type ---
if analysis_type == "Session":
    # Session data filtering (existing logic)
    filtered = df[
        (df["Direction"] == price_direction) &
        (df["TriggerLevel"] == trigger_level) &
        (df["TriggerTime"] == trigger_time)
    ].copy()
    
elif analysis_type == "StateCheck":
    # StateCheck data processing
    try:
        statecheck_file = f"statecheck_detailed_{selected_ticker}_20250710_063704.csv"
        statecheck_df = pd.read_csv(statecheck_file)
        st.success(f"✅ Loaded StateCheck data: {len(statecheck_df)} records")
        
        # Convert trigger zone string to match data format
        # "Zone 7: -0.236 to 0.0" -> "-0.236_to_0.0"
        zone_mapping = {
            "Zone 1: Above +1.0": "above_1.0",
            "Zone 2: +0.786 to +1.0": "0.786_to_1.0", 
            "Zone 3: +0.618 to +0.786": "0.618_to_0.786",
            "Zone 4: +0.5 to +0.618": "0.5_to_0.618",         # <-- Added
            "Zone 5: +0.382 to +0.5": "0.382_to_0.5",         # <-- Added
            "Zone 6: +0.236 to +0.382": "0.236_to_0.382",
            "Zone 7: 0.0 to +0.236": "0.0_to_0.236",
            "Zone 8: -0.236 to 0.0": "-0.236_to_0.0",
            "Zone 9: -0.382 to -0.236": "-0.382_to_-0.236",
            "Zone 10: -0.5 to -0.382": "-0.5_to_-0.382",      # <-- Added
            "Zone 11: -0.618 to -0.5": "-0.618_to_-0.5",      # <-- Added
            "Zone 12: -0.786 to -0.618": "-0.786_to_-0.618",
            "Zone 13: -1.0 to -0.786": "-1.0_to_-0.786",      # <-- Matches your data
            "Zone 14: Below -1.0": "below_-1.0"
        }
        
        trigger_zone_key = zone_mapping[trigger_zone]
        
        # Convert trigger time to numpy int64 format
        trigger_time_int = int(trigger_time)
        
        # Filter StateCheck data for current selection
        statecheck_filtered = statecheck_df[
            (statecheck_df["TriggerZone"] == trigger_zone_key) &
            (statecheck_df["TriggerTime"] == trigger_time_int)
        ].copy()
        
        if len(statecheck_filtered) == 0:
            st.warning(f"No StateCheck data found for {trigger_zone} at {trigger_time}")
            st.info("Try a different trigger zone or time combination.")
            st.stop()
        
        # Debug option
        #if st.checkbox("🔍 Debug StateCheck Data"):
         #   st.write("**Filtered StateCheck Data:**")
          #  st.dataframe(statecheck_filtered)
# st.dataframe(filtered.head())        
        # Adapt StateCheck data to Session format
        adapted_data = statecheck_filtered.copy()
        
        # Column mapping based on your actual data structure
        # Replace the column mapping section with this corrected version:

# Column mapping based on your actual CSV structure
    column_mapping = {
    # Your actual CSV columns -> Session format expected by chart
        "GoalZone": "GoalLevel",                    # A: GoalZone -> GoalLevel
        "GoalTime": "GoalTime",                     # B: GoalTime (keep as is)
        "TransitionCount": "NumHits",               # C: TransitionCount -> NumHits
        "TotalTriggerOccurrences": "NumTriggers",   # F: TotalTriggerOccurrences -> NumTriggers  
        "TransitionPercentage": "PctCompletion"     # G: TransitionPercentage -> PctCompletion
    }    

# Apply column renaming
    for old_col, new_col in column_mapping.items():
        if old_col in adapted_data.columns:
            adapted_data = adapted_data.rename(columns={old_col: new_col})
        else:
            st.warning(f"Column '{old_col}' not found in data")

# Convert goal zone strings to fibonacci levels for chart compatibility
    goal_zone_to_fib = {
        "above_1.0": 1.0,
        "0.786_to_1.0": 0.786,
        "0.618_to_0.786": 0.618,
        "0.5_to_0.618": 0.5,
        "0.382_to_0.5": 0.382,
        "0.236_to_0.382": 0.236,
        "0.0_to_0.236": 0.0,
        "-0.236_to_0.0": -0.236,
        "-0.382_to_-0.236": -0.382,
        "-0.5_to_-0.382": -0.5,
        "-0.618_to_-0.5": -0.618,
        "-0.786_to_-0.618": -0.786,
        "-1.0_to_-0.786": -1.0,
        "below_-1.0": -1.0
    }

# Convert GoalZone strings to fibonacci levels
    if 'GoalLevel' in adapted_data.columns:
        adapted_data['GoalLevel'] = adapted_data['GoalLevel'].map(goal_zone_to_fib)
    
    # Handle any unmapped zones
        unmapped = adapted_data['GoalLevel'].isnull().sum()
        if unmapped > 0:
            st.warning(f"Warning: {unmapped} goal zones could not be mapped to fibonacci levels")

# Convert goal times to string format expected by chart (HHMM format)
    if 'GoalTime' in adapted_data.columns:
    # Convert to string and ensure 4-digit format
        adapted_data['GoalTime'] = adapted_data['GoalTime'].astype(str).str.zfill(4)
    
    # Convert times like 940 -> "0940"
        adapted_data.loc[adapted_data['GoalTime'].str.len() == 3, 'GoalTime'] = '0' + adapted_data['GoalTime']
        
        # Set as filtered data for chart logic
        filtered = adapted_data
        
        # Create compatibility variables for chart building
        price_direction = f"Zone Transitions from {trigger_zone}"
        trigger_level = 0.0  # Not used for StateCheck display
        
        st.success(f"✅ StateCheck data adapted: {len(filtered)} records")
        
    except Exception as e:
        st.error(f"Error loading StateCheck data: {str(e)}")
        st.info("No StateCheck data available")
        st.stop()

elif analysis_type == "Rolling":
    # Rolling data processing (placeholder for now)
    st.info("⏰ Rolling analysis - Coming soon")
    st.stop()
    
elif analysis_type == "ZoneBaseline":
    # ZoneBaseline processing (placeholder for now)
    st.info("📊 ZoneBaseline analysis - Coming soon")
    st.stop()

else:
    st.error("Unknown analysis type")
    st.stop()

# STEP 2: The rest of the data processing (lines 571+) stays the same
# because now 'filtered' exists for both Session and StateCheck

# --- Create lookup dictionary from pre-calculated data ---
data_lookup = {}
for _, row in filtered.iterrows():
    goal_time = row["GoalTime"]
    if pd.notna(goal_time):
        if isinstance(goal_time, (int, float)):
            time_int = int(goal_time)
            if time_int == 900:
                goal_time_str = "0900"
            elif time_int < 1000:
                goal_time_str = f"0{time_int}"
            else:
                goal_time_str = str(time_int)
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

# Continue with existing goal_totals and goal_remaining calculations...

# --- Create lookup dictionary from pre-calculated data ---
data_lookup = {}
for _, row in filtered.iterrows():
    goal_time = row["GoalTime"]
    if pd.notna(goal_time):
        if isinstance(goal_time, (int, float)):
            time_int = int(goal_time)
            if time_int == 900:
                goal_time_str = "0900"
            elif time_int < 1000:
                goal_time_str = f"0{time_int}"
            else:
                goal_time_str = str(time_int)
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

# --- Calculate total completion rate for each goal level ---
goal_totals = {}
goal_remaining = {}
if len(filtered) > 0:
    goal_summary = filtered.groupby('GoalLevel').agg({
        'NumHits': 'sum',
        'NumTriggers': 'first'
    }).reset_index()
    
    # Standard time order for calculations
    standard_time_order = ["OPEN", "0900", "1000", "1100", "1200", "1300", "1400", "1500"]
    
    for _, row in goal_summary.iterrows():
        goal_level = row['GoalLevel']
        if goal_level == trigger_level:
            # Include trigger level for totals - we want same-level retest data
            pass
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
            key = (goal_level, time_slot)
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

# --- Get OPEN trigger data for tooltip ---
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

# --- Build chart ---
if analysis_type != "Session":
    # Simple placeholders that respect your mobile design
    if analysis_type == "ZoneBaseline":
        st.info("📊 ZoneBaseline Heatmap - Coming Soon")
    elif analysis_type == "StateCheck":
    # Load StateCheck data
        statecheck_file = f"statecheck_detailed_{selected_ticker}_20250710_063704.csv"
        
        try:
            # Load the StateCheck data
            statecheck_df = pd.read_csv(statecheck_file)
            st.success(f"✅ Loaded StateCheck data: {len(statecheck_df)} records")

            st.write("**StateCheck Columns After Adaptation:**")
            st.write(list(filtered.columns))
            st.write("**Sample StateCheck Data:**")
            st.dataframe(filtered.head())
            #st.stop()  # Temporarily stop here to see the data
            
            # Debug: Show data structure
            if st.checkbox("🔍 Debug StateCheck Data"):
                st.write("**StateCheck Data Columns:**")
                st.write(list(statecheck_df.columns))
                st.write("**Sample StateCheck Data:**")
                st.dataframe(statecheck_df.head())
               
            # Filter StateCheck data for current selection
            statecheck_filtered = statecheck_df[
                (statecheck_df["TriggerZone"] == trigger_zone) &
                (statecheck_df["TriggerTime"] == trigger_time)
            ].copy()
            
            if len(statecheck_filtered) == 0:
                st.warning(f"No StateCheck data found for {trigger_zone} at {trigger_time}")
                st.stop()
            
            # Adapt StateCheck data to look like Session data
            # Map StateCheck columns to Session column names so existing logic works
            adapted_data = statecheck_filtered.copy()
            # Add this after: filtered = adapted_data

            
            # Rename columns to match Session expectations
            column_mapping = {
                # Map StateCheck columns to Session columns
                # We'll need to see your actual column names to do this properly
                "GoalZone": "GoalLevel",  # Example - adjust based on actual columns
                "TransitionTime": "GoalTime",  # Example - adjust based on actual columns  
                "Probability": "PctCompletion",  # Example - adjust based on actual columns
                # Add more mappings as needed
            }
            
            # Apply column renaming
            for old_col, new_col in column_mapping.items():
                if old_col in adapted_data.columns:
                    adapted_data = adapted_data.rename(columns={old_col: new_col})
            
            # Set this as the filtered data for the existing chart logic to use
            filtered = adapted_data
            
            # Create dummy variables that Session logic expects
            price_direction = "Zone Transition"  # Descriptive name
            trigger_level = 0.0  # Not used for StateCheck but needed for compatibility
            
            # Don't stop here - let the existing chart logic run with adapted data
            st.info(f"📊 StateCheck: Transitions from {trigger_zone} at {trigger_time}")
            
        except Exception as e:
            st.error(f"Error loading StateCheck data: {str(e)}")
            st.info("Falling back to Session analysis...")
            # Fall through to Session logic
        
        # Fall through to Session logic
    elif analysis_type == "Rolling":
        rolling_hours = get_rolling_8_hours(trigger_time)
        st.info(f"⏰ Rolling: {' → '.join(rolling_hours)} - Coming Soon")
    
    st.stop()  # Don't build the chart

# Session analysis - keep existing chart logic
fig = go.Figure()
text_offset = 0.03

# Add "Fib Level" title above left axis (dimmed)
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

# Add "Price Level" title above right side (dimmed)
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

# --- Price labels as annotations (locked to lines) ---
if price_levels_dict:
    for level in display_fib_levels:
        level_key = f"{level:+.3f}"
        price_val = price_levels_dict.get(level_key, 0)
        
        # Use annotations with paper coordinates for proper positioning
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
                # Use consistent font size for all levels
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
                # Use consistent font size for all levels
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
        key = (level, t)
        if key in data_lookup:
            data = data_lookup[key]
            pct = data["pct"]
            hits = data["hits"]
            total = data["triggers"]
            
            # Check if times are before trigger time (handle OPEN special case)
            if trigger_time == "OPEN":
                is_before_trigger = False
            elif time_order.index(t) < time_order.index(trigger_time):
                is_before_trigger = True
            else:
                is_before_trigger = False
            
            if is_before_trigger:
                display_text = ""
                hover = "Before trigger time"
            else:
                warn = " ⚠️" if total < 30 else ""
                display_text = f"{pct:.1f}%"
                hover = f"{pct:.1f}% ({hits}/{total}){warn}"
            
            line_color, line_width, font_size = fibo_styles.get(level, ("white", 1, 12))
            # Use consistent font size for all levels
            font_size = 12 * font_size_multiplier
            
            fig.add_trace(go.Scatter(
                x=[t], y=[level + text_offset],
                mode="text", text=[display_text],
                hovertext=[hover], hoverinfo="text",
                textfont=dict(color=line_color, size=font_size),
                showlegend=False
            ))
        else:
            if t not in ["OPEN", "TOTAL", "REMAINING"]:
                line_color, line_width, font_size = fibo_styles.get(level, ("lightgray", 1, 12))
                # Use consistent font size for all levels
                font_size = 12 * font_size_multiplier
                
                # Check if times are before trigger time (handle OPEN special case)
                if trigger_time == "OPEN":
                    is_before_trigger = False
                elif time_order.index(t) < time_order.index(trigger_time):
                    is_before_trigger = True
                else:
                    is_before_trigger = False
                
                if is_before_trigger:
                    display = ""
                    hover = "Before trigger time"
                else:
                    display = "0.0%"
                    hover = "No data available"
                    
                fig.add_trace(go.Scatter(
                    x=[t], y=[level + text_offset],
                    mode="text", text=[display],
                    hovertext=[hover], hoverinfo="text",
                    textfont=dict(color=line_color, size=font_size),
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

# --- Add trigger level highlighting ---
if trigger_level in display_fib_levels:
    trigger_index = display_fib_levels.index(trigger_level)
    
    # Green shading above trigger level (to next level up)
    if trigger_index > 0:  # Not the top level
        next_level_up = display_fib_levels[trigger_index - 1]
        fig.add_shape(
            type="rect",
            x0=0, x1=1, xref="paper",
            y0=trigger_level, y1=next_level_up, yref="y",
            fillcolor="rgba(0, 255, 0, 0.1)",  # Very light green
            line=dict(width=0),
            layer="below"
        )
    
    # Yellow shading below trigger level (to next level down)
    if trigger_index < len(display_fib_levels) - 1:  # Not the bottom level
        next_level_down = display_fib_levels[trigger_index + 1]
        fig.add_shape(
            type="rect",
            x0=0, x1=1, xref="paper",
            y0=next_level_down, y1=trigger_level, yref="y",
            fillcolor="rgba(255, 255, 0, 0.1)",  # Very light yellow
            line=dict(width=0),
            layer="below"
        )

# --- Chart layout ---
fig.update_layout(
    title=f"{ticker_config[selected_ticker]['display_name']} | {price_direction} | Trigger {trigger_level} at {trigger_time}",
    xaxis=dict(
        title="Projected Completion Time (Eastern Time)",
        categoryorder="array",
        categoryarray=display_columns,
        tickmode="array",
        tickvals=display_columns,
        ticktext=display_columns,
        tickfont=dict(color="white"),
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
    margin=dict(l=40 if not show_expanded_view else 80, r=80 if not show_expanded_view else 150, t=30 if not show_expanded_view else 60, b=80 if not show_expanded_view else 60)
)

st.plotly_chart(fig, use_container_width=use_container_width)

# --- Chart Information Footer ---
col1, col2 = st.columns([3, 1])
with col1:
    if atr_data.get("status") == "success":
        data_age = atr_data.get('data_age_days', 0)
        age_warning = f" (⚠️ {data_age} days old)" if data_age > 0 else ""
        st.caption(f"📊 ATR levels from {atr_data.get('reference_date', 'unknown')} | Close: {atr_data.get('reference_close', 'N/A')} | ATR: {atr_data.get('reference_atr', 'N/A')}{age_warning}")

# Display current market time at bottom
if current_market_slot in ["PREMARKET", "AFTERHOURS"]:
    time_color = "🔴"
elif current_market_slot == "CLOSE":
    time_color = "⚫"
else:
    time_color = "🟢"

st.info(f"{time_color} **Current ET:** {current_et_time.strftime('%I:%M %p')} | **Market Slot:** {current_market_slot}")

# --- Legend/Key ---
st.caption("📋 **Chart Key:** ⚠️ = Less than 30 historical triggers (lower confidence) | **Remaining Colors:** 🟢 >15% | 🟠 5-15% | 🔴 <5% | Percentages show probability of reaching target level by specified time")

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
