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

# --- Ticker Configuration ---
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

# Build ticker options with availability check
ticker_options = []
ticker_mapping = {}

for group_name, tickers in ticker_groups.items():
    ticker_options.append(f"--- {group_name} ---")
    group_has_available = False
    
    for ticker_display in tickers:
        ticker_symbol = ticker_display.split(" - ")[0]
        
        # Currently only SPX is available
        if ticker_symbol == "SPX":
            summary_file = "atr_summary_SPX_SESSION_20250710_064535.csv"
            if os.path.exists(summary_file):
                ticker_options.append(ticker_display)
                ticker_mapping[ticker_display] = ticker_symbol
                group_has_available = True
    
    if not group_has_available:
        ticker_options.append("--- (Coming Soon) ---")

# Ticker selector
if len([opt for opt in ticker_options if not opt.startswith("---")]) == 0:
    st.error("❌ No ticker data files found!")
    st.info("📝 Have a ticker request? Check back soon!")
    st.stop()

selected_ticker_display = st.selectbox("Select Ticker", ticker_options, index=1)

# Validate ticker selection
if not selected_ticker_display.startswith("---"):
    if selected_ticker_display in ticker_mapping:
        selected_ticker = ticker_mapping[selected_ticker_display]
        
        instrument_category = None
        for group_name, tickers in ticker_groups.items():
            if selected_ticker_display in tickers:
                instrument_category = group_name.split(" ", 1)[1]
                break
        
        st.success(f"✅ Selected: {selected_ticker_display} ({instrument_category})")
    else:
        st.error("❌ This ticker is not yet available")
        st.stop()
else:
    st.error("Please select a valid ticker (not a header)")
    st.stop()

# Create ticker config
ticker_config = {
    selected_ticker: {
        "summary_file": f"atr_summary_{selected_ticker}_SESSION_20250710_064535.csv",
        "display_name": selected_ticker_display.split(" - ")[1],
        "ticker_symbol": selected_ticker
    }
}

# --- App Header ---
st.title("📈 Daily ATR Analysis")
st.caption("🔧 App Version: v3.0.0 - Multi-Instrument Daily Viewer")

# --- Analysis Type Selection ---
st.subheader("📊 Analysis Type")
analysis_type = st.selectbox(
    "Select Analysis Type",
    ["Session", "Rolling", "StateCheck", "ZoneBaseline"],
    index=0,
    help="Session: Full session analysis | Rolling: 8-hour window | StateCheck: Zone transitions | ZoneBaseline: Static zone probability"
)

# --- Analysis Parameters ---
st.subheader("⚙️ Analysis Parameters")

if analysis_type == "ZoneBaseline":
    st.info("📍 **ZoneBaseline**: Static probability heatmap showing zone occupancy throughout the session")
    st.caption("No additional parameters needed - shows probability of being in each zone at each time")
    
elif analysis_type == "StateCheck":
    st.info("🔄 **StateCheck**: Zone transition probabilities from trigger zone over time")
    
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
        fib_levels = [1.0, 0.786, 0.618, 0.5, 0.382, 0.236, 0.0, -0.236, -0.382, -0.5, -0.618, -0.786, -1.0]
        trigger_level = st.selectbox("Trigger Level", fib_levels, index=6)
    with col3:
        trigger_time = st.selectbox("Trigger Time", ["OPEN", "0900", "1000", "1100", "1200", "1300", "1400", "1500"], index=0)
    
    rolling_hours = get_rolling_8_hours(trigger_time)
    st.caption(f"🔄 Rolling window: {' → '.join(rolling_hours[:4])} → {' → '.join(rolling_hours[4:])}")
    
else:  # Session
    st.info("📈 **Session**: Full session analysis with trigger conditions")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        price_direction = st.selectbox("Price Location", ["Above", "Below"], index=0)
    with col2:
        fib_levels = [1.0, 0.786, 0.618, 0.5, 0.382, 0.236, 0.0, -0.236, -0.382, -0.5, -0.618, -0.786, -1.0]
        trigger_level = st.selectbox("Trigger Level", fib_levels, index=6)
    with col3:
        trigger_time = st.selectbox("Trigger Time", ["OPEN", "0900", "1000", "1100", "1200", "1300", "1400", "1500"], index=0)

# Show analysis summary
st.divider()
if analysis_type == "ZoneBaseline":
    st.write(f"**Analysis**: {analysis_type}")
elif analysis_type == "StateCheck":
    st.write(f"**Analysis**: {analysis_type} | **From**: {trigger_zone} at {trigger_time}")
else:
    st.write(f"**Analysis**: {analysis_type} | **Trigger**: {price_direction} {trigger_level} at {trigger_time}")

# --- Utility Functions ---
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

# --- Load Data ---
current_et_time, current_market_slot = get_current_market_time()

try:
    df = pd.read_csv(ticker_config[selected_ticker]["summary_file"])
    st.success(f"✅ Loaded data for {ticker_config[selected_ticker]['display_name']}")
except FileNotFoundError:
    st.error(f"❌ Data file not found for {selected_ticker}: {ticker_config[selected_ticker]['summary_file']}")
    st.stop()
except Exception as e:
    st.error(f"❌ Error loading data for {selected_ticker}: {str(e)}")
    st.stop()

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
    st.success("✅ Session default updated!")

# --- Analysis-Specific Processing ---
if analysis_type == "StateCheck":
    try:
        statecheck_file = f"statecheck_detailed_{selected_ticker}_20250710_063704.csv"
        statecheck_df = pd.read_csv(statecheck_file)
        st.success(f"✅ Loaded StateCheck data: {len(statecheck_df)} records")
        
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
            "above_1.0": 1.0, "0.786_to_1.0": 0.786, "0.618_to_0.786": 0.618,
            "0.5_to_0.618": 0.5, "0.382_to_0.5": 0.382, "0.236_to_0.382": 0.236,
            "0.0_to_0.236": 0.0, "-0.236_to_0.0": -0.236, "-0.382_to_-0.236": -0.382,
            "-0.5_to_-0.382": -0.5, "-0.618_to_-0.5": -0.618, "-0.786_to_-0.618": -0.786,
            "-1.0_to_-0.786": -1.0, "below_-1.0": -1.0
        }
        
        if 'GoalLevel' in adapted_data.columns:
            adapted_data['GoalLevel'] = adapted_data['GoalLevel'].map(goal_zone_to_fib)
        
        if 'GoalTime' in adapted_data.columns:
            adapted_data['GoalTime'] = adapted_data['GoalTime'].astype(str).str.zfill(4)
        
        # Set data for chart building
        filtered = adapted_data
        available_levels = sorted(filtered['GoalLevel'].unique())
        display_fib_levels = available_levels
        available_times = sorted(filtered['GoalTime'].unique())
        display_columns = available_times
        time_order = display_columns.copy()
        
        st.success(f"✅ StateCheck data adapted: {len(filtered)} records")
        
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
        if chart_use_container_width:
            st.info(f"📊 **StateCheck Chart**: {len(display_columns)} time periods × {len(display_fib_levels)} levels")
        else:
            st.info(f"📊 **StateCheck Chart**: {len(display_columns)} time periods × {len(display_fib_levels)} levels | Scroll horizontally to see all data")
        
        # Display the chart
        st.plotly_chart(fig, use_container_width=chart_use_container_width)
        
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
    st.info("⏰ Rolling analysis - Coming soon")
    st.stop()
    
elif analysis_type == "ZoneBaseline":
    st.info("📊 ZoneBaseline analysis - Coming soon")
    st.stop()

else:  # Session
    # Session data filtering
    filtered = df[
        (df["Direction"] == price_direction) &
        (df["TriggerLevel"] == trigger_level) &
        (df["TriggerTime"] == trigger_time)
    ].copy()
    
    # Display configuration for Session
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
            display_columns = ["0900", "1000", "1100", "TOTAL", "REMAINING"]
        else:
            end_index = min(current_hour_index + 3, 7)
            time_columns = ["OPEN", "0900", "1000", "1100", "1200", "1300", "1400", "1500"][current_hour_index:end_index + 1]
            display_columns = time_columns + ["TOTAL", "REMAINING"]
        
        trigger_index = fib_levels.index(trigger_level)
        start_fib = max(0, trigger_index - 3)
        end_fib = min(len(fib_levels), trigger_index + 4)
        display_fib_levels = fib_levels[start_fib:end_fib]
        
        chart_height = 400
        chart_width = 700
        font_size_multiplier = 1.0
        use_container_width = False
    
    time_order = display_columns.copy()
    
    # TODO: Build Session chart (to be extracted to shared functions)
    st.info("📈 Session chart - To be implemented with shared functions")

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
