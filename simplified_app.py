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

# --- Updated Ticker Configuration (now matches the ATR calculator) ---
ticker_config = {
    "SPX": {
        "summary_file": "atr_dashboard_summary_ENHANCED.csv",
        "display_name": "S&P 500 (SPX)",
        "ticker_symbol": "^GSPC"
    },
    "QQQ": {
        "summary_file": "atr_dashboard_summary_QQQ.csv",  # You'll need to create this
        "display_name": "Nasdaq 100 (QQQ)",
        "ticker_symbol": "QQQ"
    },
    "IWM": {
        "summary_file": "atr_dashboard_summary_IWM.csv",  # You'll need to create this
        "display_name": "Russell 2000 (IWM)",
        "ticker_symbol": "IWM"
    },
    "NVDA": {
        "summary_file": "atr_dashboard_summary_NVDA.csv",
        "display_name": "S&P 500 (NVDA)",
        "ticker_symbol": "NVDA"
    },
    # Add more tickers as needed - make sure they match TICKER_CONFIG in daily_atr_updater.py
}

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
with col_title2:
    # Only show tickers that have summary files available
    available_tickers = []
    for ticker_key, config in ticker_config.items():
        if os.path.exists(config["summary_file"]):
            available_tickers.append(ticker_key)
        else:
            st.warning(f"⚠️ {config['summary_file']} not found for {ticker_key}")
    
    if not available_tickers:
        st.error("❌ No ticker data files found!")
        st.stop()
    
    selected_ticker = st.selectbox("Ticker", available_tickers, index=0)

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

# --- Controls (define before display configuration) ---
col1, col2, col3 = st.columns(3)

price_direction = col1.selectbox("Price Location", sorted(df["Direction"].unique()), 
                                index=sorted(df["Direction"].unique()).index("Above"))

trigger_level = col2.selectbox("Trigger Level", sorted(set(df["TriggerLevel"]).union(fib_levels)), 
                              index=sorted(set(df["TriggerLevel"]).union(fib_levels)).index(0.0))

trigger_time = col3.selectbox("Trigger Time", ["OPEN", "0900", "1000", "1100", "1200", "1300", "1400", "1500"], index=0)

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
if st.checkbox("🔍 Debug Mode - Show Data Structure"):
    st.write("**Selected Ticker Configuration:**")
    st.json(ticker_config[selected_ticker])
    
    st.write("**ATR Data Status:**")
    st.json({
        "status": atr_data.get("status"),
        "ticker": atr_data.get("ticker"),
        "reference_date": atr_data.get("reference_date"),
        "data_age_days": atr_data.get("data_age_days")
    })
    
    st.write("**Filtered Data for Current Selection:**")
    st.dataframe(filtered.head(10))
    
    st.write("**Available Goal Levels in Data:**")
    available_goals = sorted(filtered['GoalLevel'].unique())
    st.write(available_goals)
    
    st.write("**Trigger Level Being Searched:**")
    st.write(f"Trigger Level: {trigger_level} (type: {type(trigger_level)})")

filtered = df[
    (df["Direction"] == price_direction) &
    (df["TriggerLevel"] == trigger_level) &
    (df["TriggerTime"] == trigger_time)
].copy()

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