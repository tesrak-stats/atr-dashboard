import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import json
import os
from daily_atr_updater import calculate_atr_levels

# --- Ticker Configuration (expandable for future tickers) ---
ticker_config = {
    "SPX": {
        "summary_file": "atr_dashboard_summary_ENHANCED.csv",
        "display_name": "S&P 500 (SPX)",
        "ticker_symbol": "^GSPC"
    }
    # Future tickers can be added here:
    # "QQQ": {
    #     "summary_file": "atr_dashboard_summary_QQQ.csv", 
    #     "display_name": "Nasdaq 100 (QQQ)",
    #     "ticker_symbol": "QQQ"
    # }
}

def get_atr_levels_for_ticker(ticker_symbol="^GSPC"):
    """
    Get ATR levels using the daily_atr_updater function
    Returns the levels data or empty dict if error
    """
    try:
        # Try to load from saved JSON file first (if exists and recent)
        json_file = "atr_levels.json"
        if os.path.exists(json_file):
            with open(json_file, 'r') as f:
                saved_data = json.load(f)
                if saved_data.get("status") == "success":
                    return saved_data
        
        # If no saved file or error, calculate fresh
        levels_data = calculate_atr_levels(ticker=ticker_symbol)
        
        # Save the calculated data for future use
        if levels_data.get("status") == "success":
            with open(json_file, 'w') as f:
                json.dump(levels_data, f, indent=2)
        
        return levels_data
        
    except Exception as e:
        st.error(f"Error getting ATR levels: {str(e)}")
        return {"status": "error", "error": str(e)}

# --- Page Layout with Ticker Selector ---
col_title1, col_title2 = st.columns([4, 1])
with col_title1:
    st.title("📈 ATR Levels Roadmap")
with col_title2:
    selected_ticker = st.selectbox("Ticker", list(ticker_config.keys()), index=0)

# --- Load data based on selected ticker ---
try:
    df = pd.read_csv(ticker_config[selected_ticker]["summary_file"])
    st.success(f"✅ Loaded data for {ticker_config[selected_ticker]['display_name']}")
except FileNotFoundError:
    st.error(f"❌ Data file not found for {selected_ticker}")
    st.stop()
except Exception as e:
    st.error(f"❌ Error loading data for {selected_ticker}: {str(e)}")
    st.stop()

# --- Load current ATR-based price levels ---
ticker_symbol = ticker_config[selected_ticker]["ticker_symbol"]
atr_data = get_atr_levels_for_ticker(ticker_symbol)

if atr_data.get("status") == "success":
    atr_price_levels = atr_data
    st.info(f"📊 ATR levels from {atr_data.get('reference_date', 'unknown date')} | Close: {atr_data.get('reference_close', 'N/A')} | ATR: {atr_data.get('reference_atr', 'N/A')}")
    
    # Show data freshness warning if needed
    data_age = atr_data.get('data_age_days', 0)
    if data_age > 0:
        st.warning(f"⚠️ ATR data is {data_age} day(s) old")
else:
    atr_price_levels = {}
    st.error(f"❌ Could not load ATR levels: {atr_data.get('error', 'Unknown error')}")

# --- Display configuration ---
# Only include the exact columns we want to display
display_columns = ["OPEN", "0900", "1000", "1100", "1200", "1300", "1400", "1500", "TOTAL"]

# Create time_order with proper spacing
time_order = ["OPEN", "0830"]  # OPEN + spacer
for hour in ["0900", "1000", "1100", "1200", "1300", "1400", "1500"]:
    time_order.append(hour)
    time_order.append(f"{str(int(hour[:2])+1).zfill(2)}30")  # Add spacer after each hour
time_order.append("SPACER")  # Final spacer before TOTAL
time_order.append("TOTAL")

# Debug: Print what we're working with
print("Display columns:", display_columns)
print("Time order:", time_order)

fib_levels = [1.0, 0.786, 0.618, 0.5, 0.382, 0.236, 0.0,
              -0.236, -0.382, -0.5, -0.618, -0.786, -1.0]

# --- Controls ---
col1, col2, col3 = st.columns(3)

# Changed Direction to Price
price_direction = col1.selectbox("Price", sorted(df["Direction"].unique()), 
                                index=sorted(df["Direction"].unique()).index("Above"))

trigger_level = col2.selectbox("Trigger Level", sorted(set(df["TriggerLevel"]).union(fib_levels)), 
                              index=sorted(set(df["TriggerLevel"]).union(fib_levels)).index(0.0))

trigger_time = col3.selectbox("Trigger Time", ["OPEN", "0900", "1000", "1100", "1200", "1300", "1400", "1500"], index=0)

# --- Filter and simple lookup ---
filtered = df[
    (df["Direction"] == price_direction) &
    (df["TriggerLevel"] == trigger_level) &
    (df["TriggerTime"] == trigger_time)
].copy()

# --- Create lookup dictionary from pre-calculated data ---
data_lookup = {}
for _, row in filtered.iterrows():
    # Convert GoalTime to string and handle numeric times (CRITICAL for lookup)
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
if len(filtered) > 0:
    # Group by goal level and sum hits across all times
    goal_summary = filtered.groupby('GoalLevel').agg({
        'NumHits': 'sum',
        'NumTriggers': 'first'  # Should be same for all goals of same trigger
    }).reset_index()
    
    for _, row in goal_summary.iterrows():
        goal_level = row['GoalLevel']
        if goal_level == trigger_level:
            continue
        total_hits = row['NumHits']
        total_triggers = row['NumTriggers']
        total_pct = (total_hits / total_triggers * 100) if total_triggers > 0 else 0
        goal_totals[goal_level] = {
            "hits": total_hits,
            "triggers": total_triggers,
            "pct": total_pct
        }

# --- Get OPEN trigger data for tooltip (goal-specific) ---
open_trigger_data = {}
if trigger_time == "OPEN" and len(filtered) > 0:
    # Get trigger count (same for all goals)
    open_triggers = filtered['NumTriggers'].iloc[0]
    
    # Create goal-specific OPEN completion data
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

# --- Matrix cells ---
for level in fib_levels:
    for t in time_order:
        # Only process columns we want to display
        if t not in display_columns:
            continue  # Skip all spacer columns and any other unwanted columns
            
        # Handle OPEN column specially - blank text but show goal-specific tooltip
        if t == "OPEN":
            if trigger_time == "OPEN" and level in open_trigger_data:
                # Show goal-specific OPEN completion data
                triggers = open_trigger_data[level]["triggers"]
                completions = open_trigger_data[level]["completions"]
                hover = f"OPEN Triggers: {triggers}, Goal {level} Completed at OPEN: {completions}"
                
                fig.add_trace(go.Scatter(
                    x=[t], y=[level + 0.015],
                    mode="text", text=[""],  # Blank text
                    hovertext=[hover], hoverinfo="text",
                    textfont=dict(color="white", size=13),
                    showlegend=False
                ))
            else:
                # Empty OPEN column for non-OPEN triggers or missing data
                fig.add_trace(go.Scatter(
                    x=[t], y=[level + 0.015],
                    mode="text", text=[""],
                    hoverinfo="skip",
                    textfont=dict(color="white", size=13),
                    showlegend=False
                ))
            continue
        
        # Handle TOTAL column - show total completion rate for each goal
        if t == "TOTAL":
            if level in goal_totals and level != trigger_level:
                total_data = goal_totals[level]
                pct = total_data["pct"]
                hits = total_data["hits"]
                triggers = total_data["triggers"]
                
                warn = " ⚠️" if triggers < 30 else ""
                display_text = f"{pct:.1f}%"
                hover = f"Total: {pct:.1f}% ({hits}/{triggers}){warn}"
                
                fig.add_trace(go.Scatter(
                    x=[t], y=[level + 0.015],
                    mode="text", text=[display_text],
                    hovertext=[hover], hoverinfo="text",
                    textfont=dict(color="white", size=13),
                    showlegend=False
                ))
            else:
                # Same level as trigger or no data
                fig.add_trace(go.Scatter(
                    x=[t], y=[level + 0.015],
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
            
            # Blank out same level (trigger level = goal level)
            if level == trigger_level:
                display_text = ""
                hover = "Same level as trigger"
            # Blank out times before selected trigger time
            elif time_order.index(t) < time_order.index(trigger_time):
                display_text = ""
                hover = "Before trigger time"
            # Show percentage for valid combinations
            else:
                warn = " ⚠️" if total < 30 else ""
                display_text = f"{pct:.1f}%"
                hover = f"{pct:.1f}% ({hits}/{total}){warn}"
            
            fig.add_trace(go.Scatter(
                x=[t], y=[level + 0.015],
                mode="text", text=[display_text],
                hovertext=[hover], hoverinfo="text",
                textfont=dict(color="white", size=12),
                showlegend=False
            ))
        else:
            # No data for this combination - but only show for actual time columns
            if t not in ["OPEN", "TOTAL"]:  # Don't show 0.0% for special columns
                if level == trigger_level:
                    # Blank out same level
                    display = ""
                    hover = "Same level as trigger"
                elif time_order.index(t) < time_order.index(trigger_time):
                    # Blank out times before trigger time
                    display = ""
                    hover = "Before trigger time"
                else:
                    # Only show 0.0% for valid time columns
                    display = "0.0%"
                    hover = "No data available"
                    
                fig.add_trace(go.Scatter(
                    x=[t], y=[level + 0.015],
                    mode="text", text=[display],
                    hovertext=[hover], hoverinfo="text",
                    textfont=dict(color="white", size=13),
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
fibo_styles = {
    1.0: ("white", 2), 0.786: ("white", 1), 0.618: ("white", 2),
    0.5: ("white", 1), 0.382: ("white", 1), 0.236: ("cyan", 2),
    0.0: ("white", 1),
    -0.236: ("yellow", 2), -0.382: ("white", 1), -0.5: ("white", 1),
    -0.618: ("white", 2), -0.786: ("white", 1), -1.0: ("white", 2),
}

for level, (color, width) in fibo_styles.items():
    fig.add_shape(
        type="line", x0=0, x1=1, xref="paper", y0=level, y1=level, yref="y",
        line=dict(color=color, width=width), layer="below"
)

# --- Chart layout ---
fig.update_layout(
    title=f"{price_direction} | Trigger {trigger_level} at {trigger_time}",
    xaxis=dict(
        title="Hour Goal Was Reached",
        categoryorder="array",
        categoryarray=time_order,
        tickmode="array",
        tickvals=display_columns,
        ticktext=display_columns,  # Explicitly set tick text
        tickfont=dict(color="white")
    ),
    yaxis=dict(
        title="Goal Level",
        categoryorder="array",
        categoryarray=fib_levels,
        tickmode="array",
        tickvals=fib_levels,
        tickfont=dict(color="white")
    ),
    plot_bgcolor="black",
    paper_bgcolor="black",
    font=dict(color="white"),
    height=700,  # Reduced from 900
    width=1400,
    margin=dict(l=80, r=60, t=60, b=60)
)

# --- Price ladder on right Y-axis ---
if atr_price_levels and atr_price_levels.get("status") == "success":
    levels_dict = atr_price_levels.get("levels", {})
    
    # Create price labels for each fib level
    price_labels = []
    for level in fib_levels:
        # Convert level to the key format used in JSON (e.g., +1.000, -0.236)
        level_key = f"{level:+.3f}"
        price_value = levels_dict.get(level_key, "")
        price_labels.append(price_value)

    fig.update_layout(
        yaxis=dict(
            title="Fib Level",
            tickvals=fib_levels,
            ticktext=[f"{lvl:+.3f}" for lvl in fib_levels],
        ),
        yaxis2=dict(
            title="Price Level",
            overlaying="y",
            side="right",
            tickvals=fib_levels,
            ticktext=[
                f"{price:.2f}" if isinstance(price, (int, float)) else ""
                for price in price_labels
            ],
            tickfont=dict(color="lightgray"),
            showgrid=False
        )
    )
else:
    # No ATR data available - show message
    st.warning("⚠️ ATR price levels not available")

st.plotly_chart(fig, use_container_width=False)
