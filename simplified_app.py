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
    st.caption("🔧 App Version: v2.3.9 - Fixed Code Structure") # VERSION BUMP
with col_title2:
    selected_ticker = st.selectbox("Ticker", list(ticker_config.keys()), index=0)

# --- Load data based on selected ticker ---
try:
    df = pd.read_csv(ticker_config[selected_ticker]["summary_file"])
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
    price_levels_dict = atr_data.get("levels", {})
else:
    atr_price_levels = {}
    price_levels_dict = {}
    st.error(f"❌ Could not load ATR levels: {atr_data.get('error', 'Unknown error')}")

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
    
    🎯 **How to Use:**
    1. **Select Price Location:** Above or Below Trigger Level
    2. **Pick Trigger Level:** The level that has been traded at for the first time today
    3. **Choose Trigger Time:** When the trigger level was hit
    4. **Read Results:** See probability of reaching other levels throughout the day
    
    💡 **Example:** If price goes Above 0.0 at OPEN, there's a X% chance it reaches +0.618 by 1000 hours.
    """)

# --- Chart Title and Labels ---
unique_days = 2720
day_text = f"{unique_days:,} trading days"

st.subheader(f"📈 Probability of Reaching Price Levels (%) - Based on {day_text}")
st.caption("Historical success rates based on S&P 500 data")

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
    if st.button("💾 Make Default for Session", help="Remember this choice until you close your browser"):
        st.session_state.expanded_view_pref = show_expanded_view
        st.success("✅ Session default updated!")

if show_expanded_view != st.session_state.expanded_view_pref:
    st.session_state.expanded_view_pref = show_expanded_view

# --- Display configuration ---
if show_expanded_view:
    display_columns = ["OPEN", "0900", "1000", "1100", "1200", "1300", "1400", "1500", "TOTAL"]
    display_fib_levels = fib_levels
    chart_height = 700
    chart_width = 1600
    font_size_multiplier = 1.0
    use_container_width = False
else:
    current_hour_index = ["OPEN", "0900", "1000", "1100", "1200", "1300", "1400", "1500"].index(trigger_time)
    end_index = min(current_hour_index + 4, 7)
    time_columns = ["OPEN", "0900", "1000", "1100", "1200", "1300", "1400", "1500"][current_hour_index:end_index + 1]
    time_columns.append("TOTAL")
    display_columns = time_columns
    
    trigger_index = fib_levels.index(trigger_level)
    start_fib = max(0, trigger_index - 2)
    end_fib = min(len(fib_levels), trigger_index + 3)
    display_fib_levels = fib_levels[start_fib:end_fib]
    
    # Mobile focused view - optimize for readability
    chart_height = 400
    chart_width = None  # Let it auto-size
    font_size_multiplier = 1.0  # Keep text readable
    use_container_width = True

# Create time_order - only include displayed columns for focused view
if show_expanded_view:
    # Full time_order with spacers for expanded view
    time_order = ["OPEN", "0830"]
    for hour in ["0900", "1000", "1100", "1200", "1300", "1400", "1500"]:
        time_order.append(hour)
        time_order.append(f"{str(int(hour[:2])+1).zfill(2)}30")
    time_order.append("SPACER")
    time_order.append("TOTAL")
else:
    # Simplified time_order for focused view - no spacers
    time_order = display_columns

# --- Filter and simple lookup ---
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
if len(filtered) > 0:
    goal_summary = filtered.groupby('GoalLevel').agg({
        'NumHits': 'sum',
        'NumTriggers': 'first'
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
            if level in goal_totals and level != trigger_level:
                total_data = goal_totals[level]
                pct = total_data["pct"]
                hits = total_data["hits"]
                triggers = total_data["triggers"]
                
                line_color, line_width, font_size = fibo_styles.get(level, ("lightgray", 1, 12))
                
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
        
        # Regular time columns
        key = (level, t)
        if key in data_lookup:
            data = data_lookup[key]
            pct = data["pct"]
            hits = data["hits"]
            total = data["triggers"]
            
            if level == trigger_level:
                display_text = ""
                hover = "Same level as trigger"
            elif time_order.index(t) < time_order.index(trigger_time):
                display_text = ""
                hover = "Before trigger time"
            else:
                warn = " ⚠️" if total < 30 else ""
                display_text = f"{pct:.1f}%"
                hover = f"{pct:.1f}% ({hits}/{total}){warn}"
            
            line_color, line_width, font_size = fibo_styles.get(level, ("white", 1, 12))
            
            fig.add_trace(go.Scatter(
                x=[t], y=[level + text_offset],
                mode="text", text=[display_text],
                hovertext=[hover], hoverinfo="text",
                textfont=dict(color=line_color, size=font_size),
                showlegend=False
            ))
        else:
            if t not in ["OPEN", "TOTAL"]:
                line_color, line_width, font_size = fibo_styles.get(level, ("lightgray", 1, 12))
                
                if level == trigger_level:
                    display = ""
                    hover = "Same level as trigger"
                elif time_order.index(t) < time_order.index(trigger_time):
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

# --- Chart layout ---
fig.update_layout(
    title=f"{price_direction} | Trigger {trigger_level} at {trigger_time}",
    xaxis=dict(
        title="Hour Goal Was Reached",
        categoryorder="array",
        categoryarray=time_order,
        tickmode="array",
        tickvals=display_columns,
        ticktext=display_columns,
        tickfont=dict(color="white")
    ),
    yaxis=dict(
        title="Fib Level",
        categoryorder="array",
        categoryarray=display_fib_levels,
        tickmode="array",
        tickvals=display_fib_levels,  # Align Y-axis labels with actual fib levels, not offset
        ticktext=[f"{lvl:+.3f}" for lvl in display_fib_levels],
        tickfont=dict(color="white", size=12 * font_size_multiplier),
        side="left"
    ),
    plot_bgcolor="black",
    paper_bgcolor="black",
    font=dict(color="white", size=12 * font_size_multiplier),
    height=chart_height,
    width=chart_width,
    margin=dict(l=60 if not show_expanded_view else 80, r=100 if not show_expanded_view else 150, t=40 if not show_expanded_view else 60, b=40 if not show_expanded_view else 60)
)

# --- Price ladder on right Y-axis ---
if price_levels_dict:
    price_values = []
    for level in display_fib_levels:
        level_key = f"{level:+.3f}"
        price_val = price_levels_dict.get(level_key, 0)
        price_values.append(price_val)
    
    fig.add_trace(go.Scatter(
        x=["OPEN"], y=[0.0],
        mode="markers",
        marker=dict(opacity=0, size=1),
        yaxis="y2",
        showlegend=False,
        hoverinfo="skip"
    ))
    
    fig.update_layout(
        yaxis2=dict(
            title="Price Levels",
            overlaying="y",
            side="right",
            tickmode="array",
            tickvals=display_fib_levels,  # Align with horizontal lines, not offset text
            ticktext=[f"{p:.2f}" for p in price_values],
            tickfont=dict(color="white", size=10 * font_size_multiplier),
            showgrid=False,
            range=[min(display_fib_levels)-0.1, max(display_fib_levels)+0.1],
            fixedrange=True,
            anchor="free",
            position=1.0
        )
    )

# --- Price labels as text annotations (locked to lines) ---
if price_levels_dict:
    for level in display_fib_levels:
        level_key = f"{level:+.3f}"
        price_val = price_levels_dict.get(level_key, 0)
        
        # Add price text positioned far to the right
        fig.add_trace(go.Scatter(
            x=[len(display_columns) + 0.5],  # Position beyond last column
            y=[level + text_offset],  # Same offset as percentage text
            mode="text",
            text=[f"{price_val:.2f}"],
            textfont=dict(color="lightgray", size=10 * font_size_multiplier),
            textposition="middle left",
            showlegend=False,
            hoverinfo="skip",
            xaxis="x",
            yaxis="y"
        ))

# Add "Price Levels" title on the right
fig.add_annotation(
    text="Price Levels",
    x=len(display_columns) + 0.5,
    y=sum(display_fib_levels) / len(display_fib_levels),  # Center vertically
    xref="x",
    yref="y", 
    showarrow=False,
    textangle=90,
    font=dict(color="white", size=12 * font_size_multiplier),
    xanchor="center",
    yanchor="middle"
)

st.plotly_chart(fig, use_container_width=use_container_width)

# --- Chart Information Footer ---
col1, col2 = st.columns([3, 1])
with col1:
    if atr_data.get("status") == "success":
        data_age = atr_data.get('data_age_days', 0)
        age_warning = f" (⚠️ {data_age} days old)" if data_age > 0 else ""
        st.caption(f"📊 ATR levels from {atr_data.get('reference_date', 'unknown')} | Close: {atr_data.get('reference_close', 'N/A')} | ATR: {atr_data.get('reference_atr', 'N/A')}{age_warning}")

# --- Legend/Key ---
st.caption("📋 **Chart Key:** ⚠️ = Less than 30 historical triggers (lower confidence) | Percentages show probability of reaching target level by specified time")
