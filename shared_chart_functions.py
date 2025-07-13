"""
Shared chart building functions for ATR analysis
"""
import plotly.graph_objects as go
import pandas as pd


def create_zonebaseline_heatmap(detailed_data, level_totals_data, ticker_name, font_size_multiplier=1.0):
    """
    Create static zone probability heatmap for ZoneBaseline analysis
    
    Args:
        detailed_data: DataFrame with detailed zone baseline data
        level_totals_data: DataFrame with aggregated totals
        ticker_name: Name of ticker for chart title
        font_size_multiplier: Font scaling factor
    
    Returns:
        Plotly figure object
    """
    # Zone definitions (12 zones between fib levels)
    zones = [
        "Zone 1 (>+1.0)",
        "Zone 2 (+0.786 to +1.0)",
        "Zone 3 (+0.618 to +0.786)", 
        "Zone 4 (+0.382 to +0.618)",
        "Zone 5 (+0.236 to +0.382)",
        "Zone 6 (0.0 to +0.236)",
        "Zone 7 (-0.236 to 0.0)",
        "Zone 8 (-0.382 to -0.236)",
        "Zone 9 (-0.5 to -0.382)",
        "Zone 10 (-0.618 to -0.5)",
        "Zone 11 (-0.786 to -0.618)",
        "Zone 12 (<-1.0)"
    ]
    
    # Time periods (granular 10-minute intervals)
    time_periods = ["0930", "0940", "0950", "1000", "1010", "1020", "1030", "1040", "1050", 
                   "1100", "1110", "1120", "1130", "1140", "1150", "1200", "1210", "1220", 
                   "1230", "1240", "1250", "1300", "1310", "1320", "1330", "1340", "1350",
                   "1400", "1410", "1420", "1430", "1440", "1450", "1500", "1510", "1520",
                   "1530", "1540", "1550", "1600"]
    
    # Create data matrix for heatmap
    heatmap_data = []
    
    # Process filtered data into heatmap format
    for zone in zones:
        zone_row = []
        for time_period in time_periods:
            # Look up probability for this zone/time combination
            prob = get_zone_probability(filtered_data, zone, time_period)
            zone_row.append(prob)
        heatmap_data.append(zone_row)
    
    # Create heatmap
    fig = go.Figure(data=go.Heatmap(
        z=heatmap_data,
        x=time_periods,
        y=zones,
        colorscale='RdYlBu_r',  # Red-Yellow-Blue reversed (red=high, blue=low)
        showscale=True,
        colorbar=dict(title="Probability %"),
        hovertemplate='<b>%{y}</b><br>' +
                      'Time: %{x}<br>' +
                      'Probability: %{z:.1f}%' +
                      '<extra></extra>'
    ))
    
    # Update layout
    fig.update_layout(
        title=f"{ticker_name} - Zone Occupancy Probability Heatmap",
        xaxis=dict(
            title="Time (Eastern)",
            tickmode="array",
            tickvals=list(range(0, len(time_periods), 6)),  # Show every 6th time label
            ticktext=[time_periods[i] for i in range(0, len(time_periods), 6)],
            tickfont=dict(size=10 * font_size_multiplier)
        ),
        yaxis=dict(
            title="Zones",
            tickfont=dict(size=10 * font_size_multiplier)
        ),
        plot_bgcolor="black",
        paper_bgcolor="black",
        font=dict(color="white", size=12 * font_size_multiplier),
        height=600,
        width=1200
    )
    
    return fig


def get_zone_probability(detailed_data, zone, time_period):
    """
    Extract probability for specific zone/time combination from detailed CSV
    
    Args:
        detailed_data: DataFrame with detailed zone baseline data
        zone: Zone identifier (e.g., "Zone6")
        time_period: Time period identifier (e.g., "0930", "1000")
    
    Returns:
        Probability percentage (float)
    """
    try:
        # Look up in detailed CSV
        zone_data = detailed_data[
            (detailed_data['AnalysisType'] == 'ZoneBaseline') & 
            (detailed_data['Zone'] == zone) &
            (detailed_data['TimePeriod'] == time_period)
        ]
        
        if not zone_data.empty:
            return zone_data['Probability'].iloc[0]
        else:
            return 0.0
            
    except Exception:
        # Fallback to zero if data not found
        return 0.0


# --- Matrix cells ---

# REPLACE the existing create_statecheck_matrix function in shared_chart_functions.py with this:

# REPLACE the StateCheck section in daily_display.py (around lines 490-560) with this:

elif analysis_type == "StateCheck":
    # StateCheck data processing
    try:
        statecheck_file = f"statecheck_detailed_{selected_ticker}_20250710_063704.csv"
        statecheck_df = pd.read_csv(statecheck_file)
        st.success(f"✅ Loaded StateCheck data: {len(statecheck_df)} records")
        
        # Zone mapping
        zone_mapping = {
            "Zone 1: Above +1.0": "above_1.0",
            "Zone 2: +0.786 to +1.0": "0.786_to_1.0", 
            "Zone 3: +0.618 to +0.786": "0.618_to_0.786",
            "Zone 4: +0.5 to +0.618": "0.5_to_0.618",
            "Zone 5: +0.382 to +0.5": "0.382_to_0.5",
            "Zone 6: +0.236 to +0.382": "0.236_to_0.382",
            "Zone 7: 0.0 to +0.236": "0.0_to_0.236",
            "Zone 8: -0.236 to 0.0": "-0.236_to_0.0",
            "Zone 9: -0.382 to -0.236": "-0.382_to_-0.236",
            "Zone 10: -0.5 to -0.382": "-0.5_to_-0.382",
            "Zone 11: -0.618 to -0.5": "-0.618_to_-0.5",
            "Zone 12: -0.786 to -0.618": "-0.786_to_-0.618",
            "Zone 13: -1.0 to -0.786": "-1.0_to_-0.786",
            "Zone 14: Below -1.0": "below_-1.0"
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
        
        # Column mapping
        column_mapping = {
            "GoalZone": "GoalLevel",
            "GoalTime": "GoalTime",
            "TransitionCount": "NumHits",
            "TotalTriggerOccurrences": "NumTriggers",
            "TransitionPercentage": "PctCompletion"
        }
        
        # Apply column renaming
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
        
        # Convert goal times to string format
        if 'GoalTime' in adapted_data.columns:
            adapted_data['GoalTime'] = adapted_data['GoalTime'].astype(str).str.zfill(4)
        
        # Set as filtered data
        filtered = adapted_data
        
        # For StateCheck, show all fibonacci levels that exist in the data
        available_levels = sorted(filtered['GoalLevel'].unique())
        display_fib_levels = available_levels
        
        # Use standard time columns for StateCheck (no OPEN, TOTAL, REMAINING)
        available_times = sorted(filtered['GoalTime'].unique())
        display_columns = available_times
        
        # Create time_order to match display_columns
        time_order = display_columns.copy()
        
        # StateCheck chart dimensions - wide for horizontal scroll
        chart_height = 600
        chart_width = 3200  # Very wide for horizontal scroll
        font_size_multiplier = 1.2
        use_container_width = False  # Critical for horizontal scroll
        
        st.success(f"✅ StateCheck data adapted: {len(filtered)} records")
        st.info(f"📊 **StateCheck Chart**: {len(display_columns)} time periods × {len(display_fib_levels)} levels | Scroll horizontally to see all data")
        
        # Create compatibility variables for chart building
        price_direction = f"Zone Transitions from {trigger_zone}"
        
        # BUILD THE CHART using shared function
        from shared_chart_functions import create_statecheck_matrix
        
        fig = create_statecheck_matrix(
            filtered_data=filtered,
            display_fib_levels=display_fib_levels,
            display_columns=display_columns,
            time_order=time_order,
            trigger_zone=trigger_zone,
            price_direction=price_direction,
            ticker_name=ticker_config[selected_ticker]['display_name'],
            text_offset=0.03,
            font_size_multiplier=font_size_multiplier
        )
        
        # Display the chart
        st.plotly_chart(fig, use_container_width=use_container_width)
        
        # Add color legend for StateCheck
        st.markdown("""
        **🎨 StateCheck Color Legend:**
        - 🟢 **Bright Green** (≥50%): Very High Probability
        - 🌟 **Light Green** (30-49%): High Probability  
        - 🟡 **Yellow** (20-29%): Medium-High Probability
        - 🟠 **Orange** (10-19%): Medium Probability
        - 🔶 **Light Red** (5-9%): Low Probability
        - ⚫ **Gray** (<5%): Very Low Probability
        """)
        
        # Skip all the remaining chart building logic
        st.stop()
        
    except Exception as e:
        st.error(f"Error loading StateCheck data: {str(e)}")
        st.stop()

# Continue with other analysis types...


def get_total_probability(level_totals_data, analysis_type, direction=None, trigger_level=None, trigger_time=None, trigger_zone=None):
    """
    Get total probability from Level Totals CSV
    
    Args:
        level_totals_data: DataFrame with aggregated totals
        analysis_type: Type of analysis (Session, Rolling, StateCheck, ZoneBaseline)
        direction: Above/Below (for Session/Rolling)
        trigger_level: Trigger level (for Session/Rolling)
        trigger_time: Trigger time
        trigger_zone: Trigger zone (for StateCheck)
    
    Returns:
        Dictionary with total hits, triggers, and percentage
    """
    try:
        # Build filter conditions
        conditions = (level_totals_data['AnalysisType'] == analysis_type)
        
        if direction is not None:
            conditions &= (level_totals_data['Direction'] == direction)
        if trigger_level is not None:
            conditions &= (level_totals_data['TriggerLevel'] == trigger_level)
        if trigger_time is not None:
            conditions &= (level_totals_data['TriggerTime'] == trigger_time)
        if trigger_zone is not None:
            conditions &= (level_totals_data['TriggerZone'] == trigger_zone)
        
        # Get matching row
        total_data = level_totals_data[conditions]
        
        if not total_data.empty:
            row = total_data.iloc[0]
            return {
                'hits': row['TotalHits'],
                'triggers': row['TotalTriggers'],
                'pct': row['TotalPct']
            }
        else:
            return {'hits': 0, 'triggers': 0, 'pct': 0.0}
            
    except Exception:
        return {'hits': 0, 'triggers': 0, 'pct': 0.0}


def get_rolling_8_hours(trigger_time):
    """
    Generate 8-hour rolling window from trigger time, crossing days if needed
    
    Args:
        trigger_time: Starting time for rolling window
    
    Returns:
        List of 8 consecutive trading hours
    """
    trading_hours = ["OPEN", "0900", "1000", "1100", "1200", "1300", "1400", "1500"]
    
    if trigger_time == "OPEN":
        return trading_hours[:8]
    
    if trigger_time not in trading_hours:
        return trading_hours[:8]  # Fallback
    
    trigger_index = trading_hours.index(trigger_time)
    rolling_hours = []
    
    for i in range(8):
        hour_index = (trigger_index + i) % len(trading_hours)
        hour = trading_hours[hour_index]
        
        # Mark next-day hours
        if trigger_index + i >= len(trading_hours):
            rolling_hours.append(f"{hour}+1")
        else:
            rolling_hours.append(hour)
    
    return rolling_hours
