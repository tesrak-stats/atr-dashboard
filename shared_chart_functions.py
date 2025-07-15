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


"""
ADD this function to shared_chart_functions.py (replace the existing create_statecheck_matrix)
Make sure plotly.graph_objects as go and pandas as pd are imported at the top
"""

def create_statecheck_matrix(filtered_data, display_fib_levels, display_columns, time_order, 
                             trigger_zone, price_direction, ticker_name, 
                             text_offset=0.01, font_size_multiplier=1.2, price_levels_dict=None):
    """
    Create StateCheck transition probability matrix with full features
    """
    
    # Create data lookup from filtered data
    data_lookup = {}
    for _, row in filtered_data.iterrows():
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
    
    # Create figure
    fig = go.Figure()
    
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

    # Add horizontal lines for Fibonacci levels
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
    
    for level in display_fib_levels:
        if level in fibo_styles:
            color, width, _ = fibo_styles[level]
            fig.add_shape(
                type="line", x0=0, x1=1, xref="paper", y0=level, y1=level, yref="y",
                line=dict(color=color, width=width), layer="below"
            )
    
    # Add matrix cells with color coding
    for level in display_fib_levels:
        for t in time_order:
            if t not in display_columns:
                continue
                
            key = (float(level), t)
            if key in data_lookup:
                data = data_lookup[key]
                pct = data["pct"]
                hits = data["hits"]
                triggers = data["triggers"]
                
                # Color coding for StateCheck
                if pct >= 50:
                    text_color = "lime"
                elif pct >= 30:
                    text_color = "lightgreen"
                elif pct >= 20:
                    text_color = "yellow"
                elif pct >= 10:
                    text_color = "orange"
                elif pct >= 5:
                    text_color = "lightcoral"
                else:
                    text_color = "gray"
                
                # Text formatting
                if pct >= 10:
                    display_text = f"{pct:.0f}%"
                else:
                    display_text = f"{pct:.1f}%"
                
                # Hover text
                warn = " ⚠️" if triggers < 30 else ""
                hover = f"{pct:.1f}% ({hits}/{triggers}){warn}"
                
                fig.add_trace(go.Scatter(
                    x=[t], y=[level + text_offset],
                    mode="text", text=[display_text],
                    textfont=dict(color=text_color, size=12 * font_size_multiplier),
                    showlegend=False,
                    hovertext=[hover],
                    hoverinfo="text"
                ))
    
    # Zone highlighting
    trigger_zone_ranges = {
        "Zone 1: Above +1.0": (1.0, 1.2),
        "Zone 2: +0.786 to +1.0": (0.786, 1.0),
        "Zone 3: +0.618 to +0.786": (0.618, 0.786),
        "Zone 4: +0.5 to +0.618": (0.5, 0.618),
        "Zone 5: +0.382 to +0.5": (0.382, 0.5),
        "Zone 6: +0.236 to +0.382": (0.236, 0.382),
        "Zone 7: 0.0 to +0.236": (0.0, 0.236),
        "Zone 8: -0.236 to 0.0": (-0.236, 0.0),
        "Zone 9: -0.382 to -0.236": (-0.382, -0.236),
        "Zone 10: -0.5 to -0.382": (-0.5, -0.382),
        "Zone 11: -0.618 to -0.5": (-0.618, -0.5),
        "Zone 12: -0.786 to -0.618": (-0.786, -0.618),
        "Zone 13: -1.0 to -0.786": (-1.0, -0.786),
        "Zone 14: Below -1.0": (-1.2, -1.0)
    }
    
    if trigger_zone in trigger_zone_ranges:
        zone_bottom, zone_top = trigger_zone_ranges[trigger_zone]
        fig.add_shape(
            type="rect",
            x0=0, x1=1, xref="paper",
            y0=zone_bottom, y1=zone_top, yref="y",
            fillcolor="rgba(0, 150, 255, 0.2)",
            line=dict(color="rgba(0, 150, 255, 0.6)", width=2),
            layer="below"
        )
    
    # Dynamic chart layout based on data size
    # Calculate optimal width based on number of columns
    base_width_per_column = 120  # pixels per time column
    min_width = 800  # minimum width
    calculated_width = max(min_width, len(display_columns) * base_width_per_column)
    
    # Use container width for smaller charts, fixed width for large ones
    use_container_width_setting = calculated_width <= 1200
    chart_width = None if calculated_width <= 1200 else calculated_width
    
    fig.update_layout(
        title=f"{ticker_name} | {price_direction}",
        xaxis=dict(
            title="Transition Time (Eastern Time)",
            type="linear",
            categoryorder="array",
            categoryarray=display_columns,
            tickmode="array",
            tickvals=list(range(len(display_columns))),
            ticktext=display_columns,
            tickfont=dict(color="white", size=10 * font_size_multiplier),
            tickangle=45 if len(display_columns) > 10 else 0,
            fixedrange=not use_container_width_setting,  # Allow scrolling only for wide charts
            automargin=True
        ),
        yaxis=dict(
            title="Fibonacci Levels",
            categoryorder="array",
            categoryarray=display_fib_levels,
            tickmode="array",
            tickvals=display_fib_levels,
            ticktext=[f"{lvl:+.3f}" for lvl in display_fib_levels],
            tickfont=dict(color="white", size=12 * font_size_multiplier),
            side="left",
            fixedrange=True,
            automargin=True
        ),
        plot_bgcolor="black",
        paper_bgcolor="black",
        font=dict(color="white", size=12 * font_size_multiplier),
        height=600,
        width=3200,  # Dynamic width or None for container
        margin=dict(l=20, r=150, t=60, b=100),  # Restored right margin for price levels
        showlegend=False,
        autosize=False
    )
    
    # Return both figure and container setting
    return fig, use_container_width_setting
    
    return fig

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
