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


def create_statecheck_matrix(detailed_data, level_totals_data, display_columns, trigger_zone, trigger_time, ticker_name, font_size_multiplier=1.0):
    """
    Create zone transition matrix for StateCheck analysis
    Similar to session matrix but shows zone transitions
    
    Args:
        detailed_data: DataFrame with detailed StateCheck data
        level_totals_data: DataFrame with aggregated totals
        display_columns: List of time columns to display
        trigger_zone: Starting zone for transitions
        trigger_time: When trigger zone was entered
        ticker_name: Name of ticker for chart title
        font_size_multiplier: Font scaling factor
    
    Returns:
        Plotly figure object
    """
    # Zone definitions (same as baseline)
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
    
    # Create figure
    fig = go.Figure()
    
    # Add zone transition data points (similar to current session matrix)
    for zone in zones:
        for time_col in display_columns:
            if time_col in ["TOTAL", "REMAINING"]:
                continue
                
            # Get transition probability
            prob = get_zone_transition_probability(detailed_data, trigger_zone, zone, time_col)
            
            # Add data point
            fig.add_trace(go.Scatter(
                x=[time_col],
                y=[zone],
                mode="text",
                text=[f"{prob:.1f}%" if prob > 0 else ""],
                textfont=dict(color="white", size=12 * font_size_multiplier),
                hovertext=[f"From {trigger_zone} at {trigger_time} → {zone} at {time_col}: {prob:.1f}%"],
                hoverinfo="text",
                showlegend=False
            ))
    
    # Update layout
    fig.update_layout(
        title=f"{ticker_name} - Zone Transitions from {trigger_zone} at {trigger_time}",
        xaxis=dict(
            title="Time (Eastern)",
            categoryorder="array",
            categoryarray=display_columns,
            tickfont=dict(color="white", size=10 * font_size_multiplier)
        ),
        yaxis=dict(
            title="Target Zones",
            categoryorder="array", 
            categoryarray=zones,
            tickfont=dict(color="white", size=10 * font_size_multiplier)
        ),
        plot_bgcolor="black",
        paper_bgcolor="black",
        font=dict(color="white", size=12 * font_size_multiplier),
        height=600,
        width=1000
    )
    
    return fig


def get_zone_transition_probability(detailed_data, trigger_zone, target_zone, time_period):
    """
    Get probability of transitioning from trigger zone to target zone by time period
    
    Args:
        detailed_data: DataFrame with detailed transition data
        trigger_zone: Starting zone (e.g., "Zone6")
        target_zone: Ending zone (e.g., "Zone4")
        time_period: Time period (e.g., "1000")
    
    Returns:
        Probability percentage (float)
    """
    try:
        # Look up in detailed CSV
        transition_data = detailed_data[
            (detailed_data['AnalysisType'] == 'StateCheck') &
            (detailed_data['TriggerZone'] == trigger_zone) &
            (detailed_data['TargetZone'] == target_zone) &
            (detailed_data['TimePeriod'] == time_period)
        ]
        
        if not transition_data.empty:
            return transition_data['Probability'].iloc[0]
        else:
            return 0.0
            
    except Exception:
        # Fallback to zero if data not found
        return 0.0


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