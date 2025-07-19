"""
Shared chart building functions for ATR analysis
"""
import plotly.graph_objects as go
import pandas as pd
import streamlit as st


def create_zonebaseline_heatmap(filtered_data, display_fib_levels, display_columns, time_order, 
                               ticker_name, text_offset=0.03, font_size_multiplier=1.0, price_levels_dict=None):
    """
    Create static zone probability heatmap for ZoneBaseline analysis
    Fixed to properly align zones between Fibonacci levels
    """
    import streamlit as st
    import plotly.graph_objects as go
    
    # Create data lookup
    data_lookup = {}
    zone_ranges = {}
    
    for _, row in filtered_data.iterrows():
        goal_time = str(row["GoalTime"])
        fib_level = float(row["GoalLevel"])
        key = (fib_level, goal_time)
        data_lookup[key] = {
            "pct": row["PctCompletion"]
        }
        
        # Store zone information for hover
        if "Zone" in row:
            zone_ranges[fib_level] = row["Zone"]
    
    # Standard Fibonacci levels in descending order (top to bottom on chart)
    standard_fib_levels = [1.0, 0.786, 0.618, 0.5, 0.382, 0.236, 0.0, -0.236, -0.382, -0.5, -0.618, -0.786, -1.0]
    
    # Filter to only include levels we have data for
    available_levels = [lvl for lvl in standard_fib_levels if lvl in display_fib_levels]
    available_levels.sort(reverse=True)  # Top to bottom for proper display
    
    # Create zone boundaries - zones exist BETWEEN Fibonacci levels
    # Each zone occupies the space between two consecutive Fibonacci levels
    zone_boundaries = []
    zone_labels = []
    zone_midpoints = []
    zone_data_levels = []  # Track which fib level represents each zone's data
    
    # Add zone above highest level
    if available_levels:
        top_level = available_levels[0]
        zone_top = top_level + 0.2  # Extend above
        zone_bottom = top_level
        zone_mid = (zone_top + zone_bottom) / 2
        
        zone_boundaries.append((zone_bottom, zone_top))
        zone_midpoints.append(zone_mid)
        zone_labels.append(f"Above +{top_level:.3f}")
        zone_data_levels.append(1.15 if 1.15 in display_fib_levels else top_level)
    
    # Add zones between consecutive levels
    for i in range(len(available_levels) - 1):
        upper_level = available_levels[i]
        lower_level = available_levels[i + 1]
        
        zone_top = upper_level
        zone_bottom = lower_level
        zone_mid = (zone_top + zone_bottom) / 2
        
        zone_boundaries.append((zone_bottom, zone_top))
        zone_midpoints.append(zone_mid)
        zone_labels.append(f"{lower_level:+.3f} to {upper_level:+.3f}")
        
        # Use the upper level as the data key for this zone
        zone_data_levels.append(upper_level)
    
    # Add zone below lowest level
    if available_levels:
        bottom_level = available_levels[-1]
        zone_top = bottom_level
        zone_bottom = bottom_level - 0.2  # Extend below
        zone_mid = (zone_top + zone_bottom) / 2
        
        zone_boundaries.append((zone_bottom, zone_top))
        zone_midpoints.append(zone_mid)
        zone_labels.append(f"Below {bottom_level:+.3f}")
        zone_data_levels.append(-1.15 if -1.15 in display_fib_levels else bottom_level)
    
    # Build matrix data for heatmap
    heatmap_data = []
    hover_data = []
    
    for i, data_level in enumerate(zone_data_levels):
        row_data = []
        row_hover = []
        zone_label = zone_labels[i]
        
        for time_col in display_columns:
            key = (float(data_level), time_col)
            if key in data_lookup:
                data = data_lookup[key]
                pct_val = data["pct"]
                row_data.append(pct_val)
                row_hover.append(f"Zone: {zone_label}<br>Time: {time_col}<br>Occupancy: {pct_val:.1f}%")
            else:
                row_data.append(0)
                row_hover.append(f"Zone: {zone_label}<br>Time: {time_col}<br>No data")
        
        heatmap_data.append(row_data)
        hover_data.append(row_hover)
    
    # Create custom colorscale optimized for occupancy percentages
    custom_colorscale = [
        [0.0, '#1A1A1A'],      # Dark Gray for 0-1%
        [0.01, '#2D2D2D'],     # Gray for 1-2%
        [0.02, '#FF4444'],     # Light Red for 2-4%
        [0.04, '#FF6B6B'],     # Red for 4-7%
        [0.07, '#FFA500'],     # Orange for 7-12%
        [0.12, '#FFD700'],     # Yellow for 12-18%
        [0.18, '#ADFF2F'],     # Yellow-Green for 18-25%
        [0.25, '#90EE90'],     # Light Green for 25-35%
        [0.35, '#32CD32'],     # Green for 35-45%
        [0.45, '#00FF00'],     # Bright Green for 45-55%
        [0.55, '#00FF7F'],     # Spring Green for 55%+
        [1.0, '#00FF7F']
    ]
    
    # Create heatmap
    fig = go.Figure(data=go.Heatmap(
        z=heatmap_data,
        x=list(range(len(display_columns))),
        y=zone_midpoints,  # Position heatmap cells at zone midpoints
        colorscale=custom_colorscale,
        showscale=True,
        colorbar=dict(title="Occupancy %"),
        hovertemplate='%{customdata}<extra></extra>',
        customdata=hover_data,
        zmin=0,
        zmax=100,
        opacity=0.8
    ))
    
    # Add horizontal lines at Fibonacci levels (zone boundaries)
    for fib_level in available_levels:
        fig.add_hline(
            y=fib_level,
            line=dict(color="white", width=1),
            layer="above"  # Draw lines above the heatmap
        )
    
    # Add Fibonacci level labels on the left
    for fib_level in available_levels:
        fig.add_annotation(
            text=f"{fib_level:+.3f}",
            x=-0.02,
            y=fib_level,
            xref="paper",
            yref="y",
            showarrow=False,
            font=dict(color="white", size=12),
            xanchor="right",
            yanchor="middle"
        )
    
    # Add price level labels on the right if available
    if price_levels_dict:
        for fib_level in available_levels:
            level_key = f"{fib_level:+.3f}"
            if level_key in price_levels_dict:
                price_val = price_levels_dict[level_key]
                fig.add_annotation(
                    text=f"${price_val:.2f}",
                    x=1.02,
                    y=fib_level,
                    xref="paper",
                    yref="y",
                    showarrow=False,
                    font=dict(color="white", size=12),
                    xanchor="left",
                    yanchor="middle"
                )
    
    # Calculate chart dimensions
    chart_width = max(1000, len(display_columns) * 40)
    
    fig.update_layout(
        title=f"{ticker_name} | Zone Occupancy Heatmap",
        xaxis=dict(
            title="Time (Eastern)",
            tickfont=dict(color="white", size=10),
            tickangle=45 if len(display_columns) > 10 else 0,
            tickmode='array',
            tickvals=list(range(len(display_columns))),
            ticktext=display_columns,
            side="bottom"
        ),
        yaxis=dict(
            title="Fibonacci Levels (Zone Boundaries)",
            tickfont=dict(color="white", size=10),
            showticklabels=False,  # Hide default y-axis labels (we add custom ones)
            range=[min(zone_midpoints) - 0.1, max(zone_midpoints) + 0.1]
        ),
        plot_bgcolor="black",
        paper_bgcolor="black",
        font=dict(color="white", size=12),
        height=600,
        width=chart_width,
        margin=dict(l=80, r=100, t=60, b=100)
    )
    
    return fig, True
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
    fib_levels = [1.0, 0.786, 0.618, 0.5, 0.382, 0.236, 0.0, -0.236, -0.382, -0.5, -0.618, -0.786, -1.0]
                                 
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

     # Add price level annotations on the right side
    if price_levels_dict:
        for level in display_fib_levels:
            level_key = f"{level:+.3f}"
            price_val = price_levels_dict.get(level_key, 0)

            # Skip artificial levels (not in standard fib_levels)
            if level not in fib_levels:
                continue
            
            fig.add_annotation(
                text=f"{price_val:.2f}",
                x=1.05,
                y=level + text_offset,
                xref="paper",
                yref="y", 
                showarrow=False,
                font=dict(color="white", size=14),
                xanchor="left",
                yanchor="middle"
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

        

    # No horizontal lines needed - zone transitions show boundaries naturally
            
   
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
                display_text = f"{pct:.0f}%"
                
                # Hover text
                warn = " ⚠️" if triggers < 30 else ""
                hover = f"{pct:.1f}% ({hits}/{triggers}){warn}"
                
                fig.add_trace(go.Scatter(
                    x=[display_columns.index(t)], y=[level + text_offset],
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

    # Add this after the matrix cells are created
    # Add vertical lines between hourly groups (not through them)
    for i, time_col in enumerate(display_columns):
        # Add vertical line BEFORE each hour (times ending in 00)
        if time_col.endswith('00') and i > 0:  # Skip the first column
            fig.add_vline(
                x=i - 0.5,  # Position between columns (i-1 and i)
                line=dict(color="gray", width=0.5, dash="dot"),
                layer="below"
            )
    
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
            tickvals=[lvl for lvl in display_fib_levels if lvl in fib_levels],  # Filter axis labels
            ticktext=[f"{lvl:+.3f}" for lvl in display_fib_levels if lvl in fib_levels],  # Filter axis labels
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
