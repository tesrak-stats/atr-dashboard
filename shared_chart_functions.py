"""
Shared chart building functions for ATR analysis
"""
import plotly.graph_objects as go
import pandas as pd


def create_zonebaseline_heatmap(filtered_data, display_fib_levels, display_columns, time_order, 
                               ticker_name, text_offset=0.03, font_size_multiplier=1.0, price_levels_dict=None):
    """
    Create static zone probability heatmap for ZoneBaseline analysis
    """
    # Create data lookup
    data_lookup = {}
    for _, row in filtered_data.iterrows():
        goal_time = str(row["GoalTime"])
        key = (float(row["GoalLevel"]), goal_time)
        data_lookup[key] = {
            "pct": row["PctCompletion"],
        st.write("Sample data_lookup entries:")
           for i, (key, value) in enumerate(list(data_lookup.items())[:10]):
        st.write(f"  {key}: {value}")  
            
        }
    print("Sample data_lookup entries:")
        for i, (key, value) in enumerate(list(data_lookup.items())[:10]):
    print(f"  {key}: {value}")
    print(f"Total entries: {len(data_lookup)}")
    print(f"Display columns: {display_columns}")
    st.write(f"Sorted levels: {sorted_levels}")
    
    # Filter out artificial levels for proper display
    fib_levels = [1.0, 0.786, 0.618, 0.5, 0.382, 0.236, 0.0, -0.236, -0.382, -0.5, -0.618, -0.786, -1.0]
    display_levels = [lvl for lvl in display_fib_levels if lvl in fib_levels]
    sorted_levels = sorted(display_levels, reverse=False)  # Changed: Low to high for proper display
    
    # Build matrix data for heatmap
    heatmap_data = []
    hover_data = []
    
    for level in sorted_levels:
        row_data = []
        row_hover = []
        for time_col in display_columns:
            key = (float(level), time_col)
            if key in data_lookup:
                data = data_lookup[key]
                row_data.append(data["pct"])
                row_hover.append(f"Zone {level:+.3f}<br>Time: {time_col}<br>Occupancy: {data['pct']:.1f}%")
            else:
                row_data.append(0)
                row_hover.append(f"Zone {level:+.3f}<br>Time: {time_col}<br>No data")
        heatmap_data.append(row_data)
        hover_data.append(row_hover)
    
    # Create custom colorscale that matches our legend - adjusted for zone occupancy
    custom_colorscale = [
        [0.0, '#2D2D2D'],      # Gray for 0-2%
        [0.02, '#2D2D2D'],     # Gray
        [0.02, '#FF6B6B'],     # Light Red for 2-5%
        [0.05, '#FF6B6B'],     # Light Red
        [0.05, '#FFA500'],     # Orange for 5-10%
        [0.10, '#FFA500'],     # Orange
        [0.10, '#FFD700'],     # Yellow for 10-20%
        [0.20, '#FFD700'],     # Yellow
        [0.20, '#90EE90'],     # Light Green for 20-50%
        [0.50, '#90EE90'],     # Light Green
        [0.50, '#00FF00'],     # Bright Green for 50%+
        [1.0, '#00FF00']       # Bright Green
    ]
    
    # Create heatmap with proper y-axis mapping (more transparent)
    fig = go.Figure(data=go.Heatmap(
        z=heatmap_data,
        x=display_columns,
        y=list(range(len(sorted_levels))),  # Use indices 0, 1, 2, 3...
        colorscale=custom_colorscale,
        showscale=True,
        colorbar=dict(title="Occupancy %"),
        hovertemplate='%{customdata}<extra></extra>',
        customdata=hover_data,
        zmin=0,
        zmax=100,
        opacity=0.7  # Make heatmap more transparent
    ))
    
    # Add text overlays to debug what data is actually being displayed
    for level_idx, level in enumerate(sorted_levels):
        for time_idx, time_col in enumerate(display_columns):
            key = (float(level), time_col)
            if key in data_lookup:
                data = data_lookup[key]
                pct = data["pct"]
                
                # Add text overlay
                fig.add_trace(go.Scatter(
                    x=[time_idx], y=[level_idx],
                    mode="text", 
                    text=[f"{pct:.1f}%"],
                    textfont=dict(color="white", size=12, family="Arial Black"),
                    showlegend=False,
                    hoverinfo="skip"
                ))
    
    # Add price level annotations on the right side
    if price_levels_dict:
        for i, level in enumerate(sorted_levels):
            level_key = f"{level:+.3f}"
            price_val = price_levels_dict.get(level_key, 0)
            
            fig.add_annotation(
                text=f"${price_val:.2f}",
                x=1.02,
                y=i,
                xref="paper",
                yref="y",
                showarrow=False,
                font=dict(color="white", size=12),
                xanchor="left",
                yanchor="middle"
            )
    
    fig.update_layout(
        title=f"{ticker_name} | Zone Occupancy Heatmap",
        xaxis=dict(
            title="Time (Eastern)",
            tickfont=dict(color="white", size=10),
            tickangle=45 if len(display_columns) > 15 else 0
        ),
        yaxis=dict(
            title="Fibonacci Levels",
            tickfont=dict(color="white", size=10),
            tickmode='array',
            tickvals=list(range(len(sorted_levels))),  # Position at 0, 1, 2, 3...
            ticktext=[f"{lvl:+.3f}" if lvl in fib_levels else "" for lvl in sorted_levels]  # Show labels for standard fib levels only
        ),
        plot_bgcolor="black",
        paper_bgcolor="black",
        font=dict(color="white", size=12),
        height=600,
        width=max(1000, len(display_columns) * 25),
        margin=dict(l=80, r=120, t=60, b=100)
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
