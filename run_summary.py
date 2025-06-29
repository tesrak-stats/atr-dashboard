import streamlit as st
import pandas as pd
import numpy as np
import io
from datetime import datetime
import openpyxl
from openpyxl.styles import Font, PatternFill, Alignment
from openpyxl.utils.dataframe import dataframe_to_rows

def bucket_time(time_value):
    """Convert numeric times to hour buckets for dashboard display"""
    if pd.isna(time_value):
        return "Unknown"
    
    # Handle string times (like "OPEN")
    if isinstance(time_value, str):
        if time_value.upper() == "OPEN":
            return "OPEN"
        try:
            time_value = float(time_value)
        except:
            return str(time_value)
    
    # Convert numeric times to hour buckets
    if time_value < 930:
        return "OPEN"
    elif time_value < 1000:
        return "0900"
    elif time_value < 1100:
        return "1000"
    elif time_value < 1200:
        return "1100"
    elif time_value < 1300:
        return "1200"
    elif time_value < 1400:
        return "1300"
    elif time_value < 1500:
        return "1400"
    elif time_value < 1600:
        return "1500"
    else:
        return "1600"

def generate_excel_report(summary_df, metadata):
    """Generate Excel report grouped by trigger-goal pairs showing raw hits and percentages across time periods"""
    
    # Create workbook and worksheet
    wb = openpyxl.Workbook()
    ws = wb.active
    ws.title = "ATR Analysis Report"
    
    # Define styles
    header_font = Font(bold=True, size=12)
    subheader_font = Font(bold=True, size=10)
    title_font = Font(bold=True, size=14)
    percentage_font = Font(italic=True, color="666666")
    
    header_fill = PatternFill(start_color="E3F2FD", end_color="E3F2FD", fill_type="solid")
    trigger_fill = PatternFill(start_color="F5F5F5", end_color="F5F5F5", fill_type="solid")
    
    center_align = Alignment(horizontal="center", vertical="center")
    
    # Write title and metadata
    current_row = 1
    ws.cell(row=current_row, column=1, value="ATR Analysis Report").font = title_font
    current_row += 1
    ws.cell(row=current_row, column=1, value=f"Date Range: {metadata['date_range']}")
    current_row += 1
    ws.cell(row=current_row, column=1, value=f"Total Records: {metadata['total_records']:,}")
    current_row += 1
    ws.cell(row=current_row, column=1, value=f"Trading Days: {metadata['unique_dates']:,}")
    current_row += 1
    ws.cell(row=current_row, column=1, value=f"Overall Success Rate: {metadata['overall_rate']:.1f}%")
    current_row += 3  # Add spacing
    
    # Define time periods (columns)
    time_periods = ['OPEN', '0900', '1000', '1100', '1200', '1300', '1400', '1500']
    
    # Get unique trigger-goal combinations
    trigger_goal_combinations = summary_df.groupby(['Direction', 'TriggerLevel', 'GoalLevel']).first().reset_index()
    
    # Group by direction for organization
    directions = ['Above', 'Below']
    
    for direction in directions:
        direction_data = trigger_goal_combinations[trigger_goal_combinations['Direction'] == direction]
        if len(direction_data) == 0:
            continue
        
        # Direction header
        ws.cell(row=current_row, column=1, value=f"{direction} Trigger Analysis").font = title_font
        current_row += 2
        
        # Group by trigger level
        trigger_levels = sorted(direction_data['TriggerLevel'].unique())
        
        for trigger_level in trigger_levels:
            trigger_data = direction_data[direction_data['TriggerLevel'] == trigger_level]
            
            # Trigger level header
            ws.cell(row=current_row, column=1, value=f"Trigger Level: {trigger_level:+.3f}").font = subheader_font
            for col in range(1, 16):  # Extend range to cover all columns
                ws.cell(row=current_row, column=col).fill = trigger_fill
            current_row += 1
            
            # Column headers
            headers = ['Goal Level', 'Trigger Time', 'Total Triggers', 'Open Comps', 'Actionable'] + time_periods + ['Total Hits', 'Success Rate']
            for col, header in enumerate(headers, 1):
                cell = ws.cell(row=current_row, column=col, value=header)
                cell.font = header_font
                cell.fill = header_fill
                cell.alignment = center_align
            current_row += 1
            
            # Get goals for this trigger level
            goals = sorted(trigger_data['GoalLevel'].unique())
            
            for goal_level in goals:
                # Get all data for this trigger-goal combination
                combo_data = summary_df[
                    (summary_df['Direction'] == direction) & 
                    (summary_df['TriggerLevel'] == trigger_level) & 
                    (summary_df['GoalLevel'] == goal_level)
                ]
                
                if len(combo_data) == 0:
                    continue
                
                # Group by trigger time for this goal
                trigger_times = sorted(combo_data['TriggerTime'].unique())
                
                for i, trigger_time in enumerate(trigger_times):
                    time_data = combo_data[combo_data['TriggerTime'] == trigger_time]
                    
                    if len(time_data) == 0:
                        continue
                    
                    # Get summary stats for this trigger-goal-time combination
                    # The summary DataFrame uses 'ActionableTriggers' as the column name
                    total_triggers = time_data['TotalTriggers'].iloc[0] if len(time_data) > 0 else 0
                    open_completions = time_data['OpenCompletions'].iloc[0] if len(time_data) > 0 else 0
                    actionable_triggers = time_data['ActionableTriggers'].iloc[0] if len(time_data) > 0 else 0
                    
                    # Goal level (only show for first trigger time)
                    goal_cell = ws.cell(row=current_row, column=1, value=f"{goal_level:+.3f}" if i == 0 else "")
                    goal_cell.font = subheader_font
                    
                    # Trigger time
                    ws.cell(row=current_row, column=2, value=trigger_time)
                    
                    # Summary stats
                    ws.cell(row=current_row, column=3, value=total_triggers)
                    ws.cell(row=current_row, column=4, value=open_completions)
                    ws.cell(row=current_row, column=5, value=actionable_triggers)
                    
                    # Time period data
                    total_hits = 0
                    for col_idx, time_period in enumerate(time_periods, 6):
                        period_data = time_data[time_data['GoalTime'] == time_period]
                        if len(period_data) > 0:
                            hits = period_data['NumHits'].iloc[0]
                            pct = period_data['PctCompletion'].iloc[0]
                            total_hits += hits
                            
                            # Show hits and percentage
                            if hits > 0:
                                ws.cell(row=current_row, column=col_idx, value=f"{hits} ({pct:.1f}%)")
                            else:
                                ws.cell(row=current_row, column=col_idx, value="0 (0.0%)")
                        else:
                            ws.cell(row=current_row, column=col_idx, value="0 (0.0%)")
                    
                    # Total hits and success rate
                    ws.cell(row=current_row, column=14, value=total_hits)
                    success_rate = (total_hits / actionable_triggers * 100) if actionable_triggers > 0 else 0
                    ws.cell(row=current_row, column=15, value=f"{success_rate:.1f}%")
                    
                    current_row += 1
                
                # Add spacing between goals
                current_row += 1
            
            # Add spacing between trigger levels
            current_row += 2
        
        # Add spacing between directions
        current_row += 3
    
    # Auto-adjust column widths
    for column in ws.columns:
        max_length = 0
        column_letter = column[0].column_letter
        for cell in column:
            try:
                if len(str(cell.value)) > max_length:
                    max_length = len(str(cell.value))
            except:
                pass
        adjusted_width = min(max_length + 2, 20)  # Cap at 20 characters
        ws.column_dimensions[column_letter].width = adjusted_width
    
    # Save to BytesIO buffer
    excel_buffer = io.BytesIO()
    wb.save(excel_buffer)
    excel_buffer.seek(0)
    
    return excel_buffer.getvalue()

def generate_html_report(summary_df, metadata):
    """Generate comprehensive HTML report of all trigger-goal combinations"""
    
    # Get unique trigger levels for organizing sections
    trigger_levels = sorted(summary_df['TriggerLevel'].unique())
    directions = ['Above', 'Below']
    trigger_times = ['OPEN', '0900', '1000', '1100', '1200', '1300', '1400', '1500']
    goal_times = ['0900', '1000', '1100', '1200', '1300', '1400', '1500']
    
    html_content = f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>ATR Analysis Report - {metadata['date_range']}</title>
        <style>
            body {{
                font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
                margin: 0;
                padding: 20px;
                background-color: #f8f9fa;
                color: #333;
                line-height: 1.6;
            }}
            .container {{
                max-width: 1400px;
                margin: 0 auto;
                background: white;
                border-radius: 8px;
                box-shadow: 0 2px 10px rgba(0,0,0,0.1);
                overflow: hidden;
            }}
            .header {{
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                color: white;
                padding: 30px;
                text-align: center;
            }}
            .header h1 {{
                margin: 0 0 10px 0;
                font-size: 2.5em;
            }}
            .header p {{
                margin: 0;
                opacity: 0.9;
                font-size: 1.1em;
            }}
            .summary-stats {{
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
                gap: 20px;
                padding: 30px;
                background: #f8f9fa;
                border-bottom: 1px solid #dee2e6;
            }}
            .stat-card {{
                background: white;
                padding: 20px;
                border-radius: 8px;
                text-align: center;
                box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            }}
            .stat-number {{
                font-size: 2em;
                font-weight: bold;
                color: #667eea;
                margin-bottom: 5px;
            }}
            .stat-label {{
                color: #6c757d;
                font-size: 0.9em;
            }}
            .content {{
                padding: 30px;
            }}
            .direction-section {{
                margin-bottom: 40px;
            }}
            .direction-title {{
                font-size: 1.8em;
                color: #495057;
                border-bottom: 3px solid #667eea;
                padding-bottom: 10px;
                margin-bottom: 25px;
            }}
            .trigger-section {{
                margin-bottom: 30px;
                border: 1px solid #dee2e6;
                border-radius: 8px;
                overflow: hidden;
            }}
            .trigger-header {{
                background: #e9ecef;
                padding: 15px 20px;
                font-weight: bold;
                color: #495057;
                border-bottom: 1px solid #dee2e6;
            }}
            .data-table {{
                width: 100%;
                border-collapse: collapse;
                font-size: 0.9em;
            }}
            .data-table th {{
                background: #f8f9fa;
                padding: 12px 8px;
                text-align: center;
                border-bottom: 2px solid #dee2e6;
                border-right: 1px solid #dee2e6;
                font-weight: 600;
                color: #495057;
            }}
            .data-table td {{
                padding: 10px 8px;
                text-align: center;
                border-bottom: 1px solid #f1f3f4;
                border-right: 1px solid #f1f3f4;
            }}
            .data-table tbody tr:hover {{
                background-color: #f8f9fa;
            }}
            .trigger-time {{
                background: #e3f2fd !important;
                font-weight: 600;
                color: #1565c0;
            }}
            .percentage {{
                font-weight: 500;
            }}
            .high-pct {{ color: #28a745; }}
            .med-pct {{ color: #ffc107; }}
            .low-pct {{ color: #dc3545; }}
            .zero-pct {{ color: #6c757d; }}
            .summary-row {{
                background: #f8f9fa !important;
                font-weight: 600;
                border-top: 2px solid #dee2e6;
            }}
            .footer {{
                background: #f8f9fa;
                padding: 20px;
                text-align: center;
                color: #6c757d;
                border-top: 1px solid #dee2e6;
            }}
            @media (max-width: 768px) {{
                .container {{ margin: 10px; }}
                .header {{ padding: 20px; }}
                .header h1 {{ font-size: 1.8em; }}
                .content {{ padding: 15px; }}
                .data-table {{ font-size: 0.8em; }}
                .data-table th, .data-table td {{ padding: 6px 4px; }}
            }}
        </style>
    </head>
    <body>
        <div class="container">
            <div class="header">
                <h1>📊 ATR Analysis Report</h1>
                <p>Comprehensive Trigger-Goal Analysis | {metadata['date_range']}</p>
            </div>
            
            <div class="summary-stats">
                <div class="stat-card">
                    <div class="stat-number">{metadata['total_records']:,}</div>
                    <div class="stat-label">Total Records</div>
                </div>
                <div class="stat-card">
                    <div class="stat-number">{metadata['unique_dates']:,}</div>
                    <div class="stat-label">Trading Days</div>
                </div>
                <div class="stat-card">
                    <div class="stat-number">{metadata['total_triggers']:,}</div>
                    <div class="stat-label">Total Triggers</div>
                </div>
                <div class="stat-card">
                    <div class="stat-number">{metadata['overall_rate']:.1f}%</div>
                    <div class="stat-label">Overall Success Rate</div>
                </div>
            </div>
            
            <div class="content">
    """
    
    # Generate sections for each direction
    for direction in directions:
        direction_data = summary_df[summary_df['Direction'] == direction]
        if len(direction_data) == 0:
            continue
            
        html_content += f"""
                <div class="direction-section">
                    <h2 class="direction-title">📈 {direction} Trigger Analysis</h2>
        """
        
        # Group by trigger level ranges for better organization
        for trigger_level in trigger_levels:
            trigger_data = direction_data[direction_data['TriggerLevel'] == trigger_level]
            if len(trigger_data) == 0:
                continue
                
            html_content += f"""
                    <div class="trigger-section">
                        <div class="trigger-header">
                            Trigger Level: {trigger_level:+.3f}
                        </div>
                        <table class="data-table">
                            <thead>
                                <tr>
                                    <th>Trigger Time</th>
                                    <th>Goal Level</th>
                                    <th>0900</th>
                                    <th>1000</th>
                                    <th>1100</th>
                                    <th>1200</th>
                                    <th>1300</th>
                                    <th>1400</th>
                                    <th>1500</th>
                                    <th>Total Triggers</th>
                                    <th>Total Hits</th>
                                    <th>Success Rate</th>
                                </tr>
                            </thead>
                            <tbody>
            """
            
            # Organize data by trigger time
            for trigger_time in trigger_times:
                time_data = trigger_data[trigger_data['TriggerTime'] == trigger_time]
                if len(time_data) == 0:
                    continue
                
                # Get unique goals for this trigger time
                goals = sorted(time_data['GoalLevel'].unique())
                
                for i, goal_level in enumerate(goals):
                    goal_data = time_data[time_data['GoalLevel'] == goal_level]
                    
                    # Build row data
                    row_data = {}
                    total_hits = 0
                    total_triggers = 0
                    
                    for goal_time in goal_times:
                        goal_time_data = goal_data[goal_data['GoalTime'] == goal_time]
                        if len(goal_time_data) > 0:
                            pct = goal_time_data['PctCompletion'].iloc[0]
                            hits = goal_time_data['NumHits'].iloc[0]
                            triggers = goal_time_data['NumTriggers'].iloc[0]
                            total_hits += hits
                            total_triggers = triggers  # Should be same for all goals
                            
                            # Color code percentages
                            if pct >= 20:
                                pct_class = "high-pct"
                            elif pct >= 10:
                                pct_class = "med-pct"
                            elif pct > 0:
                                pct_class = "low-pct"
                            else:
                                pct_class = "zero-pct"
                            
                            row_data[goal_time] = f'<span class="{pct_class}">{pct:.1f}%</span>'
                        else:
                            row_data[goal_time] = '<span class="zero-pct">0.0%</span>'
                    
                    success_rate = (total_hits / total_triggers * 100) if total_triggers > 0 else 0
                    success_class = "high-pct" if success_rate >= 20 else "med-pct" if success_rate >= 10 else "low-pct" if success_rate > 0 else "zero-pct"
                    
                    # Show trigger time only for first goal
                    trigger_time_cell = f'<td class="trigger-time">{trigger_time}</td>' if i == 0 else '<td></td>'
                    
                    html_content += f"""
                                <tr>
                                    {trigger_time_cell}
                                    <td><strong>{goal_level:+.3f}</strong></td>
                                    <td>{row_data.get('0900', '<span class="zero-pct">0.0%</span>')}</td>
                                    <td>{row_data.get('1000', '<span class="zero-pct">0.0%</span>')}</td>
                                    <td>{row_data.get('1100', '<span class="zero-pct">0.0%</span>')}</td>
                                    <td>{row_data.get('1200', '<span class="zero-pct">0.0%</span>')}</td>
                                    <td>{row_data.get('1300', '<span class="zero-pct">0.0%</span>')}</td>
                                    <td>{row_data.get('1400', '<span class="zero-pct">0.0%</span>')}</td>
                                    <td>{row_data.get('1500', '<span class="zero-pct">0.0%</span>')}</td>
                                    <td><strong>{total_triggers}</strong></td>
                                    <td><strong>{total_hits}</strong></td>
                                    <td><span class="{success_class}"><strong>{success_rate:.1f}%</strong></span></td>
                                </tr>
                    """
            
            html_content += """
                            </tbody>
                        </table>
                    </div>
            """
        
        html_content += """
                </div>
        """
    
    # Close HTML
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    html_content += f"""
            </div>
            
            <div class="footer">
                <p>Generated on {current_time} | ATR Analysis System</p>
                <p>📊 This report contains {len(summary_df):,} trigger-goal combinations across {metadata['unique_dates']} trading days</p>
            </div>
        </div>
    </body>
    </html>
    """
    
    return html_content

st.title("🎯 Enhanced Summary with OPEN Completion Data")
st.write("**Includes HTML and Excel report generation with OPEN completion counts**")

uploaded_file = st.file_uploader("Upload combined_trigger_goal_results_PERFECT.csv", type="csv")

if uploaded_file is not None:
    # Load results with OPEN completions
    df = pd.read_csv(uploaded_file)
    
    st.success(f"📊 Loaded {len(df)} records with OPEN completions")
    
    # Verify this has OPEN completions
    same_time_count = len(df[df['SameTime'] == True]) if 'SameTime' in df.columns else 0
    open_goals = len(df[df['GoalTime'] == 'OPEN']) if 'GoalTime' in df.columns else 0
    
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Same-Time Records", same_time_count)
    with col2:
        st.metric("OPEN Goal Records", open_goals)
    
    if same_time_count == 0 and open_goals == 0:
        st.error("❌ This doesn't appear to have OPEN completion data!")
        st.info("Make sure you're uploading the output from the PERFECT systematic generator")
    else:
        st.success("✅ Confirmed data has OPEN completion data for processing")
    
    # Apply time bucketing
    st.write("🕐 Applying time bucketing...")
    df['TriggerTimeBucket'] = df['TriggerTime'].apply(bucket_time)
    df['GoalTimeBucket'] = df['GoalTime'].apply(lambda x: bucket_time(x) if pd.notna(x) and x != '' else 'N/A')
    
    # Show basic stats
    st.write("## Basic Statistics")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Records", len(df))
    with col2:
        st.metric("Unique Dates", df['Date'].nunique())
    with col3:
        total_hits = len(df[df['GoalHit'] == 'Yes'])
        st.metric("Total Goal Hits", total_hits)
    with col4:
        hit_rate = (total_hits / len(df) * 100) if len(df) > 0 else 0
        st.metric("Overall Hit Rate", f"{hit_rate:.1f}%")
    
    # STEP 1: Count total triggers per trigger combination
    st.write("## Step 1: Count Total Triggers")
    
    trigger_events = df[['Date', 'TriggerLevel', 'TriggerTimeBucket', 'Direction']].drop_duplicates()
    total_trigger_counts = (
        trigger_events
        .groupby(['TriggerLevel', 'TriggerTimeBucket', 'Direction'])
        .size()
        .reset_index(name='TotalTriggers')
    )
    
    st.write(f"✅ Found {len(total_trigger_counts)} unique trigger combinations")
    st.write(f"✅ Total trigger events: {total_trigger_counts['TotalTriggers'].sum():,}")
    
    # STEP 2: Count OPEN completions per trigger-goal combination
    st.write("## Step 2: Count OPEN Completions per Goal")
    
    open_completions = df[
        (df['GoalHit'] == 'Yes') & 
        (df['GoalTimeBucket'] == 'OPEN')
    ].copy()
    
    open_completion_counts = (
        open_completions
        .groupby(['TriggerLevel', 'TriggerTimeBucket', 'Direction', 'GoalLevel'])
        .size()
        .reset_index(name='OpenCompletions')
    )
    
    st.write(f"✅ Found {len(open_completion_counts)} trigger-goal combinations with OPEN completions")
    st.write(f"✅ Total OPEN completions: {open_completion_counts['OpenCompletions'].sum():,}")
    
    # STEP 2.5: Count total OPEN completions per trigger (for dashboard tooltip)
    st.write("## Step 2.5: Count Total OPEN Completions per Trigger")
    
    open_completions_per_trigger = (
        open_completions
        .groupby(['TriggerLevel', 'TriggerTimeBucket', 'Direction'])
        .size()
        .reset_index(name='TotalOpenCompletions')
    )
    
    st.write(f"✅ OPEN completions per trigger calculated")
    
    # STEP 3: Count non-OPEN goal hits per trigger-goal-time combination
    st.write("## Step 3: Count Non-OPEN Goal Hits")
    
    non_open_hits = df[
        (df['GoalHit'] == 'Yes') & 
        (df['GoalTimeBucket'] != 'OPEN') &
        (df['GoalTimeBucket'] != 'N/A')
    ].copy()
    
    goal_hit_counts = (
        non_open_hits
        .groupby(['TriggerLevel', 'TriggerTimeBucket', 'Direction', 'GoalLevel', 'GoalTimeBucket'])
        .size()
        .reset_index(name='NonOpenHits')
    )
    
    st.write(f"✅ Found {len(goal_hit_counts)} non-OPEN goal hit combinations")
    st.write(f"✅ Total non-OPEN hits: {goal_hit_counts['NonOpenHits'].sum():,}")
    
    # STEP 4: Calculate goal-specific denominators
    st.write("## Step 4: Calculate Goal-Specific Denominators")
    
    # Create summary with goal-specific denominators
    summary_rows = []
    
    # Get all unique combinations
    all_triggers = total_trigger_counts[['TriggerLevel', 'TriggerTimeBucket', 'Direction']].drop_duplicates()
    all_goals = df['GoalLevel'].unique()
    all_goal_times = [t for t in df['GoalTimeBucket'].unique() if t not in ['OPEN', 'N/A']]
    
    for _, trigger_row in all_triggers.iterrows():
        trigger_level = trigger_row['TriggerLevel']
        trigger_time = trigger_row['TriggerTimeBucket']
        direction = trigger_row['Direction']
        
        # Get total triggers for this combination
        total_triggers = total_trigger_counts[
            (total_trigger_counts['TriggerLevel'] == trigger_level) &
            (total_trigger_counts['TriggerTimeBucket'] == trigger_time) &
            (total_trigger_counts['Direction'] == direction)
        ]['TotalTriggers'].iloc[0]
        
        # Get total OPEN completions for this trigger (for dashboard tooltip)
        total_open_comps = open_completions_per_trigger[
            (open_completions_per_trigger['TriggerLevel'] == trigger_level) &
            (open_completions_per_trigger['TriggerTimeBucket'] == trigger_time) &
            (open_completions_per_trigger['Direction'] == direction)
        ]
        total_open_completions = total_open_comps['TotalOpenCompletions'].iloc[0] if len(total_open_comps) > 0 else 0
        
        for goal_level in all_goals:
            if goal_level == trigger_level:  # Skip same level
                continue
            
            # Get OPEN completions for this specific trigger-goal combination
            open_comps = open_completion_counts[
                (open_completion_counts['TriggerLevel'] == trigger_level) &
                (open_completion_counts['TriggerTimeBucket'] == trigger_time) &
                (open_completion_counts['Direction'] == direction) &
                (open_completion_counts['GoalLevel'] == goal_level)
            ]
            
            open_completions_count = open_comps['OpenCompletions'].iloc[0] if len(open_comps) > 0 else 0
            
            # Calculate goal-specific denominator
            actionable_triggers = total_triggers - open_completions_count
            
            for goal_time in all_goal_times:
                # Get non-OPEN hits for this combination
                hits = goal_hit_counts[
                    (goal_hit_counts['TriggerLevel'] == trigger_level) &
                    (goal_hit_counts['TriggerTimeBucket'] == trigger_time) &
                    (goal_hit_counts['Direction'] == direction) &
                    (goal_hit_counts['GoalLevel'] == goal_level) &
                    (goal_hit_counts['GoalTimeBucket'] == goal_time)
                ]
                
                num_hits = hits['NonOpenHits'].iloc[0] if len(hits) > 0 else 0
                
                # Calculate percentage with goal-specific denominator
                if actionable_triggers > 0:
                    pct_completion = (num_hits / actionable_triggers * 100)
                else:
                    pct_completion = 0.0
                
                summary_rows.append({
                    'Direction': direction,
                    'TriggerLevel': trigger_level,
                    'TriggerTime': trigger_time,
                    'GoalLevel': goal_level,
                    'GoalTime': goal_time,
                    'TotalTriggers': total_triggers,
                    'OpenCompletions': open_completions_count,
                    'ActionableTriggers': actionable_triggers,
                    'NumHits': num_hits,
                    'PctCompletion': round(pct_completion, 2),
                    'TotalOpenCompletions': total_open_completions  # For dashboard tooltip
                })
    
    summary = pd.DataFrame(summary_rows)
    
    # Remove combinations with 0 actionable triggers
    summary = summary[summary['ActionableTriggers'] > 0]
    
    st.write(f"✅ Complete summary: {len(summary)} combinations with goal-specific denominators")
    
    # Validation: Check for impossible percentages
    over_100 = summary[summary['PctCompletion'] > 100]
    if len(over_100) > 0:
        st.error(f"❌ Found {len(over_100)} combinations with >100% completion!")
        st.dataframe(over_100)
    else:
        st.success("✅ All completion rates ≤ 100% - logic is correct!")
    
    # Show validation example
    st.write("## Validation Example")
    st.write("**Compare the -1 OPEN → 1 vs other goals example:**")
    
    example_filter = (
        (summary['Direction'] == 'Below') &
        (summary['TriggerLevel'] == -1.0) &
        (summary['TriggerTime'] == 'OPEN') &
        (summary['GoalTime'] == '0900') &
        (summary['GoalLevel'].isin([0.236, 1.0]))
    )
    
    example_data = summary[example_filter][['GoalLevel', 'TotalTriggers', 'OpenCompletions', 'ActionableTriggers', 'NumHits', 'PctCompletion', 'TotalOpenCompletions']]
    
    if len(example_data) > 0:
        st.dataframe(example_data)
        st.write("**Key insight:** Different ActionableTriggers denominators with OPEN completion data!")
    
    # Show top performing combinations
    st.write("### Top Performing Combinations:")
    top_performers = summary[summary['ActionableTriggers'] >= 20].nlargest(10, 'PctCompletion')
    st.dataframe(top_performers[['Direction', 'TriggerLevel', 'TriggerTime', 'GoalLevel', 'GoalTime', 'ActionableTriggers', 'NumHits', 'PctCompletion']])
    
    # Create final summary for dashboard (with OPEN completion data)
    dashboard_summary = summary[['Direction', 'TriggerLevel', 'TriggerTime', 'GoalLevel', 'GoalTime', 'ActionableTriggers', 'NumHits', 'PctCompletion', 'OpenCompletions', 'TotalOpenCompletions']].copy()
    dashboard_summary = dashboard_summary.rename(columns={'ActionableTriggers': 'NumTriggers'})
    
    # Report Generation Section
    st.write("## 📊 Report Generation")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Checkbox for generating HTML report
        generate_html = st.checkbox("📄 Generate HTML Report", 
                                   help="Creates a detailed HTML report with all trigger-goal combinations for web viewing")
    
    with col2:
        # Checkbox for generating Excel report
        generate_excel = st.checkbox("📊 Generate Excel Report", 
                                    help="Creates a detailed Excel report grouped by trigger-goal pairs with raw hits and percentages")
    
    if generate_html:
        st.write("🔄 Generating HTML report...")
        
        # Prepare metadata for the report
        date_range = f"{df['Date'].min()} to {df['Date'].max()}"
        total_actionable = summary.groupby(['Direction', 'TriggerLevel', 'TriggerTime'])['ActionableTriggers'].first().sum()
        total_hits = summary['NumHits'].sum()
        overall_rate = (total_hits / total_actionable * 100) if total_actionable > 0 else 0
        
        metadata = {
            'date_range': date_range,
            'total_records': len(summary),
            'unique_dates': df['Date'].nunique(),
            'total_triggers': total_actionable,
            'overall_rate': overall_rate
        }
        
        # Generate HTML content
        html_content = generate_html_report(summary, metadata)
        
        # Create download button for HTML
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        html_filename = f"atr_analysis_report_{timestamp}.html"
        
        st.download_button(
            label="📥 Download HTML Report",
            data=html_content,
            file_name=html_filename,
            mime="text/html",
            help="Download comprehensive HTML report for web viewing and sharing"
        )
        
        st.success("✅ HTML report generated successfully!")
    
    if generate_excel:
        st.write("🔄 Generating Excel report...")
        
        # Prepare metadata for the report
        date_range = f"{df['Date'].min()} to {df['Date'].max()}"
        total_actionable = summary.groupby(['Direction', 'TriggerLevel', 'TriggerTime'])['ActionableTriggers'].first().sum()
        total_hits = summary['NumHits'].sum()
        overall_rate = (total_hits / total_actionable * 100) if total_actionable > 0 else 0
        
        metadata = {
            'date_range': date_range,
            'total_records': len(summary),
            'unique_dates': df['Date'].nunique(),
            'total_triggers': total_actionable,
            'overall_rate': overall_rate
        }
        
        try:
            # Generate Excel content
            excel_content = generate_excel_report(summary, metadata)
            
            # Create download button for Excel
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            excel_filename = f"atr_analysis_report_{timestamp}.xlsx"
            
            st.download_button(
                label="📥 Download Excel Report",
                data=excel_content,
                file_name=excel_filename,
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                help="Download detailed Excel report grouped by trigger-goal pairs"
            )
            
            st.success("✅ Excel report generated successfully!")
            st.info("💡 **Excel Format:** Grouped by trigger-goal pairs with raw hits and percentages across time periods")
            
        except Exception as e:
            st.error(f"❌ Error generating Excel report: {str(e)}")
            st.info("💡 **Note:** Excel generation requires openpyxl library. If running locally, install with: pip install openpyxl")
    
    # Save enhanced summary
    csv_buffer = io.StringIO()
    dashboard_summary.to_csv(csv_buffer, index=False)
    
    st.download_button(
        label="📥 Download Enhanced Summary CSV",
        data=csv_buffer.getvalue(),
        file_name="atr_dashboard_summary_ENHANCED.csv",
        mime="text/csv"
    )
    
    # Final statistics
    st.write("## Final Statistics")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Summary Records", len(summary))
    with col2:
        total_actionable = summary.groupby(['Direction', 'TriggerLevel', 'TriggerTime'])['ActionableTriggers'].first().sum()
        st.metric("Total Actionable Triggers", f"{total_actionable:,}")
    with col3:
        total_hits = summary['NumHits'].sum()
        st.metric("Total Non-OPEN Hits", f"{total_hits:,}")
    with col4:
        overall_rate = (total_hits / total_actionable * 100) if total_actionable > 0 else 0
        st.metric("Overall Actionable Rate", f"{overall_rate:.1f}%")
    
    st.success("🎉 **Enhanced summary complete!** Includes HTML and Excel report generation with OPEN completion data.")
    st.write("**Key improvement:** Now generates both HTML and Excel reports with trigger-goal analysis.")

else:
    st.info("👆 Upload your PERFECT trigger-goal results CSV to generate enhanced summary and reports")
