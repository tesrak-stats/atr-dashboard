import streamlit as st
import pandas as pd
import os

def detect_triggers_and_goals(daily, intraday):
    fib_levels = [1.000, 0.786, 0.618, 0.500, 0.382, 0.236, 0.000,
                  -0.236, -0.382, -0.500, -0.618, -0.786, -1.000]
    
    results = []

    for i in range(1, len(daily)):
        date = daily.iloc[i]['Date']
        
        # BULLETPROOF string comparison
        try:
            if hasattr(date, 'strftime'):
                date_str = date.strftime('%Y-%m-%d')
            elif isinstance(date, str):
                date_str = date[:10]
            else:
                date_str = str(date)[:10]
            
            # Simple string comparison
            if date_str < '2014-01-02':
                continue
                
        except Exception as e:
            continue

        day_row = daily.iloc[i]
        prev_row = daily.iloc[i - 1]

        level_map = {}
        for level in fib_levels:
            level_str = f'{level:.3f}'.rstrip('0').rstrip('.') if '.' in f'{level:.3f}' else str(level)
            if level_str in day_row:
                level_map[level] = day_row[level_str]

        day_data = intraday[intraday['Date'] == pd.to_datetime(date).date()].copy()
        if day_data.empty:
            continue

        day_data['Time'] = day_data['Datetime'].dt.strftime('%H%M')
        day_data.reset_index(drop=True, inplace=True)

        triggered_up = {}
        triggered_down = {}

        for idx, row in day_data.iterrows():
            open_price = row['Open']
            high = row['High']
            low = row['Low']
            time_label = '0000' if idx == 0 and level_map and (open_price >= min(level_map.values())) else row['Time']
            hour_block = '0000' if time_label == '0000' else time_label[:2] + '00'

            for level in sorted([lvl for lvl in fib_levels if lvl > 0]):
                if level in triggered_up:
                    continue
                if high >= level_map.get(level, float('inf')):
                    triggered_up[level] = {
                        'TriggerLevel': level,
                        'TriggerTime': time_label,
                        'TriggeredRow': idx
                    }

            for level in sorted([lvl for lvl in fib_levels if lvl < 0], reverse=True):
                if level in triggered_down:
                    continue
                if low <= level_map.get(level, float('-inf')):
                    triggered_down[level] = {
                        'TriggerLevel': level,
                        'TriggerTime': time_label,
                        'TriggeredRow': idx
                    }

        for level, trigger_info in triggered_up.items():
            for goal_level in [l for l in fib_levels if l > level]:
                goal_hit = False
                goal_time = ''
                for _, row in day_data.iloc[trigger_info['TriggeredRow']+1:].iterrows():
                    if row['High'] >= level_map.get(goal_level, float('inf')):
                        goal_hit = True
                        goal_time = row['Time']
                        break
                results.append({
                    'Date': date,
                    'Direction': 'Upside',
                    'TriggerLevel': level,
                    'TriggerTime': trigger_info['TriggerTime'],
                    'GoalLevel': goal_level,
                    'GoalHit': 'Yes' if goal_hit else 'No',
                    'GoalTime': goal_time if goal_hit else '',
                    'Type': 'Continuation',
                    'RetestedTrigger': 'No'
                })

            for retrace_level in [l for l in fib_levels if l < 0]:
                goal_hit = False
                goal_time = ''
                for _, row in day_data.iloc[trigger_info['TriggeredRow']+1:].iterrows():
                    if row['Low'] <= level_map.get(retrace_level, float('-inf')):
                        goal_hit = True
                        goal_time = row['Time']
                        break
                results.append({
                    'Date': date,
                    'Direction': 'Upside',
                    'TriggerLevel': level,
                    'TriggerTime': trigger_info['TriggerTime'],
                    'GoalLevel': retrace_level,
                    'GoalHit': 'Yes' if goal_hit else 'No',
                    'GoalTime': goal_time if goal_hit else '',
                    'Type': 'Retracement',
                    'RetestedTrigger': 'No'
                })

        for level, trigger_info in triggered_down.items():
            for goal_level in [l for l in fib_levels if l < level]:
                goal_hit = False
                goal_time = ''
                for _, row in day_data.iloc[trigger_info['TriggeredRow']+1:].iterrows():
                    if row['Low'] <= level_map.get(goal_level, float('-inf')):
                        goal_hit = True
                        goal_time = row['Time']
                        break
                results.append({
                    'Date': date,
                    'Direction': 'Downside',
                    'TriggerLevel': level,
                    'TriggerTime': trigger_info['TriggerTime'],
                    'GoalLevel': goal_level,
                    'GoalHit': 'Yes' if goal_hit else 'No',
                    'GoalTime': goal_time if goal_hit else '',
                    'Type': 'Continuation',
                    'RetestedTrigger': 'No'
                })

            for retrace_level in [l for l in fib_levels if l > 0]:
                goal_hit = False
                goal_time = ''
                for _, row in day_data.iloc[trigger_info['TriggeredRow']+1:].iterrows():
                    if row['High'] >= level_map.get(retrace_level, float('inf')):
                        goal_hit = True
                        goal_time = row['Time']
                        break
                results.append({
                    'Date': date,
                    'Direction': 'Downside',
                    'TriggerLevel': level,
                    'TriggerTime': trigger_info['TriggerTime'],
                    'GoalLevel': retrace_level,
                    'GoalHit': 'Yes' if goal_hit else 'No',
                    'GoalTime': goal_time if goal_hit else '',
                    'Type': 'Retracement',
                    'RetestedTrigger': 'No'
                })

    return pd.DataFrame(results)

def main():
    debug_info = []
    
    try:
        debug_info.append("Loading data files...")
        daily = pd.read_excel('SPXdailycandles.xlsx', header=4)
        debug_info.append(f"Daily data shape: {daily.shape}")
        debug_info.append(f"Daily columns: {list(daily.columns)}")
        debug_info.append(f"First few dates: {daily['Date'].head().tolist()}")
        
        intraday = pd.read_csv('SPX_10min.csv', parse_dates=['Datetime'])
        intraday['Date'] = intraday['Datetime'].dt.date
        debug_info.append(f"Intraday data shape: {intraday.shape}")
        debug_info.append(f"Intraday date range: {intraday['Date'].min()} to {intraday['Date'].max()}")
        
        # Check for level columns
        fib_levels = [1.000, 0.786, 0.618, 0.500, 0.382, 0.236, 0.000, -0.236, -0.382, -0.500, -0.618, -0.786, -1.000]
        found_levels = []
        for level in fib_levels:
            level_str = f'{level:.3f}'.rstrip('0').rstrip('.') if '.' in f'{level:.3f}' else str(level)
            if level_str in daily.columns:
                found_levels.append(level_str)
        debug_info.append(f"Found level columns: {found_levels}")
        
        df = detect_triggers_and_goals(daily, intraday)
        debug_info.append(f"Results generated: {len(df)} rows")
        
        return df, debug_info
        
    except Exception as e:
        debug_info.append(f"Error: {str(e)}")
        return pd.DataFrame(), debug_info

st.title('ATR Trigger & Goal Generator')

output_path = 'combined_trigger_goal_results.csv'

# Update the button section too:
if st.button('Generate combined_trigger_goal_results.csv'):
    with st.spinner('Running detection...'):
        try:
            result_df, debug_messages = main()
            
            # Show debug info
            st.subheader('Debug Information')
            for msg in debug_messages:
                st.write(msg)
            
            if not result_df.empty:
                result_df['Source'] = 'Full'
                result_df.to_csv(output_path, index=False)
                st.success('File generated and saved!')
                
                # Preview
                st.subheader('Preview of Results')
                st.dataframe(result_df.head(30))
                st.download_button('Download CSV', data=result_df.to_csv(index=False), file_name=output_path, mime='text/csv')
            else:
                st.warning('No results generated - check debug info above')
                
        except Exception as e:
            st.error(f'Error: {e}')
