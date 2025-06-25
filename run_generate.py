import streamlit as st
import pandas as pd
import os

def detect_triggers_and_goals(daily, intraday):
    fib_levels = [1.000, 0.786, 0.618, 0.500, 0.382, 0.236, 0.000, -0.236, -0.382, -0.500, -0.618, -0.786, -1.000]
    results = []
    for i in range(1, len(daily)):
        date = daily.iloc[i]['Date']
        try:
            if hasattr(date, 'strftime'):
                date_str = date.strftime('%Y-%m-%d')
            elif isinstance(date, str):
                date_str = date[:10]
            else:
                date_str = str(date)[:10]
            if date_str < '2014-01-02':
                continue
        except Exception as e:
            continue
        day_row = daily.iloc[i]
        prev_row = daily.iloc[i - 1]
        level_map = {}
        for level in fib_levels:
            found_col = None
            test_names = [str(level), str(int(level)) if level == int(level) else str(level), f'{level:.1f}' if level != int(level) else str(int(level))]
            for test_name in test_names:
                if test_name in day_row.index and pd.notna(day_row[test_name]):
                    found_col = test_name
                    break
            if found_col:
                level_map[level] = day_row[found_col]
        if not level_map:
            continue
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
            for level in sorted([lvl for lvl in fib_levels if lvl > 0 and lvl in level_map]):
                if level in triggered_up:
                    continue
                if high >= level_map[level]:
                    triggered_up[level] = {'TriggerLevel': level, 'TriggerTime': time_label, 'TriggeredRow': idx}
            for level in sorted([lvl for lvl in fib_levels if lvl < 0 and lvl in level_map], reverse=True):
                if level in triggered_down:
                    continue
                if low <= level_map[level]:
                    triggered_down[level] = {'TriggerLevel': level, 'TriggerTime': time_label, 'TriggeredRow': idx}
        for level, trigger_info in triggered_up.items():
            for goal_level in [l for l in fib_levels if l > level and l in level_map]:
                goal_hit = False
                goal_time = ''
                for _, row in day_data.iloc[trigger_info['TriggeredRow']+1:].iterrows():
                    if row['High'] >= level_map[goal_level]:
                        goal_hit = True
                        goal_time = row['Time']
                        break
                results.append({'Date': date, 'Direction': 'Upside', 'TriggerLevel': level, 'TriggerTime': trigger_info['TriggerTime'], 'GoalLevel': goal_level, 'GoalHit': 'Yes' if goal_hit else 'No', 'GoalTime': goal_time if goal_hit else '', 'Type': 'Continuation', 'RetestedTrigger': 'No'})
    return pd.DataFrame(results)

def main():
    debug_info = []
    
    try:
        debug_info.append("Loading data files...")
        daily = pd.read_excel('SPXdailycandles.xlsx', header=4)
        debug_info.append(f"Daily data shape: {daily.shape}")
        debug_info.append(f"Daily columns: {list(daily.columns)}")
        
        # Test the column matching logic
        fib_levels = [1.000, 0.786, 0.618, 0.500, 0.382, 0.236, 0.000, -0.236, -0.382, -0.500, -0.618, -0.786, -1.000]
        test_row = daily.iloc[1]  # Test with second row
        found_levels = []
        
        for level in fib_levels:
            test_names = [str(level), str(int(level)) if level == int(level) else str(level), f'{level:.1f}' if level != int(level) else str(int(level))]
            debug_info.append(f"Testing level {level}, trying names: {test_names}")
            
            found_col = None
            for test_name in test_names:
                if test_name in test_row.index:
                    found_col = test_name
                    debug_info.append(f"  Found match: {test_name}")
                    break
            
            if found_col:
                found_levels.append(found_col)
            else:
                debug_info.append(f"  No match found for {level}")
        
        debug_info.append(f"Found level columns: {found_levels}")
        
        intraday = pd.read_csv('SPX_10min.csv', parse_dates=['Datetime'])
        intraday['Date'] = intraday['Datetime'].dt.date
        debug_info.append(f"Intraday data shape: {intraday.shape}")
        
        df = detect_triggers_and_goals(daily, intraday)
        debug_info.append(f"Results generated: {len(df)} rows")
        
        return df, debug_info
        
    except Exception as e:
        debug_info.append(f"Error: {str(e)}")
        return pd.DataFrame(), debug_info
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
