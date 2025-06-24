import pandas as pd

def detect_triggers_and_goals(daily, intraday):
    fib_levels = [1.000, 0.786, 0.618, 0.500, 0.382, 0.236, 0.000,
                  -0.236, -0.382, -0.500, -0.618, -0.786, -1.000]

    results = []

    for i in range(1, len(daily)):
        date = daily.loc[i, 'Date']
        prev_close = daily.loc[i - 1, 'Close']

        day_row = daily.loc[i]
        level_map = {}
        for level in fib_levels:
            level_str = f"{level:.3f}".rstrip('0').rstrip('.') if '.' in f"{level:.3f}" else str(level)
            try:
                level_map[level] = day_row[level_str]
            except KeyError:
                continue

        day_data = intraday[intraday['Date'] == date].copy()
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

            # Detect true OPEN trigger
            time_label = row['Time']
            if idx == 0:
                for level in fib_levels:
                    if level > 0 and open_price >= level_map.get(level, float('inf')):
                        time_label = '0000'
                        break
                    elif level < 0 and open_price <= level_map.get(level, float('-inf')):
                        time_label = '0000'
                        break

            for level in sorted([lvl for lvl in fib_levels if lvl > 0]):
                if level in triggered_up:
                    continue
                if high >= level_map[level]:
                    triggered_up[level] = {
                        'TriggerLevel': level,
                        'TriggerTime': time_label,
                        'TriggeredRow': idx
                    }

            for level in sorted([lvl for lvl in fib_levels if lvl < 0], reverse=True):
                if level in triggered_down:
                    continue
                if low <= level_map[level]:
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
                    if row['High'] >= level_map[goal_level]:
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
                    if row['Low'] <= level_map[retrace_level]:
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
                    if row['Low'] <= level_map[goal_level]:
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
                    if row['High'] >= level_map[retrace_level]:
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

# ✅ Streamlit/production-compatible entry point
def main():
    daily = pd.read_excel("SPXdailycandles.xlsx", header=4)
    intraday = pd.read_csv("SPX_10min.csv", parse_dates=['Datetime'])
    intraday['Date'] = intraday['Datetime'].dt.date

    df = detect_triggers_and_goals(daily, intraday)
    df.to_csv("combined_trigger_goal_results.csv", index=False)
    print("✅ Output saved to combined_trigger_goal_results.csv")
