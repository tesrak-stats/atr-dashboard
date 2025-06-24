
import pandas as pd

def detect_triggers_and_goals(daily, intraday):
    fib_levels = [1.000, 0.786, 0.618, 0.500, 0.382, 0.236, 0.000,
                  -0.236, -0.382, -0.500, -0.618, -0.786, -1.000]

    results = []

    for i in range(1, len(daily)):  # start from index 1 (2-Jan-2014)
        date = daily.iloc[i]['Date']
        day_row = daily.iloc[i]
        prev_row = daily.iloc[i - 1]
        prev_close = prev_row['Close']

        # Build level map from headers like "0.236", "-0.382", etc.
        level_map = {}
        for level in fib_levels:
            level_str = f"{level:.3f}".rstrip('0').rstrip('.') if '.' in f"{level:.3f}" else str(level)
            if level_str in day_row:
                level_map[level] = day_row[level_str]

        # Filter intraday data for current date
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
            time_label = '0000' if idx == 0 else row['Time']

            for level in sorted([l for l in fib_levels if l > 0]):
                if level in triggered_up:
                    continue
                if (idx == 0 and prev_close < level_map[level] <= open_price) or high >= level_map[level]:
                    triggered_up[level] = {
                        'TriggerLevel': level,
                        'TriggerTime': time_label,
                        'TriggeredRow': idx
                    }

            for level in sorted([l for l in fib_levels if l < 0], reverse=True):
                if level in triggered_down:
                    continue
                if (idx == 0 and prev_close > level_map[level] >= open_price) or low <= level_map[level]:
                    triggered_down[level] = {
                        'TriggerLevel': level,
                        'TriggerTime': time_label,
                        'TriggeredRow': idx
                    }

        # Upside: check continuation and retracement
        for level, trig in triggered_up.items():
            for goal in [l for l in fib_levels if l > level]:
                goal_hit = False
                goal_time = ''
                for _, row in day_data.iloc[trig['TriggeredRow']+1:].iterrows():
                    if row['High'] >= level_map.get(goal, float('inf')):
                        goal_hit = True
                        goal_time = row['Time']
                        break
                results.append({
                    'Date': date,
                    'Direction': 'Upside',
                    'TriggerLevel': level,
                    'TriggerTime': trig['TriggerTime'],
                    'GoalLevel': goal,
                    'GoalHit': 'Yes' if goal_hit else 'No',
                    'GoalTime': goal_time if goal_hit else '',
                    'Type': 'Continuation',
                    'RetestedTrigger': 'No'
                })

            for retrace in [l for l in fib_levels if l < 0]:
                goal_hit = False
                goal_time = ''
                for _, row in day_data.iloc[trig['TriggeredRow']+1:].iterrows():
                    if row['Low'] <= level_map.get(retrace, float('-inf')):
                        goal_hit = True
                        goal_time = row['Time']
                        break
                results.append({
                    'Date': date,
                    'Direction': 'Upside',
                    'TriggerLevel': level,
                    'TriggerTime': trig['TriggerTime'],
                    'GoalLevel': retrace,
                    'GoalHit': 'Yes' if goal_hit else 'No',
                    'GoalTime': goal_time if goal_hit else '',
                    'Type': 'Retracement',
                    'RetestedTrigger': 'No'
                })

        # Downside: check continuation and retracement
        for level, trig in triggered_down.items():
            for goal in [l for l in fib_levels if l < level]:
                goal_hit = False
                goal_time = ''
                for _, row in day_data.iloc[trig['TriggeredRow']+1:].iterrows():
                    if row['Low'] <= level_map.get(goal, float('-inf')):
                        goal_hit = True
                        goal_time = row['Time']
                        break
                results.append({
                    'Date': date,
                    'Direction': 'Downside',
                    'TriggerLevel': level,
                    'TriggerTime': trig['TriggerTime'],
                    'GoalLevel': goal,
                    'GoalHit': 'Yes' if goal_hit else 'No',
                    'GoalTime': goal_time if goal_hit else '',
                    'Type': 'Continuation',
                    'RetestedTrigger': 'No'
                })

            for retrace in [l for l in fib_levels if l > 0]:
                goal_hit = False
                goal_time = ''
                for _, row in day_data.iloc[trig['TriggeredRow']+1:].iterrows():
                    if row['High'] >= level_map.get(retrace, float('inf')):
                        goal_hit = True
                        goal_time = row['Time']
                        break
                results.append({
                    'Date': date,
                    'Direction': 'Downside',
                    'TriggerLevel': level,
                    'TriggerTime': trig['TriggerTime'],
                    'GoalLevel': retrace,
                    'GoalHit': 'Yes' if goal_hit else 'No',
                    'GoalTime': goal_time if goal_hit else '',
                    'Type': 'Retracement',
                    'RetestedTrigger': 'No'
                })

    return pd.DataFrame(results)

# --- Main block for Streamlit or CLI use ---
if __name__ == "__main__":
    daily = pd.read_excel("SPXdailycandles.xlsx", header=4)
    intraday = pd.read_csv("SPX_10min.csv", parse_dates=['Datetime'])
    intraday['Date'] = intraday['Datetime'].dt.date

    df = detect_triggers_and_goals(daily, intraday)
    df.to_csv("combined_trigger_goal_results.csv", index=False)
    print("✅ Output saved to combined_trigger_goal_results.csv")
