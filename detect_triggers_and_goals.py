
import pandas as pd

def detect_triggers_and_goals(daily, intraday):
    fib_levels = [1.000, 0.786, 0.618, 0.500, 0.382, 0.236, 0.000,
                  -0.236, -0.382, -0.500, -0.618, -0.786, -1.000]

    results = []

    for i in range(1, len(daily)):
        date = daily.iloc[i]['Date']
        day_row = daily.iloc[i]
        prev_close = daily.iloc[i - 1]['Close']

        # Build mapping from fib level to level value
        level_map = {}
        for level in fib_levels:
            level_str = f"{level:.3f}".rstrip('0').rstrip('.') if '.' in f"{level:.3f}" else str(level)
            if level_str in day_row:
                level_map[level] = day_row[level_str]

        if not level_map:
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
            time_str = row['Time']

            if idx == 0:
                # Check OPEN trigger condition
                for level in sorted([lvl for lvl in fib_levels if lvl > 0]):
                    if level in level_map and level_map[level] <= open_price:
                        triggered_up[level] = {
                            'TriggerLevel': level,
                            'TriggerTime': '0000',
                            'TriggeredRow': idx
                        }
                for level in sorted([lvl for lvl in fib_levels if lvl < 0], reverse=True):
                    if level in level_map and level_map[level] >= open_price:
                        triggered_down[level] = {
                            'TriggerLevel': level,
                            'TriggerTime': '0000',
                            'TriggeredRow': idx
                        }

            # Regular candle triggers
            for level in sorted([lvl for lvl in fib_levels if lvl > 0]):
                if level in triggered_up or level not in level_map:
                    continue
                if high >= level_map[level]:
                    triggered_up[level] = {
                        'TriggerLevel': level,
                        'TriggerTime': time_str,
                        'TriggeredRow': idx
                    }

            for level in sorted([lvl for lvl in fib_levels if lvl < 0], reverse=True):
                if level in triggered_down or level not in level_map:
                    continue
                if low <= level_map[level]:
                    triggered_down[level] = {
                        'TriggerLevel': level,
                        'TriggerTime': time_str,
                        'TriggeredRow': idx
                    }

        # Evaluate goals for upside triggers
        for level, trigger_info in triggered_up.items():
            # Continuation goals: higher levels
            for goal_level in [l for l in fib_levels if l > level and l in level_map]:
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
                    'GoalTime': goal_time,
                    'Type': 'Continuation',
                    'RetestedTrigger': 'No'
                })

            # Retracement goals: lower fibs (negative side)
            for retrace_level in [l for l in fib_levels if l < 0 and l in level_map]:
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
                    'GoalTime': goal_time,
                    'Type': 'Retracement',
                    'RetestedTrigger': 'No'
                })

        # Evaluate goals for downside triggers
        for level, trigger_info in triggered_down.items():
            # Continuation goals: lower fibs
            for goal_level in [l for l in fib_levels if l < level and l in level_map]:
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
                    'GoalTime': goal_time,
                    'Type': 'Continuation',
                    'RetestedTrigger': 'No'
                })

            # Retracement goals: upside fibs
            for retrace_level in [l for l in fib_levels if l > 0 and l in level_map]:
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
                    'GoalTime': goal_time,
                    'Type': 'Retracement',
                    'RetestedTrigger': 'No'
                })

    return pd.DataFrame(results)


# ✅ Streamlit-compatible entry point
def main():
    daily = pd.read_excel("SPXdailycandles.xlsx", header=4)
    intraday = pd.read_csv("SPX_10min.csv", parse_dates=['Datetime'])
    intraday['Date'] = intraday['Datetime'].dt.date

    df = detect_triggers_and_goals(daily, intraday)
    df.to_csv("combined_trigger_goal_results.csv", index=False)
    print("✅ Output saved to combined_trigger_goal_results.csv")

if __name__ == "__main__":
    main()
