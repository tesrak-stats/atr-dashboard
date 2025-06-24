import pandas as pd

# Version: detect_triggers_and_goals.py — June 24 full logic rebuild (cleaned headers, 0000 OPEN)

def main():
    # --- Load daily data ---
    daily = pd.read_excel("SPXdailycandles.xlsx", header=4)
    daily['Date'] = pd.to_datetime(daily['Date'])

    for col in ['Open', 'High', 'Low', 'Close']:
        daily[col] = pd.to_numeric(daily[col], errors='coerce')

    # --- Build level map (keys are float values) ---
    level_map = {}
    for col in daily.columns[9:22]:
        try:
            float_label = float(col)
            level_map[float_label] = col
        except ValueError:
            continue

    print("✅ Level map built:", level_map)

    # --- Load intraday data ---
    intraday = pd.read_csv("SPX_10min.csv")
    intraday['Datetime'] = pd.to_datetime(intraday['Datetime'])
    intraday['Date'] = intraday['Datetime'].dt.date
    intraday['Time'] = intraday['Datetime'].dt.strftime('%H%M')

    results = []

    for i in range(1, len(daily)):
        day = daily.iloc[i]
        date = day['Date'].date()
        prev_close = daily.iloc[i - 1]['Close']

        fib_levels = sorted(level_map.keys())
        upside_levels = [f for f in fib_levels if f >= 0]
        downside_levels = [f for f in fib_levels if f <= 0]

        day_data = intraday[intraday['Date'] == date].copy()
        if day_data.empty:
            continue

        day_data.reset_index(drop=True, inplace=True)

        # --- Handle upside triggers ---
        for trigger_level in upside_levels:
            trigger_col = level_map[trigger_level]
            trigger_value = day[trigger_col]
            triggered = False
            trigger_time = None

            for j, row in day_data.iterrows():
                if row['High'] >= trigger_value:
                    if j == 0 and row['Open'] >= trigger_value:
                        trigger_time = '0000'
                    else:
                        trigger_time = row['Time']
                    triggered = True
                    break

            if not triggered or trigger_time == '0000':
                continue

            for goal_level in upside_levels:
                if goal_level <= trigger_level:
                    continue

                goal_col = level_map[goal_level]
                goal_value = day[goal_col]
                goal_hit = False
                goal_time = ''
                retested = False

                for k in range(j + 1, len(day_data)):
                    r = day_data.iloc[k]
                    if r['High'] >= goal_value:
                        goal_hit = True
                        goal_time = r['Time'] if r['Time'] != '0930' else '0900'
                        break

                for k in range(j + 1, len(day_data)):
                    r = day_data.iloc[k]
                    if r['Low'] <= trigger_value:
                        retested = True
                        break

                results.append({
                    'Date': date,
                    'Direction': 'Upside',
                    'TriggerLevel': trigger_level,
                    'TriggerTime': trigger_time,
                    'GoalLevel': goal_level,
                    'GoalTime': goal_time if goal_hit else '',
                    'GoalHit': 'Yes' if goal_hit else 'No',
                    'Type': 'Continuation',
                    'RetestedTrigger': 'Yes' if retested else 'No'
                })

        # --- Handle downside triggers ---
        for trigger_level in downside_levels:
            trigger_col = level_map[trigger_level]
            trigger_value = day[trigger_col]
            triggered = False
            trigger_time = None

            for j, row in day_data.iterrows():
                if row['Low'] <= trigger_value:
                    if j == 0 and row['Open'] <= trigger_value:
                        trigger_time = '0000'
                    else:
                        trigger_time = row['Time']
                    triggered = True
                    break

            if not triggered or trigger_time == '0000':
                continue

            for goal_level in downside_levels:
                if goal_level >= trigger_level:
                    continue

                goal_col = level_map[goal_level]
                goal_value = day[goal_col]
                goal_hit = False
                goal_time = ''
                retested = False

                for k in range(j + 1, len(day_data)):
                    r = day_data.iloc[k]
                    if r['Low'] <= goal_value:
                        goal_hit = True
                        goal_time = r['Time'] if r['Time'] != '0930' else '0900'
                        break

                for k in range(j + 1, len(day_data)):
                    r = day_data.iloc[k]
                    if r['High'] >= trigger_value:
                        retested = True
                        break

                results.append({
                    'Date': date,
                    'Direction': 'Downside',
                    'TriggerLevel': trigger_level,
                    'TriggerTime': trigger_time,
                    'GoalLevel': goal_level,
                    'GoalTime': goal_time if goal_hit else '',
                    'GoalHit': 'Yes' if goal_hit else 'No',
                    'Type': 'Continuation',
                    'RetestedTrigger': 'Yes' if retested else 'No'
                })

    return pd.DataFrame(results)
