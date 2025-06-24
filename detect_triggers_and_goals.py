# detect_triggers_and_goals.py
# Version 4.0 – Full production version with continuation, retracement, and OPEN/0900 duplicate handling

import pandas as pd

# --- Load Data ---
daily_df = pd.read_excel("SPXdailycandles.xlsx", header=4)
intraday_df = pd.read_csv("SPX_10min.csv")

# --- Preprocess Dates ---
daily_df['Date'] = pd.to_datetime(daily_df['Date'])
intraday_df['Datetime'] = pd.to_datetime(intraday_df['Datetime'])
intraday_df['Date'] = intraday_df['Datetime'].dt.date

# --- Extract Fibonacci Levels ---
fib_columns = [col for col in daily_df.columns if '%' in str(col)]
level_map = {col: float(str(col).replace('%', '').strip()) / 100 for col in fib_columns}

# --- Prepare Results ---
results = []

# --- Main Loop ---
for idx, day_row in daily_df.iterrows():
    date = day_row['Date'].date()
    fib_levels = {level_map[col]: day_row[col] for col in fib_columns if pd.notna(day_row[col])}

    if date not in intraday_df['Date'].values:
        continue

    day_data = intraday_df[intraday_df['Date'] == date].copy()
    day_data['Time'] = day_data['Datetime'].dt.strftime('%H%M')

    open_price = day_data.iloc[0]['Open']
    high_series = day_data['High']
    low_series = day_data['Low']

    for direction in ['Upside', 'Downside']:
        sorted_levels = sorted(fib_levels.keys(), reverse=(direction == 'Downside'))

        for i, trigger_level in enumerate(sorted_levels):
            trigger_price = fib_levels[trigger_level]

            if direction == 'Upside':
                trigger_condition = (high_series >= trigger_price)
                zone_upper = fib_levels[sorted_levels[i + 1]] if i + 1 < len(sorted_levels) else float('inf')
                open_trigger = trigger_price <= open_price < zone_upper
            else:
                trigger_condition = (low_series <= trigger_price)
                zone_lower = fib_levels[sorted_levels[i + 1]] if i + 1 < len(sorted_levels) else float('-inf')
                open_trigger = zone_lower < open_price <= trigger_price

            triggered = False
            trigger_time = None
            trigger_idx = None

            if open_trigger:
                trigger_time = 'OPEN'
                trigger_idx = 0
                triggered = True
            else:
                for j, row in day_data.iterrows():
                    if trigger_condition.iloc[j]:
                        trigger_time = row['Time'] if row['Time'] != '0930' else '0900'
                        if trigger_time == '0900' and j == 0 and open_trigger:
                            trigger_time = 'OPEN'
                        trigger_idx = j
                        triggered = True
                        break

            if not triggered:
                continue

            trigger_row = day_data.iloc[trigger_idx]
            trigger_datetime = trigger_row['Datetime']

            # --- CONTINUATION GOALS ---
            for goal_level in sorted_levels[i + 1:] if direction == 'Upside' else sorted_levels[i + 1:]:
                goal_price = fib_levels[goal_level]
                goal_hit = False
                goal_time = None

                for k in range(trigger_idx, len(day_data)):
                    row = day_data.iloc[k]
                    if direction == 'Upside' and row['High'] >= goal_price:
                        if trigger_time == 'OPEN' and k == trigger_idx:
                            break
                        goal_hit = True
                        goal_time = row['Time'] if row['Time'] != '0930' else '0900'
                        break
                    elif direction == 'Downside' and row['Low'] <= goal_price:
                        if trigger_time == 'OPEN' and k == trigger_idx:
                            break
                        goal_hit = True
                        goal_time = row['Time'] if row['Time'] != '0930' else '0900'
                        break

                results.append({
                    'Date': date,
                    'Direction': direction,
                    'TriggerLevel': trigger_level,
                    'TriggerTime': trigger_time,
                    'GoalLevel': goal_level,
                    'GoalTime': goal_time if goal_hit else '',
                    'GoalHit': 'Yes' if goal_hit else 'No',
                    'Type': 'Continuation',
                    'RetestedTrigger': ''
                })

            # --- RETRACEMENT GOALS ---
            for goal_level in sorted_levels[:i][::-1] if direction == 'Upside' else sorted_levels[:i][::-1]:
                goal_price = fib_levels[goal_level]
                goal_hit = False
                goal_time = None
                retested = False

                for k in range(trigger_idx + 1, len(day_data)):
                    row = day_data.iloc[k]

                    if direction == 'Upside':
                        if row['Low'] <= fib_levels[trigger_level]:
                            retested = True
                        if row['Low'] <= goal_price:
                            goal_hit = True
                            goal_time = row['Time'] if row['Time'] != '0930' else '0900'
                            break
                    else:
                        if row['High'] >= fib_levels[trigger_level]:
                            retested = True
                        if row['High'] >= goal_price:
                            goal_hit = True
                            goal_time = row['Time'] if row['Time'] != '0930' else '0900'
                            break

                results.append({
                    'Date': date,
                    'Direction': direction,
                    'TriggerLevel': trigger_level,
                    'TriggerTime': trigger_time,
                    'GoalLevel': goal_level,
                    'GoalTime': goal_time if goal_hit else '',
                    'GoalHit': 'Yes' if goal_hit else 'No',
                    'Type': 'Retracement',
                    'RetestedTrigger': 'Yes' if retested else 'No'
                })

# --- Save Output ---
results_df = pd.DataFrame(results)
results_df.to_csv("combined_trigger_goal_results.csv", index=False)
print("✅ File saved: combined_trigger_goal_results.csv")
