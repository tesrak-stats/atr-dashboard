# Version 3.0 – Full logic, no placeholders, fixed 0000 time and candle time parsing

import pandas as pd

# --- Load data ---
daily_df = pd.read_excel("SPXdailycandles.xlsx", header=4)
intraday_df = pd.read_csv("SPX_10min.csv")

# --- Parse datetime in intraday candles ---
intraday_df['Datetime'] = pd.to_datetime(intraday_df['Datetime'])
intraday_df['Date'] = intraday_df['Datetime'].dt.date
intraday_df['TimeLabel'] = intraday_df['Datetime'].dt.strftime('%H%M')  # For precision

# --- Loop through days ---
results = []

for idx, day_row in daily_df.iterrows():
    date = day_row['Date']
    atr_levels = {
        col: day_row[col]
        for col in daily_df.columns if isinstance(col, str) and '%' in col
    }
    prev_close = day_row['Previous Close']
    
    # Build sorted list of levels with values
    level_items = sorted(atr_levels.items(), key=lambda x: float(x[0].replace('%', '').replace('-', '')) * (-1 if '-' in x[0] else 1))
    levels = [name for name, _ in level_items]
    level_values = {name: value for name, value in level_items}
    
    day_data = intraday_df[intraday_df['Date'] == date].copy()
    if day_data.empty:
        continue

    # --- Determine triggers ---
    triggered_levels = set()

    for i, row in day_data.iterrows():
        candle_time = row['Datetime']
        time_label = row['TimeLabel']
        open_price = row['Open']
        high_price = row['High']
        low_price = row['Low']

        for level in levels:
            value = level_values[level]

            # --- Upside trigger logic ---
            if high_price >= value and level not in triggered_levels:
                # Determine if it qualifies as OPEN
                if candle_time.time().hour == 9 and candle_time.time().minute == 30:
                    if open_price >= value and open_price < level_values.get(levels[levels.index(level)+1], float('inf')):
                        trigger_time = '0000'
                    else:
                        trigger_time = time_label
                else:
                    trigger_time = time_label

                direction = 'Upside'
                triggered_levels.add(level)

                # --- Evaluate goal hits above ---
                goal_levels = levels[levels.index(level)+1:]
                for goal_level in goal_levels:
                    goal_value = level_values[goal_level]
                    goal_hit = None
                    goal_time = None
                    retrace_flag = None
                    trigger_candle_index = i

                    for j in range(i, len(day_data)):
                        goal_row = day_data.iloc[j]
                        goal_high = goal_row['High']
                        goal_low = goal_row['Low']
                        goal_open = goal_row['Open']
                        goal_label = goal_row['TimeLabel']

                        if j == i and goal_high >= goal_value:
                            goal_hit = 'Yes'
                            goal_time = goal_label
                            retrace_flag = 'Continuation'
                            break
                        elif j > i and goal_high >= goal_value:
                            goal_hit = 'Yes'
                            goal_time = goal_label
                            retrace_flag = 'Continuation'
                            break

                    if not goal_hit:
                        goal_hit = 'No'
                        goal_time = ''

                    results.append({
                        'Date': date,
                        'Direction': direction,
                        'TriggerLevel': level,
                        'TriggerTime': trigger_time,
                        'GoalLevel': goal_level,
                        'GoalTime': goal_time,
                        'GoalHit': goal_hit,
                        'RetraceOrCont': retrace_flag or 'Miss'
                    })

            # --- Downside trigger logic ---
            elif low_price <= value and level not in triggered_levels:
                if candle_time.time().hour == 9 and candle_time.time().minute == 30:
                    if open_price <= value and open_price > level_values.get(levels[levels.index(level)+1], float('-inf')):
                        trigger_time = '0000'
                    else:
                        trigger_time = time_label
                else:
                    trigger_time = time_label

                direction = 'Downside'
                triggered_levels.add(level)

                goal_levels = levels[levels.index(level)+1:]
                for goal_level in goal_levels:
                    goal_value = level_values[goal_level]
                    goal_hit = None
                    goal_time = None
                    retrace_flag = None

                    for j in range(i, len(day_data)):
                        goal_row = day_data.iloc[j]
                        goal_low = goal_row['Low']
                        goal_label = goal_row['TimeLabel']

                        if j == i and goal_low <= goal_value:
                            goal_hit = 'Yes'
                            goal_time = goal_label
                            retrace_flag = 'Continuation'
                            break
                        elif j > i and goal_low <= goal_value:
                            goal_hit = 'Yes'
                            goal_time = goal_label
                            retrace_flag = 'Continuation'
                            break

                    if not goal_hit:
                        goal_hit = 'No'
                        goal_time = ''

                    results.append({
                        'Date': date,
                        'Direction': direction,
                        'TriggerLevel': level,
                        'TriggerTime': trigger_time,
                        'GoalLevel': goal_level,
                        'GoalTime': goal_time,
                        'GoalHit': goal_hit,
                        'RetraceOrCont': retrace_flag or 'Miss'
                    })

# --- Export result ---
df_out = pd.DataFrame(results)
df_out.to_csv("combined_trigger_goal_results.csv", index=False)
print("✅ File saved to combined_trigger_goal_results.csv")
