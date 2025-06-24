
import pandas as pd

# --- Load daily ATR levels ---
daily = pd.read_excel("SPXdailycandles.xlsx", header=4)

# Extract float level names from columns J to V (index 9 to 21)
level_map = {}
for col in daily.columns[9:22]:
    try:
        if isinstance(col, str) and '%' in col:
            float_label = float(col.strip('%')) / 100
        else:
            float_label = float(col)
        level_map[float_label] = col
    except ValueError:
        continue
level_keys = sorted(level_map.keys(), reverse=True)  # High to low for processing

# --- Load intraday data ---
intraday = pd.read_csv("SPX_10min.csv")
intraday['Datetime'] = pd.to_datetime(intraday['Datetime'])
intraday['Date'] = pd.to_datetime(intraday['Datetime'].dt.date)
intraday['Hour'] = intraday['Datetime'].dt.strftime('%H00')
intraday.loc[intraday['Datetime'].dt.time == pd.to_datetime('09:30').time(), 'Hour'] = 'OPEN'

# --- Detect triggers and goals for first day only (for debug) ---
first_date = intraday['Date'].iloc[0]
intraday_day = intraday[intraday['Date'] == first_date]
levels_row = daily[daily['Date'] == first_date].iloc[0]
level_values = {lvl: levels_row[col] for lvl, col in level_map.items()}

results = []

# Track which triggers have occurred
trigger_tracker = {}

for idx, row in intraday_day.iterrows():
    dt = row['Datetime']
    high = row['High']
    low = row['Low']
    open_ = row['Open']
    hour = row['Hour']

    for direction in ['Upside', 'Downside']:
        for trigger_level in level_keys:
            if direction == 'Upside' and trigger_level <= 0:
                continue
            if direction == 'Downside' and trigger_level >= 0:
                continue

            trigger_price = level_values[trigger_level]

            # Trigger logic
            if direction == 'Upside':
                triggered = False
                if hour == 'OPEN':
                    upper_bound = level_values.get(next((lvl for lvl in level_keys if lvl > trigger_level), None), float('inf'))
                    if trigger_price <= open_ < upper_bound:
                        triggered = True
                else:
                    if high >= trigger_price:
                        if hour == '0900':
                            open_candle = intraday_day.iloc[0]
                            open_open = open_candle['Open']
                            upper_bound = level_values.get(next((lvl for lvl in level_keys if lvl > trigger_level), None), float('inf'))
                            if trigger_price <= open_open < upper_bound:
                                continue  # Skip if already triggered at OPEN
                        triggered = True
            else:  # Downside
                triggered = False
                if hour == 'OPEN':
                    lower_bound = level_values.get(next((lvl for lvl in reversed(level_keys) if lvl < trigger_level), None), float('-inf'))
                    if lower_bound < open_ <= trigger_price:
                        triggered = True
                else:
                    if low <= trigger_price:
                        if hour == '0900':
                            open_candle = intraday_day.iloc[0]
                            open_open = open_candle['Open']
                            lower_bound = level_values.get(next((lvl for lvl in reversed(level_keys) if lvl < trigger_level), None), float('-inf'))
                            if lower_bound < open_open <= trigger_price:
                                continue
                        triggered = True

            if triggered:
                key = (trigger_level, hour, direction)
                if key not in trigger_tracker:
                    trigger_tracker[key] = {
                        'TriggerLevel': trigger_level,
                        'TriggerTime': hour,
                        'Direction': direction,
                        'Date': row['Date'],
                        'ScenarioType': 'Continuation',
                        'Retouched': False,
                    }

                # Check for goals
                for goal_level in level_keys:
                    if direction == 'Upside' and goal_level > trigger_level:
                        goal_price = level_values[goal_level]
                        goal_hit = high >= goal_price
                    elif direction == 'Downside' and goal_level < trigger_level:
                        goal_price = level_values[goal_level]
                        goal_hit = low <= goal_price
                    else:
                        continue

                    result = trigger_tracker[key].copy()
                    result.update({
                        'GoalLevel': goal_level,
                        'GoalHit': 'Yes' if goal_hit else 'No',
                        'GoalTime': hour if goal_hit else None
                    })

                    results.append(result)

                # --- RETRACEMENT LOGIC ---
                for goal_level in level_keys:
                    if direction == 'Upside' and goal_level < trigger_level:
                        goal_price = level_values[goal_level]
                        goal_hit = low <= goal_price
                        retouched = low < trigger_price
                    elif direction == 'Downside' and goal_level > trigger_level:
                        goal_price = level_values[goal_level]
                        goal_hit = high >= goal_price
                        retouched = high > trigger_price
                    else:
                        continue

                    result = {
                        'TriggerLevel': trigger_level,
                        'TriggerTime': hour,
                        'Direction': direction,
                        'Date': row['Date'],
                        'ScenarioType': 'Retracement',
                        'Retouched': retouched,
                        'GoalLevel': goal_level,
                        'GoalHit': 'Yes' if goal_hit else 'No',
                        'GoalTime': hour if goal_hit else None
                    }
                    results.append(result)

# --- Output to CSV ---
df_out = pd.DataFrame(results)
df_out.to_csv("/mnt/data/combined_trigger_goal_results.csv", index=False)
print("✅ Results saved to combined_trigger_goal_results.csv")
