import pandas as pd

# === CONFIG ===
mode = "retracement"  # change to "continuation" or "retracement"

daily_path = 'C:/New folder/SPXdailycandles.xlsx'
intraday_path = 'C:/New folder/SPX_10min.csv'

# === LOAD DATA ===
daily_raw = pd.read_excel(daily_path, header=4)
intraday = pd.read_csv(intraday_path)

daily_raw['Date'] = pd.to_datetime(daily_raw['Date']).dt.date
intraday['Datetime'] = pd.to_datetime(intraday['Datetime'])
intraday['Date'] = intraday['Datetime'].dt.date

merged = intraday.merge(daily_raw, on='Date', how='left')
print("✅ Data merged")

# === IDENTIFY LEVEL COLUMNS ===
level_cols = [col for col in merged.columns if isinstance(col, float) or isinstance(col, int)]
level_cols = sorted(level_cols, reverse=True)

results = []
grouped = merged.groupby(merged['Datetime'].dt.date)

for date, day_data in grouped:
    day_data = day_data.sort_values('Datetime').reset_index(drop=True)

    if day_data.empty or len(day_data) < 2:
        continue

    open_row = day_data.iloc[0]
    open_price = open_row['Open_x']
    high_series = day_data['High_x']
    low_series = day_data['Low_x']
    time_series = day_data['Time']

    for i, trigger in enumerate(level_cols):
        direction = ''
        trigger_idx = None
        trigger_time = None

        trigger_value = day_data.loc[0, trigger]
        next_level_value = day_data.loc[0, level_cols[i + 1]] if i + 1 < len(level_cols) else None

        if trigger > 0:
            direction = 'Upside'
            price_series = high_series
            trigger_hit = (price_series >= trigger_value)
        elif trigger < 0:
            direction = 'Downside'
            price_series = low_series
            trigger_hit = (price_series <= trigger_value)
        else:
            continue  # Skip 0 level

        if not trigger_hit.any():
            continue

        trigger_idx = trigger_hit.idxmax()
        row = day_data.loc[trigger_idx]

        trigger_time = 'OPEN' if trigger_idx == 0 and (
            (direction == 'Upside' and open_price >= trigger_value and (next_level_value is None or open_price < next_level_value)) or
            (direction == 'Downside' and open_price <= trigger_value and (next_level_value is None or open_price > next_level_value))
        ) else row['Time'][:2] + '00'

        for goal in level_cols:
            # --- Logic switch ---
            if mode == "continuation":
                is_goal_valid = (direction == 'Upside' and goal > trigger) or (direction == 'Downside' and goal < trigger)
            elif mode == "retracement":
                is_goal_valid = (direction == 'Upside' and goal < trigger) or (direction == 'Downside' and goal > trigger)
            else:
                raise ValueError("Invalid mode. Use 'continuation' or 'retracement'.")

            if not is_goal_valid:
                continue

            goal_value = day_data.loc[0, goal]
            goal_hit = False
            goal_hit_time = None

            for idx2 in range(trigger_idx + 1, len(day_data)):
                row2 = day_data.loc[idx2]
                price2 = row2['High_x'] if direction == 'Upside' else row2['Low_x']

                if (direction == 'Upside' and price2 >= goal_value) or (direction == 'Downside' and price2 <= goal_value):
                    # In retracement mode, exclude if same candle as trigger
                    if mode == "retracement" and idx2 == trigger_idx:
                        continue

                    goal_hit = True
                    goal_hit_time = row2['Time'][:2] + '00' if idx2 != 0 else 'OPEN'
                    break

            results.append({
                'Date': date,
                'Direction': direction,
                'TriggerLevel': trigger,
                'TriggerTime': trigger_time,
                'GoalLevel': goal,
                'GoalHit': 'Yes' if goal_hit else 'No',
                'GoalTime': goal_hit_time if goal_hit else None
            })

# === SAVE RESULTS ===
results_df = pd.DataFrame(results)
print(results_df.head(10))

filename = "atr_retracement_results.csv" if mode == "retracement" else "atr_trigger_goal_results.csv"
results_df.to_csv(filename, index=False)
print(f"✅ Results saved to {filename}")
