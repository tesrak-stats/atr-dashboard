import pandas as pd

# --- File paths ---
daily_path = 'C:/New folder/SPXdailycandles.xlsx'
intraday_path = 'C:/New folder/SPX_10min.csv'

# --- Load files ---
daily_raw = pd.read_excel(daily_path, header=4)
intraday = pd.read_csv(intraday_path)

# --- Convert date columns ---
daily_raw['Date'] = pd.to_datetime(daily_raw['Date']).dt.date
intraday['Datetime'] = pd.to_datetime(intraday['Datetime'])
intraday['Date'] = intraday['Datetime'].dt.date

# --- Merge ---
merged = intraday.merge(daily_raw, on='Date', how='left')

# --- Safely detect level columns ---
level_cols = [col for col in merged.columns if isinstance(col, (float, int)) and -2.0 <= col <= 2.0]
level_cols = sorted(level_cols, reverse=True)
print("✔ Using level columns:", level_cols)

results = []

grouped = merged.groupby(merged['Datetime'].dt.date)

for date, day_data in grouped:
    day_data = day_data.sort_values('Datetime').reset_index(drop=True)
    open_row = day_data.iloc[0]
    open_price = open_row['Open_x']
    high_series = day_data['High_x']
    low_series = day_data['Low_x']
    time_series = day_data['Time']

    for i, trigger in enumerate(level_cols):
        trigger_value = day_data[trigger][0]
        direction = 'Upside' if trigger > 0 else 'Downside'
        price_series = high_series if direction == 'Upside' else low_series
        trigger_condition = price_series >= trigger_value if direction == 'Upside' else price_series <= trigger_value

        if not trigger_condition.any():
            continue

        trigger_idx = trigger_condition.idxmax()
        row = day_data.loc[trigger_idx]

        next_level = level_cols[i + 1] if i + 1 < len(level_cols) else None
        open_trigger_valid = (
            trigger_idx == 0 and
            (
                (direction == 'Upside' and open_price >= trigger_value and next_level is not None and open_price < day_data[next_level][0]) or
                (direction == 'Downside' and open_price <= trigger_value and next_level is not None and open_price > day_data[next_level][0])
            )
        )

        trigger_time = 'OPEN' if open_trigger_valid else row['Time'][:2] + '00'

        for goal in level_cols:
            if (direction == 'Upside' and goal > trigger) or (direction == 'Downside' and goal < trigger):
                goal_value = day_data[goal][0]
                goal_hit = False
                goal_time = None

                for idx2 in range(trigger_idx + 1, len(day_data)):
                    row2 = day_data.loc[idx2]
                    price2 = row2['High_x'] if direction == 'Upside' else row2['Low_x']
                    if (direction == 'Upside' and price2 >= goal_value) or (direction == 'Downside' and price2 <= goal_value):
                        goal_hit = True
                        goal_time = row2['Time'][:2] + '00' if idx2 != 0 else 'OPEN'
                        break

                results.append({
                    'Date': date,
                    'Direction': direction,
                    'TriggerLevel': trigger,
                    'TriggerTime': trigger_time,
                    'GoalLevel': goal,
                    'GoalHit': 'Yes' if goal_hit else 'No',
                    'GoalTime': goal_time if goal_hit else None
                })

# --- Save results ---
results_df = pd.DataFrame(results)
results_df.to_csv("combined_trigger_goal_results.csv", index=False)
print("✅ Combined results saved to combined_trigger_goal_results.csv")
