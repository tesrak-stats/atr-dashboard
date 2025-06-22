import pandas as pd

# --- Load Data ---
daily = pd.read_excel("SPXdailycandles.xlsx", header=4)
intraday = pd.read_csv("SPX_10min.csv")

# --- Preprocess ---
daily['Date'] = pd.to_datetime(daily['Date']).dt.date
intraday['Datetime'] = pd.to_datetime(intraday['Datetime'])
intraday['Date'] = intraday['Datetime'].dt.date
intraday['Time'] = intraday['Datetime'].dt.strftime('%H%M')

# Merge daily ATR levels into intraday data
merged = intraday.merge(daily, on='Date', how='left')

# Extract sorted level columns
level_cols = [col for col in daily.columns if isinstance(col, float) or isinstance(col, int)]
level_cols = sorted(level_cols)

results = []

# --- Iterate through each trading day ---
for date, day_data in merged.groupby('Date'):
    day_data = day_data.sort_values('Datetime').reset_index(drop=True)
    open_row = day_data.iloc[0]
    open_price = open_row['Open_x']

    for direction in ["Upside", "Downside"]:
        is_up = direction == "Upside"

        for i, trigger in enumerate(level_cols):
            trigger_value = open_row[trigger]
            open_trigger = False

            # --- OPEN logic ---
            if is_up and i + 1 < len(level_cols):
                next_up = level_cols[i + 1]
                if open_price >= trigger_value and open_price < open_row[next_up]:
                    open_trigger = True
            elif not is_up and i - 1 >= 0:
                next_down = level_cols[i - 1]
                if open_price <= trigger_value and open_price > open_row[next_down]:
                    open_trigger = True

            trigger_idx = None
            trigger_time = None

            if open_trigger:
                trigger_idx = 0
                trigger_time = "OPEN"
            else:
                for idx, row in day_data.iterrows():
                    price = row['High_x'] if is_up else row['Low_x']
                    if (is_up and price >= row[trigger]) or (not is_up and price <= row[trigger]):
                        trigger_idx = idx
                        trigger_time = "0900" if idx == 0 else row['Time'][:2] + "00"
                        break

            if trigger_idx is None:
                continue

            # --- Evaluate all other goal levels ---
            for goal in level_cols:
                if goal == trigger:
                    continue

                is_continuation = (goal > trigger if is_up else goal < trigger)
                is_retracement = not is_continuation

                goal_hit = False
                goal_time = None

                for idx2 in range(trigger_idx, len(day_data)):
                    row2 = day_data.iloc[idx2]
                    price2 = row2['High_x'] if is_up else row2['Low_x']
                    goal_value = day_data.iloc[0][goal]

                    if (is_up and price2 >= goal_value) or (not is_up and price2 <= goal_value):
                        if idx2 == trigger_idx and is_retracement:
                            continue  # skip same-candle retracements
                        goal_hit = True
                        goal_time = row2['Time'][:2] + "00" if idx2 != 0 else "OPEN"
                        break

                results.append({
                    "Date": date,
                    "Direction": direction,
                    "TriggerLevel": trigger,
                    "TriggerTime": trigger_time,
                    "GoalLevel": goal,
                    "GoalHit": "Yes" if goal_hit else "No",
                    "GoalTime": goal_time if goal_hit else None
                })

# --- Save output ---
results_df = pd.DataFrame(results)
results_df.to_csv("combined_trigger_goal_results.csv", index=False)
print("✅ Saved to combined_trigger_goal_results.csv")
