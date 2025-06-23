
import pandas as pd
import numpy as np

# Load daily ATR levels
daily = pd.read_excel("SPXdailycandles.xlsx")
intraday = pd.read_csv("SPX_10min.csv", parse_dates=["Datetime"])

# Preprocess intraday
intraday["Date"] = intraday["Datetime"].dt.date
intraday["TimeBlock"] = intraday["Datetime"].dt.strftime("%H00")
intraday.loc[intraday["TimeBlock"] == "0930", "TimeBlock"] = "OPEN"

# Extract level values from row 5 of SPXdailycandles.xlsx
level_row = daily.iloc[4]
fib_levels = level_row.loc["J5":"V5"].values.astype(float)
fib_labels = [1.0, 0.786, 0.618, 0.5, 0.382, 0.236, 0.0, -0.236, -0.382, -0.5, -0.618, -0.786, -1.0]
fib_map = dict(zip(fib_labels, fib_levels))

results = []

# Loop through each day
for date in intraday["Date"].unique():
    try:
        day_intraday = intraday[intraday["Date"] == date]
        day_daily = daily[daily["Date"] == pd.to_datetime(date)]
        if day_daily.empty:
            continue

        for direction in ["Upside", "Downside"]:
            for trigger_level in fib_labels:
                if direction == "Upside":
                    trigger_price = fib_map.get(trigger_level)
                    level_col = "High"
                    hit = day_intraday[day_intraday[level_col] >= trigger_price]
                else:
                    trigger_price = fib_map.get(trigger_level)
                    level_col = "Low"
                    hit = day_intraday[day_intraday[level_col] <= trigger_price]

                if hit.empty:
                    continue

                first_hit = hit.iloc[0]
                trigger_time = first_hit["TimeBlock"]
                trigger_candle_index = day_intraday.index.get_loc(first_hit.name)

                for goal_level in fib_labels:
                    if goal_level == trigger_level:
                        continue

                    is_continuation = (goal_level > trigger_level) if direction == "Upside" else (goal_level < trigger_level)
                    goal_price = fib_map.get(goal_level)

                    if goal_price is None:
                        continue

                    goal_hit = False
                    goal_time = None

                    # Define which prices to scan and how to scan them
                    if is_continuation:
                        scan = day_intraday.iloc[trigger_candle_index:]
                        price_col = "High" if direction == "Upside" else "Low"
                        goal_hit_rows = scan[scan[price_col] >= goal_price] if direction == "Upside" else scan[scan[price_col] <= goal_price]
                        if not goal_hit_rows.empty:
                            goal_time = goal_hit_rows.iloc[0]["TimeBlock"]
                            goal_hit = True
                    else:
                        scan = day_intraday.iloc[trigger_candle_index + 1:]
                        price_col = "Low" if direction == "Upside" else "High"
                        goal_hit_rows = scan[scan[price_col] <= goal_price] if direction == "Upside" else scan[scan[price_col] >= goal_price]
                        if not goal_hit_rows.empty:
                            goal_time = goal_hit_rows.iloc[0]["TimeBlock"]
                            goal_hit = True

                    results.append({
                        "Date": date,
                        "Direction": direction,
                        "TriggerLevel": trigger_level,
                        "TriggerTime": trigger_time,
                        "GoalLevel": goal_level,
                        "GoalTime": goal_time if goal_time else "",
                        "GoalHit": "Yes" if goal_hit else "No"
                    })
    except Exception as e:
        print(f"Error on {date}: {e}")
        continue

# Save result
df_out = pd.DataFrame(results)
df_out.to_csv("combined_trigger_goal_results.csv", index=False)
print("✅ combined_trigger_goal_results.csv generated.")
