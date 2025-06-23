import pandas as pd
import numpy as np
from datetime import datetime
import os

# --- Load input files ---
daily = pd.read_excel("SPXdailycandles.xlsx", header=4)  # Row 5 = header (0-indexed as 4)
intraday = pd.read_csv("SPX_10min.csv", parse_dates=['Datetime'])

# --- Ensure proper formatting ---
daily['Date'] = pd.to_datetime(daily['Date'])
intraday['Date'] = intraday['Datetime'].dt.date
intraday['Time'] = intraday['Datetime'].dt.time
intraday['HourBlock'] = intraday['Datetime'].dt.strftime('%H00')

# --- Define Fibonacci levels (must match columns J–V in Excel) ---
fib_levels = [1, 0.786, 0.618, 0.5, 0.382, 0.236, 0, -0.236, -0.382, -0.5, -0.618, -0.786, -1]
level_cols = [str(level) for level in fib_levels]

# --- Container for all results ---
all_results = []

# --- Main loop ---
for idx, row in daily.iterrows():
    try:
        date = row['Date'].date()
        day_intraday = intraday[intraday['Date'] == date].copy()
        if day_intraday.empty:
            continue

        for direction in ['Upside', 'Downside']:
            for trigger_level in fib_levels:
                trigger_price = row[str(trigger_level)]
                if pd.isna(trigger_price):
                    continue

                # Trigger logic
                if direction == 'Upside':
                    trigger_candles = day_intraday[
                        (day_intraday['High'] >= trigger_price)
                    ]
                else:
                    trigger_candles = day_intraday[
                        (day_intraday['Low'] <= trigger_price)
                    ]

                if trigger_candles.empty:
                    continue

                trigger_idx = trigger_candles.index[0]
                trigger_candle = day_intraday.loc[trigger_idx]
                trigger_time = trigger_candle['HourBlock']

                # OPEN logic
                open_candle = day_intraday.iloc[0]
                if direction == 'Upside':
                    if open_candle['Open'] >= trigger_price and open_candle['High'] >= trigger_price:
                        trigger_time = 'OPEN'
                else:
                    if open_candle['Open'] <= trigger_price and open_candle['Low'] <= trigger_price:
                        trigger_time = 'OPEN'

                # Collect goal results for levels in **both directions**
                for goal_level in fib_levels:
                    if direction == 'Upside' and goal_level <= trigger_level:
                        continue
                    if direction == 'Downside' and goal_level >= trigger_level:
                        continue

                    goal_price = row[str(goal_level)]
                    if pd.isna(goal_price):
                        continue

                    goal_hit = 'No'
                    goal_time = None

                    future_intraday = day_intraday.loc[trigger_idx + 1:]
                    for i, goal_row in future_intraday.iterrows():
                        if direction == 'Upside' and goal_row['High'] >= goal_price:
                            goal_hit = 'Yes'
                            goal_time = goal_row['HourBlock']
                            break
                        elif direction == 'Downside' and goal_row['Low'] <= goal_price:
                            goal_hit = 'Yes'
                            goal_time = goal_row['HourBlock']
                            break

                    result = {
                        'Date': date,
                        'Direction': direction,
                        'TriggerLevel': trigger_level,
                        'TriggerTime': trigger_time,
                        'GoalLevel': goal_level,
                        'GoalHit': goal_hit,
                        'GoalTime': goal_time
                    }
                    all_results.append(result)
    except Exception as e:
        print(f"Error on {row['Date']}: {e}")

# --- Save results ---
output = pd.DataFrame(all_results)
output.to_csv("combined_trigger_goal_results.csv", index=False)
print("✅ Done. Saved to combined_trigger_goal_results.csv")
