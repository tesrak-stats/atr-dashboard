# -----------------------------
# detect_triggers_and_goals.py
# Version: v1.0.0
# Last Updated: 2025-06-23
# Description:
#   - Processes SPX daily and intraday data to detect ATR-based trigger and goal levels
#   - Supports both continuation and retracement logic
#   - Flags trigger time, goal time, continuation, and future retest
#   - Normalizes result keys for downstream CSV export
#   - Logs missing fields if present
# -----------------------------

import pandas as pd

# --- Load data ---
daily_df = pd.read_excel("SPXdailycandles.xlsx", header=4)
intraday_df = pd.read_csv("SPX_10min.csv")

# --- Prepare intraday time columns ---
intraday_df['Datetime'] = pd.to_datetime(intraday_df['Datetime'])
intraday_df['Date'] = intraday_df['Datetime'].dt.date
intraday_df['Time'] = intraday_df['Datetime'].dt.strftime('%H%M')

# --- Extract ATR levels from daily ---
fib_labels = list(daily_df.columns[9:22])  # columns J through V
results = []

for idx, row in daily_df.iterrows():
    date = row['Date']
    atr_levels = {label: row[label] for label in fib_labels}
    previous_close = row['0.0%']

    day_data = intraday_df[intraday_df['Date'] == pd.to_datetime(date).date()]
    if day_data.empty:
        continue

    first_candle = day_data.iloc[0]
    open_price = first_candle['Open']
    
    # --- Detect Triggers ---
    for direction in ['Upside', 'Downside']:
        for i, level_label in enumerate(fib_labels):
            trigger_level = atr_levels[level_label]
            if pd.isna(trigger_level):
                continue

            if direction == 'Upside':
                condition_open = open_price >= trigger_level
                condition_intraday = day_data['High'] >= trigger_level
                next_level_idx = i + 1 if i + 1 < len(fib_labels) else None
            else:
                condition_open = open_price <= trigger_level
                condition_intraday = day_data['Low'] <= trigger_level
                next_level_idx = i + 1 if i + 1 < len(fib_labels) else None

            # --- Check if Open Trigger ---
            if condition_open and next_level_idx is not None:
                next_level = atr_levels[fib_labels[next_level_idx]]
                if direction == 'Upside' and open_price < next_level:
                    trigger_time = 'OPEN'
                elif direction == 'Downside' and open_price > next_level:
                    trigger_time = 'OPEN'
                else:
                    trigger_time = None
            else:
                # --- Find intraday trigger ---
                trigger_time = None
                for _, candle in day_data.iterrows():
                    high, low = candle['High'], candle['Low']
                    if direction == 'Upside' and high >= trigger_level:
                        trigger_time = candle['Time']
                        break
                    elif direction == 'Downside' and low <= trigger_level:
                        trigger_time = candle['Time']
                        break

            if not trigger_time:
                continue

            # --- Goal tracking ---
            for j in range(i + 1, len(fib_labels)):
                goal_level_label = fib_labels[j]
                goal_level = atr_levels[goal_level_label]
                goal_hit = False
                goal_time = None
                continuation = False
                retest_before_goal = False

                if pd.isna(goal_level):
                    continue

                # Skip same candle hits for retracements
                if trigger_time == 'OPEN':
                    trigger_idx = 0
                else:
                    trigger_idx = day_data.index[day_data['Time'] == trigger_time].tolist()
                    if not trigger_idx:
                        continue
                    trigger_idx = trigger_idx[0]

                candles_after = day_data.iloc[trigger_idx + 1:]

                for _, candle in candles_after.iterrows():
                    if direction == 'Upside' and candle['High'] >= goal_level:
                        goal_hit = True
                        goal_time = candle['Time']
                        continuation = True
                        break
                    elif direction == 'Downside' and candle['Low'] <= goal_level:
                        goal_hit = True
                        goal_time = candle['Time']
                        continuation = True
                        break

                # --- Result append ---
                results.append({
                    'Date': date,
                    'Direction': direction,
                    'TriggerLevel': float(level_label.strip('%')) / 100,
                    'TriggerTime': trigger_time,
                    'GoalLevel': float(goal_level_label.strip('%')) / 100,
                    'GoalHit': 'Yes' if goal_hit else 'No',
                    'GoalTime': goal_time,
                    'Continuation': continuation,
                    'RetestBeforeGoal': retest_before_goal
                })

# --- Normalize dictionary keys and log missing ---
required_keys = ['Date', 'Direction', 'TriggerLevel', 'TriggerTime', 'GoalLevel', 'GoalHit', 'GoalTime', 'Continuation', 'RetestBeforeGoal']
missing_fields_log = []

for idx, result in enumerate(results):
    missing = [k for k in required_keys if k not in result]
    if missing:
        missing_fields_log.append(f"Row {idx} is missing: {missing}")
    for k in required_keys:
        result.setdefault(k, None)

if missing_fields_log:
    with open("missing_fields_log.txt", "w") as f:
        for line in missing_fields_log:
            f.write(line + "\n")

# --- Save final output ---
df = pd.DataFrame(results)
df.to_csv("combined_trigger_goal_results.csv", index=False)
print("✅ Saved: combined_trigger_goal_results.csv")
