import pandas as pd

# --- Load input files ---
daily_df = pd.read_excel("SPXdailycandles.xlsx")
intraday_df = pd.read_csv("SPX_10min.csv")

# --- Define fib levels and time buckets ---
fib_levels = [1.0, 0.786, 0.618, 0.5, 0.382, 0.236, 0.0,
              -0.236, -0.382, -0.5, -0.618, -0.786, -1.0]
time_order = ["OPEN", "0900", "1000", "1100", "1200", "1300", "1400", "1500", "1600"]
time_rank = {label: i for i, label in enumerate(time_order)}

# --- Result collector ---
records = []

for _, row in daily_df.iterrows():
    date = row["Date"]
    atr = row["ATR"]
    levels = {float(k): row[str(k)] for k in fib_levels}

    day_candles = intraday_df[intraday_df["Date"] == date]
    if day_candles.empty:
        continue

    first_candle = day_candles.iloc[0]
    open_price = first_candle["Open"]
    high_price = first_candle["High"]
    low_price = first_candle["Low"]

    triggered = {}

    # --- Trigger detection ---
    for direction in ["Upside", "Downside"]:
        levels_to_check = [lvl for lvl in fib_levels if (lvl > 0 if direction == "Upside" else lvl < 0)]
        for i, lvl in enumerate(levels_to_check):
            lvl_val = levels[lvl]
            next_lvl_val = levels[levels_to_check[i + 1]] if i + 1 < len(levels_to_check) else None

            # Check for OPEN trigger
            triggered_open = False
            if direction == "Upside" and next_lvl_val and lvl_val <= open_price < next_lvl_val:
                triggered[lvl] = ("OPEN", direction)
                triggered_open = True
            elif direction == "Downside" and next_lvl_val and lvl_val >= open_price > next_lvl_val:
                triggered[lvl] = ("OPEN", direction)
                triggered_open = True

            # If not triggered at OPEN, check for high/low breach
            if not triggered_open:
                if direction == "Upside" and high_price >= lvl_val:
                    triggered[lvl] = ("0900", direction)
                elif direction == "Downside" and low_price <= lvl_val:
                    triggered[lvl] = ("0900", direction)

    # --- Goal checking ---
    for lvl, (trigger_time, direction) in triggered.items():
        lvl_val = levels[lvl]
        relevant_levels = [l for l in fib_levels if (l > lvl if direction == "Upside" else l < lvl)]

        # sort goals in directional order
        sorted_goals = sorted(relevant_levels, reverse=(direction == "Downside"))
        start_checking = False
        for _, candle in day_candles.iterrows():
            candle_time = candle["Time"]
            hour_label = "OPEN" if candle_time == "09:30:00" else candle_time[:2] + "00"

            if not start_checking:
                if hour_label == trigger_time:
                    start_checking = True
                continue

            goal_failed = False
            for goal in sorted_goals:
                goal_val = levels[goal]
                hit = False

                if direction == "Upside" and candle["High"] >= goal_val:
                    hit = True
                elif direction == "Downside" and candle["Low"] <= goal_val:
                    hit = True

                if hit:
                    records.append({
                        "Date": date,
                        "Direction": direction,
                        "TriggerLevel": lvl,
                        "TriggerTime": trigger_time,
                        "GoalLevel": goal,
                        "GoalHit": "Yes",
                        "GoalTime": hour_label
                    })
                else:
                    # mark first failure and break directional goal search for this hour
                    records.append({
                        "Date": date,
                        "Direction": direction,
                        "TriggerLevel": lvl,
                        "TriggerTime": trigger_time,
                        "GoalLevel": goal,
                        "GoalHit": "No",
                        "GoalTime": hour_label
                    })
                    goal_failed = True
                    break  # stop searching further levels in this direction for this hour

            if goal_failed:
                break  # stop processing candles until next hour

# --- Save results ---
result_df = pd.DataFrame(records)
result_df.to_csv("combined_trigger_goal_results.csv", index=False)
print("✅ combined_trigger_goal_results.csv saved.")