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
    triggered_at_open = set()

    for direction in ["Upside", "Downside"]:
        for i, lvl in enumerate(fib_levels):
            lvl_val = levels[lvl]

            # Identify next level in the same direction
            if direction == "Upside":
                next_lvls = [l for l in fib_levels if l > lvl]
            else:
                next_lvls = [l for l in fib_levels if l < lvl]
            next_lvl = next_lvls[0] if next_lvls else None
            next_lvl_val = levels[next_lvl] if next_lvl else None

            is_open_trigger = False
            if next_lvl_val is not None:
                if direction == "Upside" and lvl_val <= open_price < next_lvl_val:
                    is_open_trigger = True
                elif direction == "Downside" and lvl_val >= open_price > next_lvl_val:
                    is_open_trigger = True

            if is_open_trigger:
                triggered[(lvl, direction)] = "OPEN"
                triggered_at_open.add((lvl, direction))
                continue

            # If not OPEN, check 0900 (high/low breach) but not if already triggered
            if (lvl, direction) in triggered_at_open:
                continue

            if direction == "Upside" and high_price >= lvl_val:
                triggered[(lvl, direction)] = "0900"
            elif direction == "Downside" and low_price <= lvl_val:
                triggered[(lvl, direction)] = "0900"

    # --- Goal evaluation ---
    for (trigger_lvl, direction), trigger_time in triggered.items():
        trigger_val = levels[trigger_lvl]

        goal_levels = [lvl for lvl in fib_levels if lvl != trigger_lvl]
        sorted_goals = sorted(goal_levels, key=lambda x: fib_levels.index(x))  # maintain defined order

        start_checking = False
        for _, candle in day_candles.iterrows():
            candle_time = candle["Time"]
            hour_label = "OPEN" if candle_time == "09:30:00" else candle_time[:2] + "00"

            if not start_checking:
                if hour_label == trigger_time:
                    start_checking = True
                continue

            goals_this_hour = set()
            for goal in sorted_goals:
                goal_val = levels[goal]
                hit = False

                if goal_val > trigger_val and candle["High"] >= goal_val:
                    hit = True
                elif goal_val < trigger_val and candle["Low"] <= goal_val:
                    hit = True

                if hit and (goal, hour_label) not in goals_this_hour:
                    records.append({
                        "Date": date,
                        "Direction": direction,
                        "TriggerLevel": trigger_lvl,
                        "TriggerTime": trigger_time,
                        "GoalLevel": goal,
                        "GoalHit": "Yes",
                        "GoalTime": hour_label
                    })
                    goals_this_hour.add((goal, hour_label))

# --- Save results ---
result_df = pd.DataFrame(records)
result_df.to_csv("combined_trigger_goal_results.csv", index=False)
print("✅ combined_trigger_goal_results.csv saved.")