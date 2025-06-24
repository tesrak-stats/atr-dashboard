# detect_triggers_and_goals.py
# v1.0.1 – Full logic for upside/downside, continuation/retracement, goal tracking, time filtering, retouch flag

import pandas as pd

# Load data
intraday_df = pd.read_csv("SPX_10min.csv")
daily_df = pd.read_excel("SPXdailycandles.xlsx", header=4)

# Format time columns
intraday_df["Datetime"] = pd.to_datetime(intraday_df["Datetime"])
intraday_df["Date"] = intraday_df["Datetime"].dt.date
intraday_df["Time"] = intraday_df["Datetime"].dt.strftime("%H%M")
intraday_df.loc[intraday_df["Time"] == "0930", "Time"] = "OPEN"  # Convert 0930 to OPEN

# Clean daily level labels
daily_df.columns = [str(c).replace("%", "").strip() for c in daily_df.columns]
daily_df["Date"] = pd.to_datetime(daily_df["Date"]).dt.date

# Identify level columns (excluding Date, ATR, etc.)
level_cols = [col for col in daily_df.columns if col not in ["Date", "ATR", "ATR Wilder", "Close"]]
level_map = {col: round(float(col) / 100, 3) for col in level_cols}

# Storage
results = []

# Main loop
for date in intraday_df["Date"].unique():
    try:
        day_intraday = intraday_df[intraday_df["Date"] == date]
        day_daily = daily_df[daily_df["Date"] == date]

        if day_intraday.empty or day_daily.empty:
            continue

        open_price = day_intraday.iloc[0]["Open"]
        levels = day_daily.iloc[0][level_cols]
        level_values = levels.to_dict()

        # Sort levels numerically
        sorted_levels = sorted(level_map.items(), key=lambda x: x[1])
        level_keys = [k for k, _ in sorted_levels]
        level_floats = [level_map[k] for k in level_keys]

        level_vals = [level_values[k] for k in level_keys]

        # Iterate for both directions
        for direction in ["Upside", "Downside"]:
            ascending = direction == "Upside"
            levels_to_use = list(zip(level_keys, level_vals)) if ascending else list(zip(level_keys, level_vals))[::-1]

            for i, (trigger_key, trigger_level) in enumerate(levels_to_use):
                # Get the next level (for open zone filtering)
                if i + 1 < len(levels_to_use):
                    next_level_val = levels_to_use[i + 1][1]
                else:
                    next_level_val = None

                # Determine trigger at OPEN
                if direction == "Upside":
                    open_trigger = (
                        open_price >= trigger_level and
                        (next_level_val is None or open_price < next_level_val)
                    )
                else:
                    open_trigger = (
                        open_price <= trigger_level and
                        (next_level_val is None or open_price > next_level_val)
                    )

                # Detect trigger
                trigger_found = False
                trigger_time = None

                for idx, row in day_intraday.iterrows():
                    high = row["High"]
                    low = row["Low"]
                    tlabel = row["Time"]

                    if tlabel == "OPEN" and open_trigger:
                        trigger_time = "OPEN"
                        trigger_price = open_price
                        trigger_found = True
                        break

                    if direction == "Upside" and high >= trigger_level:
                        if tlabel == "OPEN":
                            continue
                        trigger_time = tlabel
                        trigger_price = high
                        trigger_found = True
                        break

                    if direction == "Downside" and low <= trigger_level:
                        if tlabel == "OPEN":
                            continue
                        trigger_time = tlabel
                        trigger_price = low
                        trigger_found = True
                        break

                if not trigger_found:
                    continue

                # CONTINUATION GOALS
                for j in range(i + 1, len(levels_to_use)):
                    goal_key, goal_val = levels_to_use[j]
                    goal_hit = False
                    goal_time = ""
                    retouched = False

                    for _, row2 in day_intraday.iterrows():
                        t2 = row2["Time"]
                        if t2 < trigger_time or t2 == "OPEN":
                            continue

                        if t2 == trigger_time:
                            continue  # continuation allows same candle, already skipped OPEN

                        if direction == "Upside":
                            if row2["High"] >= goal_val:
                                goal_hit = True
                                goal_time = t2
                                break
                            if row2["Low"] < trigger_level:
                                retouched = True

                        else:  # Downside
                            if row2["Low"] <= goal_val:
                                goal_hit = True
                                goal_time = t2
                                break
                            if row2["High"] > trigger_level:
                                retouched = True

                    results.append({
                        "Date": date,
                        "Direction": direction,
                        "TriggerLevel": level_map[trigger_key],
                        "TriggerTime": trigger_time,
                        "GoalLevel": level_map[goal_key],
                        "GoalHit": "Yes" if goal_hit else "No",
                        "GoalTime": goal_time,
                        "GoalType": "Continuation",
                        "Retouched": "Yes" if retouched else "No"
                    })

                # RETRACEMENT GOALS
                for j in range(i + 1, len(levels_to_use)):
                    goal_key, goal_val = levels_to_use[j]
                    goal_hit = False
                    goal_time = ""
                    retouched = False

                    for _, row2 in day_intraday.iterrows():
                        t2 = row2["Time"]
                        if t2 <= trigger_time or t2 == "OPEN":
                            continue

                        if direction == "Upside":
                            if row2["Low"] <= goal_val:
                                goal_hit = True
                                goal_time = t2
                                break
                            if row2["Low"] < trigger_level:
                                retouched = True
                        else:
                            if row2["High"] >= goal_val:
                                goal_hit = True
                                goal_time = t2
                                break
                            if row2["High"] > trigger_level:
                                retouched = True

                    results.append({
                        "Date": date,
                        "Direction": direction,
                        "TriggerLevel": level_map[trigger_key],
                        "TriggerTime": trigger_time,
                        "GoalLevel": level_map[goal_key],
                        "GoalHit": "Yes" if goal_hit else "No",
                        "GoalTime": goal_time,
                        "GoalType": "Retracement",
                        "Retouched": "Yes" if retouched else "No"
                    })

    except Exception as e:
        print(f"❌ Error on {date}: {e}")
        continue

# Save output
output_df = pd.DataFrame(results)
output_df.to_csv("combined_trigger_goal_results.csv", index=False)
print("✅ Exported: combined_trigger_goal_results.csv")
