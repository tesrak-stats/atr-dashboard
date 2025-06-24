# Version 7 – Full logic, cleanly formatted, no placeholders

import pandas as pd

def main():
    # --- Load input files ---
    daily_df = pd.read_excel("SPXdailycandles.xlsx", header=4)
    intraday_df = pd.read_csv("SPX_10min.csv")

    # --- Preprocess dates ---
    daily_df["Date"] = pd.to_datetime(daily_df["Date"])
    intraday_df["Datetime"] = pd.to_datetime(intraday_df["Datetime"])
    intraday_df["Date"] = intraday_df["Datetime"].dt.date
    intraday_df["Hour"] = intraday_df["Datetime"].dt.strftime('%H00')

    # --- Extract Fibonacci level columns ---
    fib_columns = [col for col in daily_df.columns if "%" in str(col)]
    level_map = {}
    for col in fib_columns:
        try:
            percent = float(str(col).replace("%", "")) / 100
            level_map[percent] = col
        except:
            continue
    sorted_levels = sorted(level_map.keys())

    all_results = []

    for i, day_row in daily_df.iterrows():
        date = day_row["Date"].date()
        day_data = intraday_df[intraday_df["Date"] == date].copy()
        if day_data.empty:
            continue

        levels = {lvl: day_row[level_map[lvl]] for lvl in sorted_levels}
        prev_close = day_row["Close"]

        for direction in ["upside", "downside"]:
            for trigger_level in sorted_levels:
                trigger_price = levels[trigger_level]

                if direction == "upside":
                    level_range = [lvl for lvl in sorted_levels if lvl > trigger_level]
                    condition = (day_data["High"] >= trigger_price)
                else:
                    level_range = [lvl for lvl in sorted_levels if lvl < trigger_level]
                    condition = (day_data["Low"] <= trigger_price)

                triggered = day_data[condition]
                if triggered.empty:
                    continue

                first_trigger_idx = triggered.index[0]
                trigger_row = triggered.loc[first_trigger_idx]
                trigger_time = trigger_row["Datetime"]
                trigger_hour_label = "OPEN" if trigger_row.name == day_data.index[0] and \
                    ((trigger_row["Open"] >= trigger_price and direction == "upside") or
                     (trigger_row["Open"] <= trigger_price and direction == "downside")) else \
                    trigger_time.strftime('%H00')

                post_trigger = day_data.loc[first_trigger_idx:]
                for goal_level in level_range:
                    goal_price = levels[goal_level]
                    if direction == "upside":
                        hit_condition = post_trigger["High"] >= goal_price
                    else:
                        hit_condition = post_trigger["Low"] <= goal_price

                    hit_rows = post_trigger[hit_condition]
                    goal_hit = "Yes" if not hit_rows.empty else "No"
                    goal_hour = hit_rows["Datetime"].iloc[0].strftime('%H00') if goal_hit == "Yes" else ""

                    all_results.append({
                        "Date": date,
                        "Direction": direction,
                        "TriggerLevel": trigger_level,
                        "TriggerTime": trigger_hour_label,
                        "GoalLevel": goal_level,
                        "GoalHit": goal_hit,
                        "GoalTime": goal_hour
                    })

    # --- Save output ---
    results_df = pd.DataFrame(all_results)
    results_df.to_csv("combined_trigger_goal_results.csv", index=False)
    print("✅ Saved: combined_trigger_goal_results.csv")
    return results_df


if __name__ == "__main__":
    main()
