import pandas as pd

# === Robust Daily Excel Load (header in row 5) ===
daily = pd.read_excel("SPXdailycandles.xlsx", header=None)
daily = daily.iloc[4:].reset_index(drop=True)     # Start from row 5
daily.columns = daily.iloc[0]                     # Set row 5 as header
daily = daily[1:]                                 # Remove header row from data
daily.columns = daily.columns.str.strip()

# === Build Fibonacci Level Map ===
level_map = {}
for col in daily.columns[9:22]:  # Columns J through V
    try:
        if isinstance(col, str) and '%' in col:
            float_label = float(col.strip('%')) / 100
        else:
            float_label = float(col)
        level_map[float_label] = col
    except ValueError:
        continue

fib_levels = sorted(level_map.keys(), reverse=True)

# === Load Intraday and Prepare ===
intraday = pd.read_csv("SPX_10min.csv")
intraday['Datetime'] = pd.to_datetime(intraday['Datetime'])
intraday['Date'] = intraday['Datetime'].dt.date
intraday['Time'] = intraday['Datetime'].dt.strftime('%H%M')

results = []

for idx, row in daily.iterrows():
    date = row["Date"]
    levels = {level: row[col] for level, col in level_map.items()}
    close_prev = levels.get(0)

    intraday_day = intraday[intraday["Date"] == date]
    if intraday_day.empty:
        continue

    open_price = intraday_day.iloc[0]["Open"]

    for direction in ["Upside", "Downside"]:
        sorted_levels = [lvl for lvl in fib_levels if (lvl > 0 if direction == "Upside" else lvl < 0)]
        check_col = "High" if direction == "Upside" else "Low"
        open_check = (
            lambda lvl, nxt: open_price >= levels[lvl] and open_price < levels[nxt]
            if direction == "Upside"
            else open_price <= levels[lvl] and open_price > levels[nxt]
        )
        intraday_check = (
            lambda bar, lvl: bar["High"] >= levels[lvl]
            if direction == "Upside"
            else bar["Low"] <= levels[lvl]
        )

        for i, trigger_level in enumerate(sorted_levels):
            if trigger_level not in levels:
                continue
            next_level = sorted_levels[i + 1] if i + 1 < len(sorted_levels) else (1.0 if direction == "Upside" else -1.0)
            if next_level not in levels:
                continue

            if open_check(trigger_level, next_level):
                trigger_time = "OPEN"
            else:
                triggered = False
                trigger_time = None
                for _, bar in intraday_day.iterrows():
                    if intraday_check(bar, trigger_level):
                        if bar["Time"] == "0900" and open_check(trigger_level, next_level):
                            trigger_time = "OPEN"
                        else:
                            trigger_time = bar["Time"]
                        triggered = True
                        break
                if not triggered:
                    continue

            filtered = intraday_day[intraday_day["Time"] >= "0900"]
            for j, goal_level in enumerate(sorted_levels):
                if (direction == "Upside" and goal_level <= trigger_level) or (direction == "Downside" and goal_level >= trigger_level):
                    continue
                goal_hit = False
                for _, bar in filtered.iterrows():
                    hour = bar["Time"]
                    if intraday_check(bar, goal_level):
                        if trigger_time == "OPEN" and open_check(trigger_level, goal_level):
                            continue
                        goal_hit = True
                        results.append({
                            "Date": date,
                            "Direction": direction,
                            "TriggerLevel": trigger_level,
                            "TriggerTime": trigger_time,
                            "GoalLevel": goal_level,
                            "GoalHit": "Yes",
                            "GoalTime": hour
                        })
                    else:
                        results.append({
                            "Date": date,
                            "Direction": direction,
                            "TriggerLevel": trigger_level,
                            "TriggerTime": trigger_time,
                            "GoalLevel": goal_level,
                            "GoalHit": "No",
                            "GoalTime": hour
                        })
                        break

# Export result
pd.DataFrame(results).to_csv("combined_trigger_goal_results.csv", index=False)
print("✅ Results saved to combined_trigger_goal_results.csv")
