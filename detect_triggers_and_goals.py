import pandas as pd

def detect_triggers_and_goals(daily, intraday):
    fib_levels = [1.000, 0.786, 0.618, 0.500, 0.382,
                  0.236, -0.236, -0.382, -0.500, -0.618, -0.786, -1.000]

    results = []

    for direction in ['Upside', 'Downside']:
        for date in intraday['Date'].unique():
            if pd.to_datetime(date) < pd.Timestamp("2014-01-02"):
                continue

            day_data = intraday[intraday['Date'] == date].copy()
            daily_row = daily[daily['Date'] == pd.to_datetime(date)]
            if daily_row.empty:
                continue
            day_row = daily_row.iloc[0]

            prev_row = daily[daily['Date'] < pd.to_datetime(date)].iloc[-1]
            prev_close = prev_row['Close']

            # === Build level map from fib levels ===
            level_map = {}
            for level in fib_levels:
                level_str = f"{level:.3f}".rstrip("0").rstrip(".")
                if level_str in day_row:
                    level_map[level] = day_row[level_str]

            # === Trigger/Goal Logic ===
            for level, trigger_price in level_map.items():
                if direction == 'Upside':
                    triggered = day_data[day_data['High'] >= trigger_price]
                else:
                    triggered = day_data[day_data['Low'] <= trigger_price]

                if triggered.empty:
                    continue

                first_trigger = triggered.iloc[0]
                trigger_time = first_trigger['Datetime']
                if first_trigger['Datetime'].hour == 6 and                    ((direction == 'Upside' and first_trigger['Open'] >= trigger_price and
                     first_trigger['Open'] < day_data['High'].iloc[0]) or
                    (direction == 'Downside' and first_trigger['Open'] <= trigger_price and
                     first_trigger['Open'] > day_data['Low'].iloc[0])):
                    trigger_hour_label = 'OPEN'
                else:
                    trigger_hour_label = trigger_time.strftime('%H00')

                goal_results = {}
                goal_levels = sorted(
                    [lvl for lvl in fib_levels if
                     (lvl > level if direction == 'Upside' else lvl < level)],
                    reverse=(direction == 'Downside'))

                for goal in goal_levels:
                    goal_price = level_map.get(goal)
                    if goal_price is None:
                        continue

                    if direction == 'Upside':
                        hit = day_data[day_data['High'] >= goal_price]
                    else:
                        hit = day_data[day_data['Low'] <= goal_price]

                    if hit.empty:
                        goal_results[f"{goal:.3f}"] = 'Fail'
                    else:
                        hit_time = hit.iloc[0]['Datetime']
                        if hit_time.hour == 6 and hit.iloc[0]['Open'] == goal_price:
                            goal_results[f"{goal:.3f}"] = 'Fail'
                        else:
                            goal_results[f"{goal:.3f}"] = hit_time.strftime('%H00')

                results.append({
                    'Date': date,
                    'Direction': direction,
                    'TriggerLevel': level,
                    'TriggerHour': trigger_hour_label,
                    **goal_results
                })

    return pd.DataFrame(results)

# === Streamlit-compatible main() wrapper ===
def main():
    daily = pd.read_excel("SPXdailycandles.xlsx", header=4)
    daily['Date'] = pd.to_datetime(daily['Date'])

    intraday = pd.read_csv("SPX_10min.csv")
    intraday['Datetime'] = pd.to_datetime(intraday['Datetime'])
    intraday['Date'] = intraday['Datetime'].dt.date

    df = detect_triggers_and_goals(daily, intraday)
    df.to_csv("combined_trigger_goal_results.csv", index=False)

if __name__ == "__main__":
    main()
