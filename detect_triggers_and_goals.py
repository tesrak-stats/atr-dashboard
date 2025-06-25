import pandas as pd

def detect_triggers_and_goals(daily, intraday):
    fib_levels = [1.000, 0.786, 0.618, 0.500, 0.382, 0.236,
                  -0.236, -0.382, -0.500, -0.618, -0.786, -1.000]

    results = []

    for i in range(1, len(daily)):
        date = daily.iloc[i]['Date']
        if date < pd.to_datetime("2014-01-02"):
            continue

        day_row = daily.iloc[i]
        prev_row = daily.iloc[i - 1]
        prev_close = prev_row['Close']

        level_map = {}
        for level in fib_levels:
            level_str = f"{level:.3f}".rstrip('0').rstrip('.')
            if level_str in day_row:
                level_map[level] = day_row[level_str]

        day_data = intraday[intraday['Date'] == date]
        if day_data.empty:
            continue

        day_data['Time'] = day_data['Datetime'].dt.strftime('%H%M')
        first_row = day_data.iloc[0]

        for direction in ['Upside', 'Downside']:
            trigger_hit = {}
            for level in fib_levels:
                if direction == 'Upside' and level <= 0:
                    continue
                if direction == 'Downside' and level >= 0:
                    continue
                level_val = level_map.get(level)
                if pd.isna(level_val):
                    continue

                for idx, row in day_data.iterrows():
                    if direction == 'Upside' and row['High'] >= level_val:
                        trig_time = '0000' if row.name == day_data.index[0] and row['Open'] >= level_val else row['Time']
                        trigger_hit[(direction, level)] = (trig_time, row)
                        break
                    if direction == 'Downside' and row['Low'] <= level_val:
                        trig_time = '0000' if row.name == day_data.index[0] and row['Open'] <= level_val else row['Time']
                        trigger_hit[(direction, level)] = (trig_time, row)
                        break

            for (dir_key, trig_level), (trig_time, trig_row) in trigger_hit.items():
                trig_idx = day_data.index.get_loc(trig_row.name)
                for goal_level in fib_levels:
                    if dir_key == 'Upside' and goal_level <= trig_level:
                        continue
                    if dir_key == 'Downside' and goal_level >= trig_level:
                        continue
                    goal_val = level_map.get(goal_level)
                    if pd.isna(goal_val):
                        continue

                    goal_time = None
                    for idx in day_data.index[trig_idx+1:]:
                        row = day_data.loc[idx]
                        if dir_key == 'Upside' and row['High'] >= goal_val:
                            goal_time = row['Time']
                            break
                        if dir_key == 'Downside' and row['Low'] <= goal_val:
                            goal_time = row['Time']
                            break

                    results.append({
                        'Date': date,
                        'Direction': dir_key,
                        'Trigger Level': trig_level,
                        'Goal Level': goal_level,
                        'Trigger Time': trig_time,
                        'Goal Time': goal_time
                    })

    return pd.DataFrame(results)

def main():
    daily = pd.read_csv("SPX_daily.csv")
    intraday = pd.read_csv("SPX_10min.csv")

    daily['Date'] = pd.to_datetime(daily['Date'])
    intraday['Datetime'] = pd.to_datetime(intraday['Datetime'])
    intraday['Date'] = intraday['Datetime'].dt.normalize()

    df = detect_triggers_and_goals(daily, intraday)
    df.to_csv("combined_trigger_goal_results.csv", index=False)

if __name__ == "__main__":
    main()
