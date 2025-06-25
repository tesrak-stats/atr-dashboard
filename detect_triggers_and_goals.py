
import pandas as pd

def detect_triggers_and_goals(daily, intraday):
    fib_levels = [1.000, 0.786, 0.618, 0.500, 0.382, 0.236,
                  -0.236, -0.382, -0.500, -0.618, -0.786, -1.000]
    results = []

    for date in daily['Date'].unique():
        if date < pd.to_datetime("2014-01-02").date():
            continue

        day_row = daily[daily['Date'] == date]
        if day_row.empty:
            continue
        day_row = day_row.iloc[0]

        prev_idx = daily[daily['Date'] == date].index[0] - 1
        if prev_idx < 0:
            continue
        prev_close = daily.iloc[prev_idx]['Close']

        level_map = {level: day_row.get(level, None) for level in fib_levels}

        day_data = intraday[intraday['Date'] == date].copy()
        if day_data.empty:
            continue

        day_data['Time'] = day_data['Datetime'].dt.strftime('%H%M')
        day_data.loc[day_data['Time'] == '0930', 'Time'] = 'OPEN'
        time_buckets = ['OPEN'] + [f'{{hour:02}}00' for hour in range(9, 17)]

        for direction in ['Upside', 'Downside']:
            for trigger_level in fib_levels:
                if direction == 'Upside' and trigger_level <= 0:
                    continue
                if direction == 'Downside' and trigger_level >= 0:
                    continue

                trigger_value = level_map.get(trigger_level)
                if pd.isna(trigger_value):
                    continue

                triggered = False
                for t in day_data.itertuples():
                    price = t.High if direction == 'Upside' else t.Low
                    if (direction == 'Upside' and price >= trigger_value) or                            (direction == 'Downside' and price <= trigger_value):
                        trigger_time = t.Time
                        if trigger_time == 'OPEN' and not (
                            (direction == 'Upside' and t.Open >= trigger_value and t.Open < t.High) or
                            (direction == 'Downside' and t.Open <= trigger_value and t.Open > t.Low)):
                            continue
                        triggered = True
                        break
                if not triggered:
                    continue

                for goal_level in fib_levels:
                    if direction == 'Upside' and goal_level <= trigger_level:
                        continue
                    if direction == 'Downside' and goal_level >= trigger_level:
                        continue

                    goal_value = level_map.get(goal_level)
                    if pd.isna(goal_value):
                        continue

                    hit = False
                    for t in day_data.itertuples():
                        price = t.High if direction == 'Upside' else t.Low
                        if (direction == 'Upside' and price >= goal_value) or                                (direction == 'Downside' and price <= goal_value):
                            hit = True
                            break

                    results.append({
                        'Date': date,
                        'Direction': direction,
                        'Trigger_Level': trigger_level,
                        'Goal_Level': goal_level,
                        'Hit': hit,
                        'Trigger_Time': trigger_time
                    })
    return pd.DataFrame(results)

def main():
    daily = pd.read_csv("SPXdailycandles.csv")
    intraday = pd.read_csv("SPX_10min.csv")
    daily['Date'] = pd.to_datetime(daily['Date']).dt.date
    intraday['Datetime'] = pd.to_datetime(intraday['Datetime'])
    intraday['Date'] = intraday['Datetime'].dt.date

    df = detect_triggers_and_goals(daily, intraday)
    df.to_csv("combined_trigger_goal_results.csv", index=False)
    print("✅ Trigger/goal results written to combined_trigger_goal_results.csv")

if __name__ == "__main__":
    main()
