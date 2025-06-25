
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
            prev_close = prev_row['0']

            # Build level map
            level_map = {}
            for level in fib_levels:
                level_str = f"{level:.3f}".rstrip("0").rstrip(".")
                if level_str in day_row:
                    level_map[level] = day_row[level_str]

            for level, trigger_price in level_map.items():
                if direction == 'Upside':
                    if day_data.iloc[0]['Open'] >= trigger_price and day_data.iloc[0]['Open'] < level_map.get(level + 0.236, float('inf')):
                        trigger_hour = 'OPEN'
                    else:
                        triggered = day_data[day_data['High'] >= trigger_price]
                        if triggered.empty:
                            continue
                        trigger_time = triggered.iloc[0]['Datetime']
                        trigger_hour = 'OPEN' if (trigger_time == day_data.iloc[0]['Datetime'] and day_data.iloc[0]['Open'] >= trigger_price) else trigger_time.strftime('%H00')
                    for goal_level, goal_price in level_map.items():
                        if goal_level > level:
                            goal_hit = day_data[day_data['High'] >= goal_price]
                            result = {
                                'Date': date,
                                'Direction': direction,
                                'TriggerLevel': level,
                                'GoalLevel': goal_level,
                                'Triggered': True,
                                'GoalHit': not goal_hit.empty,
                                'TriggerHour': trigger_hour,
                            }
                            results.append(result)
                elif direction == 'Downside':
                    if day_data.iloc[0]['Open'] <= trigger_price and day_data.iloc[0]['Open'] > level_map.get(level - 0.236, float('-inf')):
                        trigger_hour = 'OPEN'
                    else:
                        triggered = day_data[day_data['Low'] <= trigger_price]
                        if triggered.empty:
                            continue
                        trigger_time = triggered.iloc[0]['Datetime']
                        trigger_hour = 'OPEN' if (trigger_time == day_data.iloc[0]['Datetime'] and day_data.iloc[0]['Open'] <= trigger_price) else trigger_time.strftime('%H00')
                    for goal_level, goal_price in level_map.items():
                        if goal_level < level:
                            goal_hit = day_data[day_data['Low'] <= goal_price]
                            result = {
                                'Date': date,
                                'Direction': direction,
                                'TriggerLevel': level,
                                'GoalLevel': goal_level,
                                'Triggered': True,
                                'GoalHit': not goal_hit.empty,
                                'TriggerHour': trigger_hour,
                            }
                            results.append(result)

    return pd.DataFrame(results)


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
