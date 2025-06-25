
import pandas as pd

def detect_triggers_and_goals(daily, intraday):
    fib_levels = [1.000, 0.786, 0.618, 0.500, 0.382,
                  -0.236, -0.382, -0.500, -0.618, -0.786, -1.000]

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

            level_map = {}
            for col in daily.columns[9:22]:
                try:
                    col_str = str(col).strip().replace('%', '')
                    float_label = float(col_str) / 100 if '%' in str(col) else float(col_str)
                    level_map[float_label] = day_row[col]
                except ValueError:
                    continue

            if direction == 'Upside':
                trigger_levels = [lvl for lvl in fib_levels if lvl > 0 and lvl in level_map]
                goal_levels = [lvl for lvl in fib_levels if lvl > 0 and lvl in level_map]
                trigger_check = lambda row, lvl: row['High'] >= level_map[lvl]
                goal_check = lambda row, lvl: row['High'] >= level_map[lvl]
                open_check = lambda row, lvl, next_lvl: (
                    level_map[lvl] <= row['Open'] < level_map[next_lvl]
                )
            else:
                trigger_levels = [lvl for lvl in fib_levels if lvl < 0 and lvl in level_map]
                goal_levels = [lvl for lvl in fib_levels if lvl < 0 and lvl in level_map]
                trigger_check = lambda row, lvl: row['Low'] <= level_map[lvl]
                goal_check = lambda row, lvl: row['Low'] <= level_map[lvl]
                open_check = lambda row, lvl, next_lvl: (
                    level_map[next_lvl] < row['Open'] <= level_map[lvl]
                )

            for lvl in trigger_levels:
                triggered = False
                trigger_time = None
                for idx, row in day_data.iterrows():
                    if idx == day_data.index[0]:
                        try:
                            next_lvl = trigger_levels[trigger_levels.index(lvl) + 1]
                        except IndexError:
                            next_lvl = lvl
                        if open_check(row, lvl, next_lvl):
                            trigger_time = '0000'
                            triggered = True
                            break
                    elif trigger_check(row, lvl):
                        trigger_time = row['Time'][:2] + '00'
                        triggered = True
                        break

                if not triggered:
                    continue

                for goal_lvl in goal_levels:
                    if (direction == 'Upside' and goal_lvl <= lvl) or (direction == 'Downside' and goal_lvl >= lvl):
                        continue

                    goal_hit = False
                    for idx, row in day_data.iterrows():
                        if goal_check(row, goal_lvl):
                            goal_hit = True
                            break

                    results.append({
                        'Date': date,
                        'Direction': direction,
                        'Trigger Level': lvl,
                        'Trigger Time': trigger_time,
                        'Goal Level': goal_lvl,
                        'Goal Hit': goal_hit,
                        'Trigger Price': level_map[lvl],
                        'Goal Price': level_map[goal_lvl],
                    })

    return pd.DataFrame(results)


def main():
    daily = pd.read_excel("SPXdailycandles.xlsx", header=4)
    daily['Date'] = pd.to_datetime(daily['Date'])

    intraday = pd.read_csv("SPX_10min.csv")
    intraday['Datetime'] = pd.to_datetime(intraday['Datetime'])
    intraday['Date'] = intraday['Datetime'].dt.date

    df = detect_triggers_and_goals(daily, intraday)
    df.to_csv("combined_trigger_goal_results.csv", index=False)
    print("✅ Output written to combined_trigger_goal_results.csv")


if __name__ == "__main__":
    main()
