import pandas as pd

def detect_triggers_and_goals(daily, intraday):
    fib_levels = [1.000, 0.786, 0.618, 0.500, 0.382, 0.236,
                 -0.236, -0.382, -0.500, -0.618, -0.786, -1.000]

    results = []

    for direction in ['Upside', 'Downside']:
        for date in intraday['Date'].unique():
            if date < pd.Timestamp("2014-01-02").date():
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

            # === Trigger/goal detection logic ===
            for goal_level, goal_price in level_map.items():
                results.append({
                    'Date': date,
                    'Direction': direction,
                    'Level': goal_level,
                    'GoalPrice': goal_price,
                    'Hit': False  # placeholder, actual logic should update this
                })

    return pd.DataFrame(results)


def main():
    # Load daily data from Excel (header row is on Excel row 5 → header=4)
    daily = pd.read_excel("SPXdailycandles.xlsx", header=4)
    daily['Date'] = pd.to_datetime(daily['Date'])

    # Load intraday data and normalize date column
    intraday = pd.read_csv("SPX_10min.csv")
    intraday['Datetime'] = pd.to_datetime(intraday['Datetime'])
    intraday['Date'] = intraday['Datetime'].dt.date

    df = detect_triggers_and_goals(daily, intraday)
    df.to_csv("combined_trigger_goal_results.csv", index=False)

if __name__ == "__main__":
    main()
