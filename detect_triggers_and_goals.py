
import pandas as pd

def detect_triggers_and_goals(daily, intraday):
    fib_levels = [1.000, 0.786, 0.618, 0.500, 0.382, 0.236,
                  -0.236, -0.382, -0.500, -0.618, -0.786, -1.000]
    results = []

    start_date = pd.to_datetime("2014-01-02").date()

    for date in daily['Date'].unique():
        if date < start_date:
            continue

        day_row = daily[daily['Date'] == date].iloc[0]
        prev_row = daily[daily['Date'] < date].iloc[-1]
        prev_close = prev_row['Close']

        level_map = {}
        for level in fib_levels:
            level_map[level] = day_row.get(level, None)

        day_data = intraday[intraday['Date'] == date]

        if day_data.empty:
            continue

        # Example logic (custom trigger/goal logic goes here)
        for _, row in day_data.iterrows():
            for level in fib_levels:
                level_value = level_map.get(level)
                if pd.notna(level_value) and row['High'] >= level_value:
                    results.append({
                        'Date': date,
                        'Time': row['Time'],
                        'Trigger_Level': level,
                        'Reached': True
                    })

    return pd.DataFrame(results)

def main():
    daily = pd.read_csv("SPX_daily.csv")
    daily['Date'] = pd.to_datetime(daily['Date']).dt.date

    intraday = pd.read_csv("SPX_10min.csv")
    intraday['Datetime'] = pd.to_datetime(intraday['Datetime'])
    intraday['Date'] = intraday['Datetime'].dt.date

    df = detect_triggers_and_goals(daily, intraday)
    df.to_csv("combined_trigger_goal_results.csv", index=False)

if __name__ == "__main__":
    main()
