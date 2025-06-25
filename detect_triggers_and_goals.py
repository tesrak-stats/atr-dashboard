import pandas as pd
from datetime import datetime

def detect_triggers_and_goals(daily, intraday):
    fib_levels = [1.000, 0.786, 0.618, 0.500, 0.382, 0.236,
                 -0.236, -0.382, -0.500, -0.618, -0.786, -1.000, 0.000]

    # Ensure date column is datetime
    daily['Date'] = pd.to_datetime(daily['Date'])
    intraday['Datetime'] = pd.to_datetime(intraday['Datetime'])
    intraday['Date'] = intraday['Datetime'].dt.date

    # Map: fib level float -> column name (e.g. 0.236 -> '0.236')
    level_map = {}
    for col in daily.columns[9:22]:
        try:
            col_str = str(col).strip().replace('%', '')
            float_label = float(col_str) / 100 if '%' in col_str else float(col_str)
            level_map[float_label] = col
        except ValueError:
            continue

    results = []
    cutoff = datetime(2014, 1, 2).date()

    for direction in ['Upside', 'Downside']:
        for date in intraday['Date'].unique():
            if date < cutoff:
                continue

            day_data = intraday[intraday['Date'] == date]
            if day_data.empty:
                continue

            daily_row = daily[daily['Date'].dt.date == date]
            if daily_row.empty:
                continue

            day_row = daily_row.iloc[0]
            prev_row = daily[daily['Date'].dt.date < date].iloc[-1]
            prev_close = prev_row['Close']

            for level in fib_levels:
                if level == 0:
                    continue
                level_col = level_map.get(level)
                if pd.isna(level_col):
                    continue

                trigger_level = day_row[level_col]

                # OPEN time logic
                first_row = day_data.iloc[0]
                open_price = first_row['Open']
                hour_label = '0000' if (
                    (direction == 'Upside' and trigger_level <= open_price < prev_close)
                    or (direction == 'Downside' and trigger_level >= open_price > prev_close)
                ) else first_row['Datetime'].strftime('%H00')

                # Trigger check
                triggered = False
                for _, row in day_data.iterrows():
                    price = row['High'] if direction == 'Upside' else row['Low']
                    if (direction == 'Upside' and price >= trigger_level) or                        (direction == 'Downside' and price <= trigger_level):
                        triggered = True
                        hour = row['Datetime'].strftime('%H00')
                        hour = '0000' if (
                            row.name == 0 and (
                                (direction == 'Upside' and trigger_level <= open_price < prev_close) or
                                (direction == 'Downside' and trigger_level >= open_price > prev_close)
                            )
                        ) else hour
                        break

                if not triggered:
                    continue

                for goal_level in fib_levels:
                    if direction == 'Upside' and goal_level <= level:
                        continue
                    if direction == 'Downside' and goal_level >= level:
                        continue

                    goal_col = level_map.get(goal_level)
                    if pd.isna(goal_col):
                        continue

                    goal_price = day_row[goal_col]
                    goal_hit = False

                    for _, row in day_data.iterrows():
                        price = row['High'] if direction == 'Upside' else row['Low']
                        if (direction == 'Upside' and price >= goal_price) or                            (direction == 'Downside' and price <= goal_price):
                            goal_hit = True
                            break

                    results.append({
                        'Date': date,
                        'Direction': direction,
                        'Trigger Level': level,
                        'Goal Level': goal_level,
                        'Hour': hour,
                        'Hit': int(goal_hit)
                    })

    return pd.DataFrame(results)

def main():
    daily = pd.read_csv("SPXdailycandles.csv")
    intraday = pd.read_csv("SPX_10min.csv")
    df = detect_triggers_and_goals(daily, intraday)
    df.to_csv("combined_trigger_goal_results.csv", index=False)
