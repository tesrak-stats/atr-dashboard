import pandas as pd

# Load data
daily_df = pd.read_excel("SPXdailycandles.xlsx", header=4)
intraday_df = pd.read_csv("SPX_10min.csv")

# Ensure proper datetime
intraday_df['Datetime'] = pd.to_datetime(intraday_df['Datetime'])
intraday_df['Date'] = intraday_df['Datetime'].dt.date
intraday_df['Hour'] = intraday_df['Datetime'].dt.strftime('%H00')

# Build output
results = []

# List of Fibonacci levels and their column names
level_columns = [
    ('1.000', '100.0%'), ('0.786', '78.6%'), ('0.618', '61.8%'), ('0.500', '50.0%'),
    ('0.382', '38.2%'), ('0.236', '23.6%'), ('0.000', '0.0%'),
    ('-0.236', '-23.6%'), ('-0.382', '-38.2%'), ('-0.500', '-50.0%'),
    ('-0.618', '-61.8%'), ('-0.786', '-78.6%'), ('-1.000', '-100.0%')
]

level_map = {float(k): v for k, v in level_columns}
fib_levels = list(level_map.keys())

# Loop over each day
for i in range(len(daily_df)):
    date = pd.to_datetime(daily_df.loc[i, 'Date']).date()
    daily_levels = {
        float(k): daily_df.loc[i, v]
        for k, v in level_columns
        if pd.notnull(daily_df.loc[i, v])
    }

    day_data = intraday_df[intraday_df['Date'] == date].copy()
    if day_data.empty:
        continue

    first_candle = day_data.iloc[0]
    open_price = first_candle['Open']
    open_time = '0000'

    for direction in ['upside', 'downside']:
        if direction == 'upside':
            trigger_check = lambda high, level: high >= level
            goal_check = lambda high, level: high >= level
            price_column = 'High'
            fib_sorted = sorted(fib_levels)
        else:
            trigger_check = lambda low, level: low <= level
            goal_check = lambda low, level: low <= level
            price_column = 'Low'
            fib_sorted = sorted(fib_levels, reverse=True)

        for idx, trigger_level in enumerate(fib_sorted[:-1]):
            trigger_value = daily_levels.get(trigger_level)
            if trigger_value is None:
                continue

            next_level = fib_sorted[idx + 1]
            next_value = daily_levels.get(next_level)

            # Trigger detection logic
            open_trigger = False
            if direction == 'upside':
                if open_price >= trigger_value and (next_value is None or open_price < next_value):
                    open_trigger = True
            else:
                if open_price <= trigger_value and (next_value is None or open_price > next_value):
                    open_trigger = True

            trigger_row = None
            trigger_hour = None

            if open_trigger:
                trigger_row = first_candle
                trigger_hour = open_time
            else:
                for j, row in day_data.iterrows():
                    if trigger_check(row[price_column], trigger_value):
                        trigger_row = row
                        trigger_hour = row['Hour']
                        break

            if trigger_row is None:
                continue  # No trigger occurred

            trigger_time = trigger_hour

            for goal_level in fib_levels:
                if (direction == 'upside' and goal_level <= trigger_level) or \
                   (direction == 'downside' and goal_level >= trigger_level):
                    continue  # Only evaluate goals beyond the trigger

                goal_value = daily_levels.get(goal_level)
                if goal_value is None:
                    continue

                goal_hit = False
                goal_time = None
                open_completion = 'No'

                for j, row in day_data.iterrows():
                    row_hour = row['Hour']
                    if pd.to_datetime(row['Datetime']) < pd.to_datetime(trigger_row['Datetime']):
                        continue  # Must be after trigger

                    if goal_check(row[price_column], goal_value):
                        goal_hit = True
                        goal_time = row_hour
                        if row['Datetime'] == trigger_row['Datetime'] and trigger_time == open_time:
                            goal_time = open_time
                            open_completion = 'Yes'
                        break

                results.append({
                    'Date': date,
                    'Direction': direction,
                    'TriggerLevel': trigger_level,
                    'TriggerTime': trigger_time,
                    'GoalLevel': goal_level,
                    'GoalHit': 'Yes' if goal_hit else 'No',
                    'GoalTime': goal_time if goal_hit else '',
                    'OpenCompletionFlag': open_completion
                })

# Save
df_out = pd.DataFrame(results)
df_out.to_csv("combined_trigger_goal_results.csv", index=False)
print("✅ File saved: combined_trigger_goal_results.csv")
