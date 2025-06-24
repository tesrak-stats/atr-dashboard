# detect_triggers_and_goals.py
# Version 1.1 — includes fix for percent-formatted level headers

import pandas as pd

def main():
    # === Load daily candle levels ===
    daily = pd.read_excel("SPXdailycandles.xlsx", header=4)  # header row is 5th row (0-indexed)

    # === Build a map of level names to float values ===
    level_map = {}
    for col in daily.columns[9:22]:  # Columns J through V
        try:
            # Always cast to string and remove any percent signs
            col_str = str(col).strip().replace('%', '')
            float_label = float(col_str) / 100 if '%' in str(col) else float(col_str)
            level_map[float_label] = col
        except ValueError:
            continue

    print("✅ Level map built:", level_map)

    # === Load intraday data and normalize datetime ===
    intraday = pd.read_csv("SPX_10min.csv")
    intraday['Datetime'] = pd.to_datetime(intraday['Datetime'])
    intraday['Date'] = intraday['Datetime'].dt.date

    results = []

    # === Begin trigger/goal detection logic ===
    for direction in ['Upside', 'Downside']:
        for date in intraday['Date'].unique():
            day_data = intraday[intraday['Date'] == date].copy()

            # Get the daily level row matching this date
            daily_row = daily[daily['Date'] == pd.to_datetime(date)]
            if daily_row.empty:
                continue

            for trigger_level, trigger_col in level_map.items():
                try:
                    trigger_value = daily_row.iloc[0][trigger_col]
                except KeyError:
                    print(f"⚠️ Missing trigger column: {trigger_col}")
                    continue

                # Track whether trigger was hit
                trigger_hit = False
                trigger_time = None
                direction_flag = (direction == 'Upside')

                for idx, row in day_data.iterrows():
                    price = row['High'] if direction_flag else row['Low']
                    open_price = row['Open']

                    if not trigger_hit:
                        # Determine if this row hits the trigger level
                        if (row.name == day_data.index[0] and  # First row
                            ((direction_flag and open_price >= trigger_value) or
                             (not direction_flag and open_price <= trigger_value))):
                            trigger_time = 'OPEN'
                            trigger_hit = True
                        elif ((direction_flag and price >= trigger_value) or
                              (not direction_flag and price <= trigger_value)):
                            trigger_time = row['Datetime'].strftime('%H%M')
                            trigger_hit = True
                    else:
                        # Already triggered — check goals
                        for goal_level, goal_col in level_map.items():
                            if (direction_flag and goal_level <= trigger_level) or \
                               (not direction_flag and goal_level >= trigger_level):
                                continue  # Skip irrelevant goals

                            try:
                                goal_value = daily_row.iloc[0][goal_col]
                            except KeyError:
                                continue

                            price_check = row['High'] if direction_flag else row['Low']
                            if (direction_flag and price_check >= goal_value) or \
                               (not direction_flag and price_check <= goal_value):
                                goal_time = row['Datetime'].strftime('%H%M')
                                results.append({
                                    'Date': date,
                                    'Direction': direction,
                                    'TriggerLevel': trigger_level,
                                    'TriggerTime': trigger_time,
                                    'GoalLevel': goal_level,
                                    'GoalHit': 'Yes',
                                    'GoalTime': goal_time
                                })

                # If trigger was hit but no goal reached
                if trigger_hit:
                    for goal_level, goal_col in level_map.items():
                        if (direction_flag and goal_level <= trigger_level) or \
                           (not direction_flag and goal_level >= trigger_level):
                            continue
                        goal_hits = [r for r in results if r['Date'] == date and
                                     r['TriggerLevel'] == trigger_level and
                                     r['GoalLevel'] == goal_level]
                        if not goal_hits:
                            results.append({
                                'Date': date,
                                'Direction': direction,
                                'TriggerLevel': trigger_level,
                                'TriggerTime': trigger_time,
                                'GoalLevel': goal_level,
                                'GoalHit': 'No',
                                'GoalTime': None
                            })

    df = pd.DataFrame(results)
    return df
