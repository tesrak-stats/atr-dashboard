
import pandas as pd

def main():
    # Load daily ATR level map
    daily = pd.read_excel("SPXdailycandles.xlsx", header=4)
    level_map = {}
    for col in daily.columns[9:22]:  # Columns J through V
        try:
            if isinstance(col, str) and '%' in col:
                float_label = float(col.strip('%')) / 100
            else:
                float_label = float(col)
            level_map[float_label] = col
        except ValueError:
            continue
    print("✅ Level map built:", level_map)

    # Load intraday data
    intraday = pd.read_csv("SPX_10min.csv")
    intraday['Datetime'] = pd.to_datetime(intraday['Datetime'])
    intraday['Date'] = intraday['Datetime'].dt.date

    # Get target date from daily row
    target_date = pd.to_datetime(daily.at[0, 'Date']).date()
    levels_row = daily.iloc[0]
    intraday_day = intraday[intraday['Date'] == target_date].copy()

    # Assign hour label (OPEN, 0900, 1000, ...)
    def get_hour_label(row):
        dt = row['Datetime']
        if dt.time().hour == 6 and dt.time().minute == 30:
            return 'OPEN'
        return dt.strftime('%H00')
    intraday_day['Time'] = intraday_day.apply(get_hour_label, axis=1)

    results = []

    for direction in ['up', 'down']:
        for level, colname in level_map.items():
            if direction == 'up' and level <= 0:
                continue
            if direction == 'down' and level >= 0:
                continue

            trigger_found = False
            goal_levels = sorted([l for l in level_map.keys() if (l > level if direction == 'up' else l < level)],
                                 reverse=(direction == 'down'))
            trigger_idx = None
            trigger_time = None

            for i, row in intraday_day.iterrows():
                high = row['High']
                low = row['Low']
                open_price = row['Open']
                time_label = row['Time']

                if not trigger_found:
                    if direction == 'up':
                        if time_label == 'OPEN' and open_price >= levels_row[colname] and high >= levels_row[colname]:
                            trigger_found = True
                            trigger_idx = i
                            trigger_time = time_label
                        elif high >= levels_row[colname]:
                            trigger_found = True
                            trigger_idx = i
                            trigger_time = time_label
                    else:
                        if time_label == 'OPEN' and open_price <= levels_row[colname] and low <= levels_row[colname]:
                            trigger_found = True
                            trigger_idx = i
                            trigger_time = time_label
                        elif low <= levels_row[colname]:
                            trigger_found = True
                            trigger_idx = i
                            trigger_time = time_label
                else:
                    break

            if not trigger_found:
                continue

            # Now scan for goal levels hit after the trigger
            trigger_row = intraday_day.loc[trigger_idx]
            following_rows = intraday_day.loc[trigger_idx+1:]

            for goal_level in goal_levels:
                goal_colname = level_map[goal_level]
                goal_hit = False
                goal_time = 'N/A'

                for _, row in following_rows.iterrows():
                    if direction == 'up' and row['High'] >= levels_row[goal_colname]:
                        goal_hit = True
                        goal_time = row['Time']
                        break
                    elif direction == 'down' and row['Low'] <= levels_row[goal_colname]:
                        goal_hit = True
                        goal_time = row['Time']
                        break

                results.append({
                    'Direction': direction,
                    'TriggerLevel': level,
                    'GoalLevel': goal_level,
                    'TriggerTime': trigger_time,
                    'GoalHit': 'Yes' if goal_hit else 'No',
                    'GoalTime': goal_time,
                    'Scenario': 'Continuation'
                })

            # Check retracement (first opposite-side level hit)
            retrace_levels = sorted([l for l in level_map.keys() if (l < level if direction == 'up' else l > level)],
                                    reverse=(direction == 'up'))

            for retrace_level in retrace_levels:
                retrace_colname = level_map[retrace_level]
                retrace_hit = False
                retrace_time = 'N/A'

                for _, row in following_rows.iterrows():
                    if direction == 'up' and row['Low'] <= levels_row[retrace_colname]:
                        retrace_hit = True
                        retrace_time = row['Time']
                        break
                    elif direction == 'down' and row['High'] >= levels_row[retrace_colname]:
                        retrace_hit = True
                        retrace_time = row['Time']
                        break

                results.append({
                    'Direction': direction,
                    'TriggerLevel': level,
                    'GoalLevel': retrace_level,
                    'TriggerTime': trigger_time,
                    'GoalHit': 'Yes' if retrace_hit else 'No',
                    'GoalTime': retrace_time,
                    'Scenario': 'Retracement'
                })
                break  # Only log first retracement level

    return pd.DataFrame(results)
