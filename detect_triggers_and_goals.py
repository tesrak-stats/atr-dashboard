
import pandas as pd

def run_continuation_logic(daily_df, intraday_df, fib_col_map):
    fib_levels = list(map(float, fib_col_map.keys()))
    intraday_df["Datetime"] = pd.to_datetime(intraday_df["Datetime"])
    intraday_df["Date"] = intraday_df["Datetime"].dt.date
    intraday_df["Time"] = intraday_df["Datetime"].dt.strftime("%H%M")
    intraday_df.loc[intraday_df["Time"] == "0930", "Time"] = "OPEN"

    first_date = intraday_df["Date"].iloc[0]
    intraday_day = intraday_df[intraday_df["Date"] == first_date]
    daily_row = daily_df[daily_df["Date"] == pd.to_datetime(first_date)].iloc[0]
    levels = {float(k): daily_row[v] for k, v in fib_col_map.items()}

    output_rows = []
    triggered_levels = {"up": set(), "down": set()}

    for i, row in intraday_day.iterrows():
        high = row["High"]
        low = row["Low"]
        open_price = row["Open"]
        time = row["Time"]
        date = row["Date"]

        for direction in ["up", "down"]:
            is_up = direction == "up"
            level_list = sorted(fib_levels, reverse=is_up)
            for idx, level in enumerate(level_list):
                trigger_cond = high >= levels[level] if is_up else low <= levels[level]
                open_trigger_cond = (
                    (open_price >= levels[level] and open_price < levels[level_list[idx - 1]])
                    if (is_up and idx > 0)
                    else (open_price <= levels[level] and open_price > levels[level_list[idx - 1]])
                    if (not is_up and idx > 0)
                    else False
                )
                trig_key = (level, time)
                if trig_key in triggered_levels[direction]:
                    continue

                if time == "OPEN" and open_trigger_cond:
                    triggered_levels[direction].add(trig_key)
                    output_rows.append({
                        "Date": date,
                        "Direction": "Upside" if is_up else "Downside",
                        "TriggerLevel": level,
                        "TriggerTime": time,
                        "Scenario": "continuation"
                    })
                elif trigger_cond:
                    triggered_levels[direction].add(trig_key)
                    output_rows.append({
                        "Date": date,
                        "Direction": "Upside" if is_up else "Downside",
                        "TriggerLevel": level,
                        "TriggerTime": time,
                        "Scenario": "continuation"
                    })
    return pd.DataFrame(output_rows)

def run_retracement_logic(daily_df, intraday_df, fib_col_map):
    # Placeholder for retracement logic, structure matched to continuation
    return pd.DataFrame(columns=[
        "Date", "Direction", "TriggerLevel", "TriggerTime", "Scenario"
    ])

def detect_triggers_and_goals(daily_df, intraday_df):
    fib_col_map = {
        "1.0": "100.0%",
        "0.786": "78.6%",
        "0.618": "61.8%",
        "0.5": "50.0%",
        "0.382": "38.2%",
        "0.236": "23.6%",
        "0.0": "0.0%",
        "-0.236": "-23.6%",
        "-0.382": "-38.6%",
        "-0.5": "-50.0%",
        "-0.618": "-61.8%",
        "-0.786": "-78.6%",
        "-1.0": "-100.0%",
    }
    continuation_df = run_continuation_logic(daily_df, intraday_df, fib_col_map)
    retracement_df = run_retracement_logic(daily_df, intraday_df, fib_col_map)

    return pd.concat([continuation_df, retracement_df], ignore_index=True)
