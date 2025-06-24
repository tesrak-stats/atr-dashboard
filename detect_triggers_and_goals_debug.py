
import pandas as pd

def run_debug():
    try:
        daily = pd.read_excel("SPXdailycandles.xlsx", header=4)
        intraday = pd.read_csv("SPX_10min.csv", parse_dates=["Datetime"])
        intraday['Date'] = intraday['Datetime'].dt.date

        first_day = daily['Date'].iloc[0]
        day_row = daily[daily['Date'] == first_day].iloc[0]
        debug_log = []

        debug_log.append(f"--- Debug for {first_day} ---")
        debug_log.append(f"Day Row: {day_row.to_dict()}")

        if '0' in day_row:
            prev_close = day_row['0']
            debug_log.append(f"Previous Close from '0': {prev_close}")
        elif 0.0 in day_row:
            prev_close = day_row[0.0]
            debug_log.append(f"Previous Close from 0.0: {prev_close}")
        elif 'Close' in day_row:
            prev_close = day_row['Close']
            debug_log.append(f"Previous Close from 'Close': {prev_close}")
        else:
            debug_log.append("Previous Close not found.")
            prev_close = None

        first_intraday = intraday[intraday['Date'] == first_day]
        debug_log.append(f"First Intraday Rows:\n{first_intraday.head().to_string(index=False)}")

        with open("debug_log_output.csv", "w") as f:
            for line in debug_log:
                f.write(line + "\n")

    except Exception as e:
        with open("debug_log_output.csv", "w") as f:
            f.write("Error during debug:\n")
            f.write(str(e))
