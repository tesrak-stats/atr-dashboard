
import pandas as pd

def main():
    try:
        daily = pd.read_excel("SPXdailycandles.xlsx", header=4)
        intraday = pd.read_csv("SPX_10min.csv", parse_dates=['Datetime'])
        intraday['Date'] = intraday['Datetime'].dt.date
    except Exception as e:
        raise RuntimeError(f"Error loading input files: {e}")

    fib_levels = [1.000, 0.786, 0.618, 0.500, 0.382, 0.236, 0.000,
                 -0.236, -0.382, -0.500, -0.618, -0.786, -1.000]

    results = []

    try:
        first_date = daily['Date'].iloc[0]
        day_row = daily[daily['Date'] == first_date].iloc[0]
        prev_close = day_row.get('0') or day_row.get(0.0) or daily[daily['Date'] == first_date].iloc[-2]['Close']
        if pd.isna(prev_close):
            raise ValueError("Previous close is NaN")

        level_map = {}
        for level in fib_levels:
            level_str = f"{level:.3f}".rstrip('0').rstrip('.') if '.' in f"{level:.3f}" else str(level)
            if level_str in day_row:
                level_map[level] = day_row[level_str]

        day_data = intraday[intraday['Date'] == first_date].copy()
        if day_data.empty:
            raise ValueError("No intraday data for first date")

        day_data['Time'] = day_data['Datetime'].dt.strftime('%H%M')
        day_data.reset_index(drop=True, inplace=True)

        open_price = day_data.iloc[0]['Open']
        results.append({
            'Date': first_date,
            'OpenPrice': open_price,
            'LevelMapCount': len(level_map),
            'TriggerCheck': any(day_data['High'] >= min(level_map.values()))
        })

    except Exception as e:
        raise RuntimeError(f"Error during debug logic execution: {e}")

    try:
        pd.DataFrame(results).to_csv("debug_trigger_goal_output.csv", index=False)
    except Exception as e:
        raise RuntimeError(f"Failed to save debug output: {e}")

if __name__ == "__main__":
    main()
