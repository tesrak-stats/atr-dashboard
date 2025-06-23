
import pandas as pd

# Load daily and intraday data
daily = pd.read_excel("SPXdailycandles.xlsx", header=5)
intraday = pd.read_csv("SPX_10min.csv")

# Ensure datetime conversion and day filtering
intraday['Date'] = pd.to_datetime(intraday['Datetime']).dt.date
first_day = intraday['Date'].min()
intraday_day = intraday[intraday['Date'] == first_day]

# Pull ATR levels from columns J–V
level_map = {}
for col in daily.columns[9:22]:  # J to V
    try:
        label = col
        if isinstance(label, str) and '%' in label:
            float_label = float(label.strip('%')) / 100
        else:
            float_label = float(label)
        level_map[float_label] = col
    except ValueError:
        continue

# Get the first day's levels
daily_first = daily.iloc[0]
levels = {k: daily_first[v] for k, v in level_map.items()}

results = []
triggered_levels = set()

# Check each candle for triggers and goals
for _, row in intraday_day.iterrows():
    row_result = {
        "Datetime": row["Datetime"],
        "Open": row["Open"],
        "High": row["High"],
        "Low": row["Low"],
        "Close": row["Last"]
    }

    # Track which levels were triggered/goals hit
    for level, value in levels.items():
        trigger_hit = None
        goal_hit = None

        # Upside trigger: high crosses at or above level
        if row["High"] >= value and level not in triggered_levels:
            trigger_hit = "YES"
            triggered_levels.add(level)

        # Goal hit after trigger
        if level in triggered_levels:
            if level > 0 and row["High"] >= value:
                goal_hit = "YES"
            elif level < 0 and row["Low"] <= value:
                goal_hit = "YES"

        row_result[f"Trigger_{level}"] = trigger_hit or ""
        row_result[f"Goal_{level}"] = goal_hit or ""

    results.append(row_result)

# Save results to CSV
debug_df = pd.DataFrame(results)
debug_df.to_csv("debug_first_day_results.csv", index=False)
print("✅ Debug CSV generated: debug_first_day_results.csv")
