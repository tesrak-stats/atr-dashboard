
import pandas as pd

# Load daily candle levels
daily = pd.read_excel("SPXdailycandles.xlsx", header=5)

# Build a map of level names to float values
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

# Load intraday data and fix date column
intraday = pd.read_csv("SPX_10min.csv")
intraday['Datetime'] = pd.to_datetime(intraday['Datetime'])
intraday['Date'] = intraday['Datetime'].dt.date

# Placeholder for full trigger/goal logic
print("🚧 Placeholder: trigger/goal logic not yet inserted")
