
import pandas as pd

# Load daily candle levels
daily = pd.read_excel("SPXdailycandles.xlsx", header=5)
print("✅ Loaded columns:", daily.columns.tolist())

level_map = {}
for col in daily.columns[9:22]:  # Columns J through V
    try:
        if isinstance(col, str) and '%' in col:
            float_label = float(col.strip('%')) / 100
        else:
            float_label = float(col)  # fallback for numeric labels like 0, 1, -1
        level_map[float_label] = col
    except ValueError:
        continue

print("✅ Level map built:", level_map)

# Placeholder for full logic, with debug on Date access
for index, row in daily.iterrows():
    try:
        date = row["Date"]
    except KeyError:
        print("❌ 'Date' KeyError - Available columns in row:", row.index.tolist())
        raise
