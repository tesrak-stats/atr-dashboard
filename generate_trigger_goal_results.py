# Version 1.1 – Fix for string headers like '0.0%' or '100.0%'

import pandas as pd

# Load daily candle levels
daily = pd.read_excel("SPXdailycandles.xlsx", header=4)  # ✅ header=4 confirmed

# Build a map of level names to float values
level_map = {}
for col in daily.columns[9:22]:  # Columns J through V
    try:
        label = col
        if isinstance(label, str) and '%' in label:
            label = float(label.strip('%')) / 100  # e.g., "0.0%" -> 0.0
        else:
            label = float(label)
        level_map[label] = col
    except (ValueError, TypeError):
        continue

print("✅ Level map built:", level_map)
