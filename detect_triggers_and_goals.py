import pandas as pd
import numpy as np

# --- Load daily and intraday data ---
daily = pd.read_excel("SPXdailycandles.xlsx", header=4)
intraday = pd.read_csv("SPX_10min.csv")

# --- Build mapping from float goal levels to column names in Excel ---
level_map = {}
for col in daily.columns[9:22]:  # Columns J to V expected here
    try:
        # Convert labels like '23.6%' to 0.236, '-38.2%' to -0.382, '50%' to 0.5, etc.
        float_label = float(col.strip('%')) / 100 if '%' in col else float(col)
        level_map[float_label] = col
    except ValueError:
        continue

# --- Output for debugging ---
print("✅ Level mapping:")
for k, v in sorted(level_map.items()):
    print(f"  {k} => {v}")

# --- Placeholder for full detection logic ---
# You would insert your full pipeline logic here
# For demonstration, we're just saving the level map to confirm

pd.Series(level_map).to_csv("level_mapping_debug.csv")

print("✅ Script completed at 2025-06-23T18:17:35.687689 UTC")
