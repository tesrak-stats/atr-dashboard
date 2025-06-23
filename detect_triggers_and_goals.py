
import pandas as pd

# ✅ Load daily candle levels with correct header row (row 5 = index 4)
daily = pd.read_excel("SPXdailycandles.xlsx", header=4)

# Preview to confirm
print("✅ Columns loaded:", daily.columns.tolist())

# Proceed with logic (placeholder)
for index, row in daily.iterrows():
    try:
        date = row["Date"]
    except KeyError:
        print("❌ 'Date' KeyError on row", index)
        raise

# Placeholder for future logic:
# - Trigger/goal detection
# - Level parsing
# - Time block matching, etc.

print("✅ Script ran successfully.")
