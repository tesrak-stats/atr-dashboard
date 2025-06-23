
import pandas as pd

# Load daily candle levels
daily = pd.read_excel("SPXdailycandles.xlsx", header=5)
level_map = {}

# Convert column headers like '23.6%' to float keys like 0.236
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
intraday = pd.read_csv("SPX_10min.csv", header=1)
intraday['Date'] = pd.to_datetime(intraday['Datetime']).dt.date

all_results = []

# Placeholder: your detection logic goes here
# Right now just testing that this path executes
print("🚀 Starting loop over daily data...")
for idx, row in daily.iterrows():
    date = row["Date"]
    intraday_day = intraday[intraday['Date'] == date]
    if intraday_day.empty:
        continue
    # Simulate result
    all_results.append({
        "Date": date,
        "Example": "Test row"
    })

# Write CSV if results exist
if all_results:
    df = pd.DataFrame(all_results)
    df.to_csv("combined_trigger_goal_results.csv", index=False)
    print(f"✅ File written with {len(df)} rows.")
else:
    print("⚠️ No data to write to CSV. all_results is empty.")
