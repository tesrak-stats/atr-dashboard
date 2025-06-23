import pandas as pd

# === Load Daily Levels ===
daily = pd.read_excel("SPXdailycandles.xlsx", header=5)

# Extract Fibonacci level mapping from column headers
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

# === Load Intraday 10-min candles ===
intraday = pd.read_csv("SPX_10min.csv")

# Split combined Datetime column if necessary
if 'Datetime' in intraday.columns and 'Date' not in intraday.columns:
    intraday['Datetime'] = pd.to_datetime(intraday['Datetime'])
    intraday['Date'] = intraday['Datetime'].dt.date
    intraday['Time'] = intraday['Datetime'].dt.strftime('%H%M')

# Now safe to proceed
for index, row in daily.iterrows():
    try:
        date = row["Date"]
    except KeyError:
        print("❌ 'Date' KeyError in daily row:", row)
        raise

    # Extract trigger levels for this date
    trigger_levels = {level: row[col_name] for level, col_name in level_map.items()}

    # Pull intraday data for that date
    intraday_day = intraday[intraday["Date"] == date]
    if intraday_day.empty:
        continue

    # Placeholder: add detection logic here using intraday_day and trigger_levels

print("🚀 Detection script setup complete.")
