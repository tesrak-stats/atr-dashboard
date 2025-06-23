import pandas as pd

# === Load daily candle levels ===
daily = pd.read_excel("SPXdailycandles.xlsx", header=4)
level_map = {}

# Parse levels from column headers J through V (index 9:22)
for col in daily.columns[9:22]:
    try:
        if isinstance(col, str) and '%' in col:
            float_label = float(col.strip('%')) / 100
        else:
            float_label = float(col)  # Fallback for 0, 1, -1
        level_map[float_label] = col
    except ValueError:
        continue

# === Load intraday 10-min candle data ===
intraday = pd.read_csv("SPX_10min.csv")
intraday['Date'] = pd.to_datetime(intraday['Datetime']).dt.date

# === Select first day from daily ===
daily['Date'] = pd.to_datetime(daily['Date']).dt.date
first_day = daily.iloc[0]
first_day_date = first_day['Date']
atr = first_day['ATR']

# === Extract ATR levels for the day ===
level_values = {}
for ratio, label in level_map.items():
    try:
        level_values[ratio] = first_day[label]
    except KeyError:
        level_values[ratio] = None  # Some levels might be missing

# === Subset intraday data to just the first day ===
day_candles = intraday[intraday['Date'] == first_day_date].copy()

# === Add ATR level columns to the intraday data ===
for ratio, level in level_values.items():
    colname = f"level_{ratio}"
    day_candles[colname] = level

# === Trigger/Goal analysis ===
results = []
triggered_up = {}
triggered_down = {}

for i, row in day_candles.iterrows():
    high = row['High']
    low = row['Low']
    open_price = row['Open']
    hit = {}

    for ratio, level in level_values.items():
        if pd.isna(level):
            continue

        # Determine if it's a trigger or a goal hit
        if ratio > 0:
            if high >= level:
                if ratio not in triggered_up:
                    hit[f'trigger_{ratio}'] = True
                    triggered_up[ratio] = True
                else:
                    hit[f'goal_{ratio}'] = True
        elif ratio < 0:
            if low <= level:
                if ratio not in triggered_down:
                    hit[f'trigger_{ratio}'] = True
                    triggered_down[ratio] = True
                else:
                    hit[f'goal_{ratio}'] = True

    results.append(hit)

# === Merge results into DataFrame ===
expanded = pd.DataFrame(results).fillna(False).astype(bool)
full_trace = pd.concat([day_candles.reset_index(drop=True), expanded], axis=1)

# === Save debug trace CSV ===
full_trace.to_csv("debug_first_day_trace.csv", index=False)
print("✅ Debug trace written to debug_first_day_trace.csv")
