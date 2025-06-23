
import pandas as pd

# Load SPX daily candle data (header in row 5)
daily = pd.read_excel("SPXdailycandles.xlsx", header=4)
daily.columns = daily.columns.str.strip()
levels_row = daily.iloc[0]

# Extract ATR levels from columns J to V (indices 9 to 21)
level_map = {}
for col in daily.columns[9:22]:
    try:
        label = col
        if isinstance(label, str) and "%" in label:
            float_label = float(label.strip("%")) / 100
        else:
            float_label = float(label)
        level_map[float_label] = levels_row[col]
    except:
        continue

# Load intraday 10-minute candles
intraday = pd.read_csv("SPX_10min.csv")
intraday['Datetime'] = pd.to_datetime(intraday['Datetime'])
intraday['Date'] = intraday['Datetime'].dt.date

# Filter to only first day in the daily file
target_date = pd.to_datetime(levels_row["Date"]).date()
intraday_day = intraday[intraday["Date"] == target_date].copy()

# Sort and reset index
intraday_day = intraday_day.sort_values("Datetime").reset_index(drop=True)

# Define columns to record trigger/goal results
results = []
triggered_levels = {"Upside": set(), "Downside": set()}

for _, row in intraday_day.iterrows():
    entry = {
        "Datetime": row["Datetime"],
        "Open": row["Open"],
        "High": row["High"],
        "Low": row["Low"],
        "Close": row["Last"]
    }

    for direction in ["Upside", "Downside"]:
        for lvl, value in level_map.items():
            key = f"{direction}_@{lvl}"
            if direction == "Upside":
                # Trigger condition
                if lvl not in triggered_levels["Upside"] and row["High"] >= value:
                    entry[f"Triggered_{key}"] = "Yes"
                    triggered_levels["Upside"].add(lvl)
                else:
                    entry[f"Triggered_{key}"] = ""
                # Goal condition (if already triggered)
                entry[f"Goal_{key}"] = "Yes" if lvl in triggered_levels["Upside"] and row["High"] >= value else ""
            else:
                # Downside logic
                if lvl not in triggered_levels["Downside"] and row["Low"] <= value:
                    entry[f"Triggered_{key}"] = "Yes"
                    triggered_levels["Downside"].add(lvl)
                else:
                    entry[f"Triggered_{key}"] = ""
                entry[f"Goal_{key}"] = "Yes" if lvl in triggered_levels["Downside"] and row["Low"] <= value else ""

    # Add ATR levels as a row reference
    for lvl, val in level_map.items():
        entry[f"Level_{lvl}"] = val

    results.append(entry)

# Export to CSV
debug_df = pd.DataFrame(results)
debug_df.to_csv("debug_first_day_detailed.csv", index=False)
print("✅ Debug trace saved as debug_first_day_detailed.csv")
