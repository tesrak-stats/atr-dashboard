
import yfinance as yf
import pandas as pd
import json
from datetime import datetime, timedelta

# Settings
ticker = "SPY"  # Proxy for SPX
lookback_days = 21  # Includes buffer for non-trading days
atr_window = 14
output_file = "daily_atr_levels.json"

# Dates
end_date = datetime.today()
start_date = end_date - timedelta(days=lookback_days)

# Download historical data
df = yf.download(ticker, start=start_date.strftime('%Y-%m-%d'), end=end_date.strftime('%Y-%m-%d'))

# Clean and prepare
df = df[["Open", "High", "Low", "Close"]].dropna()
df["Prev Close"] = df["Close"].shift(1)

# True Range
df["TR"] = df.apply(lambda row: max(
    row["High"] - row["Low"],
    abs(row["High"] - row["Prev Close"]),
    abs(row["Low"] - row["Prev Close"])
), axis=1)

# ATR Calculation
df["ATR_14"] = df["TR"].rolling(window=atr_window).mean()

# Get most recent row (yesterday's close), and latest open (today's open if available)
latest = df.iloc[-1]
prior = df.iloc[-2]

# Calculate ATR-based levels from prior day's close
fibs = [+1.0, +0.786, +0.618, +0.5, +0.382, +0.236, 0.0,
        -0.236, -0.382, -0.5, -0.618, -0.786, -1.0]

levels = {f"{fib:+.3f}": round(prior["Close"] + fib * latest["ATR_14"], 2) for fib in fibs}

# Prepare output
result = {
    "date_generated": datetime.today().strftime("%Y-%m-%d"),
    "ticker": ticker,
    "atr": round(latest["ATR_14"], 2),
    "prev_close": round(prior["Close"], 2),
    "latest_open": round(latest["Open"], 2),
    "levels": levels
}

# Save to JSON
with open(output_file, "w") as f:
    json.dump(result, f, indent=2)

print(f"Daily ATR levels saved to {output_file}")
