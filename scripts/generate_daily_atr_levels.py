
import yfinance as yf
import pandas as pd
import json
from datetime import datetime

# Parameters
ticker = "^GSPC"
lookback_days = 14
output_file = "data/daily_atr_levels.json"

# 1. Fetch 100 days to ensure we get 14 full closes
data = yf.download(ticker, period="100d", interval="1d", auto_adjust=True)

# 2. Keep only required columns
data = data[["Open", "High", "Low", "Close"]]

# 3. Calculate True Range and ATR
data["H-L"] = data["High"] - data["Low"]
data["H-PC"] = abs(data["High"] - data["Close"].shift(1))
data["L-PC"] = abs(data["Low"] - data["Close"].shift(1))
data["TR"] = data[["H-L", "H-PC", "L-PC"]].max(axis=1)
data["ATR"] = data["TR"].rolling(window=lookback_days).mean()

# 4. Get latest day's ATR and close
latest = data.dropna().iloc[-1]
atr = latest["ATR"]
close = latest["Close"]

# 5. Calculate ATR-based levels
fib_levels = [-1.000, -0.786, -0.618, -0.500, -0.382, -0.236, 0.000,
               0.236, 0.382, 0.500, 0.618, 0.786, 1.000]

levels = {f"{f:+.3f}": round(close + (f * atr), 2) for f in fib_levels}

# 6. Save to JSON
output = {
    "date_generated": datetime.utcnow().isoformat(),
    "ticker": ticker,
    "close": round(close, 2),
    "atr": round(atr, 2),
    "levels": levels
}

with open(output_file, "w") as f:
    json.dump(output, f, indent=2)

print("✅ ATR levels generated and saved.")
