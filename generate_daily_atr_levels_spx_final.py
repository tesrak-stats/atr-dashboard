
import yfinance as yf
import pandas as pd
from datetime import datetime

def get_latest_atr_levels(ticker="^GSPC", atr_window=14):
    spx = yf.Ticker(ticker)
    df = spx.history(period="40d")  # Fetch extra in case of non-trading days

    if df.empty or df.shape[0] < atr_window + 1:
        raise ValueError(f"Not enough data to calculate ATR. Got {df.shape[0]} rows.")

    df = df[["Open", "High", "Low", "Close"]].dropna()
    df["Prev Close"] = df["Close"].shift(1)

    df["TR"] = df.apply(lambda row: max(
        row["High"] - row["Low"],
        abs(row["High"] - row["Prev Close"]),
        abs(row["Low"] - row["Prev Close"])
    ) if pd.notnull(row["Prev Close"]) else None, axis=1)

    # ✅ Wilder's smoothing for ATR
    df["ATR_14"] = df["TR"].ewm(alpha=1/atr_window, adjust=False).mean()

    df = df.dropna(subset=["ATR_14"])
    if df.shape[0] < 1:
        raise ValueError("Not enough valid rows with ATR calculated.")

    latest = df.iloc[-1]  # Use the just-closed day's values

    fibs = [+1.0, +0.786, +0.618, +0.5, +0.382, +0.236, 0.0,
            -0.236, -0.382, -0.5, -0.618, -0.786, -1.0]

    levels = {
        f"{fib:+.3f}": round(latest["Close"] + fib * latest["ATR_14"], 2)
        for fib in fibs
    }

    return {
        "date_generated": datetime.today().strftime("%Y-%m-%d"),
        "ticker": ticker,
        "atr": round(latest["ATR_14"], 2),
        "prev_close": round(latest["Close"], 2),
        "latest_open": round(latest["Open"], 2),
        "levels": levels
    }
