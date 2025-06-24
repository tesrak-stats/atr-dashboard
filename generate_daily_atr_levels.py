
import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta

def get_latest_atr_levels(ticker="SPY", lookback_days=21, atr_window=14):
    # Date range for data pull
    end_date = datetime.today()
    start_date = end_date - timedelta(days=lookback_days)

    # Download historical data
    df = yf.download(ticker, start=start_date.strftime('%Y-%m-%d'), end=end_date.strftime('%Y-%m-%d'))

    # Clean and prepare
    df = df[["Open", "High", "Low", "Close"]].dropna()
    df["Prev Close"] = df["Close"].shift(1)

    # True Range (with NaN guard for first row)
    df["TR"] = df.apply(lambda row: max(
        row["High"] - row["Low"],
        abs(row["High"] - row["Prev Close"]),
        abs(row["Low"] - row["Prev Close"])
    ) if pd.notnull(row["Prev Close"]) else None, axis=1)

    # ATR Calculation
    df["ATR_14"] = df["TR"].rolling(window=atr_window).mean()

    # Get latest and prior day
    latest = df.iloc[-1]
    prior = df.iloc[-2]

    # Fib levels
    fibs = [+1.0, +0.786, +0.618, +0.5, +0.382, +0.236, 0.0,
            -0.236, -0.382, -0.5, -0.618, -0.786, -1.0]

    levels = {
        f"{fib:+.3f}": round(prior["Close"] + fib * latest["ATR_14"], 2)
        for fib in fibs
    }

    # Return as dictionary (not saving to file)
    return {
        "date_generated": datetime.today().strftime("%Y-%m-%d"),
        "ticker": ticker,
        "atr": round(latest["ATR_14"], 2),
        "prev_close": round(prior["Close"], 2),
        "latest_open": round(latest["Open"], 2),
        "levels": levels
    }

if __name__ == "__main__":
    result = get_latest_atr_levels()
    print(result)
