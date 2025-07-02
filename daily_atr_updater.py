# Daily ATR Levels Updater - Multi-Ticker Windows Compatible
"""
Daily ATR Levels Updater - Multi-Ticker Support
Runs at 5:16 PM Eastern to calculate next day's ATR levels for multiple tickers
Saves results to JSON file for dashboard consumption
"""

import yfinance as yf
import pandas as pd
import json
from datetime import datetime
import pytz
import sys
import os

# Configuration for multiple tickers
TICKER_CONFIG = {
    "SPX": {
        "symbol": "^GSPC",
        "display_name": "S&P 500 (SPX)"
    },
    "QQQ": {
        "symbol": "QQQ",
        "display_name": "Nasdaq 100 (QQQ)"
    },
     "NVDA": {
        "symbol": "NVDA",
        "display_name": "NVidia Corporation (NVDA)"
    },
    "IWM": {
        "symbol": "IWM", 
        "display_name": "Russell 2000 (IWM)"
    },
    # Add more tickers as needed
}

def calculate_true_range(row):
    """Calculate True Range for a given row"""
    if pd.isna(row["Prev_Close"]):
        return row["High"] - row["Low"]
    
    return max(
        row["High"] - row["Low"],
        abs(row["High"] - row["Prev_Close"]),
        abs(row["Low"] - row["Prev_Close"])
    )

def calculate_true_wilders_atr(tr_series, period=14):
    """
    Calculate TRUE Wilder's ATR exactly like your Excel formula:
    Uses EXACT Excel logic - OFFSET includes current row in average
    """
    atr_values = [None] * len(tr_series)
    
    for i in range(len(tr_series)):
        if i < period - 1:
            # No ATR until we have enough data (need 14 periods)
            atr_values[i] = None
        elif i == period - 1:
            # First ATR = average of first 14 TR values (index 0 to 13)
            # This matches Excel OFFSET(current,-13,0,14,1) logic
            atr_values[i] = tr_series.iloc[0:i+1].mean()
        else:
            # Subsequent ATR = 1/14*current_TR + (1-1/14)*previous_ATR
            # Exact Excel formula: 1/I$4*F2915+(1-1/I$4)*I2914
            prev_atr = atr_values[i-1]
            current_tr = tr_series.iloc[i]
            atr_values[i] = (1/period) * current_tr + (1 - 1/period) * prev_atr
    
    return atr_values

def calculate_atr_levels(ticker="^GSPC", atr_window=14):
    """
    Calculate ATR levels using the same methodology as run_generate.py
    Uses today's close + ATR to generate tomorrow's levels
    """
    try:
        # Fetch data - GET 6 MONTHS using period parameter
        # Yahoo Finance sometimes limits start/end date ranges, so use period instead
        
        print(f"🔍 DEBUG: Requesting 6 months of data for {ticker} using period='6mo'")
        
        spx = yf.Ticker(ticker)
        df = spx.history(period="6mo")  # Use period instead of start/end dates
        
        if len(df) < atr_window + 1:
            raise ValueError(f"Not enough data. Got {len(df)} days, need {atr_window + 1}")
        
        print(f"📊 Downloaded {len(df)} days of data for {ticker} for proper ATR baseline")
        print(f"📅 Date range: {df.index[0].strftime('%Y-%m-%d')} to {df.index[-1].strftime('%Y-%m-%d')}")
        print(f"🔄 Using 6 months of historical data for ATR convergence")
        
        # Calculate ATR using TRUE Wilder's method (Excel-matching)
        df["Prev_Close"] = df["Close"].shift(1)
        df["TR"] = df.apply(calculate_true_range, axis=1)
        
        # Use TRUE Wilder's ATR calculation (Excel-matching method)
        atr_values = calculate_true_wilders_atr(df["TR"], atr_window)
        df["ATR"] = atr_values
        
        # Use latest complete day's data (must have valid ATR)
        # Find the latest row with a valid ATR value
        valid_atr_rows = df[df["ATR"].notna()]
        
        # DEBUG: Check what's happening with valid ATR rows
        print(f"🔍 DEBUG INFO for {ticker}:")
        print(f"   Total downloaded rows: {len(df)}")
        print(f"   Valid ATR rows: {len(valid_atr_rows)}")
        if not valid_atr_rows.empty:
            print(f"   First valid ATR date: {valid_atr_rows.iloc[0].name.strftime('%Y-%m-%d')}")
            print(f"   Last valid ATR date: {valid_atr_rows.iloc[-1].name.strftime('%Y-%m-%d')}")
            print(f"   First valid ATR value: {valid_atr_rows.iloc[0]['ATR']:.4f}")
            print(f"   Last valid ATR value: {valid_atr_rows.iloc[-1]['ATR']:.4f}")
        
        if valid_atr_rows.empty:
            raise ValueError(f"No valid ATR values found. Need at least {atr_window + 1} days of data.")
        
        latest = valid_atr_rows.iloc[-1]
        close_price = latest["Close"]
        atr_value = latest["ATR"]
        
        # Validate data freshness (optional warning)
        latest_date = latest.name.date()
        today = datetime.now(pytz.timezone('US/Eastern')).date()
        days_old = (today - latest_date).days
        
        if days_old > 1:
            print(f"⚠️  Warning: Latest data for {ticker} is {days_old} days old ({latest_date})")
        
        # Generate Fibonacci levels (same as run_generate)
        fib_ratios = [1.0, 0.786, 0.618, 0.5, 0.382, 0.236, 0.0,
                      -0.236, -0.382, -0.5, -0.618, -0.786, -1.0]
        
        levels = {}
        for ratio in fib_ratios:
            level_price = close_price + (ratio * atr_value)
            levels[f"{ratio:+.3f}"] = round(level_price, 2)
        
        # Get timezone info
        eastern = pytz.timezone('US/Eastern')
        now = datetime.now(eastern)
        
        return {
            "generated_at": now.strftime("%Y-%m-%d %H:%M:%S %Z"),
            "generated_timestamp": now.timestamp(),
            "ticker": ticker,
            "reference_date": latest.name.strftime("%Y-%m-%d"),
            "reference_close": round(close_price, 2),
            "reference_atr": round(atr_value, 2),
            "data_age_days": days_old,
            "target_date": "next_trading_day",
            "levels": levels,
            "status": "success"
        }
        
    except Exception as e:
        eastern = pytz.timezone('US/Eastern')
        now = datetime.now(eastern)
        
        return {
            "generated_at": now.strftime("%Y-%m-%d %H:%M:%S %Z"),
            "generated_timestamp": now.timestamp(),
            "ticker": ticker,
            "error": str(e),
            "status": "error"
        }

def calculate_all_tickers():
    """Calculate ATR levels for all configured tickers"""
    results = {}
    
    for ticker_key, config in TICKER_CONFIG.items():
        print(f"\n{'='*50}")
        print(f"🔄 Processing {ticker_key}: {config['display_name']} ({config['symbol']})")
        print(f"{'='*50}")
        
        ticker_data = calculate_atr_levels(config['symbol'])
        
        # Add additional metadata
        ticker_data['ticker_key'] = ticker_key
        ticker_data['display_name'] = config['display_name']
        
        results[ticker_key] = ticker_data
        
        if ticker_data['status'] == 'success':
            print(f"✅ {ticker_key}: Close=${ticker_data['reference_close']:.2f}, ATR=${ticker_data['reference_atr']:.2f}")
        else:
            print(f"❌ {ticker_key}: ERROR - {ticker_data['error']}")
    
    return results

def save_levels_to_json(levels_data, filename="atr_levels.json"):
    """Save levels data to JSON file"""
    try:
        with open(filename, 'w') as f:
            json.dump(levels_data, f, indent=2)
        print(f"✅ Levels saved to {filename}")
        return True
    except Exception as e:
        print(f"❌ Error saving to {filename}: {e}")
        return False

def main():
    """Main execution function"""
    print("🔄 Calculating daily ATR levels for all tickers...")
    
    eastern = pytz.timezone('US/Eastern')
    start_time = datetime.now(eastern)
    print(f"🕐 Started at: {start_time.strftime('%Y-%m-%d %H:%M:%S %Z')}")
    
    # Calculate levels for all tickers
    all_ticker_data = calculate_all_tickers()
    
    # Create final data structure
    final_data = {
        "last_updated": start_time.strftime("%Y-%m-%d %H:%M:%S %Z"),
        "last_updated_timestamp": start_time.timestamp(),
        "tickers": all_ticker_data
    }
    
    # Count successes and failures
    success_count = sum(1 for data in all_ticker_data.values() if data['status'] == 'success')
    error_count = len(all_ticker_data) - success_count
    
    print(f"\n{'='*60}")
    print(f"📊 SUMMARY")
    print(f"{'='*60}")
    print(f"✅ Successful: {success_count}/{len(all_ticker_data)} tickers")
    print(f"❌ Errors: {error_count}/{len(all_ticker_data)} tickers")
    
    if success_count > 0:
        print(f"\n🎯 Successful Calculations:")
        for ticker_key, data in all_ticker_data.items():
            if data['status'] == 'success':
                print(f"   {ticker_key}: ${data['reference_close']:.2f} (ATR: ${data['reference_atr']:.2f}) [{data['reference_date']}]")
    
    if error_count > 0:
        print(f"\n❌ Failed Calculations:")
        for ticker_key, data in all_ticker_data.items():
            if data['status'] == 'error':
                print(f"   {ticker_key}: {data['error']}")
    
    # Save to JSON
    if save_levels_to_json(final_data):
        print(f"\n🎉 Daily multi-ticker ATR levels update complete!")
        
        end_time = datetime.now(eastern)
        duration = (end_time - start_time).total_seconds()
        print(f"⏱️  Total execution time: {duration:.1f} seconds")
        
        if error_count == 0:
            return 0  # All tickers successful
        elif success_count > 0:
            return 0  # At least some tickers successful (partial success)
        else:
            return 1  # All tickers failed
    else:
        print("❌ Failed to save levels")
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
