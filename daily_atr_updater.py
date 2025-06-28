# Daily ATR Levels Updater - Windows Compatible
"""
Daily ATR Levels Updater
Runs at 5:16 PM Eastern to calculate next day's ATR levels
Saves results to JSON file for dashboard consumption
"""

import yfinance as yf
import pandas as pd
import json
from datetime import datetime
import pytz
import sys
import os

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
        # Fetch data - GET 6 MONTHS EXTRA for proper ATR baseline
        # Based on testing: 3-4 months needed for convergence, 6 months for safety
        from datetime import datetime, timedelta
        
        # Calculate start date: 6 months ago
        six_months_ago = datetime.now() - timedelta(days=180)
        start_date = six_months_ago.strftime("%Y-%m-%d")
        
        spx = yf.Ticker(ticker)
        df = spx.history(start=start_date, end=None)
        
        if len(df) < atr_window + 1:
            raise ValueError(f"Not enough data. Got {len(df)} days, need {atr_window + 1}")
        
        print(f"📊 Downloaded {len(df)} days of data for proper ATR baseline")
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
        print(f"🔍 DEBUG INFO:")
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
            print(f"⚠️  Warning: Latest data is {days_old} days old ({latest_date})")
        
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
            "error": str(e),
            "status": "error"
        }

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
    print("🔄 Calculating daily ATR levels...")
    
    # Calculate levels
    levels_data = calculate_atr_levels()
    
    if levels_data["status"] == "success":
        print(f"✅ Levels calculated successfully")
        print(f"📊 Reference: {levels_data['reference_date']} Close={levels_data['reference_close']}, ATR={levels_data['reference_atr']}")
        print(f"🎯 Generated at: {levels_data['generated_at']}")
        
        # Save to JSON
        if save_levels_to_json(levels_data):
            print("🎉 Daily ATR levels update complete!")
            return 0
        else:
            print("❌ Failed to save levels")
            return 1
    else:
        print(f"❌ Error calculating levels: {levels_data['error']}")
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
