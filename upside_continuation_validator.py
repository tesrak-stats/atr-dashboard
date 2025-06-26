import streamlit as st
import pandas as pd
import numpy as np

def calculate_atr(df, period=14):
    """Calculate Wilder's ATR"""
    df = df.copy()
    df['High-Low'] = df['High'] - df['Low']
    df['High-PrevClose'] = abs(df['High'] - df['Close'].shift(1))
    df['Low-PrevClose'] = abs(df['Low'] - df['Close'].shift(1))
    df['TR'] = df[['High-Low', 'High-PrevClose', 'Low-PrevClose']].max(axis=1)
    df['ATR'] = df['TR'].ewm(alpha=1/period, adjust=False).mean()
    return df

def get_fib_levels(prev_close, atr):
    """Generate Fibonacci-based ATR levels"""
    fib_ratios = [1.0, 0.786, 0.618, 0.5, 0.382, 0.236, 0.0, -0.236, -0.382, -0.5, -0.618, -0.786, -1.0]
    levels = {}
    for ratio in fib_ratios:
        levels[ratio] = prev_close + (atr * ratio)
    return levels

st.title("🔍 Core Logic Validator - From Raw Data")
st.write("Validate the fundamental trigger detection and goal classification logic")

# File uploads
col1, col2 = st.columns(2)
with col1:
    daily_file = st.file_uploader("Upload SPXdailycandles.xlsx", type="xlsx")
with col2:
    intraday_file = st.file_uploader("Upload SPX_10min.csv", type="csv")

if daily_file is not None and intraday_file is not None:
    # Load daily data for ATR calculation
    st.write("## Step 1: Load and Process Daily Data")
    
    try:
        # Read Excel file (assuming header on row 5 as mentioned)
        daily_df = pd.read_excel(daily_file, header=4)
        st.write(f"✅ Loaded daily data: {len(daily_df)} rows")
        st.write("Daily data columns:", list(daily_df.columns))
        st.dataframe(daily_df.head())
        
        # Calculate ATR
        daily_with_atr = calculate_atr(daily_df)
        st.write("✅ ATR calculated")
        
    except Exception as e:
        st.error(f"Error loading daily data: {e}")
        st.stop()
    
    # Load intraday data
    st.write("## Step 2: Load Intraday Data")
    
    try:
        intraday_df = pd.read_csv(intraday_file)
        st.write(f"✅ Loaded intraday data: {len(intraday_df)} rows")
        st.write("Intraday data columns:", list(intraday_df.columns))
        st.dataframe(intraday_df.head())
        
    except Exception as e:
        st.error(f"Error loading intraday data: {e}")
        st.stop()
    
    # Step 3: Pick a specific test date
    st.write("## Step 3: Test Specific Date")
    
    # Let user pick a test date
    available_dates = sorted(daily_with_atr['Date'].unique()) if 'Date' in daily_with_atr.columns else []
    
    if available_dates:
        test_date = st.selectbox("Select test date:", available_dates[-30:])  # Last 30 dates
        
        if test_date:
            st.write(f"### Testing date: {test_date}")
            
            # Get daily data for test date
            daily_row = daily_with_atr[daily_with_atr['Date'] == test_date]
            
            if len(daily_row) > 0:
                daily_row = daily_row.iloc[0]
                prev_close = daily_row['Close']
                atr = daily_row['ATR']
                
                st.write(f"**Previous Close**: {prev_close:.2f}")
                st.write(f"**ATR**: {atr:.2f}")
                
                # Generate Fibonacci levels
                fib_levels = get_fib_levels(prev_close, atr)
                
                st.write("### Generated Fibonacci Levels:")
                levels_df = pd.DataFrame([
                    {'Ratio': k, 'Price Level': v} 
                    for k, v in fib_levels.items()
                ])
                st.dataframe(levels_df)
                
                # Step 4: Analyze intraday data for this date
                st.write("## Step 4: Analyze Intraday Price Action")
                
                # Filter intraday data for test date
                intraday_day = intraday_df[intraday_df['Date'] == test_date] if 'Date' in intraday_df.columns else pd.DataFrame()
                
                if len(intraday_day) > 0:
                    st.write(f"✅ Found {len(intraday_day)} intraday records for {test_date}")
                    st.dataframe(intraday_day.head(10))
                    
                    # Step 5: Test trigger detection logic
                    st.write("## Step 5: Test Trigger Detection")
                    
                    # For each Fibonacci level, check if price hit it
                    triggers_found = []
                    
                    for ratio, level_price in fib_levels.items():
                        # Check if high >= level (upside trigger) or low <= level (downside trigger)
                        upside_hits = intraday_day[intraday_day['High'] >= level_price]
                        downside_hits = intraday_day[intraday_day['Low'] <= level_price]
                        
                        if len(upside_hits) > 0:
                            first_hit = upside_hits.iloc[0]
                            triggers_found.append({
                                'Level': ratio,
                                'Price': level_price,
                                'Direction': 'Upside',
                                'TriggerTime': first_hit.get('Time', 'Unknown'),
                                'ActualPrice': first_hit['High']
                            })
                        
                        if len(downside_hits) > 0:
                            first_hit = downside_hits.iloc[0]
                            triggers_found.append({
                                'Level': ratio,
                                'Price': level_price,
                                'Direction': 'Downside', 
                                'TriggerTime': first_hit.get('Time', 'Unknown'),
                                'ActualPrice': first_hit['Low']
                            })
                    
                    if triggers_found:
                        triggers_df = pd.DataFrame(triggers_found)
                        st.write("### Triggers Found:")
                        st.dataframe(triggers_df)
                        
                        # Step 6: Test goal classification logic
                        st.write("## Step 6: Test Goal Classification")
                        
                        # For upside triggers, check which higher levels were hit
                        upside_triggers = triggers_df[triggers_df['Direction'] == 'Upside']
                        
                        if len(upside_triggers) > 0:
                            st.write("### Upside Trigger Analysis:")
                            
                            for _, trigger in upside_triggers.iterrows():
                                trigger_level = trigger['Level']
                                st.write(f"**Trigger**: {trigger_level} at {trigger['TriggerTime']}")
                                
                                # Find potential goals (levels higher than trigger for continuation)
                                potential_goals = [lvl for lvl in fib_levels.keys() if lvl > trigger_level]
                                
                                goals_hit = []
                                for goal_level in potential_goals:
                                    goal_price = fib_levels[goal_level]
                                    goal_hits = intraday_day[intraday_day['High'] >= goal_price]
                                    
                                    if len(goal_hits) > 0:
                                        first_goal_hit = goal_hits.iloc[0]
                                        goals_hit.append({
                                            'GoalLevel': goal_level,
                                            'GoalPrice': goal_price,
                                            'GoalTime': first_goal_hit.get('Time', 'Unknown'),
                                            'Classification': 'Continuation'
                                        })
                                
                                if goals_hit:
                                    goals_df = pd.DataFrame(goals_hit)
                                    st.dataframe(goals_df)
                                else:
                                    st.write("  No continuation goals hit")
                        
                        # Summary for validation
                        st.write("## Step 7: Validation Summary")
                        st.write(f"**Test Date**: {test_date}")
                        st.write(f"**Previous Close**: {prev_close:.2f}")
                        st.write(f"**ATR**: {atr:.2f}")
                        st.write(f"**Total Triggers**: {len(triggers_found)}")
                        st.write(f"**Upside Triggers**: {len(upside_triggers)}")
                        
                        # Focus on 0.0 level
                        zero_triggers = triggers_df[triggers_df['Level'] == 0.0]
                        if len(zero_triggers) > 0:
                            st.success(f"✅ Found {len(zero_triggers)} triggers at 0.0 (Previous Close)")
                        else:
                            st.warning("⚠️ No triggers found at 0.0 level")
                    
                    else:
                        st.warning("No triggers found for this date")
                else:
                    st.error(f"No intraday data found for {test_date}")
            else:
                st.error(f"No daily data found for {test_date}")
    else:
        st.error("No dates available in daily data")

else:
    st.info("👆 Please upload both files to begin validation")

st.write("---")
st.write("**Goal**: Verify that our logic matches your Excel methodology for detecting triggers and classifying goals")
