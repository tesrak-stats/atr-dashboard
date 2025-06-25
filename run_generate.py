import streamlit as st
import pandas as pd
import os

def detect_triggers_and_goals(daily, intraday):
fib_levels = [1.000, 0.786, 0.618, 0.500, 0.382, 0.236, 0.000,
-0.236, -0.382, -0.500, -0.618, -0.786, -1.000]

```
results = []

for i in range(1, len(daily)):
    date = daily.iloc[i]['Date']
    
    # BULLETPROOF string comparison
    try:
        if hasattr(date, 'strftime'):
            date_str = date.strftime('%Y-%m-%d')
        elif isinstance(date, str):
            date_str = date[:10]
        else:
            date_str = str(date)[:10]
        
        # Simple string comparison
        if date_str < "2014-01-02":
            continue
            
    except Exception as e:
        continue

    day_row = daily.iloc[i]
    prev_row = daily.iloc[i - 1]

    level_map = {}
    for level in fib_levels:
        level_str = f"{level:.3f}".rstrip('0').rstrip('.') if '.' in f"{level:.3f}" else str(level)
        if level_str in day_row:
            level_map[level] = day_row[level_str]

    day_data = intraday[intraday['Date'] == pd.to_datetime(date).date()].copy()
    if day_data.empty:
        continue

    day_data['Time'] = day_data['Datetime'].dt.strftime('%H%M')
    day_data.reset_index(drop=True, inplace=True)

    triggered_up = {}
    triggered_down = {}

    for idx, row in day_data.iterrows():
        open_price = row['Open']
        high = row['High']
        low = row['Low']
        time_label = '0000' if idx == 0 and (open_price >= min(level_map.values())) else row['Time']
        hour_block = '0000' if time_label == '0000' else time_label[:2] + '00'

        for level in sorted([lvl for lvl in fib_levels if lvl > 0]):
            if level in triggered_up:
                continue
            if high >= level_map.get(level, float('inf')):
                triggered_up[level] = {
                    'TriggerLevel': level,
                    'TriggerTime': time_label,
                    'TriggeredRow': idx
                }

        for level in sorted([lvl for lvl in fib_levels if lvl < 0], reverse=True):
            if level in triggered_down:
                continue
            if low <= level_map.get(level, float('-inf')):
                triggered_down[level] = {
                    'TriggerLevel': level,
                    'TriggerTime': time_label,
                    'TriggeredRow': idx
                }

    for level, trigger_info in triggered_up.items():
        for goal_level in [l for l in fib_levels if l > level]:
            goal_hit = False
            goal_time = ''
            for _, row in day_data.iloc[trigger_info['TriggeredRow']+1:].iterrows():
                if row['High'] >= level_map.get(goal_level, float('inf')):
                    goal_hit = True
                    goal_time = row['Time']
                    break
            results.append({
                'Date': date,
                'Direction': 'Upside',
                'TriggerLevel': level,
                'TriggerTime': trigger_info['TriggerTime'],
                'GoalLevel': goal_level,
                'GoalHit': 'Yes' if goal_hit else 'No',
                'GoalTime': goal_time if goal_hit else '',
                'Type': 'Continuation',
                'RetestedTrigger': 'No'
            })

        for retrace_level in [l for l in fib_levels if l < 0]:
            goal_hit = False
            goal_time = ''
            for _, row in day_data.iloc[trigger_info['TriggeredRow']+1:].iterrows():
                if row['Low'] <= level_map.get(retrace_level, float('-inf')):
                    goal_hit = True
                    goal_time = row['Time']
                    break
            results.append({
                'Date': date,
                'Direction': 'Upside',
                'TriggerLevel': level,
                'TriggerTime': trigger_info['TriggerTime'],
                'GoalLevel': retrace_level,
                'GoalHit': 'Yes' if goal_hit else 'No',
                'GoalTime': goal_time if goal_hit else '',
                'Type': 'Retracement',
                'RetestedTrigger': 'No'
            })

    for level, trigger_info in triggered_down.items():
        for goal_level in [l for l in fib_levels if l < level]:
            goal_hit = False
            goal_time = ''
            for _, row in day_data.iloc[trigger_info['TriggeredRow']+1:].iterrows():
                if row['Low'] <= level_map.get(goal_level, float('-inf')):
                    goal_hit = True
                    goal_time = row['Time']
                    break
            results.append({
                'Date': date,
                'Direction': 'Downside',
                'TriggerLevel': level,
                'TriggerTime': trigger_info['TriggerTime'],
                'GoalLevel': goal_level,
                'GoalHit': 'Yes' if goal_hit else 'No',
                'GoalTime': goal_time if goal_hit else '',
                'Type': 'Continuation',
                'RetestedTrigger': 'No'
            })

        for retrace_level in [l for l in fib_levels if l > 0]:
            goal_hit = False
            goal_time = ''
            for _, row in day_data.iloc[trigger_info['TriggeredRow']+1:].iterrows():
                if row['High'] >= level_map.get(retrace_level, float('inf')):
                    goal_hit = True
                    goal_time = row['Time']
                    break
            results.append({
                'Date': date,
                'Direction': 'Downside',
                'TriggerLevel': level,
                'TriggerTime': trigger_info['TriggerTime'],
                'GoalLevel': retrace_level,
                'GoalHit': 'Yes' if goal_hit else 'No',
                'GoalTime': goal_time if goal_hit else '',
                'Type': 'Retracement',
                'RetestedTrigger': 'No'
            })

return pd.DataFrame(results)
```

def main():
daily = pd.read_excel(‘SPXdailycandles.xlsx’, header=4)
intraday = pd.read_csv(‘SPX_10min.csv’, parse_dates=[‘Datetime’])
intraday[‘Date’] = intraday[‘Datetime’].dt.date

```
df = detect_triggers_and_goals(daily, intraday)
df.to_csv('combined_trigger_goal_results.csv', index=False)
print('✅ Output saved to combined_trigger_goal_results.csv')
return df
```

st.title(“📊 ATR Trigger & Goal Generator”)

output_path = “combined_trigger_goal_results.csv”

if st.button(“Generate combined_trigger_goal_results.csv”):
with st.spinner(“Running detection…”):
try:
result_df = main()
result_df[“Source”] = “Full”
result_df.to_csv(output_path, index=False)
st.success(“✅ File generated and saved!”)

```
        # Preview
        if os.path.exists(output_path):
            df = pd.read_csv(output_path)
            st.subheader("🔍 Preview of Most Recent Output")
            st.dataframe(df.head(30))
            st.download_button("⬇️ Download CSV", data=df.to_csv(index=False), file_name=output_path, mime="text/csv")
    except Exception as e:
        st.error(f"❌ Error running detect_triggers_and_goals.py: {e}")
```