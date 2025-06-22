import pandas as pd

# --- Load retracement results ---
df = pd.read_csv("atr_trigger_goal_results.csv")

# --- Normalize 'GoalHit' to boolean ---
df['GoalHit'] = df['GoalHit'].map({'Yes': True, 'No': False})

# --- Calculate total triggers per TriggerLevel + TriggerTime ---
trigger_counts = df.groupby(['TriggerLevel', 'TriggerTime'])['Date'].nunique().reset_index()
trigger_counts.rename(columns={'Date': 'TotalTriggers'}, inplace=True)

# --- Calculate goal hits per TriggerLevel + TriggerTime + GoalLevel ---
goal_hits = df[df['GoalHit']].groupby(['TriggerLevel', 'TriggerTime', 'GoalLevel'])['Date'].nunique().reset_index()
goal_hits.rename(columns={'Date': 'GoalHits'}, inplace=True)

# --- Merge and calculate completion percentage ---
summary = pd.merge(goal_hits, trigger_counts, on=['TriggerLevel', 'TriggerTime'], how='left')
summary['Completion%'] = (summary['GoalHits'] / summary['TotalTriggers'] * 100).round(2)

# --- Now calculate total success (ANY goal hit) per TriggerLevel + TriggerTime ---
# This uses a drop-duplicates trick to count only once per trigger-day when any goal was hit
any_hits = df[df['GoalHit']].drop_duplicates(subset=['Date', 'TriggerLevel', 'TriggerTime'])
overall_success = any_hits.groupby(['TriggerLevel', 'TriggerTime'])['Date'].nunique().reset_index()
overall_success.rename(columns={'Date': 'AnyGoalHits'}, inplace=True)

# Merge into main summary
summary = pd.merge(summary, overall_success, on=['TriggerLevel', 'TriggerTime'], how='left')
summary['AnyGoalHits'].fillna(0, inplace=True)
summary['OverallCompletion%'] = (summary['AnyGoalHits'] / summary['TotalTriggers'] * 100).round(2)

# --- Final sorting ---
summary = summary.sort_values(by=['TriggerLevel', 'TriggerTime', 'GoalLevel'])

# --- Save to CSV ---
summary.to_csv("retracement_summary_by_hour.csv", index=False)
print("✅ Summary saved to retracement_summary_by_hour.csv")
