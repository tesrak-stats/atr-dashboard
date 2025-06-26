import pandas as pd

# Load combined results
df = pd.read_csv("combined_trigger_goal_results.csv")

print(f"📊 Loaded {len(df)} total records")

# Check if 0.0 exists in raw data
zero_triggers = df[df['TriggerLevel'] == 0.0]
print(f"🔍 Found {len(zero_triggers)} records with TriggerLevel = 0.0")

# Count unique triggers per day
trigger_occurrences = df[['Date', 'TriggerLevel', 'TriggerTime', 'Direction']].drop_duplicates()

print(f"📈 Total unique trigger occurrences: {len(trigger_occurrences)}")

# Check if 0.0 survives the deduplication
zero_after_dedup = trigger_occurrences[trigger_occurrences['TriggerLevel'] == 0.0]
print(f"🔍 Zero triggers after deduplication: {len(zero_after_dedup)}")

trigger_counts = (
    trigger_occurrences
    .value_counts(subset=['TriggerLevel', 'TriggerTime', 'Direction'])
    .reset_index()
)

trigger_counts.columns = ['TriggerLevel', 'TriggerTime', 'Direction', 'NumTriggers']

# Check if 0.0 is in trigger_counts
zero_in_counts = trigger_counts[trigger_counts['TriggerLevel'] == 0.0]
print(f"🔍 Zero triggers in trigger_counts: {len(zero_in_counts)}")
if len(zero_in_counts) > 0:
    print("Zero trigger details:")
    print(zero_in_counts)

# Count successful goal hits per group
goal_hits = df[df['GoalHit'] == 'Yes']

print(f"🎯 Total goal hits: {len(goal_hits)}")

# Check if 0.0 triggers ever hit goals
zero_goals = goal_hits[goal_hits['TriggerLevel'] == 0.0]
print(f"🔍 Goal hits FROM 0.0 triggers: {len(zero_goals)}")

goal_counts = (
    goal_hits
    .groupby(['TriggerLevel', 'TriggerTime', 'Direction', 'GoalLevel', 'GoalTime'])
    .size()
    .reset_index(name='NumHits')
)

# Check if 0.0 survives goal grouping
zero_in_goal_counts = goal_counts[goal_counts['TriggerLevel'] == 0.0]
print(f"🔍 Zero triggers in goal_counts: {len(zero_in_goal_counts)}")

# Merge hits with trigger totals
summary = pd.merge(
    goal_counts,
    trigger_counts,
    on=['TriggerLevel', 'TriggerTime', 'Direction'],
    how='left'
)

print(f"📋 Final summary records: {len(summary)}")

# Check final result
zero_in_summary = summary[summary['TriggerLevel'] == 0.0]
print(f"🔍 Zero triggers in final summary: {len(zero_in_summary)}")

# Calculate % completion
summary['PctCompletion'] = (summary['NumHits'] / summary['NumTriggers'] * 100).round(2)

# Final column order
summary = summary[[
    'Direction',
    'TriggerLevel',
    'TriggerTime',
    'GoalLevel',
    'GoalTime',
    'NumTriggers',
    'NumHits',
    'PctCompletion'
]]

# Save
summary.to_csv("atr_dashboard_summary.csv", index=False)
print("✅ Summary saved to atr_dashboard_summary.csv")

# Final check of unique trigger levels in summary
unique_levels = sorted(summary['TriggerLevel'].unique())
print(f"🎯 Final unique trigger levels: {unique_levels}")
