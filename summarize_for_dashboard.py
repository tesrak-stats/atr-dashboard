import pandas as pd

# --- Load the combined results file ---
df = pd.read_csv("combined_trigger_goal_results.csv")

# --- Count unique triggers per day ---
trigger_occurrences = df[['Date', 'TriggerLevel', 'TriggerTime', 'Direction']].drop_duplicates()

trigger_counts = (
    trigger_occurrences
    .value_counts(subset=['TriggerLevel', 'TriggerTime', 'Direction'])
    .reset_index()
)

trigger_counts.columns = ['TriggerLevel', 'TriggerTime', 'Direction', 'NumTriggers']

# --- Count successful goal hits per group ---
goal_hits = df[df['GoalHit'] == 'Yes']

goal_counts = (
    goal_hits
    .groupby(['TriggerLevel', 'TriggerTime', 'Direction', 'GoalLevel'])
    .size()
    .reset_index(name='NumHits')
)

# --- Merge hits with trigger totals ---
summary = pd.merge(
    goal_counts,
    trigger_counts,
    on=['TriggerLevel', 'TriggerTime', 'Direction'],
    how='left'
)

# --- Calculate % completion ---
summary['PctCompletion'] = (summary['NumHits'] / summary['NumTriggers'] * 100).round(2)

# --- Final column order ---
summary = summary[[
    'Direction',
    'TriggerLevel',
    'TriggerTime',
    'GoalLevel',
    'NumTriggers',
    'NumHits',
    'PctCompletion'
]]

# --- Save to CSV ---
summary.to_csv("atr_dashboard_summary.csv", index=False)

print("✅ Summary saved to atr_dashboard_summary.csv")
print(summary.head(10))
