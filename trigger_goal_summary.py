import pandas as pd

# Load the combined trigger-goal results
df = pd.read_csv("combined_trigger_goal_results.csv")

# Group and count occurrences
summary = (
    df.groupby(['TriggerLevel', 'GoalLevel', 'TriggerTime'])
    .agg(
        Triggers=('GoalHit', 'count'),
        GoalsHit=('GoalHit', lambda x: (x == 'Yes').sum())
    )
    .reset_index()
)

# Calculate completion rate
summary['CompletionRate'] = (summary['GoalsHit'] / summary['Triggers'] * 100).round(2)

# Save to CSV
summary.to_csv("trigger_goal_summary.csv", index=False)

print("✅ Summary saved to trigger_goal_summary.csv")
print(summary.head(10))
