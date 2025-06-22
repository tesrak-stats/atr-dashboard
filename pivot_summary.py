import pandas as pd

# Load summary file
summary_path = "retracement_summary_by_hour.csv"
df = pd.read_csv(summary_path)

# Calculate percent completions (fixed column name)
df['PercentCompletion'] = (df['GoalHits'] / df['TotalTriggers']) * 100

# Optional: clean TriggerTime
df['TriggerTime'] = df['TriggerTime'].astype(str).str.zfill(4).str.upper()

# Pivot: GoalLevel as rows, TriggerTime as columns
pivot = df.pivot_table(
    index='GoalLevel',
    columns='TriggerTime',
    values='PercentCompletion',
    aggfunc='mean'
)

# Sort goal levels descending
pivot = pivot.sort_index(ascending=False)

# Save
pivot.to_csv("retracement_percent_matrix.csv")
print("✅ Matrix saved to retracement_percent_matrix.csv")
