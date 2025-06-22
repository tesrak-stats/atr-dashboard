import pandas as pd

# Load the corrected trigger-goal results
df = pd.read_csv("combined_trigger_goal_results.csv")

# Normalize TriggerTime and GoalTime as integers (0900, 1000, etc.)
df["TriggerTime"] = df["TriggerTime"].str.replace(":", "").str.zfill(4)
df["GoalTime"] = df["GoalTime"].astype(str).str.replace(":", "").str.zfill(4)

# Keep only successful hits
success_df = df[df["GoalHit"] == "Yes"].copy()

# Count goal hits
hit_counts = success_df.groupby([
    "Direction", "TriggerLevel", "TriggerTime", "GoalLevel", "GoalTime"
]).size().reset_index(name="NumHits")

# Count unique triggers for each (Direction, TriggerLevel, TriggerTime)
unique_triggers = df[["Date", "Direction", "TriggerLevel", "TriggerTime"]].drop_duplicates()
trigger_counts = (
    unique_triggers
    .groupby(["Direction", "TriggerLevel", "TriggerTime"])
    .size()
    .reset_index(name="NumTriggers")
)

# Merge
summary = pd.merge(hit_counts, trigger_counts, on=["Direction", "TriggerLevel", "TriggerTime"], how="left")
summary["PctCompletion"] = (summary["NumHits"] / summary["NumTriggers"] * 100).round(2)

# Save
summary = summary.sort_values(["Direction", "TriggerLevel", "TriggerTime", "GoalLevel", "GoalTime"])
summary.to_csv("atr_matrix_summary_by_goal_time.csv", index=False)
print("✅ Saved to atr_matrix_summary_by_goal_time.csv")
