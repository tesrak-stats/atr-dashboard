
import streamlit as st
import pandas as pd
import os

st.title("📊 Generate ATR Dashboard Summary")

def generate_summary():
    df = pd.read_csv("combined_trigger_goal_results.csv")
    df["TriggerTime"] = df["TriggerTime"].astype(str)
    df["GoalTime"] = df["GoalTime"].astype(str)

    # Count unique triggers per day
    trigger_occurrences = df[['Date', 'TriggerLevel', 'TriggerTime', 'Direction']].drop_duplicates()

    trigger_counts = (
        trigger_occurrences
        .value_counts(subset=['TriggerLevel', 'TriggerTime', 'Direction'])
        .reset_index()
    )
    trigger_counts.columns = ['TriggerLevel', 'TriggerTime', 'Direction', 'NumTriggers']

    # Count successful goal hits per group
    goal_hits = df[df['GoalHit'] == 'Yes']

    goal_counts = (
        goal_hits
        .groupby(['TriggerLevel', 'TriggerTime', 'Direction', 'GoalLevel', 'GoalTime'])
        .size()
        .reset_index(name='NumHits')
    )

    # Merge hits with trigger totals
    summary = pd.merge(
        goal_counts,
        trigger_counts,
        on=['TriggerLevel', 'TriggerTime', 'Direction'],
        how='left'
    )

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

    summary.to_csv("atr_dashboard_summary.csv", index=False)
    return summary

if st.button("Generate atr_dashboard_summary.csv"):
    with st.spinner("Processing summary..."):
        summary = generate_summary()
    st.success("✅ Summary file generated!")

    st.subheader("🔍 Preview of Summary")
    st.dataframe(summary.head(25))
