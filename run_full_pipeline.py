
import streamlit as st
import pandas as pd
import numpy as np

st.title("🧠 Full Trigger → Summary Generator (Patched)")

def detect_triggers_and_goals():
    st.write("📥 Reading input files...")
    daily = pd.read_excel("SPXdailycandles.xlsx", header=4)
    daily.columns = daily.columns.astype(str)  # 🔧 Fix column key error
    intraday = pd.read_csv("SPX_10min.csv", parse_dates=["Datetime"])

    intraday["Date"] = intraday["Datetime"].dt.date
    intraday["TimeBlock"] = intraday["Datetime"].dt.strftime("%H00")
    intraday.loc[intraday["TimeBlock"] == "0930", "TimeBlock"] = "OPEN"

    st.write("📊 Preview of column names from daily data:")
    st.write(daily.columns.tolist())

    fib_labels = [1.0, 0.786, 0.618, 0.5, 0.382, 0.236, 0.0, -0.236, -0.382, -0.5, -0.618, -0.786, -1.0]
    fib_labels_str = [str(f) for f in fib_labels]

    results = []

    for date in intraday["Date"].unique():
        try:
            day_intraday = intraday[intraday["Date"] == date]
            day_daily = daily[daily["Date"] == pd.to_datetime(date)]
            if day_daily.empty:
                continue
            row = day_daily.iloc[0]

            for direction in ["Upside", "Downside"]:
                for trigger_level in fib_labels_str:
                    if trigger_level not in row or pd.isna(row[trigger_level]):
                        continue

                    trigger_price = row[trigger_level]
                    if direction == "Upside":
                        level_col = "High"
                        hit = day_intraday[day_intraday[level_col] >= trigger_price]
                    else:
                        level_col = "Low"
                        hit = day_intraday[day_intraday[level_col] <= trigger_price]

                    if hit.empty:
                        continue

                    first_hit = hit.iloc[0]
                    trigger_time = first_hit["TimeBlock"]
                    trigger_candle_index = day_intraday.index.get_loc(first_hit.name)

                    for goal_level in fib_labels_str:
                        if goal_level == trigger_level:
                            continue

                        # Skip invalid direction goals
                        if direction == "Upside" and float(goal_level) <= float(trigger_level):
                            continue
                        if direction == "Downside" and float(goal_level) >= float(trigger_level):
                            continue

                        if goal_level not in row or pd.isna(row[goal_level]):
                            continue

                        goal_price = row[goal_level]
                        goal_hit = False
                        goal_time = None

                        if float(goal_level) > float(trigger_level):
                            scan = day_intraday.iloc[trigger_candle_index:]
                            price_col = "High" if direction == "Upside" else "Low"
                            goal_hit_rows = scan[scan[price_col] >= goal_price] if direction == "Upside" else scan[scan[price_col] <= goal_price]
                        else:
                            scan = day_intraday.iloc[trigger_candle_index + 1:]
                            price_col = "Low" if direction == "Upside" else "High"
                            goal_hit_rows = scan[scan[price_col] <= goal_price] if direction == "Upside" else scan[scan[price_col] >= goal_price]

                        if not goal_hit_rows.empty:
                            goal_time = goal_hit_rows.iloc[0]["TimeBlock"]
                            goal_hit = True

                        results.append({
                            "Date": date,
                            "Direction": direction,
                            "TriggerLevel": float(trigger_level),
                            "TriggerTime": trigger_time,
                            "GoalLevel": float(goal_level),
                            "GoalTime": goal_time if goal_time else "",
                            "GoalHit": "Yes" if goal_hit else "No"
                        })
        except Exception as e:
            st.error(f"⚠️ Error processing {date}: {e}")
            continue

    df_out = pd.DataFrame(results)
    st.write("✅ Detection complete. Sample output:")
    st.dataframe(df_out.head())
    return df_out

def generate_summary(df):
    st.write("📊 Generating summary...")
    if not all(col in df.columns for col in ["TriggerTime", "GoalTime", "Date", "Direction"]):
        st.error("❌ Required columns missing. Cannot generate summary.")
        return pd.DataFrame()

    df["TriggerTime"] = df["TriggerTime"].fillna("").astype(str)
    df["GoalTime"] = df["GoalTime"].fillna("").astype(str)

    trigger_occurrences = df[['Date', 'TriggerLevel', 'TriggerTime', 'Direction']].drop_duplicates()
    trigger_counts = (
        trigger_occurrences
        .value_counts(subset=['TriggerLevel', 'TriggerTime', 'Direction'])
        .reset_index()
    )
    trigger_counts.columns = ['TriggerLevel', 'TriggerTime', 'Direction', 'NumTriggers']

    goal_hits = df[df['GoalHit'] == 'Yes']
    goal_counts = (
        goal_hits
        .groupby(['TriggerLevel', 'TriggerTime', 'Direction', 'GoalLevel', 'GoalTime'])
        .size()
        .reset_index(name='NumHits')
    )

    summary = pd.merge(
        goal_counts,
        trigger_counts,
        on=['TriggerLevel', 'TriggerTime', 'Direction'],
        how='left'
    )
    summary['PctCompletion'] = (summary['NumHits'] / summary['NumTriggers'] * 100).round(2)

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

    return summary

if st.button("🔁 Run Full Trigger + Summary Pipeline"):
    with st.spinner("Running full logic..."):
        df_trigger = detect_triggers_and_goals()
        if not df_trigger.empty:
            df_trigger.to_csv("combined_trigger_goal_results.csv", index=False)
            df_summary = generate_summary(df_trigger)
            if not df_summary.empty:
                df_summary.to_csv("atr_dashboard_summary.csv", index=False)
                st.success("✅ Files generated.")
                st.subheader("📄 Preview of atr_dashboard_summary.csv")
                st.dataframe(df_summary.head(25))
                st.download_button(
                    label="⬇️ Download atr_dashboard_summary.csv",
                    data=df_summary.to_csv(index=False),
                    file_name="atr_dashboard_summary.csv",
                    mime="text/csv"
                )
