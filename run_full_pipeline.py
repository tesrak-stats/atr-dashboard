
import streamlit as st
import pandas as pd
import numpy as np

st.title("🧠 Full Trigger → Summary Generator (Debug Mode)")

def detect_triggers_and_goals():
    st.write("📥 Reading input files...")
    daily = pd.read_excel("SPXdailycandles.xlsx", skiprows=4)
    intraday = pd.read_csv("SPX_10min.csv", parse_dates=["Datetime"])

    intraday["Date"] = intraday["Datetime"].dt.date
    intraday["TimeBlock"] = intraday["Datetime"].dt.strftime("%H00")
    intraday.loc[intraday["TimeBlock"] == "0930", "TimeBlock"] = "OPEN"

    # Show debug info for first row of levels
    st.write("🔎 Preview of daily.iloc[0, 9:22] (should contain fib levels):")
    st.write(daily.iloc[0, 9:22])

    try:
        level_row = daily.iloc[0, 9:22]
        fib_levels = level_row.values.astype(float)
    except Exception as e:
        st.error(f"❌ Error reading fib levels: {e}")
        return pd.DataFrame()

    fib_labels = [1.0, 0.786, 0.618, 0.5, 0.382, 0.236, 0.0, -0.236, -0.382, -0.5, -0.618, -0.786, -1.0]
    fib_map = dict(zip(fib_labels, fib_levels))

    results = []
    for date in intraday["Date"].unique():
        try:
            day_intraday = intraday[intraday["Date"] == date]
            day_daily = daily[daily["Date"] == pd.to_datetime(date)]
            if day_daily.empty:
                continue

            for direction in ["Upside", "Downside"]:
                for trigger_level in fib_labels:
                    if direction == "Upside":
                        trigger_price = fib_map.get(trigger_level)
                        level_col = "High"
                        hit = day_intraday[day_intraday[level_col] >= trigger_price]
                    else:
                        trigger_price = fib_map.get(trigger_level)
                        level_col = "Low"
                        hit = day_intraday[day_intraday[level_col] <= trigger_price]

                    if hit.empty:
                        continue

                    first_hit = hit.iloc[0]
                    trigger_time = first_hit["TimeBlock"]
                    trigger_candle_index = day_intraday.index.get_loc(first_hit.name)

                    for goal_level in fib_labels:
                        if goal_level == trigger_level:
                            continue

                        is_continuation = (goal_level > trigger_level) if direction == "Upside" else (goal_level < trigger_level)
                        goal_price = fib_map.get(goal_level)
                        if goal_price is None:
                            continue

                        goal_hit = False
                        goal_time = None

                        if is_continuation:
                            scan = day_intraday.iloc[trigger_candle_index:]
                            price_col = "High" if direction == "Upside" else "Low"
                            goal_hit_rows = scan[scan[price_col] >= goal_price] if direction == "Upside" else scan[scan[price_col] <= goal_price]
                            if not goal_hit_rows.empty:
                                goal_time = goal_hit_rows.iloc[0]["TimeBlock"]
                                goal_hit = True
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
                            "TriggerLevel": trigger_level,
                            "TriggerTime": trigger_time,
                            "GoalLevel": goal_level,
                            "GoalTime": goal_time if goal_time else "",
                            "GoalHit": "Yes" if goal_hit else "No"
                        })
        except Exception as e:
            st.error(f"⚠️ Error processing {date}: {e}")
            continue

    df_out = pd.DataFrame(results)
    st.write("✅ Finished detection. Preview of results:")
    st.dataframe(df_out.head())
    return df_out

def generate_summary(df):
    st.write("📊 Generating summary...")

    required_cols = ['TriggerTime', 'GoalTime', 'Date', 'Direction']
    if not all(col in df.columns for col in required_cols):
        st.error(f"❌ Required columns missing from trigger data: {set(required_cols) - set(df.columns)}")
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

if st.button("🔁 Run Full Trigger + Summary Pipeline (Debug Mode)"):
    with st.spinner("Running full logic..."):
        df_trigger = detect_triggers_and_goals()
        if not df_trigger.empty:
            df_trigger.to_csv("combined_trigger_goal_results.csv", index=False)
            df_summary = generate_summary(df_trigger)
            if not df_summary.empty:
                df_summary.to_csv("atr_dashboard_summary.csv", index=False)
                st.success("✅ Both files generated!")
                st.subheader("📄 Preview of atr_dashboard_summary.csv")
                st.dataframe(df_summary.head(25))
                st.download_button(
                    label="⬇️ Download atr_dashboard_summary.csv",
                    data=df_summary.to_csv(index=False),
                    file_name="atr_dashboard_summary.csv",
                    mime="text/csv"
                )
