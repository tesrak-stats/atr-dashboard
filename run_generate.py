
import streamlit as st
import pandas as pd
import importlib
import detect_triggers_and_goals

# Force reload to ensure latest version is used
importlib.reload(detect_triggers_and_goals)

st.title("🔁 ATR Trigger & Goal Generator (Debug v1)")

# File paths
daily_path = "SPXdailycandles.xlsx"
intraday_path = "SPX_10min.csv"
output_path = "combined_trigger_goal_results.csv"

if st.button("Generate combined_trigger_goal_results.csv"):
    with st.spinner("Running detection logic (debug version)..."):
        # Load inputs
        daily_df = pd.read_excel(daily_path, header=4)
        intraday_df = pd.read_csv(intraday_path)

        # Run detection logic
        result_df = detect_triggers_and_goals.detect_triggers_and_goals(daily_df, intraday_df)
        result_df["Source"] = "Debug v1"  # Mark this version for clarity
        result_df.to_csv(output_path, index=False)

    st.success("✅ File generated and saved!")

# Preview + Download
if os.path.exists(output_path):
    df = pd.read_csv(output_path)
    st.subheader("🔍 Preview of Debug Output")
    st.dataframe(df.head(30))
    st.download_button("📥 Download CSV", df.to_csv(index=False), file_name="combined_trigger_goal_results.csv")
