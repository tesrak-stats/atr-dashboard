import streamlit as st
import pandas as pd
import os

st.title("🔁 ATR Trigger & Goal Generator")

if st.button("Generate combined_trigger_goal_results.csv"):
    with st.spinner("Running detection logic..."):
        import detect_triggers_and_goals  # this runs the logic
    st.success("✅ File generated: combined_trigger_goal_results.csv")

if os.path.exists("combined_trigger_goal_results.csv"):
    st.subheader("📄 Preview of Generated File")
    df = pd.read_csv("combined_trigger_goal_results.csv")
    st.dataframe(df.head(25))
    st.download_button("📥 Download CSV", df.to_csv(index=False), file_name="combined_trigger_goal_results.csv")