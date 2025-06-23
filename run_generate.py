
import streamlit as st
import pandas as pd
import os

st.title("🔁 ATR Trigger & Goal Generator")

if st.button("Generate combined_trigger_goal_results.csv"):
    with st.spinner("Running detection logic..."):
        import detect_triggers_and_goals  # Executes the script
    st.success("File generated!")

# Optional preview
if os.path.exists("combined_trigger_goal_results.csv"):
    df = pd.read_csv("combined_trigger_goal_results.csv")
    st.subheader("🔍 Preview of Generated Data")
    st.dataframe(df.head(25))
