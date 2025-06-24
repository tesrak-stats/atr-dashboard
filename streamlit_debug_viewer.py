
import streamlit as st
from detect_triggers_and_goals_debug import run_debug

st.title("🔍 Debug Viewer – First Day Trigger & Goal Trace")

if st.button("Run Debug Analysis"):
    try:
        run_debug()
        st.success("✅ Debug script executed.")
        with open("debug_log_output.csv", "r") as f:
            lines = f.readlines()
            st.text("\n".join(lines))
    except Exception as e:
        st.error(f"An error occurred while running the debug function: {e}")
