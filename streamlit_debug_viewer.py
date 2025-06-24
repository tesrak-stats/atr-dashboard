
import streamlit as st
import subprocess
import pandas as pd

st.title("🔍 Debug Viewer – First Day Trigger & Goal Trace")

if st.button("Run Debug Analysis"):
    debug_script = "detect_triggers_and_goals_debug.py"

    try:
        result = subprocess.run(
            ["python", debug_script],
            check=True,
            capture_output=True,
            text=True
        )
        st.success("✅ Debug script executed successfully.")
        st.text(result.stdout)

        try:
            debug_df = pd.read_csv("debug_trigger_goal_output.csv")
            st.dataframe(debug_df)
        except Exception as df_err:
            st.error("Debug script ran but could not load output CSV:")
            st.code(str(df_err), language="python")

    except subprocess.CalledProcessError as e:
        st.error("❌ An error occurred while running the debug script:")
        st.code(e.stderr or str(e), language="python")
