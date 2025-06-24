# streamlit_debug_viewer.py

import streamlit as st
import pandas as pd
import os
import subprocess
import time

st.title("🔍 Debug Viewer – First Day Trigger & Goal Trace")

csv_path = "debug_first_day_trace.csv"
debug_script = "detect_triggers_and_goals_debug.py"

# Run the debug detection script
if st.button("Run Debug Analysis"):
    with st.spinner("Running debug logic..."):
        try:
            # Run the Python script
            subprocess.run(["python", debug_script], check=True)

            # Small delay to ensure the file is written
            time.sleep(1)

            # Load and display the output
            if os.path.exists(csv_path):
                df = pd.read_csv(csv_path)
                if df.empty:
                    st.warning("Debug CSV was created but contains no rows. No triggers/goals may have occurred on the first day.")
                else:
                    st.success("Debug results loaded!")
                    st.dataframe(df)
            else:
                st.error("The debug script ran, but the CSV file was not found.")
        except subprocess.CalledProcessError as e:
            st.error(f"An error occurred while running the debug script:\n{e}")
else:
    st.info("Click the button above to run the debug analysis.")
