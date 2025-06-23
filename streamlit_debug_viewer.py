
import streamlit as st
import pandas as pd
import os

st.title("🔍 ATR Debug Viewer: First Day Trace")

# Trigger script (assumes debug_trace is already imported manually if needed)
if st.button("Run Debug Script"):
    with st.spinner("Running debug analysis..."):
        import run_debug_first_day  # This must be present in the environment
    st.success("✅ Debug CSV generated!")

# Load and display CSV if it exists
debug_file = "debug_first_day_results.csv"
if os.path.exists(debug_file):
    st.subheader("📊 Debug Output Preview")
    df = pd.read_csv(debug_file)
    st.dataframe(df.head(50))

    st.download_button(
        label="📥 Download Full Debug CSV",
        data=df.to_csv(index=False),
        file_name="debug_first_day_results.csv",
        mime="text/csv"
    )
else:
    st.info("Click the button above to generate the debug file.")
