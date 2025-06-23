import streamlit as st
import pandas as pd
import os

st.title("🔍 ATR Debug Viewer: First Day Trace")

# Trigger script (assumes debug_trace is auto-saved)
if st.button("Run Debug Script"):
    with st.spinner("Running debug analysis..."):
        import run_debug_first_day  # This script generates 'debug_first_day_trace.csv'
        st.success("✅ Debug CSV generated!")

csv_path = "debug_first_day_trace.csv"

# Load and display CSV if it exists
if os.path.exists(csv_path):
    df = pd.read_csv(csv_path)

    st.markdown("### 🧾 Debug CSV Preview")
    st.dataframe(df, use_container_width=True)

    # Failsafe preview text
    st.text(f"CSV preview (first 2 rows):\n{df.head(2).to_string()}")

    # Add download button
    csv = df.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="⬇️ Download Debug CSV",
        data=csv,
        file_name="debug_first_day_trace.csv",
        mime="text/csv"
    )
else:
    st.info("Click the button above to generate the debug file.")
