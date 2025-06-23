import streamlit as st
import pandas as pd
import os

st.title("🔍 ATR Debug Viewer: First Day Trace")

if st.button("Run Debug Script"):
    with st.spinner("Running debug analysis..."):
        import run_debug_first_day
        st.success("✅ Debug CSV generated!")

csv_path = "debug_first_day_trace.csv"

st.text(f"Path checked: {csv_path}")
st.text(f"Exists? {os.path.exists(csv_path)}")
if os.path.exists(csv_path):
    size = os.path.getsize(csv_path)
    st.text(f"Size of CSV: {size} bytes")

    try:
        df = pd.read_csv(csv_path)
        st.text(f"Loaded DataFrame: {df.shape}")
        st.markdown("### 🧾 Debug CSV Preview")
        st.dataframe(df, use_container_width=True)
        st.text(f"First 2 rows:\n{df.head(2).to_string()}")

        csv = df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="⬇️ Download Debug CSV",
            data=csv,
            file_name="debug_first_day_trace.csv",
            mime="text/csv"
        )
    except Exception as e:
        st.error(f"Error loading CSV: {e}")
else:
    st.info("Click the button above to generate the debug file.")
