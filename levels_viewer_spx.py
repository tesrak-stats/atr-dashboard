
import os
os.environ["STREAMLIT_WATCH_DIRECTORIES"] = "false"

import streamlit as st
from generate_daily_atr_levels_spx import get_latest_atr_levels
import pandas as pd

st.title("📊 ATR Fib Level Viewer (SPX)")

# Get the data
try:
    atr_data = get_latest_atr_levels()
except Exception as e:
    st.error(f"Failed to load ATR levels: {e}")
    st.stop()

# Show metadata
st.markdown(f"""
**Ticker**: {atr_data['ticker']}  
**Date Generated**: {atr_data['date_generated']}  
**ATR**: {atr_data['atr']}  
**Previous Close**: {atr_data['prev_close']}  
**Latest Open**: {atr_data['latest_open']}
""")

# Convert levels to a DataFrame
levels_df = pd.DataFrame({
    "Fib Level": list(atr_data["levels"].keys()),
    "ATR Price": list(atr_data["levels"].values())
})

# Sort by Fib value descending
levels_df["Fib Level"] = levels_df["Fib Level"].astype(str)
levels_df = levels_df.sort_values("Fib Level", ascending=False)

st.dataframe(levels_df, use_container_width=True)
