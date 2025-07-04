
import os
os.environ["STREAMLIT_WATCH_DIRECTORIES"] = "false"

import streamlit as st
import pandas as pd

st.title("📊 ATR Fib Level Viewer (SPX)")

try:
    # Move import inside try block to catch import-time errors
    from generate_daily_atr_levels_spx_fast import get_latest_atr_levels

    atr_data = get_latest_atr_levels()

    st.markdown(f"""
    **Ticker**: {atr_data['ticker']}  
    **Date Generated**: {atr_data['date_generated']}  
    **ATR**: {atr_data['atr']}  
    **Previous Close**: {atr_data['prev_close']}  
    **Latest Open**: {atr_data['latest_open']}
    """)

    levels_df = pd.DataFrame({
        "Fib Level": list(atr_data["levels"].keys()),
        "ATR Price": list(atr_data["levels"].values())
    }).sort_values("Fib Level", ascending=False)

    st.dataframe(levels_df, use_container_width=True)

except Exception as e:
    st.error(f"🚨 App failed to load: {e}")
