import streamlit as st
import pandas as pd
import os
import detect_triggers_and_goals

st.title("📊 ATR Trigger & Goal Generator")

output_path = "combined_trigger_goal_results.csv"

if st.button("Generate combined_trigger_goal_results.csv"):
    with st.spinner("Running detection..."):
        try:
            result_df = detect_triggers_and_goals.main()
            result_df["Source"] = "Full"
            result_df.to_csv(output_path, index=False)
            st.success("✅ File generated and saved!")

            # Preview
            if os.path.exists(output_path):
                df = pd.read_csv(output_path)
                st.subheader("🔍 Preview of Most Recent Output")
                st.dataframe(df.head(30))
                st.download_button("⬇️ Download CSV", data=df.to_csv(index=False), file_name=output_path, mime="text/csv")
        except Exception as e:
            st.error(f"❌ Error running detect_triggers_and_goals.py: {e}")
