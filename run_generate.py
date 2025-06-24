import streamlit as st
import pandas as pd
import os

st.title("📊 ATR Trigger & Goal Generator")

# Button to run the main detection script
if st.button("Generate combined_trigger_goal_results.csv"):
    with st.spinner("Running detection logic..."):
        try:
            # Try to import and run detection script
            import detect_triggers_and_goals
            result = detect_triggers_and_goals.main()  # must return a DataFrame OR None

            if isinstance(result, pd.DataFrame):
                # Production mode — full result
                result.to_csv("combined_trigger_goal_results.csv", index=False)
                st.success("✅ File generated: combined_trigger_goal_results.csv")
                st.dataframe(result.head(25))
                st.download_button(
                    label="📥 Download CSV",
                    data=result.to_csv(index=False),
                    file_name="combined_trigger_goal_results.csv",
                    mime="text/csv"
                )
            else:
                # Debug mode or no return
                st.warning("⚠️ Script ran, but no result DataFrame was returned. Possibly in debug mode.")
        except Exception as e:
            st.error(f"❌ Error running detect_triggers_and_goals.py: {e}")

# Optional preview if file exists
if os.path.exists("combined_trigger_goal_results.csv"):
    st.subheader("🔍 Preview of Most Recent Output")
    df = pd.read_csv("combined_trigger_goal_results.csv")
    st.dataframe(df.head(25))
