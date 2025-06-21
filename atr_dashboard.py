import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

st.set_page_config(page_title="ATR Dashboard", layout="wide")

st.title("ATR Level Roadmap (Sample)")
st.write("This is a placeholder Streamlit app. Replace with full app logic.")

# Example plot
df = pd.DataFrame({
    "hour": ["OPEN", "0900", "1000", "1100"],
    "goal_0.236": [42, 58, 76, 83]
})
fig = px.line(df, x="hour", y="goal_0.236", title="Goal Completion by Hour")
st.plotly_chart(fig)
