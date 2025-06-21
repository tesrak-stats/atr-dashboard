import pandas as pd
df = pd.read_csv("fake_atr_dashboard_data.csv")
import streamlit as st
import pandas as pd
import plotly.express as px

st.set_page_config(layout="wide")
st.title("ATR Roadmap – Simulated Bubble Chart")

# Simulated Data
hours = ['OPEN'] + [f"{str(h).zfill(2)}00" for h in range(9, 17)]
goal_levels = [round(x, 3) for x in [round(i, 3) for i in list(pd.Series(range(-1000, 1001, 167)) / 1000)]]

import numpy as np
np.random.seed(42)
data = []
for level in goal_levels:
    for hour in hours:
        count = np.random.randint(5, 120)
        pct = np.clip(np.random.normal(loc=0.5, scale=0.2), 0, 1)
        data.append({
            'ATR Level': level,
            'Hour': hour,
            'Trigger Count': count,
            'Completion %': round(100 * pct, 1)
        })
df = pd.DataFrame(data)

# Create custom hover text
df['Hover'] = df.apply(
    lambda row: f"Hour: {row['Hour']}<br>Level: {row['ATR Level']}<br>Completion: {row['Completion %']}%<br>Triggers: {row['Trigger Count']}" +
                (" ⚠️ Low sample size" if row['Trigger Count'] < 30 else ""),
    axis=1
)

# Plotly Chart
fig = px.scatter(
    df,
    x='Hour',
    y='ATR Level',
    size='Trigger Count',
    color='Completion %',
    color_continuous_scale='RdYlGn',
    size_max=40,
    hover_name='Hover',
)

fig.update_traces(hovertemplate='%{hovertext}<extra></extra>')
fig.update_layout(
    template="plotly_dark",
    height=700,
    yaxis=dict(dtick=0.236, title="ATR Level"),
    xaxis=dict(title="Hour of Day", type='category'),
    title_font_size=24,
    title="Simulated ATR Roadmap"
)

st.plotly_chart(fig, use_container_width=True)
