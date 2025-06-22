
import pandas as pd
import plotly.graph_objects as go

def render_atr_chart(data: pd.DataFrame):
    atr_order = ['+1.000', '+0.786', '+0.618', '+0.500', '+0.382', '+0.236', '0.000',
                 '-0.236', '-0.382', '-0.500', '-0.618', '-0.786', '-1.000']
    hour_order = ['OPEN', '0900', '1000', '1100', '1200', '1300', '1400', '1500', '1600']

    data['atr_level'] = pd.Categorical(data['atr_level'], categories=atr_order[::-1], ordered=True)
    data['hour'] = pd.Categorical(data['hour'], categories=hour_order, ordered=True)

    fig = go.Figure()

    for level in atr_order:
        line_style = dict(color='white', width=1)
        if level in ['+0.618', '-0.618']:
            line_style['width'] = 2
        elif level in ['+1.000', '-1.000']:
            line_style['width'] = 3
        elif level == '+0.236':
            line_style['color'] = 'cyan'
        elif level == '-0.236':
            line_style['color'] = 'yellow'

        fig.add_shape(
            type="line",
            x0=0,
            x1=len(hour_order) - 1,
            y0=level,
            y1=level,
            xref="x",
            yref="y",
            line=line_style
        )

    for _, row in data.iterrows():
        fig.add_trace(go.Scatter(
            x=[row["hour"]],
            y=[row["atr_level"]],
            mode="markers+text",
            marker=dict(size=row["percent"] * 3, color='deepskyblue', line=dict(width=1, color='black')),
            text=f"{row['percent']}%",
            textposition="top center",
            hovertemplate=f"Hour: {row['hour']}<br>Level: {row['atr_level']}<br>% Complete: {row['percent']}%<br>Raw: {row['raw_count']}"
        ))

    fig.update_layout(
        title="ATR Levels Roadmap (Simulated Data)",
        xaxis=dict(title="Hour of Day", tickvals=hour_order),
        yaxis=dict(title="ATR Level", tickvals=atr_order, autorange="reversed"),
        plot_bgcolor="black",
        paper_bgcolor="black",
        font=dict(color="white"),
        height=700,
        margin=dict(t=60, l=80, r=80, b=60)
    )

    return fig
