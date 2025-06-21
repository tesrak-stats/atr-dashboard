import plotly.graph_objects as go

def render_atr_chart(df, title="ATR Levels Roadmap"):
    levels = sorted(df['goal_level'].unique())
    hours = ['OPEN', '0900', '1000', '1100', '1200', '1300', '1400', '1500', '1600']

    fig = go.Figure()

    # Add horizontal ATR level lines
    for lvl in levels:
        line_color = "white"
        width = 1
        if lvl == 0.236:
            line_color = "cyan"
        elif lvl == -0.236:
            line_color = "yellow"
        elif abs(lvl) == 0.618:
            width = 3

        fig.add_shape(
            type="line",
            x0=hours[0], x1=hours[-1],
            y0=lvl, y1=lvl,
            line=dict(color=line_color, width=width),
            xref="x", yref="y"
        )

    # Add bubbles for each scenario
    for direction in df['direction'].unique():
        subset = df[df['direction'] == direction]
        fig.add_trace(go.Scatter(
            x=subset["goal_hour"],
            y=subset["goal_level"],
            mode="markers+text",
            text=[f"{pct:.1f}%" for pct in subset["percent_complete"]],
            textposition="top center",
            marker=dict(
                size=subset["percent_complete"],
                sizemode="area",
                sizeref=2.*max(df["percent_complete"])/100**2,
                sizemin=4,
                color="blue" if direction == "Below" else "orange",
                opacity=0.7,
                line=dict(width=1, color='DarkSlateGrey')
            ),
            name=direction,
            hovertemplate="<b>%{y:+.3f} ATR</b><br>Hour: %{x}<br>% Complete: %{text}<extra></extra>"
        ))

    fig.update_layout(
        title=title,
        xaxis=dict(title="Hour of Day", tickvals=hours),
        yaxis=dict(title="ATR Level"),
        height=700,
        paper_bgcolor="black",
        plot_bgcolor="black",
        font=dict(color="white"),
        margin=dict(l=40, r=40, t=60, b=40),
        showlegend=True
    )

    return fig
