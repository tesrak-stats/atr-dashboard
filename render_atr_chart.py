import plotly.graph_objects as go

def render_atr_chart(df):
    # Expect columns: 'hour', 'atr_level', 'percent'
    fig = go.Figure()

    levels = sorted(df['atr_level'].unique(), reverse=True)
    hours = df['hour'].unique()

    # Plot lines for each level
    for lvl in levels:
        level_df = df[df['atr_level'] == lvl]
        fig.add_trace(go.Scatter(
            x=level_df['hour'],
            y=[lvl]*len(level_df),
            mode='lines+markers',
            line=dict(color='grey'),
            marker=dict(size=10 * level_df['percent']),
            name=f"{lvl:+.3f}"
        ))
        # Annotate percentages
        for _, row in level_df.iterrows():
            fig.add_annotation(
                x=row['hour'], y=lvl,
                text=f"{row['percent']:.0%}",
                showarrow=False,
                yshift=12,
                font=dict(size=12, color='white')
            )

    fig.update_layout(
        yaxis=dict(title="ATR Level", tickmode='array', tickvals=levels, ticktext=[f"{lvl:+.3f}" for lvl in levels]),
        xaxis=dict(title="Hour of Day", tickmode='array', tickvals=hours),
        template='plotly_dark', height=600
    )
    return fig
