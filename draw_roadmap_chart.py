
import plotly.graph_objects as go

def draw_roadmap_chart(levels, labels, level_annotations, open_price=None, trigger_hour_idx=None, ticker="SPX"):
    hours = ["OPEN", "0900", "1000", "1100", "1200", "1300", "1400", "1500", "1600"]
    fig = go.Figure()

    for y, label, price in zip(levels, labels, level_annotations):
        if label == "-0.236":
            color = "yellow"
        elif ".236" in label:
            color = "cyan"
        elif label == "+0.000" or label == "0.000":
            color = "white"
        else:
            color = "gray"

        if label in ["+1.000", "-1.000"]:
            width = 3
        elif label in ["+0.618", "-0.618"]:
            width = 2
        else:
            width = 1

        fig.add_shape(
            type="line",
            x0=0, x1=len(hours)-1,
            y0=y, y1=y,
            line=dict(color=color, width=width)
        )

        # Left: ATR level label
        fig.add_annotation(x=-0.5, y=y, text=label, showarrow=False,
                           font=dict(color="lightgray"), xanchor="right")

        # Right: Price value
        fig.add_annotation(x=len(hours)-0.5, y=y, text=price, showarrow=False,
                           font=dict(color="lightgray"), xanchor="left")

    # Show Open marker only if open_price is passed and default view
    if open_price is not None and trigger_hour_idx is None:
        fig.add_trace(go.Scatter(
            x=[0],
            y=[open_price],
            mode="markers+text",
            marker=dict(color="red", size=10),
            text=["Open"],
            textposition="bottom center"
        ))

    fig.update_layout(
        height=500,
        xaxis=dict(
            tickmode="array",
            tickvals=list(range(len(hours))),
            ticktext=hours,
            title="Hour of Day",
            showgrid=False
        ),
        yaxis=dict(
            title="ATR Level",
            showgrid=False,
            autorange=True,
            showticklabels=False  # hide price values
        ),
        plot_bgcolor="#111",
        paper_bgcolor="#111",
        font_color="white",
        margin=dict(l=100, r=100, t=30, b=40),
        title=dict(
            text=f"ATR Levels Roadmap – {ticker}",
            x=0.5,
            xanchor="center"
        )
    )

    return fig
