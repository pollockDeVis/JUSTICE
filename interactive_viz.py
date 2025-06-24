import dash
from dash import dcc, html, Input, Output
import plotly.graph_objs as go
import pandas as pd
from justice.util.enumerations import WelfareFunction, SSP

# --- Data Loading ---
swf = WelfareFunction.UTILITARIAN
ssp = SSP.SSP2
path1 = f"data/temporary/NU_DATA/combined/{str(ssp).split('.')[1]}/"
df1 = pd.read_csv(path1 + f"{swf.value[1]}_reference_set.csv")
df1 = df1.iloc[:, -2:]

x = df1.iloc[:, 0]
y = df1.iloc[:, 1]
indices = df1.index  # <-- preserves original CSV row index

# Normalize the x axis values
x = (x - x.min()) / (x.max() - x.min())
x = 1 - x

app = dash.Dash(__name__)

x_min, x_max = float(x.min()), float(x.max())
y_min, y_max = float(y.min()), float(y.max())

app.layout = html.Div(
    [
        dcc.Graph(id="scatter-plot"),
        html.Div(
            [
                "Welfare range:",
                dcc.RangeSlider(
                    id="x-slider",
                    min=x_min,
                    max=x_max,
                    step=(x_max - x_min) / 100,
                    value=[x_min, x_max],
                    marks={
                        float(f"{v:.2f}"): f"{v:.2f}"
                        for v in [x_min, (x_min + x_max) / 2, x_max]
                    },
                ),
            ],
            style={"width": "48%", "display": "inline-block", "padding": "20px"},
        ),
        html.Div(
            [
                "Temperature range:",
                dcc.RangeSlider(
                    id="y-slider",
                    min=y_min,
                    max=y_max,
                    step=(y_max - y_min) / 100,
                    value=[y_min, y_max],
                    marks={
                        float(f"{v:.2f}"): f"{v:.2f}"
                        for v in [y_min, (y_min + y_max) / 2, y_max]
                    },
                ),
            ],
            style={"width": "48%", "display": "inline-block", "padding": "20px"},
        ),
    ]
)


@app.callback(
    Output("scatter-plot", "figure"),
    Input("x-slider", "value"),
    Input("y-slider", "value"),
)
def update_scatter(x_range, y_range):
    mask = (x >= x_range[0]) & (x <= x_range[1]) & (y >= y_range[0]) & (y <= y_range[1])
    marker_colors = ["red" if show else "rgba(200,200,200,0.1)" for show in mask]
    fig = go.Figure(
        data=go.Scatter(
            x=x,
            y=y,
            mode="markers",
            marker=dict(size=10, color=marker_colors),
            name="combined",
            customdata=indices,  # Pass original index for hover!
            hovertemplate="Index: %{customdata}<br>Welfare: %{x}<br>Temperature: %{y}<extra></extra>",
        ),
        layout=go.Layout(
            plot_bgcolor="white",
            paper_bgcolor="white",
            font=dict(color="black"),
            title="Scatter Plot of Normalized Column 1 vs Column 2",
            xaxis_title="Welfare",
            yaxis_title="Temperature",
            width=900,
            height=600,
        ),
    )
    return fig


if __name__ == "__main__":
    app.run(debug=True)
