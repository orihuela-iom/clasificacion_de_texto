from dash import Dash, html, dcc, callback, Output, Input, State
from dash.exceptions import PreventUpdate
import pandas as pd
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
from train_model import make_predit
import math


def probs_plot(data:pd.DataFrame, limit_cat:int = 5,
               features:bool = False):

    data.columns = ["p"]
    data = data.sort_values(by="p", ascending=True)
    data = data.tail(limit_cat)

    if features:
        colors = ["#7570b3" if x < 0 else "#e7298a" for x in data["p"]]
        xaxis_range = [data["p"].min() -0.2, data["p"].max() + 0.2]
        title = "Relvancia de cada palabra"
    else:
        colors = ["lightslategray"] * (limit_cat -1) + ["crimson"]
        xaxis_range = [0,1]
        title = "Probabilidades de predicción"


    fig = go.Figure(go.Bar(y=data.index,
                        x=data["p"].values,
                            orientation='h',
                            text=[math.floor(x * 100) / 100 for x in data["p"].to_list()],
                            textposition="outside",
                            marker_color = colors
                            )
                    )

    fig.update_layout(
                margin={'l': 20, 'r': 20, 't': 30, 'b': 20},
                xaxis_range=xaxis_range,
                title=title,
                height=300,
                plot_bgcolor="#FFF")

    return fig


def card_cointainer(some_object) -> dbc.Card:
    """
    Crea una tarjeta
    """
    card = html.Div([
        dbc.Card(
            dbc.CardBody([some_object]),
            className="mb-3",
        )
    ])
    return card


app = Dash(__name__, external_stylesheets=[dbc.themes.LUX])
app.title = "Ismael Orihuela"


navbar = dbc.NavbarSimple(
    children=[""],
    brand="Shasa Challenge",
    brand_href="#",
    color="primary",
    dark=True, style={"height": 70})


text_container = dbc.Textarea(id="input-text",
    className="mb-3", placeholder="Ingresa un texto", style={"height": 100}
    )


text_row = dbc.Row([card_cointainer(text_container),
                    dbc.Button("Clasificar", color="primary", className="me-1", id="predecir")])

main_layout = dbc.Row(
    children=[
        dbc.Col(
            html.Div([
                card_cointainer(dcc.Loading(dcc.Graph(
                    id='features_plot',
                    config={'displayModeBar': False}
                    ), id = "load_1"))
                ]),
            #width=6,
            lg=5),

        dbc.Col(
            html.Div([
                card_cointainer(dcc.Loading(dcc.Graph(id='probability_plot',
                                            config={'displayModeBar': False}), id="load_2"))
                ]),
            #offset=2,
            lg=5)
    ])


app.layout = html.Div(
    children=[
        navbar,
        html.Br(),
        dbc.Container([
            text_row,
            main_layout
        ], fluid=False)
    ])




@callback(
    Output('probability_plot', component_property='figure'),
    Output('features_plot', component_property='figure'),
    Input('predecir', 'n_clicks'),
    State('input-text', 'value', ),
    prevent_initial_call=True
    )
def update_graph(clicks, my_text: str):
    if clicks is None or my_text is None or my_text == "":
        raise PreventUpdate

    prediction = make_predit(my_text)

    probas = pd.DataFrame.from_dict(prediction["proba"], orient="index")
    words = pd.DataFrame.from_dict(prediction["Key words"], orient="index")

    return  probs_plot(words, features=True), probs_plot(probas, features=False)


if __name__ == "__main__":
    app.run_server(debug=False, host="0.0.0.0", port="8080")
