import dash
from dash import html, dcc, Input, Output, State
import dash_bootstrap_components as dbc
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
from dash_bootstrap_templates import ThemeSwitchAIO
import os

# ========= App ============== #
FONT_AWESOME = ["https://use.fontawesome.com/releases/v5.10.2/css/all.css"]
dbc_css = "https://cdn.jsdelivr.net/gh/AnnMarieW/dash-bootstrap-templates@V1.0.4/dbc.min.css"

app = dash.Dash(__name__, external_stylesheets=[dbc.themes.FLATLY, dbc_css])
app.scripts.config.serve_locally = True
server = app.server

# ========== Styles ============ #

template_theme1 = "flatly"
template_theme2 = "vapor"
url_theme1 = dbc.themes.FLATLY
url_theme2 = dbc.themes.VAPOR
tab_style = {"height": "100%"}
main_config = {
    "hovermode": "x unified",
    "legend": {"yanchor":"top", 
                "y":0.9, 
                "xanchor":"left",
                "x":0.1,
                "title": {"text": None},
                "font" :{"color":"white"},
                "bgcolor": "rgba(0,0,0,0.5)"},
    "margin": {"l":0, "r":0, "t":10, "b":0}
}

# ===== Reading n cleaning File ====== #

df = pd.read_csv("data_gas.csv")

df["DATA INICIAL"] = pd.to_datetime(df["DATA INICIAL"])
df["DATA FINAL"] = pd.to_datetime(df["DATA FINAL"])

df["NÚMERO DE POSTOS PESQUISADOS"] = df["NÚMERO DE POSTOS PESQUISADOS"].astype(int)
df["DATA MEDIA"] = ((df["DATA FINAL"] -  df["DATA INICIAL"])/2 + df["DATA FINAL"])

df = df.sort_values(by="DATA MEDIA", ascending=True)
df.rename(columns= {"DATA MEDIA" : "DATA"}, inplace=True)
df["ANO"] = df["DATA"].apply(lambda x: str(x.year))
df = df[df["PRODUTO"] == "GASOLINA COMUM"].reset_index()
df["ANO"] = df["DATA"].apply(lambda x: str(x.year)).astype(int)

df.drop(['UNIDADE DE MEDIDA', 'COEF DE VARIAÇÃO REVENDA', 'COEF DE VARIAÇÃO DISTRIBUIÇÃO', 
    'NÚMERO DE POSTOS PESQUISADOS', 'DATA INICIAL', 'DATA FINAL', 'PREÇO MÁXIMO DISTRIBUIÇÃO', 'PREÇO MÍNIMO DISTRIBUIÇÃO', 
    'DESVIO PADRÃO DISTRIBUIÇÃO', 'MARGEM MÉDIA REVENDA', 'PREÇO MÍNIMO REVENDA', 'PREÇO MÁXIMO REVENDA', 'DESVIO PADRÃO REVENDA', 
    'PRODUTO', 'PREÇO MÉDIO DISTRIBUIÇÃO'], inplace=True, axis=1)
df_store = df.to_dict()

# =========  Layout  =========== #
app.layout = dbc.Container(children=[

    dcc.Store(id="dataset", data=df_store),
    dcc.Store(id="dataset_fixed", data=df_store),
    dcc.Store(id="controller", data={'play':False}),
    #Linha 1:
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    dbc.Row([
                        dbc.Col([  
                            html.Legend("Preço de gasolina")
                        ], sm=8),
                        dbc.Col([        
                            html.I(className='fa fa-filter', style={'font-size': '300%'})
                        ], sm=4, align="center")
                    ]),
                    dbc.Row([
                        dbc.Col([
                            ThemeSwitchAIO(aio_id="theme", themes=[url_theme1, url_theme2]),
                            html.Legend("MaxforceDev")
                        ])
                    ], style={'margin-top': '10px'}),
                    dbc.Row([
                        dbc.Button("Visite o Site", href="https://github.com/maxforcedev/", target="_blank")
                    ], style={'margin-top': '10px'})
                ])
            ], style=tab_style)
        ], sm=4, lg=2),

        # Coluna 2
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    dbc.Row([
                        dbc.Col([
                            html.H3("Maximos e minimos"),
                            dcc.Graph(id="graph-maxmin", config={"displayModeBar":False, "showTips": False})
                        ])
                    ])
                ])
            ], style=tab_style)
        ], sm=8, lg=3),
        # Coluna 3
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    dbc.Row([
                        dbc.Col([
                            html.H6("Ano de análise:"),
                            dcc.Dropdown(
                                id="dropdown-ano",
                                value= df.at[df.index[1], "ANO"],
                                clearable=False,
                                className="dbc",
                                options=[{"label": x, "value":x} for x in df.ANO.unique()]
                            ),
                            dcc.Graph(id="graph-ano", config={"displayModeBar":False, "showTips": False})
                        ], sm=6),
                        dbc.Col([
                            html.H6("Região de análise:"),
                            dcc.Dropdown(
                                id="dropdown-regiao",
                                value= df.at[df.index[1], "REGIÃO"],
                                clearable=False,
                                className="dbc",
                                options=[{"label": x, "value":x} for x in df.REGIÃO.unique()]
                            ),
                            dcc.Graph(id="graph-regiao", config={"displayModeBar":False, "showTips": False})
                        ], sm=6),
                    ])
                ])
            ], style=tab_style)
        ], sm=12, lg=7),
    ], className='main_row g-2 my-auto', style={'margin-top': '7px'}),

    #Linha 2:
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.H3("Preço x Estado"),
                    html.H6("Comparação temporal entre estados"),
                    dbc.Row([
                        dbc.Col([
                            dcc.Dropdown(
                                id="dropdown-precoestado",
                                value= [df.at[df.index[1], "ESTADO"], df.at[df.index[6], "ESTADO"], df.at[df.index[4], "ESTADO"]],
                                clearable=False,
                                className="dbc",
                                multi=True,
                                options=[{"label": x, "value":x} for x in df.ESTADO.unique()]
                            ),
                        ], sm=10)
                    ]),
                    dbc.Row([
                        dbc.Col([
                            dcc.Graph(id="graph-precoestado", config={"displayModeBar":False, "showTips": False})
                        ])
                    ])
                ]),
            ], style=tab_style)
        ], sm=12, md=6, lg=5),
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.H3("Comparação Direta"),
                    html.H6("Qual preço é menor em um dado período de tempo?"),
                    dbc.Row([
                        dbc.Col([
                            dcc.Dropdown(
                                id="dropdown1-comparacao",
                                value= df.at[df.index[1], "ESTADO"],
                                clearable=False,
                                className="dbc",
                                options=[{"label": x, "value":x} for x in df.ESTADO.unique()]
                            ),
                        ], sm=12, md=5, lg=5),
                        dbc.Col([
                            dcc.Dropdown(
                                id="dropdown2-comparacao",
                                value= df.at[df.index[3], "ESTADO"],
                                clearable=False,
                                className="dbc",
                                options=[{"label": x, "value":x} for x in df.ESTADO.unique()]
                            ),
                        ], sm=12, md=5, lg=5),
                    ], style={'margin-top': '20px'}, justify='center'),
                    dbc.Row([
                        dbc.Col([
                            dcc.Graph(id="graph-comparacao", config={"displayModeBar":False, "showTips": False})
                        ])
                    ]),
                    html.P(id='desc_comparison', style={'color': 'gray', 'font-size': '80%'}),
                ]),
            ], style=tab_style)
        ], sm=12, md=6, lg=4),
        dbc.Col([
            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardBody([
                            dcc.Graph(id="indicator-estado1", config={"displayModeBar":False, "showTips": False})
                        ])
                    ], style=tab_style)
                ]),
            ], justify='center', style={'padding-bottom': '7px', 'height': '50%'}),
            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardBody([
                            dcc.Graph(id="indicator-estado2", config={"displayModeBar":False, "showTips": False})
                        ])
                    ], style=tab_style)
                ]),
            ], justify='center', style={'height': '50%'})
        ], sm=12, lg=3, style={'height': '100%'})
    ], style={'margin-top': '7px'}, className='main_row g-2 my-auto'),

    dbc.Row([
        dbc.Col([
            dbc.Card([                
                dbc.Row([
                    dbc.Col([
                        dbc.Button([html.I(className='fa fa-play')], id="play-button", style={'margin-right': '15px'}),  
                        dbc.Button([html.I(className='fa fa-stop')], id="stop-button")
                    ], sm=12, md=1, style={'justify-content': 'center', 'margin-top': '10px'}),
                    dbc.Col([
                        dcc.RangeSlider(
                            id='rangeslider',
                            marks= {int(x): f'{x}' for x in df['ANO'].unique()},
                            step=3,                
                            min=2004,
                            max=2021,
                            value=[2004,2021],   
                            dots=True,             
                            pushable=3,
                            tooltip={'always_visible':False, 'placement':'bottom'},
                        )
                    ], sm=12, md=10, style={'margin-top': '15px'}),
                    dcc.Interval(id='interval', interval=10000),
                ], className='g-1', style={'height': '20%', 'justify-content': 'center'})
            ], style=tab_style)
        ])
    ], className='main_row g-2 my-auto')

], fluid=True, style={'height': '100%'})


# ======== Callbacks ========== #

@app.callback(
    Output("graph-maxmin", "figure"),
    [
    Input('dataset', 'data'),
    Input(ThemeSwitchAIO.ids.switch("theme"), "value")
    ]
)
def graph_maxmin(data, toggle):
    template = template_theme1 if toggle else template_theme2

    df = pd.DataFrame(data)

    max = df.groupby(['ANO'])['PREÇO MÉDIO REVENDA'].max()
    min = df.groupby(['ANO'])['PREÇO MÉDIO REVENDA'].min()

    final_df = pd.concat([max, min], axis=1)
    final_df.columns = ['Máximo', 'Mínimo']

    fig = px.line(final_df, x=final_df.index, y=final_df.columns, template=template)
    
    # updates
    fig.update_layout(main_config, height=150, xaxis_title=None, yaxis_title=None)

    return fig

@app.callback(
    Output("graph-ano", "figure"),
    Output("graph-regiao", "figure"),


    [
    Input('dataset', 'data'),
    Input('dropdown-ano', 'value'),
    Input('dropdown-regiao', 'value'),

    Input(ThemeSwitchAIO.ids.switch("theme"), "value")
    ]
)
def graphs_no_regiao(data, ano, regiao, toggle):
    template = template_theme1 if toggle else template_theme2

    df = pd.DataFrame(data)
    df_filtrado_ANO = df[df["ANO"].isin([ano])]
    df_filtrado_REGIAO = df[df["REGIÃO"].isin([regiao])]

    df_regiao = df_filtrado_ANO.groupby(["ANO", "REGIÃO"])["PREÇO MÉDIO REVENDA"].mean().reset_index().sort_values(by="PREÇO MÉDIO REVENDA", ascending=True)
    df_estado = df_filtrado_REGIAO.groupby(["REGIÃO", "ESTADO"])["PREÇO MÉDIO REVENDA"].mean().reset_index().sort_values(by="PREÇO MÉDIO REVENDA", ascending=True)

    fig1_text = [f'{x} - R${y}' for x,y in zip(df_regiao.REGIÃO.unique(), df_regiao['PREÇO MÉDIO REVENDA'].unique().round(decimals = 2))]
    fig2_text = [f'R${y} - {x}' for x,y in zip(df_estado.ESTADO.unique(), df_estado['PREÇO MÉDIO REVENDA'].unique().round(decimals = 2))]

    fig1 = go.Figure(go.Bar(
        x=df_regiao['PREÇO MÉDIO REVENDA'],
        y=df_regiao['REGIÃO'],
        orientation='h',
        text=fig1_text,
        textposition='auto',
        insidetextanchor='end',
        insidetextfont=dict(family='Times', size=12)
    ))
    fig2 = go.Figure(go.Bar(
        x=df_estado['PREÇO MÉDIO REVENDA'],
        y=df_estado['ESTADO'],
        orientation='h',
        text=fig2_text,
        insidetextanchor='end',
        insidetextfont=dict(family='Times', size=12) 
    ))

    fig1.update_layout(main_config, yaxis={'showticklabels':False}, height=140, template=template)
    fig2.update_layout(main_config, yaxis={'showticklabels':False}, height=140, template=template)

    fig1.update_layout(xaxis_range=[df_regiao['PREÇO MÉDIO REVENDA'].max(), df_regiao['PREÇO MÉDIO REVENDA'].min() - 0.15])
    fig2.update_layout(xaxis_range=[df_estado['PREÇO MÉDIO REVENDA'].min() - 0.15, df_estado['PREÇO MÉDIO REVENDA'].max()])

    return [fig1, fig2]

@app.callback(
    Output("graph-precoestado", "figure"),
    [
    Input('dataset', 'data'),
    Input('dropdown-precoestado', 'value'),
    Input(ThemeSwitchAIO.ids.switch("theme"), "value")]
)
def graph_precoestado(data, estado, toggle):
    template = template_theme1 if toggle else template_theme2
    df = pd.DataFrame(data)
    estados = df[df.ESTADO.isin(estado)]

    fig = px.line(estados, x="DATA", y="PREÇO MÉDIO REVENDA", color="ESTADO", template=template)
    fig.update_layout(main_config, height=425, xaxis_title=None)

    return fig
    

@app.callback(
    [Output("graph-comparacao", "figure"),
     Output("desc_comparison", "children"),
    ],
    [
        Input('dataset', 'data'),
        Input('dropdown1-comparacao', 'value'),
        Input('dropdown2-comparacao', 'value'),
        Input(ThemeSwitchAIO.ids.switch("theme"), "value")
    ]
)
def graph_comparacao(data, estado1, estado2, toggle):
    template = template_theme1 if toggle else template_theme2

    dff = pd.DataFrame(data)
    df1 = dff[dff.ESTADO == estado1]
    df2 = dff[dff.ESTADO == estado2]

    df_estado1 = df1.groupby(pd.PeriodIndex(df1['DATA'], freq="M"))['PREÇO MÉDIO REVENDA'].mean().reset_index()
    df_estado2 = df2.groupby(pd.PeriodIndex(df2['DATA'], freq="M"))['PREÇO MÉDIO REVENDA'].mean().reset_index()

    df_estado1['DATA'] = df_estado1['DATA'].astype(str)
    df_estado2['DATA'] = df_estado2['DATA'].astype(str)

    df_merged = pd.merge(df_estado1, df_estado2, on='DATA', suffixes=(f'_{estado1}', f'_{estado2}'))

    df_merged['DIFERENÇA'] = df_merged[f'PREÇO MÉDIO REVENDA_{estado1}'] - df_merged[f'PREÇO MÉDIO REVENDA_{estado2}']

    fig = go.Figure()
    fig.add_scattergl(name=f'{estado1} - {estado2}', x=df_merged['DATA'], y=df_merged['DIFERENÇA'])

    fig.add_scattergl(name=f'{estado2} mais barato',
                      x=df_merged['DATA'],
                      y=df_merged['DIFERENÇA'].where(df_merged['DIFERENÇA'] > 0.0))

    fig.update_layout(main_config, height=350, template=template)
    fig.update_yaxes(range=[-0.7, 0.7])

    fig.add_annotation(text=f'{estado2} mais barato',
                       xref="paper", yref="paper",
                       font=dict(size=12, color="#ffffff"),
                       align="center", bgcolor="rgba(0,0,0,0.5)", opacity=0.8,
                       x=0.1, y=0.75, showarrow=False)

    fig.add_annotation(text=f'{estado1} mais barato',
                       xref="paper", yref="paper",
                       font=dict(size=12, color="#ffffff"),
                       align="center", bgcolor="rgba(0,0,0,0.5)", opacity=0.8,
                       x=0.1, y=0.25, showarrow=False)

    text = f"Comparando {estado1} e {estado2}. Se a linha estiver acima do eixo X, {estado2} tinha menor preço, do contrário, {estado1} tinha um valor inferior"

    return [fig, text]


@app.callback(
    Output("indicator-estado1", "figure"),
    [
        Input('dataset', 'data'),
        Input('dropdown1-comparacao', 'value'),
        Input(ThemeSwitchAIO.ids.switch("theme"), "value")
    ]
)
def indicator1(data, estado, toggle):
    template = template_theme1 if toggle else template_theme2

    dff = pd.DataFrame(data)
    df_final = dff[dff.ESTADO.isin([estado])]

    data1 = dff.ANO.min()
    data2 = dff.ANO.max()
    
    fig = go.Figure()

    fig.add_trace(go.Indicator(
        mode = "number+delta",
        title = {"text": f"<span style='size:60%'>{estado}</span><br><span style='font-size:0.7em'>{data1} - {data2}</span>"},
        value = df_final.at[df_final.index[-1],'PREÇO MÉDIO REVENDA'],
        number = {'prefix': "R$", 'valueformat': '.2f'},
        delta = {'relative': True, 'valueformat': '.1%', 'reference': df_final.at[df_final.index[0],'PREÇO MÉDIO REVENDA']}
    ))
    
    fig.update_layout(main_config, height=250, template=template)
    
    return fig

@app.callback(
    Output("indicator-estado2", "figure"),
    [
        Input('dataset', 'data'),
        Input('dropdown2-comparacao', 'value'),
        Input(ThemeSwitchAIO.ids.switch("theme"), "value")
    ]
)
def indicator2(data, estado, toggle):
    template = template_theme1 if toggle else template_theme2

    dff = pd.DataFrame(data)
    df_final = dff[dff.ESTADO.isin([estado])]

    data1 = dff.ANO.min()
    data2 = dff.ANO.max()
    
    fig = go.Figure()

    fig.add_trace(go.Indicator(
        mode = "number+delta",
        title = {"text": f"<span style='size:60%'>{estado}</span><br><span style='font-size:0.7em'>{data1} - {data2}</span>"},
        value = df_final.at[df_final.index[-1],'PREÇO MÉDIO REVENDA'],
        number = {'prefix': "R$", 'valueformat': '.2f'},
        delta = {'relative': True, 'valueformat': '.1%', 'reference': df_final.at[df_final.index[0],'PREÇO MÉDIO REVENDA']}
    ))
    
    fig.update_layout(main_config, height=250, template=template)
    
    return fig

@app.callback(
    Output('dataset', 'data'),
    [Input('rangeslider', 'value'),
    Input('dataset_fixed', 'data')], prevent_initial_call=True
)
def range_slider(range, data):
    df = pd.DataFrame(data)
    dff = df[(df['ANO'] >= range[0]) & (df['ANO'] <= range[1])]
    return dff.to_dict()

@app.callback(
    Output('rangeslider', 'value'),
    Output('controller', 'data'), 

    Input('interval', 'n_intervals'),
    Input('play-button', 'n_clicks'),
    Input('stop-button', 'n_clicks'),

    State('rangeslider', 'value'), 
    State('controller', 'data'), 
    prevent_initial_callbacks = True)
def controller(n_intervals, play, stop, rangeslider, controller):
    trigg = dash.callback_context.triggered[0]["prop_id"]

    if ('play-button' in trigg and not controller["play"]):
        if not controller["play"]:
            controller["play"] = True
            rangeslider[1] = 2007
        
    elif 'stop-button' in trigg:
        if controller["play"]:
            controller["play"] = False

    if controller["play"]:
        if rangeslider[1] == 2021:
            controller['play'] = False
        rangeslider[1] += 1 if rangeslider[1] < 2021 else 0
    
    return rangeslider, controller



# Run server
if __name__ == '__main__':
    app.run_server(debug=True, port=int(os.environ.get("PORT", 8050)), host='0.0.0.0')
