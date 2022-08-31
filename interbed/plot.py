import dash
import yaml
import numpy as np
import pandas as pd
import librosa as lb
from . import embed2d
import sounddevice as sd
from pathlib import Path
import plotly.express as px
import plotly.graph_objects as go
from dash.dependencies import Input, Output

with open('interbed/config.yaml', 'rb') as f:
    config = yaml.safe_load(f)

def plot_wo_specs(data, timeLabels, title, centroids, classes):
    fig = px.scatter(data, x='x', y='y', color=timeLabels, opacity = 0.4,
                    hover_data = ['time_within_file', 'filename'],
                    title = 'UMAP Embedding for {}'.format(title))
    fig.add_trace(
        go.Scatter(
            x = centroids[:,0], y= centroids[:,1], mode = 'markers',
            marker = dict(
                color = classes,
                size = [20]*10
            ) ) )
    
    fig.show()
    fig.write_html('Interactive_Plot.html')

def dummy_image():
    return np.array([[[255, 0, 0], [0, 255, 0], [0, 0, 255]],
                [[0, 255, 0], [0, 0, 255], [255, 0, 0]]
                ], dtype=np.uint8)
    
def plotUMAP_Continuous_plotly(audioEmbeddingsList, percentiles, 
                               colormap, files, lengths, 
                               title = config['title'] ):

    embeddings, cen, timeLabels, classes = embed2d.compute_embeddings(audioEmbeddingsList,
                                                           percentiles)

    divisions_array, files_array = embed2d.create_timeList(lengths, files)
    
    data = pd.DataFrame({'x' : embeddings[:,0], 
                         'y':embeddings[:,1],
                        'time_within_file' : divisions_array,
                        'filename' : files_array})

    app = dash.Dash(__name__, external_stylesheets=['./styles.css'])
    app.layout = dash.html.Div(
        [
            dash.html.Div([
                dash.html.H1(children=title),
                dash.dcc.Graph(
                    id="bar_chart",
                    figure = px.scatter(data, x='x', y='y', color=timeLabels,
                                        opacity = 0.4,
                                        hover_data = ['time_within_file', 
                                                      'filename'],
                                        height = 900
                                        ),
                            )
            ], style={'width': '75%', 'display': 'inline-block',
                      'vertical-align': 'top'}),
            
            dash.html.Div([
                    dash.html.H2(children='Spectrogram'),
                    dash.html.Div(id='graph_heading', children='file: ...'),
                	dash.html.Button(id="button1", children="Play Sound", 
                                  n_clicks = 0),
                    dash.dcc.Graph(id="table_container", 
                                   figure = px.imshow(dummy_image(), 
                                                      height = 900)
                                   ),
            ], style={'width': '25%', 'display': 'inline-block',
                      'vertical-align': 'top'})
        ]
    )

    @app.callback(
        Output("table_container", "figure"),
        Output("graph_heading", "children"),
        Input("bar_chart", "clickData"),
        Input("button1", "n_clicks"))
    
    def fig_click(clickData, *_):
        if not clickData:
            return px.imshow(dummy_image(), height = 900), "file: ..."
        
        else:
            time_in_file = clickData['points'][0]['customdata'][0]
            file_path = clickData['points'][0]['customdata'][1]
            
            audio, sr, file_stem = load_audio(time_in_file, file_path)
            spec = create_specs(audio, file_path)
            
        if "button1" == dash.ctx.triggered_id:
            print('test')
            play_audio(audio, sr)
            
        title = f"file: {file_stem} at {time_in_file}"
            
        return spec, title
    
    app.run_server(debug = False)
    
def play_audio(audio, sr):
    sd.play(audio, sr)
    
def time_string_to_float(t):
    minu = int(t.split(':')[0])*60
    sec = int(t.split(':')[1].split('.')[0])
    ms = int(t.split('.')[-1][:-1])/10
    return minu+sec+ms
    
def load_audio(t_s, file):
    file_stem = Path(file).stem
    main_path = Path(config['raw_data_path'])

    t_f = time_string_to_float(t_s)
    
    audio, sr = lb.load(main_path.joinpath(file_stem), 
                        sr=16000, offset=t_f, duration = 0.96)
    return audio, sr, file_stem
 
def create_specs(audio, t):
 
    S = np.abs(lb.stft(audio, win_length = 512))

    S_dB = lb.amplitude_to_db(S, ref=np.max)

    fig = px.imshow(S_dB, origin='lower', 
                    aspect = 'auto',
                    y = np.linspace(50, 8000, S.shape[0]),
                    x = np.linspace(0, 0.96, S.shape[1]),
                    labels = {'x' : 'time in s', 
                            'y' : 'frequency in Hz'},
                    height = 850)
    return fig