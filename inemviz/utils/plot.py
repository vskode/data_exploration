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
    
def build_dash_layout(data, title, timeLabels, location=None):
    symbols = ['square', 'circle-dot', 'circle', 'circle-open']
    hoverdata = {
                'file date': False,
                'file time': True,
                'location': True,
                'time within original file': True, 
                'time within condensed file': True, 
                'filename': True}
    
    return dash.html.Div(
        [
            dash.html.Div([
                dash.html.H1(children=title),
                dash.dcc.Graph(
                    id="bar_chart",
                    figure = px.scatter(data, x='x', y='y', 
                                        color = data['file stems'],
                                        # symbol = location,
                                        # symbol_sequence = symbols,
                                        opacity = 0.4,
                                        hover_data = hoverdata,
                                        hover_name = data['file date'],
                                        height = 900
                                        ),
                            )
            ], style={'width': '75%', 'display': 'inline-block',
                      'vertical-align': 'top'}),
            
            dash.html.Div([
                    dash.html.H2(children='Spectrogram'),
                    dash.html.Div(id='graph_heading', children='file: ...'),
                	dash.html.Button(id="play_audio_btn", children="Play Sound", 
                                  n_clicks = 0),
                    dash.dcc.RadioItems(['Autoplay on', 'Autoplay off'], 
                                    'Autoplay off', id='radio_autoplay'),
                    dash.dcc.Graph(id="table_container", 
                                   figure = px.imshow(dummy_image(), 
                                                      height = 900)
                                   ),
            ], style={'width': '25%', 'display': 'inline-block',
                      'vertical-align': 'top'})
        ]
    )    

def fix_misalligned_df(meta_df, file_stems, lengths):
    a = [len(np.where(meta_df.file_stems == stem)[0]) for stem in file_stems]
    a = np.array(a)
    for i in np.where(abs(lengths-a)>0)[0]:
        bool_len = (meta_df.lengths != lengths[i]).values
        bool_stem = (meta_df.file_stems == file_stems[i]).values
        meta_df = meta_df[~(bool_len * bool_stem)]
    return meta_df

def plotUMAP_Continuous_plotly(audioEmbeddingsList, percentiles, 
                               colormap, files, lengths, 
                               title = config['title'] ):

    tup = embed2d.compute_embeddings(audioEmbeddingsList, percentiles)
    embeddings, cen, timeLabels, classes = tup

    divisions_array, files_array = embed2d.create_timeList(lengths, files)
    
    meta_df = pd.read_csv(config['raw_data_path'] + '/meta_data.csv')
    
    file_stems = list(map(lambda path: Path(path).stem.split('.Table')[0], files))
    
    meta_df.index = meta_df.file_stems
    meta_df = meta_df.loc[file_stems]
    
    if len(meta_df) != len(files_array):
        meta_df = fix_misalligned_df(meta_df, file_stems, lengths)
    
    data = pd.DataFrame({'x' : embeddings[:,0], 
                         'y':  embeddings[:,1],
                        'location': meta_df['site'],
                'file date': meta_df['file_datetime'][0].split(' ')[0],
                'file time': meta_df['file_datetime'][0].split(' ')[1],
                        'time within original file' : meta_df['call_time'],
                        'time within condensed file' : divisions_array,
                        'filename' : files_array,
                        'file stems': meta_df.file_stems})

    app = dash.Dash(__name__, external_stylesheets=['./styles.css'])
    app.layout = build_dash_layout(data, title, 
                                   meta_df['file_datetime'].values, 
                                   meta_df['site'].values)

    @app.callback(
        Output("table_container", "figure"),
        Output("graph_heading", "children"),
        Input("bar_chart", "clickData"),
        Input("play_audio_btn", "n_clicks"),
        Input("radio_autoplay", "value"))
    
    def fig_click(clickData, play_btn, autoplay_rad):
        if not clickData:
            return px.imshow(dummy_image(), height = 900), "file: ..."
        
        else:
            time_in_file = clickData['points'][0]['customdata'][-2]
            file_path = clickData['points'][0]['customdata'][-1]
            
            audio, sr, file_stem = load_audio(time_in_file, file_path)
            spec = create_specs(audio, file_path)
            if autoplay_rad == "Autoplay on":
                play_audio(audio, sr)
            
        if "play_audio_btn" == dash.ctx.triggered_id:
            play_audio(audio, sr)
            
        title = f"file: {file_stem} at {time_in_file}"
            
        return spec, title
    
    app.run_server(debug = False)
    
def play_audio(audio, sr):
    smoothing = np.cos(np.linspace(0,np.pi/2,500))
    test = [*[0]*500, 
            *audio[500:1000]*30*smoothing[::-1], 
            *audio[1000:-5000]*30, 
            *audio[-3500:-1000]*30*np.cos(np.linspace(0,np.pi/2,2500)), 
            *[0]*1000]
    sd.play(test, sr)
    # sd.play(audio, sr)
    
def time_string_to_float(t):
    minu = int(t.split(':')[0])*60
    sec = int(t.split(':')[1].split('.')[0])
    ms = int(t.split('.')[-1][:-1])/100
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

    if config['preproc']['downsample']:
        f_max = config['preproc']['downsample_sr'] / 2
        reduce = config['preproc']['model_sr'] / (f_max * 2)
        S_dB = S_dB[:int(S_dB.shape[0] / reduce), :]
    else:
        f_max = config['preproc']['model_sr'] / 2

    fig = px.imshow(S_dB, origin='lower', 
                    aspect = 'auto',
                    y = np.linspace(50, f_max, S_dB.shape[0]),
                    x = np.linspace(0, 0.96, S_dB.shape[1]),
                    labels = {'x' : 'time in s', 
                            'y' : 'frequency in Hz'},
                    height = 850)
    return fig