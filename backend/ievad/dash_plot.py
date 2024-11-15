import dash
import plotly.express as px
import plotly.graph_objects as go
from dash.dependencies import Input, Output
import numpy as np
from pathlib import Path
import yaml
import pandas as pd
import librosa as lb
from . import helpers as he
from . import plot_helpers as ph

with open('backend/ievad/config.yaml', 'rb') as f:
    config = yaml.safe_load(f)
    
LOAD_PATH = Path(config['audio_dir']).joinpath(
            Path(config['preproc']['annots_path']).stem
            )   
if not LOAD_PATH.exists():
    LOAD_PATH = LOAD_PATH.parent

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
    
def create_specs2(audio):
    hop_length = int(len(audio)/128)
    n_fft = 2048 if hop_length < 2048 else int(2**np.ceil(np.log2(hop_length))) 
    S = np.abs(lb.stft(audio, hop_length=hop_length, n_fft=n_fft))
    S = S[1:].reshape([256, 4, 129]).mean(axis=1)
    S_dB = lb.amplitude_to_db(S, ref=np.max)
    f_max, S_dB = ph.set_axis_lims_dep_sr(S_dB)
    return S_dB
    
def create_specs(audio):
    S = np.abs(lb.stft(audio, win_length = config['spec_win_len']))
    S_dB = lb.amplitude_to_db(S, ref=np.max)
    f_max, S_dB = ph.set_axis_lims_dep_sr(S_dB)
    
    if config['preproc']['downsample']:
        f_max = config['preproc']['plot_spec_sr']/2
    else:
        f_max = he.MEL_MAX_HZ

    fig = px.imshow(S_dB, origin='lower', 
                    aspect = 'auto',
                    y = np.linspace(0, 
                                    f_max, 
                                    S_dB.shape[0]),
                    x = np.linspace(0, 
                                    he.CORRECTED_CONTEXT_WIN_TIME, 
                                    S_dB.shape[1]),
                    labels = {'x' : 'time in s', 
                            'y' : 'frequency in Hz'},
                    height = config['spec_height'])
    return fig
    
def build_dash_layout(data, title, file_date=None, 
                      file_time=None, location=False,
                      orig_file_time=False):
    data = pd.DataFrame(data)
    symbols = ['square', 'circle-dot', 'circle', 'circle-open']
    # TODO clean this up! - keys should only be hard coded once
    # hoverdata = {
    #             'file_date': file_date,
    #             'file_time': file_time,
    #             'site': location,
    #             'time_in_orig_file': orig_file_time, 
    #             'time_in_condensed_file': True, 
    #             'filename': True}
    hoverdata = {key: True for key in data.columns}
    
    return dash.html.Div(
        [
            dash.html.Div([
                dash.html.H1(children=f"{title} - {config['embedding_model']}"),
                dash.dcc.Graph(
                    id="bar_chart",
                    figure = px.scatter(data, x='x', y='y', 
                                        color = data['filename'],
                                        # color = data['annot'],
                                        symbol = data['file_date'],
                                        # symbol_sequence = symbols,
                                        opacity = 0.4,
                                        hover_data = hoverdata,
                                        hover_name = data['file_date'],
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
                                   figure = px.imshow(ph.dummy_image(), 
                                                      height = 500)
                                   ),
            ], style={'width': '25%', 'display': 'inline-block',
                      'vertical-align': 'top'})
        ]
    )    

def plotUMAP_Continuous_plotly(umap_embeds, metadata_dict, divisions_array,
                               title = config['title'] ):

        
    files_array = metadata_dict['files']['audio_files']
    data = dict(x=np.array([]), y=np.array([]), 
                filename=[], file_date=[], file_time=[])
    # if LOAD_PATH.joinpath('meta_data.csv').exists():
    #     meta_df = pd.read_csv(LOAD_PATH.joinpath('meta_data.csv'))
    #     # meta_df = ph.align_df_and_embeddings(files, meta_df)
    #     meta_df = ph.get_df_to_corresponding_file_part(files_array, 
    #                                                    meta_df)
        
    #     for key in ['preds', 'site', 'file_stem', 'time_in_orig_file']:
    #         if key in meta_df.keys(): 
    #             data.update({key: meta_df[key].values})
    #     if 'file_datetime' in meta_df:
    #         data.update({
    #             'file_date': meta_df['file_datetime'][0].split(' ')[0],
    #             'file_time': meta_df['file_datetime'][0].split(' ')[1]
    #         })
    #     n = len(meta_df)
    # else:
    try:
        dtimes = list(map(ph.get_dt_strings_from_filename, files_array))
        dates, times = [[d[0] for d in dtimes], [t[1] for t in dtimes]]
    except Exception as e:
        print('time format not found in file name', e)
        dates, times = [0]*len(umap_embeds), [0]*len(umap_embeds)
    data.update({'time_in_orig_file': divisions_array})
    
    length = []
    for embed in umap_embeds:
        length.append(embed[:,0].shape[0])
        data['x'] = np.append(data['x'], embed[:,0])
        data['y'] = np.append(data['y'], embed[:,1])
        
    for key, array in zip(['filename', 'file_date', 'file_time'], 
                          [files_array, dates, times]):
        [[data[key].append(file) for _ in range(length[ind])] 
         for ind, file in enumerate(array)]
    
    if 'time_in_orig_file' in data.keys():
        orig_file_time = True
    else:
        orig_file_time = False
    
    # from ievad.annots import return_data_with_annots
    # data = return_data_with_annots(config, data)

    app = dash.Dash(__name__, external_stylesheets=['./styles.css'])
    app.layout = build_dash_layout(data, title, file_date=True, 
                                   file_time=True, 
                                   orig_file_time=orig_file_time)

    @app.callback(
        Output("table_container", "figure"),
        Output("graph_heading", "children"),
        Input("bar_chart", "clickData"),
        Input("play_audio_btn", "n_clicks"),
        Input("radio_autoplay", "value"))
    
    def fig_click(clickData, play_btn, autoplay_radio):
        if not clickData:
            return (px.imshow(ph.dummy_image(), height = config['umap_height']),
                    "file: ...")
        
        else:
            time_in_file = clickData['points'][0]['customdata'][3]
            file_path = clickData['points'][0]['customdata'][0]
            
            audio, sr, file_stem = ph.load_audio(time_in_file, file_path)
            spec = create_specs(audio)
            if autoplay_radio == "Autoplay on":
                ph.play_audio(audio, sr)
            
        if "play_audio_btn" == dash.ctx.triggered_id:
            ph.play_audio(audio, sr)
            
        title = dash.html.P([f"file: {file_stem.split('.Table')[0]}",
                             dash.html.Br(),
                            f"time in file: {time_in_file}",
                             dash.html.Br(),
                            f"location: {file_stem.split('_')[-1]}"])
            
        return spec, title
    
    app.run_server(debug = False, port=8054)
    

