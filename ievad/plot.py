import dash
import yaml
import numpy as np
import pandas as pd
import librosa as lb
import re
import datetime as dt
from . import embed2d
import sounddevice as sd
from pathlib import Path
import plotly.express as px
import plotly.graph_objects as go
from dash.dependencies import Input, Output
from ievad.helpers import get_datetime_from_filename
from ievad.vggish import vggish_params

with open('ievad/config.yaml', 'rb') as f:
    config = yaml.safe_load(f)
    
LOAD_PATH = Path(config['raw_data_path']).joinpath(
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

def dummy_image():
    return np.array([[[255, 0, 0], [0, 255, 0], [0, 0, 255]],
                [[0, 255, 0], [0, 0, 255], [255, 0, 0]]
                ], dtype=np.uint8)
    
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
                dash.html.H1(children=title),
                dash.dcc.Graph(
                    id="bar_chart",
                    figure = px.scatter(data, x='x', y='y', 
                                        color = data['file_stem'],
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
                                   figure = px.imshow(dummy_image(), 
                                                      height = 500)
                                   ),
            ], style={'width': '25%', 'display': 'inline-block',
                      'vertical-align': 'top'})
        ]
    )    

def align_df_and_embeddings(files, meta_df):
    tup = zip(pd.to_datetime(meta_df.file_datetime).values, 
                   meta_df.site.values)
    
    meta_df.index = pd.MultiIndex.from_tuples(tup, names=['datetime', 'site'])

    datetimes = map(get_datetime_from_filename, files)
    sites = map(get_site_from_filename, files)

    return meta_df.loc[zip(datetimes, sites)].drop_duplicates()

def get_site_from_filename(file_path):
    return Path(file_path).stem.split('_')[-3]

def get_stem_from_pathlib(pathlib_object):
    return pathlib_object.stem

def get_df_to_corresponding_file_part(files, meta_df):
    f = lambda p: Path(p).stem
    meta_df.index = list(map(f, meta_df.cond_file))
    return meta_df.loc[np.unique(list(map(f, files)))]

def get_dt_strings_from_filename(f):
    nums = ''.join(re.findall('[0-9]+', f))
    if len(nums) > 14:
        f = f.split('_')[-1]
        nums = ''.join(re.findall('[0-9]+', f))
        
    if len(nums) == 14:
        dtime = dt.datetime.strptime(''.join(re.findall('[0-9]+', f)), 
                             config['dt_format'])
    elif len(nums) == 12:
        dtime = dt.datetime.strptime(''.join(re.findall('[0-9]+', f)), 
                             config['dt_format'].replace('Y', 'y'))
    dt_string = dtime.strftime('%Y-%m-%d %H:%M:%S')
    return dt_string.split(' ')

def plotUMAP_Continuous_plotly(audioEmbeddingsList, percentiles, 
                               colormap, files, lengths, 
                               title = config['title'] ):

    tup = embed2d.compute_embeddings(audioEmbeddingsList, percentiles)
    embeddings, cen, timeLabels, classes = tup

    divisions_array, files_array = embed2d.create_timeList(lengths, 
                                list(map(get_stem_from_pathlib, files)))
        
        
    data = dict()
    if LOAD_PATH.joinpath('meta_data.csv').exists():
        meta_df = pd.read_csv(LOAD_PATH.joinpath('meta_data.csv'))
        meta_df = align_df_and_embeddings(files, meta_df)
        meta_df = get_df_to_corresponding_file_part(files_array, meta_df)
        
        for key in ['preds', 'site', 'file_stem', 'time_in_orig_file']:
            if key in meta_df.keys(): 
                data.update({key: meta_df[key].values})
        if 'file_datetime' in meta_df:
            data.update({
                'file_date': meta_df['file_datetime'][0].split(' ')[0],
                'file_time': meta_df['file_datetime'][0].split(' ')[1]
            })
        n = len(meta_df)
    else:
        try:
            dtimes = list(map(get_dt_strings_from_filename, files_array))
            dates, times = [[d[0] for d in dtimes], [t[1] for t in dtimes]]
        except Exception as e:
            print('time format not found in file name', e)
            dates, times = [0]*len(embeddings), [0]*len(embeddings)
        data.update({'file_date': dates, 
                     'file_time': times,
                     'file_stem': files_array,
                     'site': list(map(lambda s: s.split('_')[0], files_array))})
        n = len(embeddings)
    
    data.update({'x' : embeddings[:,0][:n],
                 'y':  embeddings[:,1][:n],
                 'time_in_condensed_file' : divisions_array[:n],
                 'filename' : files_array[:n]})
    if 'time_in_orig_file' in data.keys():
        orig_file_time = True
    else:
        orig_file_time = False

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
            return (px.imshow(dummy_image(), height = config['umap_height']),
                    "file: ...")
        
        else:
            time_in_file = clickData['points'][0]['customdata'][-2]
            file_path = clickData['points'][0]['customdata'][-1]
            
            audio, sr, file_stem = load_audio(time_in_file, file_path)
            spec = create_specs(audio)
            if autoplay_radio == "Autoplay on":
                play_audio(audio, sr)
            
        if "play_audio_btn" == dash.ctx.triggered_id:
            play_audio(audio, sr)
            
        title = dash.html.P([f"file: {file_stem.split('.Table')[0]}",
                             dash.html.Br(),
                            f"time in file: {time_in_file}",
                             dash.html.Br(),
                            f"location: {file_stem.split('_')[-1]}"])
            
        return spec, title
    
    app.run_server(debug = False)
    
def smoothing_func(num_samps, func='sin'):
    return getattr(np, func)(np.linspace(0, np.pi/2, num_samps))

def fade_audio(audio):
    perc_aud = np.linspace(0, len(audio), 100).astype(int)
    return [*[0]*perc_aud[3], 
            *audio[perc_aud[3]:perc_aud[7]]*config['amp']
                *smoothing_func(perc_aud[7]-perc_aud[3]),
            *audio[perc_aud[7]:perc_aud[70]]*config['amp'], 
            *audio[perc_aud[70]:perc_aud[93]]*config['amp']
                *smoothing_func(perc_aud[93]-perc_aud[70], func='cos'),
            *[0]*(perc_aud[-1]-perc_aud[93])]
    
def play_audio(audio, sr):
    sd.play(fade_audio(audio), sr)
    
def time_string_to_float(t):
    min = int(t.split(':')[0])*60
    sec = int(t.split(':')[1].split('.')[0])
    ms = int(t.split('.')[-1][:-1])/100
    return min+sec+ms
    
def load_audio(t_s, file):
    file_stem = file#Path(file).stem
    main_path = Path(LOAD_PATH)
    t_f = time_string_to_float(t_s)
    
    audio, sr = lb.load(main_path.joinpath(file_stem), 
                        offset=t_f, 
                        sr=vggish_params.SAMPLE_RATE, 
                        duration = vggish_params.EXAMPLE_WINDOW_SECONDS)
    return audio, sr, file_stem
 
def set_axis_lims_dep_sr(S_dB):
    if config['preproc']['downsample']:
        f_max = config['preproc']['downsample_sr'] / 2
        reduce = config['preproc']['model_sr'] / (f_max * 2)
        S_dB = S_dB[:int(S_dB.shape[0] / reduce), :]
    else:
        f_max = config['preproc']['model_sr'] / 2 
    return f_max, S_dB

def create_specs(audio):
    S = np.abs(lb.stft(audio, win_length = config['spec_win_len']))
    S_dB = lb.amplitude_to_db(S, ref=np.max)
    f_max, S_dB = set_axis_lims_dep_sr(S_dB)
    
    if config['preproc']['downsample']:
        f_max = config['preproc']['downsample_sr']
    else:
        f_max = vggish_params.MEL_MAX_HZ

    fig = px.imshow(S_dB, origin='lower', 
                    aspect = 'auto',
                    y = np.linspace(vggish_params.MEL_MIN_HZ, 
                                    f_max, 
                                    S_dB.shape[0]),
                    x = np.linspace(0, 
                                    vggish_params.EXAMPLE_WINDOW_SECONDS, 
                                    S_dB.shape[1]),
                    labels = {'x' : 'time in s', 
                            'y' : 'frequency in Hz'},
                    height = config['spec_height'])
    return fig