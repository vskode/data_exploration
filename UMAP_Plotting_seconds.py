# Written by Avery Bick, June 2021
# Modified by Vincent Kather, August 2022
# Adapted from UMAP documentation: https://umap-learn.readthedocs.io/en/latest/supervised.html
"""
Apply UMAP dimensionality reduction to Audioset embeddings
"""

from __future__ import division
import sys
import umap
import dash
import glob
import yaml
import pickle
import numpy as np
import pandas as pd
import librosa as lb
import sounddevice as sd
from pathlib import Path
import plotly.express as px
import plotly.graph_objects as go
from dash.dependencies import Input, Output
 
with open('config.yaml', 'rb') as f:
    config = yaml.safe_load(f)

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', 10)



def splitEmbeddingsByColumnValue(df, column):
	dfs = [d for _, d in df.groupby([column])]
	values = df[column].unique()
	return(dfs, values)

def createTimeLabelsList(length, percentiles):
	outList = []
	labelCount = int(length/percentiles)

	for percentile in list(range(percentiles)):
		tmpList = [percentile]*labelCount
		outList.extend(tmpList)

	if len(outList) < length:
		nextNumber = outList[-1]+1

		while len(outList) < length:
			outList.append(nextNumber)

	if len(outList) > length:
		while len(outList) > length:
			outList = outList[:-1]

	return np.array(outList)

def calculateCentroids_Continuous(embeddings, timeLabels):
	classCentroids = []
	arr = np.column_stack((embeddings,timeLabels))
	split_arrs = np.split(arr, np.where(np.diff(arr[:,2]))[0]+1)
	for array in split_arrs:
		length = array.shape[0]
		sum_x = np.sum(array[:, 0])
		sum_y = np.sum(array[:, 1])
		centroid = [sum_x/length,sum_y/length]
		classCentroids.append(centroid)
	outArray = np.array(classCentroids)
	return outArray

def calculateCentroids_Classes(l):
	array = np.array(l)
	length = array.shape[0]
	sum_x = np.sum(array[:, 0])
	sum_y = np.sum(array[:, 1])
	centroid = [sum_x/length,sum_y/length]
	return centroid

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
    
def create_timeList(lengths, files):
    lin_array = np.linspace(0, config['length_of_file'], lengths[0])
    files_array = []
    divisions_array = []
    for i in range(len(lengths)):
        for j in range(lengths[i]):
            files_array.append(files[i])
            divisions_array.append(
                f'{int(lin_array[j]/60)}:{np.mod(lin_array[j], 60):.1f}s')
    return divisions_array, files_array
    
def compute_embeddings(audioEmbeddingsList, percentiles):
    embeddings = umap.UMAP(n_neighbors=10).fit_transform(audioEmbeddingsList)
    timeLabels = createTimeLabelsList(len(audioEmbeddingsList), percentiles)
    classes = list(set(timeLabels))

    centroids = calculateCentroids_Continuous(embeddings, timeLabels)

    embeddings = np.array(embeddings)
    centroids = np.array(centroids)
    return embeddings, centroids, timeLabels, classes
    
def dummy_image():
    return np.array([[[255, 0, 0], [0, 255, 0], [0, 0, 255]],
                [[0, 255, 0], [0, 0, 255], [255, 0, 0]]
                ], dtype=np.uint8)
    
def plotUMAP_Continuous_plotly(audioEmbeddingsList, percentiles, title, 
                        colormap, files, lengths):

    embeddings, cen, timeLabels, classes = compute_embeddings(audioEmbeddingsList,
                                                           percentiles)


    
    divisions_array, files_array = create_timeList(lengths, files)
    
    
    data = pd.DataFrame({'x' : embeddings[:,0], 'y':embeddings[:,1],
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
                    hover_data = ['time_within_file', 'filename'],
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
                    dash.dcc.Graph(	id="table_container", figure = px.imshow(
                        dummy_image(), height = 900)),
            ], style={'width': '25%', 'display': 'inline-block',
                      'vertical-align': 'top'})
        ]
    )

    @app.callback(
        Output("table_container", "figure"),
        Output("graph_heading", "children"),
        Input("bar_chart", "clickData"),
        Input("button1", "n_clicks"))
    def fig_click(clickData, btn1):
        
            
        if not clickData:
            return dummy_image(), "file: ..."
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
    
def load_audio(t, file):
    file_stem = Path(file).stem
    main_path = Path(config['raw_data_path'])

    minu = int(t.split(':')[0])*60
    sec = int(t.split(':')[1].split('.')[0])
    ms = int(t.split('.')[-1][:-1])/10
    ttt = minu+sec+ms
    audio, sr = lb.load(main_path.joinpath(file_stem), sr=16000, offset=ttt, 
                        duration = 0.96)
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
                    height = 900)
    return fig


def get_embeddings():
    folders = glob.glob(config['pickled_data_path'])
    acc_embeddings, file_list, lenghts = [], [], []
    
    for folder in folders:
        files = glob.glob(folder + '/*.pickle')
        files = np.sort(files).astype(list)
        
        for file in files:
            
            with open(file, 'rb') as loadf:
                audioEmbeddings = pickle.load(loadf)
                acc_embeddings = [*acc_embeddings, *audioEmbeddings]
                lenghts.append(len(audioEmbeddings))
                file_list.append(file)
    
    return acc_embeddings, folders, file_list, lenghts

if __name__ == "__main__":

    acc_embeddings, folders, file_list, lenghts = get_embeddings()
    percentiles = 24
    plotUMAP_Continuous_plotly(acc_embeddings, 
                            percentiles, config['title'], 
                            'plasma', file_list, lenghts)

    sys.exit()

# download checkpoint either mine or googles: 
# https://drive.google.com/file/d/1k1UpQFKSMkmdjYlm1GphjP-nW1uMHEiU/view?usp=sharing
# https://storage.googleapis.com/audioset/vggish_model.ckpt