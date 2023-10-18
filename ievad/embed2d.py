# Written by Avery Bick, June 2021
# Modified by Vincent Kather, August 2022
# Adapted from UMAP documentation: https://umap-learn.readthedocs.io/en/latest/supervised.html
"""
Apply UMAP dimensionality reduction to Audioset embeddings
"""

from __future__ import division
import umap
import glob
import yaml
import pickle
import numpy as np
import pandas as pd
from pathlib import Path
 
with open('ievad/config.yaml', 'rb') as f:
    config = yaml.safe_load(f)
    
LOAD_PATH = Path(config['pickled_data_path']).joinpath(
            Path(config['raw_data_path']).stem
            )   

if not LOAD_PATH.exists():
    LOAD_PATH.parent.joinpath(Path(config['preproc']['annots_path']).stem)
       
    if not LOAD_PATH.exists():
        LOAD_PATH = LOAD_PATH.parent

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
    
def create_timeList(lengths, files):
    lin_array = np.arange(0, max(lengths), 0.96)
    files_array = []
    divisions_array = []
    for i in range(len(lengths)):
        for j in range(lengths[i]):
            files_array.append(files[i])
            divisions_array.append(
                f'{int(lin_array[j]/60)}:{np.mod(lin_array[j], 60):.2f}s')
    return divisions_array, files_array
    
def compute_embeddings(audioEmbeddingsList, percentiles):
    embeddings = umap.UMAP(n_neighbors=10).fit_transform(audioEmbeddingsList)
    timeLabels = createTimeLabelsList(len(audioEmbeddingsList), percentiles)
    classes = list(set(timeLabels))

    centroids = calculateCentroids_Continuous(embeddings, timeLabels)

    embeddings = np.array(embeddings)
    centroids = np.array(centroids)
    return embeddings, centroids, timeLabels, classes

def get_embeddings(limit = None):
    # TODO einfügen, dass man auch nicht condensed files laden kann,
    # in dem fall wären es dann ganz viele ordner
    # folders = glob.glob(config['pickled_data_path']) 
    folders = [LOAD_PATH] 
    acc_embeddings, file_list, lengths = [], [], []
    
    for folder in folders:
        files = list(folder.glob('*.pickle'))
        files = np.sort(files).astype(list)[:limit]
        
        for file in files:
            
            with open(file, 'rb') as loadf:
                audioEmbeddings = pickle.load(loadf)
                acc_embeddings = [*acc_embeddings, *audioEmbeddings]
                lengths.append(len(audioEmbeddings))
                file_list.append(file)
    
    return acc_embeddings, folders, file_list, lengths