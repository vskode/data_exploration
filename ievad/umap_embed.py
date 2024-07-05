import yaml
import pickle
import logging
import numpy as np
from pathlib import Path
from ievad.generate_embeddings import generate_embeddings
logger = logging.getLogger('ievad')
 
with open('ievad/config.yaml', 'rb') as f:
    config = yaml.safe_load(f)

def get_embeddings(check_if_combination_exists=True):
    
    generate_embeddings(model_name=config['embedding_model'], 
                        check_if_combination_exists=check_if_combination_exists)
    ld = generate_embeddings(model_name='umap', 
                        check_if_combination_exists=check_if_combination_exists)
    embeds, divisions_array = [], []
    for ind, file in enumerate(ld.files):
        embeds.append(pickle.load(open(file, 'rb')))
        append_timeList(embeds[ind].shape[0],
                        divisions_array)
        
    return embeds, ld.metadata_dict, divisions_array

def append_timeList(length, divisions_array = []):
    lin_array = np.arange(0, length, 
                          config[
                              'preproc']['model_time_length'])
    for j in range(length):
        divisions_array.append(
            f'{int(lin_array[j]/60)}:{np.mod(lin_array[j], 60):.2f}s')
    
