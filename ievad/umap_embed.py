import yaml
import pickle
import numpy as np
from pathlib import Path
from ievad.generate_embeddings import generate_embeddings, Loader
 
with open('ievad/config.yaml', 'rb') as f:
    config = yaml.safe_load(f)
    

def get_embeddings():
    generate_embeddings(model_name=config['embedding_model'], 
                        check_if_combination_exists=True)
    ld = generate_embeddings(model_name='umap', 
                        check_if_combination_exists=True)
    embeds = []
    for file in ld.files:
        embeds.append(pickle.load(open(file, 'rb')))
    
    return embeds, ld.metadata_dict

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
    
