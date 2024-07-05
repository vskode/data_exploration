import librosa as lb
import numpy as np
from pathlib import Path
import yaml
import time
import logging
logger = logging.getLogger('ievad')

class Loader():
    def __init__(self, check_if_combination_exists=True, 
                 model_name='umap', **kwargs):
        self.model_name = model_name
        
        with open('ievad/config.yaml', "r") as f:
            self.config =  yaml.safe_load(f)
            
        for key, val in self.config.items():
            setattr(self, key, val)
        
        self.check_if_combination_exists = check_if_combination_exists
        self.check_embeds_already_exist()
        if self.model_name != 'umap':
            self._get_audio_paths()
            self._init_metadata_dict()
        else:
            self.get_embeddings()
        
        if not self.combination_already_exists:
            self.embed_dir.mkdir(exist_ok=True, parents=True)
        else:
            logger.debug(
                'Combination of {} and {} already '
                'exists -> using saved embeddings in {}'
                .format(self.model_name,
                        Path(self.audio_dir).stem,
                        str(self.embed_dir))
                )
        
    def check_embeds_already_exist(self):
        self.combination_already_exists = False
        
        if self.check_if_combination_exists:
            if self.model_name == 'umap':
                existing_embed_dirs = Path(self.umap_parent_dir).iterdir()
            else:
                existing_embed_dirs = Path(self.embed_parent_dir).iterdir()
                
            for d in existing_embed_dirs:
                
                if (self.model_name in d.stem 
                    and Path(self.audio_dir).stem in d.stem
                    and self.embedding_model in d.stem):
                    
                    num_files = len([f for f in d.iterdir() 
                                    if f.suffix == '.pickle'])
                    num_audio_files = len([f for f in 
                                           Path(self.audio_dir).iterdir()])
                    
                    if num_audio_files == num_files:
                        self.combination_already_exists = True

    def _init_metadata_dict(self):
        self.metadata_dict = {
            'model_name': self.model_name,
            'sr': self.sr,
            'audio_dir': str(self.audio_dir),
            'embed_dir': str(self.embed_dir),
            'files' : {
                'audio_files': [],
                'file_lengths (s)': [],
            }
        }
        
    def _get_metadata_dict(self, folder):
        with open(folder.joinpath('metadata.yml'), "r") as f:
            self.metadata_dict =  yaml.safe_load(f)
        
    def get_embeddings(self):
        embed_dir = self.get_embedding_dir()
        self.files = [f for f in embed_dir.iterdir() 
                      if f.suffix == '.pickle']
        self.files.sort()
        self._get_metadata_dict(embed_dir)
        self.embed_dir = (Path(self.umap_parent_dir)
                            .joinpath(self.get_timestamp_dir()
                                    + f'-{self.embedding_model}'))

    def get_embedding_dir(self):
        if self.combination_already_exists and self.model_name == 'umap':
            self.embed_parent_dir = Path(self.umap_parent_dir)
        else:
            self.embed_parent_dir = Path(self.embed_parent_dir)
        self.audio_dir = Path(self.audio_dir)
        
        embed_dirs = [d for d in self.embed_parent_dir.iterdir()
                    if self.audio_dir.stem in d.stem and 
                    self.embedding_model in d.stem]
        
        most_recent_emdbed_dir = embed_dirs[0]
        return most_recent_emdbed_dir
    
    def get_annotations(self):
        pass
        
    def get_timestamp_dir(self):
        return time.strftime('%Y-%m-%d_%H-%M___'
                                      + self.model_name
                                      + '-'
                                      + self.audio_dir.stem,
                                      time.localtime())
        
    def _get_audio_paths(self):
        self.audio_dir = Path(self.audio_dir)

        self.files = self.audio_dir.iterdir()
        
        self.embed_dir = (Path(self.embed_parent_dir)
                          .joinpath(self.get_timestamp_dir()))
        
    def embed_read(self, file):
        import pickle
        with open(file, 'rb') as e:
            return pickle.load(e)
    
    def load(self, file):
        if not self.model_name in ['umap', 'tsne']:
            return self.audio_read(file)
        else:
            return self.embed_read(file)
    
    def audio_read(self, file):
        if not file.suffix in ['.WAV', '.wav', '.aif', '.m4a']:
            return None
        audio, _ = lb.load(file, sr=self.sr)
        
        self.metadata_dict['files']['audio_files'].append(
            file.stem + file.suffix
            )
        self.metadata_dict['files']['file_lengths (s)'].append(
            len(audio)//self.sr
            )
        
        return audio
    
    def write_metadata_file(self):
        with open(str(self.embed_dir.joinpath('metadata.yml')), 'w') as f:
            yaml.safe_dump(self.metadata_dict, f)
            
    def update_attributes(self, **kwargs):
        if self.model_name == 'umap':
            self.__init__(**kwargs)

class PrepareModel():
    def get_callable_aves_model(self, pooling='mean', **kwargs):
        from .get_aves_model import AvesTorchaudioWrapper
        return AvesTorchaudioWrapper(pooling=pooling)
    
    def get_callable_umap_model(self, **kwargs):
        import umap
        return umap.UMAP(n_neighbors=self.config['n_neighbors'],
                         n_components=self.config['n_components'],
                         metric=self.config['metric'],
                         random_state=self.config['random_state']
                         ).fit_transform


    def get_callable_vggish_model(self, **kwargs):
        import tensorflow_hub as hub
        return hub.load('ievad/files/models/vggish')

class PreProcessing():    
    def get_vggish_preprocessing(self, audio):
        return (audio * 32767).astype(np.int16)

    def get_umap_preprocessing(self, embeds):
        return embeds
    
    def get_aves_preprocessing(self, audio):
        import torch
        segment_length = int(self.config['sr']
                          *self.config['preproc']['model_time_length'])
        num_of_segments = int(audio.shape[0]/segment_length)
        audio = audio[:num_of_segments*segment_length]
        audio = audio.reshape(num_of_segments, segment_length)
        
        return torch.tensor(audio)

class Embedder(PrepareModel, PreProcessing):
    def __init__(self, model_name, **kwargs):
        import yaml
        with open('ievad/config.yaml', 'rb') as f:
            self.config = yaml.safe_load(f)
            
        self.model_name = model_name
        self._init_model(**kwargs)

    def _init_model(self, **kwargs):
        self.model = getattr(self, 
                             f'get_callable_{self.model_name}_model')(**kwargs)
        

    def get_embeddings_from_model(self, input):
        samples = getattr(self, f'get_{self.model_name}_preprocessing')(input)
        start = time.time()
        
        embeds = self.model(samples)
        if not isinstance(embeds, np.ndarray):
            embeds = embeds.numpy()
        
        logger.debug(f'{self.model_name} embeddings have shape: {embeds.shape}')
        logger.info(f'{self.model_name} inference took {time.time()-start:.2f}s.')
        return embeds

    def save_embeddings_as_pickle(self, embed_dir, file, embeds):
        file_dest = embed_dir.joinpath(file.stem 
                                       + f'_{self.model_name}.pickle')
        
        import pickle
        with open(file_dest, 'wb') as f:
            pickle.dump(embeds, f, protocol=pickle.HIGHEST_PROTOCOL)
    
def generate_embeddings(**kwargs):
    ld = Loader(**kwargs)
    if not ld.combination_already_exists:    
        embed = Embedder(**kwargs)
        for file in ld.files:
            input = ld.load(file)
            if input is None:
                continue
            embeds = embed.get_embeddings_from_model(input)
            embed.save_embeddings_as_pickle(ld.embed_dir, file, embeds)
        ld.write_metadata_file()
        ld.update_attributes(**kwargs)
    return ld
