import librosa as lb
import numpy as np
from pathlib import Path
import yaml
import time

class Loader():
    def __init__(self, ignore_check_if_combination_exists=False, 
                 model_name='umap', **kwargs):
        self.model_name = model_name
        
        with open('ievad/config.yaml', "r") as f:
            self.config =  yaml.safe_load(f)
            
        for key, val in self.config.items():
            setattr(self, key, val)
        
        self.check_exists = ignore_check_if_combination_exists
        if not self.embeds_already_exist():
            if not self.model_name is 'umap':
                self._get_audio_paths()
            else:
                self.get_embeddings()
            self._init_metadata_dict()
        
    def embeds_already_exist(self):
        self.combination_already_exists = False
        
        if not self.check_exists:
            existing_embed_dirs = Path(self.embed_parent_dir).iterdir()
            for d in existing_embed_dirs:
                if (self.model_name in d.stem 
                    and Path(self.audio_dir).stem in d.stem):
                    num_files = len([f for f in d.iterdir() 
                                    if f.suffix == '.pickle'])
                    num_audio_files = len([f for f in 
                                           Path(self.audio_dir).iterdir()])
                    if num_audio_files == num_files:
                        self.combination_already_exists = True
                        return 1

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
        
    def get_embeddings(self):
        self.folder = self.get_embedding_dir()
        self.files = [f for f in self.folder.iterdir() 
                      if f.suffix == '.pickle']
        
        with open(self.folder.joinpath('metadata.yml'), "r") as f:
            self.metadata =  yaml.safe_load(f)

    def get_embedding_dir(self):
        embed_dirs = [d for d in self.embed_parent_dir.iterdir()
                    if self.audio_dir.stem in d.stem and 
                    self.model_name in d.stem]
        # TODO fix paths so the the metadata file is loaded here
        most_recent_emdbed_dir = embed_dirs[0]
        return most_recent_emdbed_dir
    
    def get_annotations(self):
        pass
        # self.load_path = (Path(self.audio_dir)
        #                   .joinpath(Path(
        #                       self.preproc['annots_path']).stem
        #                             )
        #                   )
        # if not self.load_path.exists():
        #     self.load_path = self.load_path.parent
        
    def _get_audio_paths(self):
        self.audio_dir = Path(self.audio_dir)

        top_level_dir = time.strftime('%Y-%m-%d_%H-%M___'
                                      + self.model_name
                                      + '-'
                                      + self.audio_dir.stem,
                                      time.localtime())
        self.files = self.audio_dir.iterdir()
        
        self.embed_dir = (Path(self.embed_parent_dir)
                          .joinpath(top_level_dir))
        self.embed_dir.mkdir(exist_ok=True, parents=True)
        
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
        if not file.suffix in ['.WAV', '.wav', '.aif']:
            return None
        audio, _ = lb.load(file, sr=self.sr)
        
        self.metadata_dict['files']['audio_files'].append(
            file.stem + file.suffix
            )
        self.metadata_dict['files']['file_lengths (s)'].append(
            len(audio)//self.sr
            )
        
        return (audio * 32767).astype(np.int16)
    
    def write_metadata_file(self):
        with open(str(self.embed_dir.joinpath('metadata.yml')), 'w') as f:
            yaml.safe_dump(self.metadata_dict, f)

class Embedder():
    def __init__(self, model_name, **kwargs):
        self.model_name = model_name
        self._init_model(**kwargs)

    def _init_model(self, **kwargs):
        self.model = getattr(self, 
                             f'get_callable_{self.model_name}_model')(**kwargs)
        
    def get_callable_umap_model(self, **kwargs):
        import umap
        return umap.UMAP(**kwargs).fit_transform


    def get_callable_vggish_model(self, **kwargs):
        import tensorflow_hub as hub
        return hub.load('ievad/models/vggish')

    def get_embeddings_from_model(self, samples):
        embeds = self.model(samples)
        if not isinstance(embeds, np.ndarray):
            embeds = embeds.numpy()
        return embeds

    def save_embeddings_as_pickle(self, embed_dir, file, embeds):
        file_dest = embed_dir.joinpath(file.stem 
                                       + f'_{self.model_name}.pickle')
        
        import pickle
        with open(file_dest, 'wb') as f:
            pickle.dump(embeds, f, protocol=pickle.HIGHEST_PROTOCOL)

def generate_embeddings(**kwargs):
    ld = Loader(**kwargs)
    if not ld.embeds_already_exist():    
        embed = Embedder(**kwargs)
        for file in ld.files:
            sample = ld.load(file)
            if sample is None:
                continue
            embeds = embed.get_embeddings_from_model(sample)
            embed.save_embeddings_as_pickle(ld.embed_dir, file, embeds)
        ld.write_metadata_file()
