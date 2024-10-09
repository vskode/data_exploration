import librosa as lb
import numpy as np
from pathlib import Path
import yaml
import time
from tqdm import tqdm
import logging
logger = logging.getLogger('ievad')

class Loader():
    def __init__(self, check_if_combination_exists=True, 
                 model_name='umap', **kwargs):
        self.model_name = model_name
        
        with open('backend/ievad/config.yaml', "r") as f:
            self.config =  yaml.safe_load(f)
            
        for key, val in self.config.items():
            setattr(self, key, val)
        
        self.check_if_combination_exists = check_if_combination_exists
        if self.model_name == 'umap':
            self.embed_suffix = '.json'
        else:
            self.embed_suffix = '.npy'
            
        self.check_embeds_already_exist()
        if self.combination_already_exists or self.model_name == 'umap':
            self.get_embeddings()
        else:
            self._get_audio_paths()
            self._init_metadata_dict()
        
        if not self.combination_already_exists:
            self.embed_dir.mkdir(exist_ok=True, parents=True)
        else:
            if self.model_name == 'umap':
                self.embed_dir = self.umap_embed_dir
            logger.debug(
                'Combination of {} and {} already '
                'exists -> using saved embeddings in {}'
                .format(self.model_name,
                        Path(self.audio_dir).stem,
                        str(self.embed_dir))
                )
        
    def check_embeds_already_exist(self):
        self.combination_already_exists = False
        self.umap_embed_dir = False
        
        if self.check_if_combination_exists:
            if self.model_name == 'umap':
                existing_embed_dirs = list(Path(self.umap_parent_dir)
                                           .iterdir())
            else:
                existing_embed_dirs = list(Path(self.embed_parent_dir)
                                           .iterdir())
            if isinstance(self.check_if_combination_exists, str):
                existing_embed_dirs = [existing_embed_dirs[0].parent
                                       .joinpath(
                                           self.check_if_combination_exists
                                       )]
            existing_embed_dirs.sort()
            for d in existing_embed_dirs[::-1]:
                
                if (self.model_name in d.stem 
                    and Path(self.audio_dir).stem in d.stem
                    and self.embedding_model in d.stem):
                    
                    num_files = len([f for f in d.iterdir() 
                                    if f.suffix == self.embed_suffix])
                    num_audio_files = len([
                        f for f in Path(self.audio_dir).iterdir()
                        if f.suffix in self.config['audio_suffixes']
                        ])
                    
                    if num_audio_files == num_files:
                        self.combination_already_exists = True
                        self._get_metadata_dict(d)
                        break
        
    def _get_audio_paths(self):
        self.audio_dir = Path(self.audio_dir)

        self.files = self._get_audio_files()
        
        self.embed_dir = (Path(self.embed_parent_dir)
                          .joinpath(self.get_timestamp_dir()))

    def _get_audio_files(self):
        files_list = []
        [[files_list.append(ll) for ll in self.audio_dir.rglob(f'*{string}')] 
         for string in self.config['audio_suffixes']]
        return files_list
    
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
        for key, val in self.metadata_dict.items():
            if isinstance(val, str) and Path(val).is_dir():
                setattr(self, key, Path(val))
        if self.model_name == 'umap':
            self.umap_embed_dir = folder
        
    def get_embeddings(self):
        embed_dir = self.get_embedding_dir()
        self.files = [f for f in embed_dir.iterdir() 
                      if f.suffix == self.embed_suffix]
        self.files.sort()
        if not self.combination_already_exists:
            self._get_metadata_dict(embed_dir)
            self.metadata_dict['files'].update(
                {'embedding_files': [],
                'embedding_dimensions': []}
            )
            self.embed_dir = (Path(self.umap_parent_dir)
                                .joinpath(self.get_timestamp_dir()
                                        + f'-{self.embedding_model}'))

    def get_embedding_dir(self):
        if self.model_name == 'umap':
            if self.combination_already_exists:
                self.embed_parent_dir = Path(self.umap_parent_dir)
            else:
                self.embed_parent_dir = Path(self.embed_parent_dir)
                self.embed_suffix = '.npy'
        else:
            return self.embed_dir
        self.audio_dir = Path(self.audio_dir)
        
        if self.umap_embed_dir:
            # check if they are compatible
            return self.umap_embed_dir
        
        embed_dirs = [d for d in self.embed_parent_dir.iterdir()
                    if self.audio_dir.stem in d.stem and 
                    self.embedding_model in d.stem]
        # check if timestamp of umap is after timestamp of model embeddings
        embed_dirs.sort()
        most_recent_emdbed_dir = embed_dirs[-1]
        return most_recent_emdbed_dir
    
    def get_annotations(self):
        pass
        
    def get_timestamp_dir(self):
        return time.strftime('%Y-%m-%d_%H-%M___'
                                      + self.model_name
                                      + '-'
                                      + self.audio_dir.stem,
                                      time.localtime())
        
    def embed_read(self, file):
        embeds = np.load(file)
        self.metadata_dict['files']['embedding_files'].append(
            str(file)
            )
        self.metadata_dict['files']['embedding_dimensions'].append(
            embeds.shape
            )
        return embeds
    
    def load(self, file):
        if not self.model_name in ['umap', 'tsne']:
            return self.audio_read(file)
        else:
            return self.embed_read(file)
    
    def audio_read(self, file):
        if not file.suffix in self.config['audio_suffixes']:
            logger.warning(f'{str(file)} could not be read due to unsupported format.')
            return None
        with open(file, 'rb') as r:
            audio, _ = lb.load(r, sr=self.sr)
        
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
            
    def update_files(self):
        if self.model_name == 'umap':
            self.files = [f for f in self.embed_dir.iterdir() 
                          if f.suffix == '.json']

class PrepareModel():
    def get_callable_aves_model(self, pooling='mean', **kwargs):
        from .get_aves_model import AvesTorchaudioWrapper
        return AvesTorchaudioWrapper(pooling=pooling)
    
    def get_callable_perch_model(self, **kwargs):
        from ml_collections import config_dict
        from ievad.perch_chirp.chirp.inference import colab_utils
        from ievad.perch_chirp.chirp.inference import embed_lib
        from ievad.perch_chirp.chirp.projects.zoo import models
        model_choice = 'perch_8'
        config = config_dict.ConfigDict()
        config.embed_fn_config = config_dict.ConfigDict()
        config.embed_fn_config.model_config = config_dict.ConfigDict()
        model_key, embedding_dim, model_config = models.get_preset_model_config(
            model_choice)
        config.embed_fn_config.model_key = model_key
        config.embed_fn_config.model_config = model_config

        # Only write embeddings to reduce size.
        config.embed_fn_config.write_embeddings = True
        config.embed_fn_config.write_logits = False
        config.embed_fn_config.write_separated_audio = False
        config.embed_fn_config.write_raw_audio = False
        config.embed_fn_config.file_id_depth = 1
        embed_fn = embed_lib.EmbedFn(**config.embed_fn_config)
        embed_fn.setup()
        model = embed_fn.embedding_model.embed
        return lambda x: model(x).embeddings.squeeze()
    
    def get_callable_animal2vec_model(self, **kwargs):
        from ievad.animal2vec_nn.nn import chunk_and_normalize
        from fairseq import checkpoint_utils
        import torch
        path_to_pt_file = "backend/ievad/models/animal2vec/animal2vec_large_finetuned_MeerKAT_240507.pt"
        models, _ = checkpoint_utils.load_model_ensemble([path_to_pt_file])
        model = models[0].to("cpu")
        model.eval()
        
        @torch.inference_mode()
        def a2v_infer(samples):
            all_embeds = []
            for batch in tqdm(samples[0]):
                res = model(source=batch.view(1, -1))
                embeds = res['layer_results']
                np_embeds = [a.detach().numpy() for a in embeds]
                all_embeds.append(np_embeds)
            return np.array(all_embeds)
        
        return a2v_infer

    def get_callable_biolingual_model(self, **kwargs):
        from transformers import pipeline
        audio_classifier = pipeline(
            task="zero-shot-audio-classification", 
            model="davidrrobinson/BioLingual"
            )
        embedder = audio_classifier.model.get_audio_features
        return lambda x: embedder(x).detach().numpy()
    
    def get_callable_birdaves_model(self, pooling='mean', **kwargs):
        return self.get_callable_aves_model(pooling=pooling, **kwargs)
    
    def get_callable_umap_model(self, **kwargs):
        import umap
        return umap.UMAP(n_neighbors=self.config['n_neighbors'],
                         n_components=self.config['n_components'],
                         min_dist=self.config['min_dist'],
                         metric=self.config['metric'],
                         random_state=self.config['random_state']
                         ).fit_transform


    def get_callable_vggish_model(self, **kwargs):
        import tensorflow_hub as hub
        return hub.load('backend/ievad/models/vggish')
    
    def get_callable_birdnet_model(self, **kwargs):
        import tensorflow as tf
        model = tf.keras.models.load_model('backend/ievad/models/birdnet', compile=False)
        return tf.keras.Sequential(model.embeddings_model)
        # return lambda x: model.embeddings(x)['embeddings']
        
    def get_callable_hbdet_model(self, **kwargs):
        import tensorflow as tf
        from tensorflow_addons import metrics
        from ievad.models.hbdet import front_end
        orig_model = tf.keras.models.load_model('backend/ievad/models/hbdet',
                custom_objects={"FBetaScote": metrics.FBetaScore},
        )
        model_list = orig_model.layers[:-2]
        model_list.insert(0, tf.keras.layers.Input([7755]))
        model_list.insert(
            1, tf.keras.layers.Lambda(lambda t: tf.expand_dims(t, -1))
        )
        model_list.insert(2, front_end.MelSpectrogram())
        model = tf.keras.Sequential(
            layers=[layer for layer in model_list]
        )
        return model.predict

class PreProcessing():    
    def get_perch_preprocessing(self, audio):
        sr, win_len_s = 32000, 5.0
        win_len_samples = int(sr*win_len_s)
        re_audio = lb.resample(audio, 
                               orig_sr = self.config['sr'], 
                               target_sr = sr)
        num = np.ceil(len(re_audio) / (sr*win_len_s))
        # zero pad in case the end is reached
        re_audio = [*re_audio, *np.zeros([int(num * win_len_samples - len(re_audio))])]
        wins = np.array(re_audio).reshape([int(num), win_len_samples])

        return np.array(wins, dtype=np.float32)
    
    def get_vggish_preprocessing(self, audio):
        return (audio * 32767).astype(np.int16)
    
    def get_biolingual_preprocessing(self, audio):
        import torch
        from transformers import pipeline
        sr = 48000
        input_tensor_shape = [1, 1, 1001, 64]
        re_audio = lb.resample(audio, 
                        orig_sr = self.config['sr'], 
                        target_sr = sr)
        audio_classifier = pipeline(
            task="zero-shot-audio-classification", 
            model="davidrrobinson/BioLingual"
            )
        features = audio_classifier.feature_extractor(re_audio,
                                                      sampling_rate = sr)
        aud_features = features['input_features'][0]
        aud_input = torch.tensor(aud_features.reshape(input_tensor_shape))
        return aud_input
        
    
    def get_hbdet_preprocessing(self, audio):
        import tensorflow as tf
        num = np.ceil(len(audio) / 7755)
        # zero pad in case the end is reached
        audio = [*audio, *np.zeros([int(num * 7755 - len(audio))])]
        wins = np.array(audio).reshape([int(num), 7755])

        return tf.convert_to_tensor(wins)

    def get_birdnet_preprocessing(self, audio):
        import tensorflow as tf
        re_audio = lb.resample(audio, 
                               orig_sr = self.config['sr'], 
                               target_sr = 48000)
        num = np.ceil(len(re_audio) / 144000)
        # zero pad in case the end is reached
        re_audio = [*re_audio, *np.zeros([int(num * 144000 - len(re_audio))])]
        wins = np.array(re_audio).reshape([int(num), 144000])

        return tf.convert_to_tensor(wins, dtype=tf.float32)
    
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
    
    def get_birdaves_preprocessing(self, audio):        
        return self.get_aves_preprocessing(audio)
    
    def get_animal2vec_preprocessing(self, audio):
        from ievad.animal2vec_nn.nn import chunk_and_normalize
        import torch
        chunk = chunk_and_normalize(
            torch.tensor(audio),
            segment_length=10,
            sample_rate=8000,
            normalize=True,
            max_batch_size=16
        )
        # if not torch.is_tensor(chunk):
        #     chunk = torch.stack(chunk)  # stack the list of tensors
        # elif chunk.dim() == 1:  # split segments or single segment
        #     chunk = chunk.view(1, -1)
        return chunk

class Embedder(PrepareModel, PreProcessing):
    def __init__(self, model_name, **kwargs):
        import yaml
        with open('backend/ievad/config.yaml', 'rb') as f:
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

    def save_embeddings(self, file_idx, fileloader_obj, file, embeds):
        file_dest = fileloader_obj.embed_dir.joinpath(file.stem 
                                                      + '_'
                                                      + self.model_name)
        if file.suffix == '.npy':
            file_dest = str(file_dest) + '.json'
            input_len = 3
            save_embeddings_dict_with_timestamps(file_dest, embeds, input_len, 
                                                 fileloader_obj, file_idx)
        else:
            file_dest = str(file_dest) + '.npy'
            np.save(file_dest, embeds)
        
def save_embeddings_dict_with_timestamps(file_dest, embeds, input_len, 
                                         loader_obj, f_idx):
    length = embeds.shape[0]
    lin_array = np.arange(0, length*input_len, input_len)
    d = {var: embeds[:, i].tolist() 
         for i, var in zip(range(embeds.shape[1]), ['x', 'y'])}
    d['timestamp'] = lin_array.tolist()
    
    d['metadata'] = {k: (v[f_idx] if isinstance(v, list) else v) 
                     for (k, v) in loader_obj.metadata_dict['files'].items()}
    d['metadata'].update({k: v for (k, v) in loader_obj.metadata_dict.items()
                          if not isinstance(v, dict)})
    
    import json
    with open(file_dest, 'w') as f:
        json.dump(d, f)
    
def generate_embeddings(**kwargs):
    ld = Loader(**kwargs)
    if not ld.combination_already_exists:    
        embed = Embedder(**kwargs)
        for idx, file in tqdm(enumerate(ld.files)):
            input = ld.load(file)
            if input is None:
                continue
            embeds = embed.get_embeddings_from_model(input)
            embed.save_embeddings(idx, ld, file, embeds)
        ld.write_metadata_file()
        ld.update_files()
    return ld
