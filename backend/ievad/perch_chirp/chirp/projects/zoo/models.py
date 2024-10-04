# coding=utf-8
# Copyright 2024 The Perch Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Implementations of inference interfaces for applying trained models."""

import dataclasses
import tempfile
from typing import Any

from absl import logging
from ievad.perch_chirp.chirp.models import frontend
from ievad.perch_chirp.chirp.models import handcrafted_features
from ievad.perch_chirp.chirp.projects.zoo import taxonomy_model_tf
from ievad.perch_chirp.chirp.projects.zoo import zoo_interface
from ievad.perch_chirp.chirp.taxonomy import namespace
from ievad.perch_chirp.chirp.taxonomy import namespace_db
from etils import epath
from ml_collections import config_dict
import numpy as np
import tensorflow as tf
import tensorflow.compat.v1 as tf1
import tensorflow_hub as hub


def model_class_map() -> dict[str, Any]:
  """Get the mapping of model keys to classes."""
  return {
      'taxonomy_model_tf': taxonomy_model_tf.TaxonomyModelTF,
      'separator_model_tf': SeparatorModelTF,
      'birb_separator_model_tf1': BirbSepModelTF1,
      'birdnet': BirdNet,
      'placeholder_model': PlaceholderModel,
      'separate_embed_model': SeparateEmbedModel,
      'tfhub_model': TFHubModel,
      'google_whale': GoogleWhaleModel,
  }


def get_preset_model_config(preset_name):
  """Get a config_dict for a known model."""
  model_config = config_dict.ConfigDict()

  if preset_name == 'perch_8':
    model_key = 'taxonomy_model_tf'
    embedding_dim = 1280
    model_config.window_size_s = 5.0
    model_config.hop_size_s = 5.0
    model_config.sample_rate = 32000
    model_config.tfhub_version = 8
    model_config.model_path = ''
  elif preset_name == 'humpback':
    model_key = 'google_whale'
    embedding_dim = 2048
    model_config.window_size_s = 3.9124
    model_config.sample_rate = 10000
    model_config.model_url = 'https://tfhub.dev/google/humpback_whale/1'
    model_config.peak_norm = 0.02
  elif preset_name == 'multispecies_whale':
    model_key = 'google_whale'
    embedding_dim = 1280
    model_config.window_size_s = 5.0  # Is this correct?
    model_config.sample_rate = 24000
    model_config.model_url = 'https://www.kaggle.com/models/google/multispecies-whale/TensorFlow2/default/2'
    model_config.peak_norm = -1.0
  elif preset_name == 'surfperch':
    model_key = 'taxonomy_model_tf'
    embedding_dim = 1280
    model_config.window_size_s = 5.0
    model_config.hop_size_s = 5.0
    model_config.sample_rate = 32000
    model_config.tfhub_version = 1
    model_config.tfhub_path = taxonomy_model_tf.SURFPERCH_TF_HUB_URL
    model_config.model_path = ''
  elif preset_name.startswith('birdnet'):
    model_key = 'birdnet'
    birdnet_version = preset_name.split('_')[-1]
    if birdnet_version not in ('V2.1', 'V2.2', 'V2.3'):
      raise ValueError(f'Birdnet version not supported: {birdnet_version}')
    base_path = 'gs://chirp-public-bucket/models/birdnet'
    if birdnet_version == 'V2.1':
      embedding_dim = 420
      model_path = 'V2.1/BirdNET_GLOBAL_2K_V2.1_Model_FP16.tflite'
    elif birdnet_version == 'V2.2':
      embedding_dim = 320
      model_path = 'V2.2/BirdNET_GLOBAL_3K_V2.2_Model_FP16.tflite'
    elif birdnet_version == 'V2.3':
      embedding_dim = 1024
      model_path = 'V2.3/BirdNET_GLOBAL_3K_V2.3_Model_FP16.tflite'
    else:
      # TODO(tomdenton): Support V2.4.
      raise ValueError(f'Birdnet version not supported: {birdnet_version}')
    model_config.window_size_s = 3.0
    model_config.hop_size_s = 3.0
    model_config.sample_rate = 48000
    model_config.model_path = f'{base_path}/{model_path}'
    # Note: The v2_1 class list is appropriate for Birdnet 2.1, 2.2, and 2.3.
    model_config.class_list_name = 'birdnet_v2_1'
    model_config.num_tflite_threads = 4
  else:
    raise ValueError('Unsupported model preset: %s' % preset_name)
  return model_key, embedding_dim, model_config


@dataclasses.dataclass
class SeparateEmbedModel(zoo_interface.EmbeddingModel):
  """Wrapper for separate separation and embedding models.

  Note: Use the separation model's sample rate. The embedding model's sample
  rate is used to resample prior to computing the embedding.

  Attributes:
    separation_model: SeparationModelTF.
    embedding_model: TaxonomyModelTF.
    embed_raw: If True, the outputs will include embeddings of the original
      audio in addition to embeddings for the separated channels. The embeddings
      will have shape [T, C+1, D], with the raw audio embedding on channel 0.
  """

  separator_model_tf_config: config_dict.ConfigDict
  taxonomy_model_tf_config: config_dict.ConfigDict
  separation_model: 'SeparatorModelTF'
  embedding_model: taxonomy_model_tf.TaxonomyModelTF
  embed_raw: bool = True

  @classmethod
  def from_config(cls, config: config_dict.ConfigDict) -> 'SeparateEmbedModel':
    separation_model = SeparatorModelTF.from_config(
        config.separator_model_tf_config
    )
    embedding_model = taxonomy_model_tf.TaxonomyModelTF.from_config(
        config.taxonomy_model_tf_config
    )
    return cls(
        separation_model=separation_model,
        embedding_model=embedding_model,
        **config,
    )

  def __post_init__(self):
    if self.separation_model.sample_rate != self.embedding_model.sample_rate:
      raise ValueError(
          'Separation and embedding models must have matching rates.'
      )

  def embed(self, audio_array: np.ndarray) -> zoo_interface.InferenceOutputs:
    # Frame the audio according to the embedding model's config.
    # We then apply separation to each frame independently, and embed
    # the separated audio.
    framed_audio = self.frame_audio(
        audio_array,
        self.embedding_model.window_size_s,
        self.embedding_model.hop_size_s,
    )
    # framed_audio has shape [Frames, Time]
    separation_outputs = self.separation_model.batch_embed(framed_audio)
    # separated_audio has shape [F, C, T]
    separated_audio = separation_outputs.separated_audio
    if separated_audio is None:
      raise RuntimeError('Separation model returned None for separated audio.')

    if self.embed_raw:
      separated_audio = np.concatenate(
          [
              framed_audio[:, np.newaxis, : separated_audio.shape[-1]],
              separated_audio,
          ],
          axis=1,
      )
    num_frames = separated_audio.shape[0]
    num_channels = separated_audio.shape[1]
    num_samples = separated_audio.shape[2]
    separated_audio = np.reshape(separated_audio, [-1, num_samples])

    embedding_outputs = self.embedding_model.batch_embed(separated_audio)

    if embedding_outputs.embeddings is not None:
      # Batch embeddings have shape [Batch, Time, Channels, Features]
      # Time is 1 because we have framed using the embedding model's
      # window_size. The batch size is num_frames * num_channels.
      embedding_outputs.embeddings = np.reshape(
          embedding_outputs.embeddings, [num_frames, num_channels, -1]
      )

    # Take the maximum logits over the channels dimension.
    if embedding_outputs.logits is not None:
      max_logits = {}
      for k, v in embedding_outputs.logits.items():
        v = v.reshape([num_frames, num_channels, -1])
        max_logits[k] = np.max(v, axis=1)
    else:
      max_logits = None

    return zoo_interface.InferenceOutputs(
        embeddings=embedding_outputs.embeddings,
        logits=max_logits,
        # Because the separated audio is framed, it does not match the
        # outputs interface, so we do not return it.
        separated_audio=None,
    )


@dataclasses.dataclass
class BirbSepModelTF1(zoo_interface.EmbeddingModel):
  """Separation model from the Bird MixIT paper.

  Example usage:
  ```
  from ievad.perch_chirp.chirp.projects.zoo import models
  birbsep1_config = config_dict.ConfigDict({
    'model_path': $MODEL_PATH,
    'window_size_s': 60.0,
    'keep_raw_channel': False,
    'sample_rate': 22050,
  })
  birbsep1 = models.BirbSepModelTF1.from_config(birbsep1_config)
  outputs = birbsep1.embed($SOME_AUDIO)
  ```
  """

  model_path: str
  window_size_s: float
  keep_raw_channel: bool
  session: Any
  input_placeholder_ns: Any
  output_tensor_ns: Any

  @classmethod
  def _find_checkpoint(cls, model_path: str) -> str:
    # Publicly released model does not have a checkpoints directory file.
    ckpt = None
    for ckpt in sorted(
        tuple(epath.Path(model_path).glob('model.ckpt-*.index'))
    ):
      ckpt = ckpt.as_posix()[: -len('.index')]
    if ckpt is None:
      raise FileNotFoundError('Could not find checkpoint file.')
    return ckpt

  @classmethod
  def from_config(cls, config: config_dict.ConfigDict) -> 'BirbSepModelTF1':
    """Load model files and create TF1 session graph."""
    metagraph_path_ns = epath.Path(config.model_path) / 'inference.meta'
    checkpoint_path = cls._find_checkpoint(config.model_path)
    graph_ns = tf.Graph()
    sess_ns = tf1.Session(graph=graph_ns)
    with graph_ns.as_default():
      new_saver = tf1.train.import_meta_graph(metagraph_path_ns)
      new_saver.restore(sess_ns, checkpoint_path)
      input_placeholder_ns = graph_ns.get_tensor_by_name(
          'input_audio/receiver_audio:0'
      )
      output_tensor_ns = graph_ns.get_tensor_by_name('denoised_waveforms:0')
    session = sess_ns
    return cls(
        session=session,
        input_placeholder_ns=input_placeholder_ns,
        output_tensor_ns=output_tensor_ns,
        **config,
    )

  def embed(self, audio_array: Any) -> zoo_interface.InferenceOutputs:
    start_sample = 0
    window_size = int(self.window_size_s * self.sample_rate)
    sep_chunks = []
    raw_chunks = []
    while start_sample <= audio_array.shape[0]:
      audio_chunk = audio_array[start_sample : start_sample + window_size]
      raw_chunks.append(audio_chunk)
      separated_audio = self.session.run(
          self.output_tensor_ns,
          feed_dict={
              self.input_placeholder_ns: audio_chunk[np.newaxis, np.newaxis, :]
          },
      )
      # Drop the extraneous batch dimension.
      separated_audio = np.squeeze(separated_audio, axis=0)
      sep_chunks.append(separated_audio)
      start_sample += window_size

    raw_chunks = np.concatenate(raw_chunks, axis=0)
    sep_chunks = np.concatenate(sep_chunks, axis=-1)
    if self.keep_raw_channel:
      sep_chunks = np.concatenate(
          [sep_chunks, raw_chunks[np.newaxis, :]], axis=0
      )
    return zoo_interface.InferenceOutputs(separated_audio=sep_chunks)

  def batch_embed(
      self, audio_batch: np.ndarray
  ) -> zoo_interface.InferenceOutputs:
    return zoo_interface.batch_embed_from_embed_fn(self.embed, audio_batch)


@dataclasses.dataclass
class SeparatorModelTF(zoo_interface.EmbeddingModel):
  """Separator SavedModel.

  Attributes:
    model_path: Path to model files.
    frame_size: Audio frame size for separation model.
    model: Loaded TF SavedModel.
    class_list: Loaded class_list for the model's output logits.
    windows_size_s: Window size for framing audio in samples. The audio will be
      chunked into frames of size window_size_s, which may help avoid memory
      blowouts. If None, will simply treat all audio as a single frame.
  """

  model_path: str
  frame_size: int
  model: Any
  class_list: namespace.ClassList
  window_size_s: float | None = None

  @classmethod
  def from_config(cls, config: config_dict.ConfigDict) -> 'SeparatorModelTF':
    logging.info('Loading taxonomy separator model...')
    model = tf.saved_model.load(epath.Path(config.model_path) / 'savedmodel')
    label_csv_path = epath.Path(config.model_path) / 'label.csv'
    with label_csv_path.open('r') as f:
      class_list = namespace.ClassList.from_csv(f)
    return cls(model=model, class_list=class_list, **config)

  def embed(self, audio_array: np.ndarray) -> zoo_interface.InferenceOutputs:
    # Drop samples to allow reshaping to frame_size
    excess_samples = audio_array.shape[0] % self.frame_size
    if excess_samples > 0:
      audio_array = audio_array[:-excess_samples]
    framed_audio = self.frame_audio(
        audio_array, self.window_size_s, self.window_size_s
    )
    framed_audio = np.reshape(
        framed_audio,
        [
            framed_audio.shape[0],
            framed_audio.shape[1] // self.frame_size,
            self.frame_size,
        ],
    )

    sep_audio, all_logits, all_embeddings = self.model.infer_tf(
        framed_audio[:1]
    )
    for window in framed_audio[1:]:
      separated, logits, embeddings = self.model.infer_tf(window[np.newaxis, :])
      sep_audio = np.concatenate([sep_audio, separated], axis=0)
      all_logits = np.concatenate([all_logits, logits], axis=0)
      all_embeddings = np.concatenate([all_embeddings, embeddings], axis=0)
    all_embeddings = all_embeddings[:, np.newaxis, :]

    # Recombine batch and time dimensions.
    sep_audio = np.reshape(sep_audio, [-1, sep_audio.shape[-1]])
    all_logits = np.reshape(all_logits, [-1, all_logits.shape[-1]])
    all_embeddings = np.reshape(all_embeddings, [-1, all_embeddings.shape[-1]])
    return zoo_interface.InferenceOutputs(
        all_embeddings, {'label': all_logits}, sep_audio
    )

  def batch_embed(
      self, audio_batch: np.ndarray
  ) -> zoo_interface.InferenceOutputs:
    return zoo_interface.batch_embed_from_embed_fn(self.embed, audio_batch)


@dataclasses.dataclass
class BirdNet(zoo_interface.EmbeddingModel):
  """Wrapper for BirdNet models.

  Attributes:
    model_path: Path to the saved model checkpoint or TFLite file.
    model: The TF SavedModel or TFLite interpreter.
    tflite: Whether the model is a TFLite model.
    class_list: The loaded class list.
    window_size_s: Window size for framing audio in samples.
    hop_size_s: Hop size for inference.
    num_tflite_threads: Number of threads to use with TFLite model.
    class_list_name: Name of the BirdNet class list.
  """

  model_path: str
  model: Any
  tflite: bool
  class_list: namespace.ClassList
  window_size_s: float = 3.0
  hop_size_s: float = 3.0
  num_tflite_threads: int = 16
  class_list_name: str = 'birdnet_v2_1'

  @classmethod
  def from_config(cls, config: config_dict.ConfigDict) -> 'BirdNet':
    logging.info('Loading BirdNet model...')
    if config.model_path.endswith('.tflite'):
      tflite = True
      with tempfile.NamedTemporaryFile() as tmpf:
        model_file = epath.Path(config.model_path)
        model_file.copy(tmpf.name, overwrite=True)
        model = tf.lite.Interpreter(
            tmpf.name, num_threads=config.num_tflite_threads
        )
      model.allocate_tensors()
    else:
      tflite = False
      model = tf.saved_model.load(config.model_path)
    db = namespace_db.load_db()
    class_list = db.class_lists[config.class_list_name]
    return cls(
        model=model,
        tflite=tflite,
        class_list=class_list,
        **config,
    )

  def embed_saved_model(
      self, audio_array: np.ndarray
  ) -> zoo_interface.InferenceOutputs:
    """Get logits using the BirdNet SavedModel."""
    # Note that there is no easy way to get the embedding from the SavedModel.
    all_logits = self.model(audio_array[:1])
    for window in audio_array[1:]:
      logits = self.model(window[np.newaxis, :])
      all_logits = np.concatenate([all_logits, logits], axis=0)
    return zoo_interface.InferenceOutputs(
        None, {self.class_list_name: all_logits}, None
    )

  def embed_tflite(
      self, audio_array: np.ndarray
  ) -> zoo_interface.InferenceOutputs:
    """Create an embedding and logits using the BirdNet TFLite model."""
    input_details = self.model.get_input_details()[0]
    output_details = self.model.get_output_details()[0]
    embedding_idx = output_details['index'] - 1
    embeddings = []
    logits = []
    for audio in audio_array:
      self.model.set_tensor(
          input_details['index'], np.float32(audio)[np.newaxis, :]
      )
      self.model.invoke()
      logits.append(self.model.get_tensor(output_details['index']))
      embeddings.append(self.model.get_tensor(embedding_idx))
    # Create [Batch, 1, Features]
    embeddings = np.array(embeddings)
    logits = np.array(logits)
    return zoo_interface.InferenceOutputs(
        embeddings, {self.class_list_name: logits}, None
    )

  def embed(self, audio_array: np.ndarray) -> zoo_interface.InferenceOutputs:
    framed_audio = self.frame_audio(
        audio_array, self.window_size_s, self.hop_size_s
    )
    if self.tflite:
      return self.embed_tflite(framed_audio)
    else:
      return self.embed_saved_model(framed_audio)

  def batch_embed(
      self, audio_batch: np.ndarray
  ) -> zoo_interface.InferenceOutputs:
    return zoo_interface.batch_embed_from_embed_fn(self.embed, audio_batch)


@dataclasses.dataclass
class HandcraftedFeaturesModel(zoo_interface.EmbeddingModel):
  """Wrapper for simple feature extraction."""

  window_size_s: float
  hop_size_s: float
  melspec_config: config_dict.ConfigDict
  melspec_layer: frontend.Frontend
  features_config: config_dict.ConfigDict
  features_layer: handcrafted_features.HandcraftedFeatures

  @classmethod
  def from_config(
      cls, config: config_dict.ConfigDict
  ) -> 'HandcraftedFeaturesModel':
    melspec_layer = frontend.MelSpectrogram(**config.melspec_config)
    features_layer = handcrafted_features.HandcraftedFeatures(
        **config.features_config
    )
    return cls(
        melspec_layer=melspec_layer,
        features_layer=features_layer,
        **config,
    )

  @classmethod
  def beans_baseline(cls, sample_rate=32000, frame_rate=100):
    stride = sample_rate // frame_rate
    mel_config = config_dict.ConfigDict({
        'sample_rate': sample_rate,
        'features': 160,
        'stride': stride,
        'kernel_size': 2 * stride,
        'freq_range': (60.0, sample_rate / 2.0),
        'scaling_config': frontend.LogScalingConfig(),
    })
    features_config = config_dict.ConfigDict({
        'compute_mfccs': True,
        'aggregation': 'beans',
    })
    config = config_dict.ConfigDict({
        'sample_rate': sample_rate,
        'melspec_config': mel_config,
        'features_config': features_config,
        'window_size_s': 1.0,
        'hop_size_s': 1.0,
    })
    # pylint: disable=unexpected-keyword-arg
    return HandcraftedFeaturesModel.from_config(config)

  def embed(self, audio_array: np.ndarray) -> zoo_interface.InferenceOutputs:
    framed_audio = self.frame_audio(
        audio_array, self.window_size_s, self.hop_size_s
    )
    melspec = self.melspec_layer.apply({}, framed_audio)
    features = self.features_layer.apply(
        {}, melspec[:, :, :, np.newaxis], train=False
    )
    # Add a trivial channels dimension.
    features = features[:, np.newaxis, :]
    return zoo_interface.InferenceOutputs(features, None, None)

  def batch_embed(
      self, audio_batch: np.ndarray
  ) -> zoo_interface.InferenceOutputs:
    return zoo_interface.batch_embed_from_embed_fn(self.embed, audio_batch)


@dataclasses.dataclass
class GoogleWhaleModel(zoo_interface.EmbeddingModel):
  """Wrapper for Google Humpback model.

  Attributes:
    model_url: Path to model files.
    window_size_s: Window size for framing audio in seconds. 3.915 seems to work
      well for the Humpback model.
    hop_size_s: Hop size for inference.
    peak_norm: Peak normalizaiton target. Ignore if <= 0.
    model: Loaded TF SavedModel.
  """

  model_url: str
  window_size_s: float
  peak_norm: float
  class_list: namespace.ClassList
  model: Any  # TF SavedModel

  @classmethod
  def load_humpback_model(
      cls,
      model_url: str = 'https://tfhub.dev/google/humpback_whale/1',
      **kwargs,
  ) -> 'GoogleWhaleModel':
    model = hub.load(model_url)
    class_list = namespace.ClassList('humpback', ('humpback',))
    sample_rate = model.metadata()['input_sample_rate'].numpy()
    window_size_s = model.metadata()['context_width_samples'] / sample_rate
    return cls(
        model=model,
        sample_rate=sample_rate,
        window_size_s=window_size_s,
        model_url=model_url,
        peak_norm=0.02,
        class_list=class_list,
        **kwargs,
    )

  @classmethod
  def from_config(cls, config: config_dict.ConfigDict) -> 'GoogleWhaleModel':
    model = hub.load(config.model_url)
    class_names = tuple(
        [str(c.numpy(), 'utf8') for c in model.metadata()['class_names']]
    )
    class_list = namespace.ClassList('multispecies_whale', class_names)
    return cls(model=model, class_list=class_list, **config)

  @property
  def hop_size_s(self) -> float:
    return self.window_size_s

  def batch_embed(
      self, audio_batch: np.ndarray
  ) -> zoo_interface.InferenceOutputs:
    # Renormalize to 0.02 peak.
    if self.peak_norm > 0:
      audio_batch = self.normalize_audio(audio_batch, self.peak_norm)
    spectrogram = self.model.front_end(audio_batch[:, :, np.newaxis])
    framed_spec = tf.signal.frame(
        spectrogram, frame_length=128, frame_step=128, pad_end=True, axis=1
    )
    rebatched_spec = tf.reshape(framed_spec, [-1, 128, framed_spec.shape[-1]])
    embeddings = self.model.features(rebatched_spec)

    # Recover batch and time dimensions, and add a channel dimension.
    embeddings = tf.reshape(
        embeddings,
        [framed_spec.shape[0], framed_spec.shape[1], 1, embeddings.shape[-1]],
    )
    logits = self.model.logits(rebatched_spec)
    logits = tf.reshape(
        logits, [framed_spec.shape[0], framed_spec.shape[1], logits.shape[-1]]
    )
    logits = {self.class_list.namespace: logits}
    outputs = zoo_interface.InferenceOutputs(
        embeddings=embeddings,
        logits=logits,
        frontend=spectrogram,
        batched=True,
    )
    return outputs

  def embed(self, audio_array: np.ndarray) -> zoo_interface.InferenceOutputs:
    return zoo_interface.embed_from_batch_embed_fn(
        self.batch_embed, audio_array
    )


@dataclasses.dataclass
class TFHubModel(zoo_interface.EmbeddingModel):
  """Generic wrapper for TFHub models which produce embeddings."""

  model: Any  # TFHub loaded model.
  model_url: str
  embedding_index: int
  logits_index: int = -1

  @classmethod
  def from_config(cls, config: config_dict.ConfigDict) -> 'TFHubModel':
    model = hub.load(config.model_url)
    return cls(
        model=model,
        **config,
    )

  @classmethod
  def yamnet(cls):
    # Parent class takes a sample_rate arg which pylint doesn't find.
    config = config_dict.ConfigDict({
        'sample_rate': 16000,
        'model_url': 'https://tfhub.dev/google/yamnet/1',
        'embedding_index': 1,
        'logits_index': 0,
    })
    return TFHubModel.from_config(config)

  @classmethod
  def vggish(cls):
    config = config_dict.ConfigDict({
        'sample_rate': 16000,
        'model_url': 'https://tfhub.dev/google/vggish/1',
        'embedding_index': -1,
        'logits_index': -1,
    })
    return TFHubModel.from_config(config)

  def embed(
      self, audio_array: np.ndarray[Any, np.dtype[Any]]
  ) -> zoo_interface.InferenceOutputs:
    outputs = self.model(audio_array)
    if self.embedding_index < 0:
      embeddings = outputs
    else:
      embeddings = outputs[self.embedding_index]
    if len(embeddings.shape) == 1:
      embeddings = embeddings[np.newaxis, :]
    elif len(embeddings.shape) != 2:
      raise ValueError('Embeddings should have shape [Depth] or [Time, Depth].')

    if self.logits_index >= 0:
      logits = {'label': outputs[self.logits_index]}
    else:
      logits = None
    embeddings = embeddings[:, np.newaxis, :]
    return zoo_interface.InferenceOutputs(embeddings, logits, None, False)

  def batch_embed(
      self, audio_batch: np.ndarray
  ) -> zoo_interface.InferenceOutputs:
    return zoo_interface.batch_embed_from_embed_fn(self.embed, audio_batch)


@dataclasses.dataclass
class PlaceholderModel(zoo_interface.EmbeddingModel):
  """Test implementation of the EmbeddingModel zoo_interface."""

  embedding_size: int = 128
  make_embeddings: bool = True
  make_logits: bool = True
  make_separated_audio: bool = True
  make_frontend: bool = True
  do_frame_audio: bool = False
  window_size_s: float = 1.0
  hop_size_s: float = 1.0
  frontend_size: tuple[int, int] = (32, 32)

  @classmethod
  def from_config(cls, config: config_dict.ConfigDict) -> 'PlaceholderModel':
    return cls(**config)

  def __post_init__(self):
    db = namespace_db.load_db()
    self.class_list = db.class_lists['caples']

  def embed(self, audio_array: np.ndarray) -> zoo_interface.InferenceOutputs:
    outputs = {}
    if self.do_frame_audio:
      audio_array = self.frame_audio(
          audio_array, self.window_size_s, self.hop_size_s
      )
    time_size = audio_array.shape[0] // int(
        self.window_size_s * self.sample_rate
    )
    if self.make_embeddings:
      outputs['embeddings'] = np.zeros(
          [time_size, 1, self.embedding_size], np.float32
      )
    if self.make_frontend:
      outputs['frontend'] = np.zeros(
          [time_size, self.frontend_size[0], self.frontend_size[1]], np.float32
      )
    if self.make_logits:
      outputs['logits'] = {
          'label': np.zeros(
              [time_size, len(self.class_list.classes)], np.float32
          ),
          'other_label': np.ones(
              [time_size, len(self.class_list.classes)], np.float32
          ),
      }
    if self.make_separated_audio:
      outputs['separated_audio'] = np.zeros(
          [2, audio_array.shape[-1]], np.float32
      )
    return zoo_interface.InferenceOutputs(**outputs)

  def batch_embed(
      self, audio_batch: np.ndarray
  ) -> zoo_interface.InferenceOutputs:
    return zoo_interface.batch_embed_from_embed_fn(self.embed, audio_batch)
