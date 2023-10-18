# Copyright 2017 The TensorFlow Authors All Rights Reserved.
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
# ==============================================================================

import yaml
import numpy as np
with open('ievad/config.yaml', 'r') as f:
    config = yaml.safe_load(f)['preproc']
"""Global parameters for the VGGish model.

See vggish_slim.py for more information.
"""

# Architectural constants.
# NUM_FRAMES = 96  # Frames in input mel-spectrogram patch.
NUM_FRAMES = 96  # Frames in input mel-spectrogram patch.
NUM_BANDS = 64  # Frequency bands in input mel-spectrogram patch.
EMBEDDING_SIZE = 128  # Size of embedding layer.

# Hyperparameters used in feature and example generation.
SAMPLE_RATE = config['model_sr']
STFT_WINDOW_LENGTH_SECONDS = 1024/config['model_sr']
NUM_MEL_BINS = NUM_BANDS
MEL_MIN_HZ = config['model_sr']/40
MEL_MAX_HZ = config['model_sr']/2
LOG_OFFSET = 0.01  # Offset used for stabilized log of input mel-spectrogram.
time_in_samps = int(config['model_sr']*config['model_time_length'])
corrected_context_win_time = ((time_in_samps-1024) 
                              -np.mod((time_in_samps-1024), 95)
                              +1024)/SAMPLE_RATE
EXAMPLE_WINDOW_SECONDS = corrected_context_win_time
EXAMPLE_HOP_SECONDS = corrected_context_win_time
STFT_HOP_LENGTH_SECONDS = corrected_context_win_time/NUM_FRAMES

# SAMPLE_RATE = 16000
# SAMPLE_RATE = 2000 
# SAMPLE_RATE = 192000
# STFT_WINDOW_LENGTH_SECONDS = 0.025
# STFT_WINDOW_LENGTH_SECONDS = 0.512 # = 1024/SAMPLE_RATE
# MEL_MIN_HZ = 125
# MEL_MIN_HZ = 50 # = SAMPLE_RATE/40
# MEL_MAX_HZ = 7500
# MEL_MAX_HZ = 1000 # = SAMPLE_RATE/2
# MEL_MAX_HZ = 90000
# EXAMPLE_WINDOW_SECONDS = 0.96  # Each example contains 96 10ms frames
# EXAMPLE_WINDOW_SECONDS = 3.8845  # Each example contains 96 10ms frames
# EXAMPLE_HOP_SECONDS = 0.96     # with zero overlap.
# EXAMPLE_HOP_SECONDS = 3.8845     # with zero overlap.
# STFT_HOP_LENGTH_SECONDS = 0.010
# STFT_HOP_LENGTH_SECONDS = 0.0404 # = EXAMPLE_WINDOW_SECONDS/NUM_FRAMES

# Parameters used for embedding postprocessing.
PCA_EIGEN_VECTORS_NAME = 'pca_eigen_vectors'
PCA_MEANS_NAME = 'pca_means'
QUANTIZE_MIN_VAL = -2.0
QUANTIZE_MAX_VAL = +2.0

# Hyperparameters used in training.
INIT_STDDEV = 0.01  # Standard deviation used to initialize weights.
LEARNING_RATE = 1e-4  # Learning rate for the Adam optimizer.
ADAM_EPSILON = 1e-8  # Epsilon for the Adam optimizer.

# Names of ops, tensors, and features.
INPUT_OP_NAME = 'vggish/input_features'
INPUT_TENSOR_NAME = INPUT_OP_NAME + ':0'
OUTPUT_OP_NAME = 'vggish/embedding'
OUTPUT_TENSOR_NAME = OUTPUT_OP_NAME + ':0'
AUDIO_EMBEDDING_FEATURE_NAME = 'audio_embedding'
