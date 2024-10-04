
import yaml
import numpy as np
import pandas as pd
import librosa as lb
import re
import datetime as dt
import sounddevice as sd
from pathlib import Path

from ievad.helpers import (get_datetime_from_filename, 
                           CORRECTED_CONTEXT_WIN_TIME)


with open('backend/ievad/config.yaml', 'rb') as f:
    config = yaml.safe_load(f)
    
LOAD_PATH = Path(config['audio_dir']).joinpath(
            Path(config['preproc']['annots_path']).stem
            )   
if not LOAD_PATH.exists():
    LOAD_PATH = LOAD_PATH.parent

def dummy_image():
    return np.array([[[255, 0, 0], [0, 255, 0], [0, 0, 255]],
                [[0, 255, 0], [0, 0, 255], [255, 0, 0]]
                ], dtype=np.uint8)
    
def align_df_and_embeddings(files, meta_df):
    tup = zip(pd.to_datetime(meta_df.file_datetime).values, 
                   meta_df.site.values)
    
    meta_df.index = pd.MultiIndex.from_tuples(tup, names=['datetime', 'site'])

    datetimes = map(get_datetime_from_filename, files)
    sites = map(get_site_from_filename, files)

    return meta_df.loc[zip(datetimes, sites)].drop_duplicates()

def get_site_from_filename(file_path):
    return Path(file_path).stem.split('_')[-3]

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
    if not isinstance(t_s, float):
        t_s = time_string_to_float(t_s)
    
    audio, sr = lb.load(main_path.joinpath(file_stem), 
                        offset=t_s, 
                        sr=config['preproc']['plot_spec_sr'], 
                        duration = CORRECTED_CONTEXT_WIN_TIME)
    return audio, sr, file_stem
 
def set_axis_lims_dep_sr(S_dB):
    if config['preproc']['downsample']:
        f_max = config['preproc']['downsample_sr'] / 2
        reduce = config['preproc']['model_sr'] / (f_max * 2)
        S_dB = S_dB[:int(S_dB.shape[0] / reduce), :]
    else:
        f_max = config['preproc']['model_sr'] / 2 
    return f_max, S_dB

