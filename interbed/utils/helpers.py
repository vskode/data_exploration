import glob
import yaml
import numpy as np
import pandas as pd
import librosa as lb
import soundfile as sf
from pathlib import Path

with open('interbed/config.yaml', 'rb') as f:
    config = yaml.safe_load(f)


def get_corresponding_sound_file(file):
    """
    Find the wav file corresponding to the current annotations file. 
    The search is conducted with a recursive search inside the 
    sound_files_path which is specified in the config.yaml file. 

    Args:
        file (string): parent directory of sound files (can be nested)

    Returns:
        string: path to corresponding sound file
    """
    hard_drive_path = config['preproc']['sound_files_path']
    
    file_path = glob.glob(
        f'{hard_drive_path}/**/{Path(file).stem.split("Table")[0]}wav',
                      recursive = True)
    
    if not file_path:
        new_file = Path(file).stem.split("Table")[0] + 'wav'
        file_tolsta = '336097327.' + new_file[6:].replace(
                                            '_000', '').replace('_', '')
        file_path = glob.glob(f'{hard_drive_path}/**/{file_tolsta}',
                    recursive = True)
        
        if not file_path :
            file_tolsta = '335564853.' + new_file[6:].replace(
                                            '5_000', '4').replace('_', '')
            file_path = glob.glob(f'{hard_drive_path}/**/{file_tolsta}',
                        recursive = True)
            if not file_path :
                return f'{Path(file).stem.split("Table")[0]}wav'
            
            
    return file_path[0]
    
def standardize_annotations(file):
    """
    Load annotation file and find corresponding sound file, which is added
    to the created DataFrame. Label is fixed to 1 as it is assumed that only
    calls are annotated. Furthermore the column names are changed to allow 
    for less code writing. 

    Args:
        file (string): path to annotation file

    Returns:
        pandas.DataFrame: annotations 
    """
    ann = pd.read_csv(file, sep = '\t')

    ann['filename'] = get_corresponding_sound_file(file)
    ann['label']    = 1
    ann = ann.rename(columns = { 'Begin Time (s)' : 'start', 
                                'End Time (s)' : 'end' })
    ann = ann.sort_values('start', ignore_index = True)
    return ann

def get_site(file):
    """
    Get Location of Site for this specific measurement. 

    Args:
        file (string): path to annotation file

    Returns:
        string: location specified in parent directory of path
    """
    site = list(Path(file).parents)[-6].stem.split('_')[0]
    if site == 'S1':
        site = 'Gravelly sand'
    return site

def load_audio(annots):
    """
    Load the audio file and adjust sample rate. If different files are used
    and the sample rate vaires, downsample = True can be specified to resample
    all files to a common sample rate specified in the config.yaml file. 

    Args:
        annots (pd.DataFrame): annotations
        downsample (bool, optional): specifies whether file should be resampled
        or not. Defaults to False.

    Returns:
        np.array: audio array
    """
    if config['preproc']['downsample']:
        sr = config['preproc']['downsample_sr']
    else:
        sr = config['preproc']['model_sr']
        
    audio, sr = lb.load(annots.filename[0], sr = sr, offset = annots.start[0],
                        duration = annots.end.values[-1] - annots.start[0])
    
    if config['preproc']['downsample']:
        audio = lb.resample(audio, orig_sr = sr, 
                            target_sr = config['preproc']['model_sr'])
    
    return audio

def get_number_of_segs_per_call(annots):
    """
    Get the number of audio segments within one call. 
    Because the Model was trained on a specific audio length which can be 
    shorter than the actual annotated call, this method calculates the number
    of audio segments with the length defined by the model that fit into one
    call. 
    If for example a call was annotated with 3.5 seconds length and the 
    fixed audio segment length is 0.96s (VGGish) then, 3.5 / 0.96 = 3.64. 
    The function will return a 4 for this call. Three of the 0.96 s windows
    fit into the 3.5 s call, and the remaining 0.64 get rounded up. That way
    no call is lost but at the same time to audio segment that contains 
    mostly noise is included. 
    
    This idea relies heavily on the quality of annotation, but has proven 
    to work well. 

    Args:
        annots (pd.DataFrame): annotations

    Returns:
        np.array: number of audio segments per annotation
    """
    num_of_segs_per_call = np.round( ((annots.end - annots.start)/ 
                                    config['preproc']['model_time_length'] ),
                                    0).astype(int)
    num_of_segs_per_call[num_of_segs_per_call == 0] = 1
    return num_of_segs_per_call

def extract_segments(file, meta_df):
    """
    Find annotation and sound file, load audio, create the 1d audio array
    consisting only of calls, gather metadata and save the resulting 
    1d audio array. 
    This function contains the preprocesing pipeline to go from the original
    sound file to the condensed sound file that can be later used for the 
    UMAP visualization. 

    Args:
        file (string): path to annotation file
        meta_df (pd.DataFrame): metadata of recordings

    Returns:
        pd.DataFrame: metadata of recordings
    """
    
    annots = standardize_annotations(file)
    
    audio = load_audio(annots)
    
    segs_per_call = get_number_of_segs_per_call(annots)
    
    flat_call_array, segs_per_call = create_1d_call_array(segs_per_call, 
                                              annots, audio)
            
    df, site = save_metadata(file, meta_df, annots, segs_per_call)
    
    sf.write('interbed/files/raw/' + 
             Path(file).stem + f'_{site}_condensed.wav',
             flat_call_array, samplerate = config['preproc']['model_sr'])
    
    return df
    
def create_1d_call_array(num_of_segs_per_call, annots, audio):
    """
    Based on the num_of_segs_per_call array a loop iterates through the
    array, and for every element repeats the loop for as many times as
    audio segments fit into one call. The audio segment length is 
    governed by the model training, for VGGish it is 0.96s. This value
    is specified in the config.yaml file. Corresponding to the position
    within the original audio file, array of the above mentioned length 
    get extracted from the audio array and included into a matrix of 
    dimensions (number of calls x audio segment length). 
    Using a matrix speeds things up. 
    Finally the matrix is flattened into a 1d representation, thereby
    yielding an array that contains only windows of the specified length
    only containing the calls of the original file, all in sequence. 
    
    This file can later be disected by the pretrained model and used
    to visualize the gathered segments, allowing for a visualization
    of the nontrivial parts of the entire dataset. 

    Args:
        num_of_segs_per_call (np.array): segments per call
        annots (pd.DataFrame): annotations
        audio (np.array): audio array

    Returns:
        np.array: 1d audio array of only calls
    """
    call_array = np.zeros([num_of_segs_per_call.sum(), 
                           int(config['preproc']['model_time_length']*
                               config['preproc']['model_sr'])])
    cum = 0
    corr_segs_per_call = num_of_segs_per_call.copy()
    for ind, row in annots.iterrows():
        
        for segment in range( num_of_segs_per_call[ind] ):
            beg = int( (row.start -
                        annots.start.values[0] +
                        config['preproc']['model_time_length'] *
                        segment
                        ) * 
                            config['preproc']['model_sr'])
            
            end = int(beg + 
                      config['preproc']['model_time_length']*
                            config['preproc']['model_sr'])
            
            if end > len(audio):
                continue
            call_array[cum] = audio[ beg:end ] 
            cum += 1
    corr_segs_per_call.values[-1] -= sum(num_of_segs_per_call) - cum
    
    return call_array[:cum].flatten(), corr_segs_per_call

def extend_df(df, segs_per_call):
    df = pd.DataFrame(np.repeat(df.values, segs_per_call, axis=0))
    return df.rename(columns={0: 'call_time', 1: 'file', 
                              2: 'file_datetime', 3: 'site'})
    
        
def save_metadata(file, meta_df, annots, segs_per_call):
    df = pd.DataFrame()
    s_to_time = lambda s: f'{int(s/60)}:{np.mod(s, 60):.2f}s'
    df['call_time'] = list(map(s_to_time, annots.start))
    df['file'] = annots.filename
    df['file_datetime'] = get_filename_pattern(file)
    df['site'] = get_site(annots.filename[0])
    
    df = extend_df(df, segs_per_call)
    df['lengths'] = sum(segs_per_call)
    return pd.concat([meta_df, df]), df['site'].iloc[0]

def get_filename_pattern(file):
    
    if Path(file).stem[0] == 'P':
        string = Path(file).stem
        file_date = pd.to_datetime(string.split('.Table')[0], 
                            format='PAM_%Y%m%d_%H%M%S_000')
    elif Path(file).stem[0] == 'c':
        string = Path(file).stem.split('A_')[1]
        file_date = pd.to_datetime(string.split('.Table')[0],
                                    format='%Y-%m-%d_%H-%M-%S')
    else:
        string = Path(file).stem.split('.')[1]
        file_date = pd.to_datetime(string.split('.Table')[0], 
                                    format='%y%m%d%H%M%S')
    return file_date
    
def condense_files_into_only_calls():
    annotation_files = glob.glob(
                            config['preproc']['annots_path'] + '/**/*.txt',
                               recursive=True)
    meta_df = pd.DataFrame()
    for ind, file in enumerate(list(annotation_files)):
        print('\n', 
              f'{ind/len(list(annotation_files)) * 100:.0f} % complete | ',
              'now compressing file: ', Path(file).stem)
        meta_df = extract_segments(file, meta_df)
    meta_df.to_csv(config['raw_data_path'] + '/meta_data.csv')
    
if __name__ == '__main__':
    condense_files_into_only_calls()