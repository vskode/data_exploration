#%% imports
import glob
import numpy as np
import pandas as pd
from pathlib import Path
import librosa as lb
import soundfile as sf

#%% load files
    
# model = hub.load('google_humpback_model')
annotation_files = glob.glob('/home/vincent/Code/MA/Daten/Catherine_annotations/**/*.txt', 
                               recursive=True)[19:]

#%%
def get_corresponding_sound_file(file):
    hard_drive_path = '/media/vincent/Seagate Backup Plus Drive'
    # hard_drive_path = '/mnt/d'
    file_path = glob.glob(f'{hard_drive_path}/**/{Path(file).stem.split("Table")[0]}wav',
                      recursive = True)
    
    if not file_path:
        new_file = Path(file).stem.split("Table")[0] + 'wav'
        file_tolsta = '336097327.'+new_file[6:].replace('_000', '').replace('_', '')
        file_path = glob.glob(f'{hard_drive_path}/**/{file_tolsta}',
                    recursive = True)
        
        if not file_path :
            file_tolsta = '335564853.'+new_file[6:].replace('5_000', '4').replace('_', '')
            file_path = glob.glob(f'{hard_drive_path}/**/{file_tolsta}',
                        recursive = True)
            if not file_path :
                return f'{file.stem.split("Table")[0]}wav'
    return file_path[0]
    
def standardize_annotations(file):
    ann = pd.read_csv(file, sep = '\t')

    ann['filename'] = get_corresponding_sound_file(file)
    ann['label']    = 1
    ann = ann.rename(columns = { 'Begin Time (s)' : 'start', 'End Time (s)' : 'end' })
    ann = ann.sort_values('start')
    return ann

def get_location(file):
    return Path(file).parent.stem

def extract_segments(file, meta_df):
    
    annots = standardize_annotations(file)
    
    location = get_location(file)
    
    audio, sr = lb.load(annots.filename[0], sr = 16000, offset = annots.start[0], 
                        duration = annots.end.values[-1] - annots.start[0])
    
    num_of_segs_per_call = np.round( ((annots.end - annots.start)/ 0.96 ), 
                                    0).astype(int)
    num_of_segs_per_call[num_of_segs_per_call == 0] = 1
    
    call_array = np.zeros([num_of_segs_per_call.sum(), int(0.96*16000)])
    cum = 0
    for ind, row in annots.iterrows():
        
        for segment in range( num_of_segs_per_call[ind] ):
            beg = int( ( row.start + 0.96 *
                        segment - annots.start.values[0]) * 16000)
            end = int( beg + 0.96*16000 )
            if end > len(audio):
                continue
            call_array[cum] = audio[ beg:end ] 
            cum += 1
            
    df = save_metadata(file, df, annots)
    
    flat_call_array = call_array[:cum].flatten()
    sf.write('interbed/files/raw/' + Path(file).stem + f'_{location}_condensed.wav', 
             flat_call_array, samplerate = 16000)
    return df
    
    
def save_ket_annot_only_existing_paths(df):
    check_if_full_path_func = lambda x: x[0] == '/'
    df[list( map(check_if_full_path_func, 
        df.index.get_level_values(0)) )].to_csv(
        'Daten/ket_annot_file_exists.csv')
        
def save_metadata(file, df, annots):
    annots.start
    

if __name__ == '__main__':
    
    meta_df = pd.DataFrame()
    for file in list(annotation_files):
        print('now compressing file: ', Path(file).stem)
        meta_df = extract_segments(file, meta_df)
    
