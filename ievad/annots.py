import numpy as np
import pandas as pd
from pathlib import Path


def return_data_with_annots(config, data):
    annot_files = Path(config['preproc']['annots_path']).rglob('*txt')
    annot_files = list(annot_files)
    ldf = pd.DataFrame()
    for f in annot_files:
        df = pd.read_csv(f, sep='\t')
        ldf = pd.concat([ldf, df])

    data_df = pd.DataFrame(data)
    from ievad.plot_helpers import time_string_to_float
    ttimes = list(map(time_string_to_float, data['time_in_orig_file']))
    data_df['times'] = ttimes
    audio_fnames = np.unique(data['filename'])
    count, ac_alist = 0, []
    for af in audio_fnames:
        df = ldf[ldf['Begin File'] == af]
        dat = data_df[data_df['filename'] == af]
        alist = [0]*len(dat)
        for bt, et in zip(df['Begin Time (s)'], df['End Time (s)']):
            for idx, emb in enumerate(dat['times']):
                if emb > bt and emb < et:
                    count +=1
                    print(count)
                    alist[idx] = 1
        ac_alist = [*ac_alist, *alist]
    data_df.loc[:, 'annot'] = ac_alist
    return data_df.to_dict('list')