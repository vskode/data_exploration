import numpy as np
from pathlib import Path
import pickle

e_dir = 'ievad/files/embeds'
u_dir = 'ievad/files/umap_embeds'

aves_embeds = [f for f in Path(e_dir).iterdir() if 'aves' in f.stem]
aves_umap_embeds = [f for f in Path(u_dir).iterdir() if 'vggish' in f.stem]

embed_arrs = {}
for d in aves_embeds:
    for f in [i for i in d.iterdir() if i.suffix == '.pickle']:
        em = pickle.load(open(f, 'rb'))
        embed_arrs.update({f: em})

umap_embed_arrs = {}
for u in aves_umap_embeds:
    for f in [i for i in u.iterdir() if i.suffix == '.pickle']:
        em = pickle.load(open(f, 'rb'))
        umap_embed_arrs.update({f: em})
    
print('fin')

import umap
umap.UMAP()