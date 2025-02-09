from fastapi.middleware.cors import CORSMiddleware
from fastapi import FastAPI
from pydantic import BaseModel
from pathlib import Path
from backend.ievad.plot_helpers import load_audio
from backend.ievad.dash_plot import create_specs2

global_path = Path('frontend/public')

class Item(BaseModel):
    x: float
    y: float
    z: float
    source_file: str
    meta: dict
    index: int

class DataPath(BaseModel):
    path: str

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  
    # Adjust the origin to your frontend's URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/getDictionaries")
async def get_folders(path: DataPath):
    jsons = [str(p.relative_to(global_path))
             for p in global_path.joinpath(path.path).rglob('*umap.json')]
    
    # print(jsons)
    return {'message': 'dictionaries successfully retrieved', 
            'dicts': jsons}
    

@app.post("/getDataPoint")
async def create_spectrogram(item: Item):
    
    path = Path(item.meta['audio_dir']).joinpath(
        item.source_file
        )
    sr = item.meta['sample_rate (Hz)']
    segment_length = item.meta['segment_length (samples)'] / sr
        
    audio, sr, file_stem = load_audio(item.z, 
                                      path,
                                      sr,
                                      segment_length)
    spec = create_specs2(audio)

    return {'message': 'values successfully received', 
            'spectrogram_data': spec.tolist()}

@app.get("/")

async def read_item():
    return {'message': 'laeuft'}