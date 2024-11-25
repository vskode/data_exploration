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
    meta: dict
    index: int
    # label: float

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
             for p in global_path.joinpath(path.path).rglob('*json')][:3]
    # print(jsons)
    return {'message': 'dictionaries successfully retrieved', 
            'dicts': jsons}
    

@app.post("/getDataPoint")
async def create_item(item: Item):
    # print(item)
    path = Path(Path(item.meta['audio_dir']).stem).joinpath(item.meta['audio_files'][item.index])
    sr = item.meta['sample_rate (Hz)']
    segment_length = item.meta['segment_length (samples)'] / sr
    # if isinstance(item.label, float):
    #     t_s = item.label
    # else:
    #     t_s = item.z
        
    audio, sr, file_stem = load_audio(0., 
                                      path,
                                      sr,
                                      segment_length)
    spec = create_specs2(audio)
    # print(spec)
    return {'message': 'values successfully received', 
            'spectrogram_data': spec.tolist()}

@app.get("/")
# async def create_item(item: Item):
async def read_item():
    return {'message': 'laeuft'}