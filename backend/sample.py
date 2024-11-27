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
#DOMRect {x: 65.62946319580078, y: 262.20831298828125, width: 8, height: 8, top: 262.20831298828125, …}
        #   <rect
        #     x={0}
        #     y={0}
        #     width={boundsWidth}
        #     height={boundsHeight}
        #     onMouseMove={onMouseMove}
        #     onMouseLeave={() => setCursorPosition(null)}
        #     visibility={"hidden"}
        #     pointerEvents={"all"}
        #     // onClick={(e) => handleClick(e, getClosestPoint(cursorPosition))}
        #     onClick={handleClick}
        #   />

@app.post("/getDictionaries")
async def get_folders(path: DataPath):
    jsons = [str(p.relative_to(global_path))
             for p in global_path.joinpath(path.path).rglob('*json')]
    
    # print(jsons)
    return {'message': 'dictionaries successfully retrieved', 
            'dicts': jsons}
    

@app.post("/getDataPoint")
async def create_spectrogram(item: Item):
    # print(item)
    path = Path(item.meta['audio_dir']).joinpath(
        item.meta['audio_files'][item.index]
        )
    # path = Path(Path(item.meta['audio_dir']).stem).joinpath(
    #     item.meta['audio_files'][item.index]
    #     )
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