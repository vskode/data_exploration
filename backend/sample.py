from fastapi.middleware.cors import CORSMiddleware
from fastapi import FastAPI
from pydantic import BaseModel
from pathlib import Path
from backend.ievad.plot_helpers import load_audio
from backend.ievad.dash_plot import create_specs2


class Item(BaseModel):
    x: float
    y: float
    z: float
    meta: dict

# with open('public/data.json', 'r') as f:
#     data = f.read()
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  
    # Adjust the origin to your frontend's URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/getDataPoint")
async def create_item(item: Item):
    print(item)
    path = Path(Path(item.meta['audio_dir']).stem).joinpath(item.meta['audio_files'])
    audio, sr, file_stem = load_audio(item.z, 
                                      path)
    spec = create_specs2(audio)
    # print(spec)
    return {'message': 'values successfully received', 
            'spectrogram_data': spec.tolist()}

@app.get("/")
# async def create_item(item: Item):
async def read_item():
    return {'message': 'laeuft'}