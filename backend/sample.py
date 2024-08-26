from fastapi.middleware.cors import CORSMiddleware
from fastapi import FastAPI
from pydantic import BaseModel


class Item(BaseModel):
    x: int
    y: int
    z: int

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
    return {'message': 'values successfully received'}

@app.get("/")
# async def create_item(item: Item):
async def read_item():
    return {'message': 'laeuft'}