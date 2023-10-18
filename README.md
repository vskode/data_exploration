# IEVAD - Interactive Embedding Visualizations of Acoustic Datasets
## Combine UMAP and interactive Visualizations to explore large acoustic datasets intuitively

ievad uses plotly's dash library to provide an interactive visualization for your acoustic dataset.

<!-- ![example of visualization](docs/imgs/example.png) -->
![example of visualization](docs/imgs/example.gif)

UMAP code provided by [@avery-b](https://github.com/avery-b)

Best performance of this code is achieved using python version 3.10

## Installation

- clone repository

`git clone https://github.com/vskode/ievad.git`
- create environmen (on windows replace python3.10 with path to python 3.10)
- this step might require installation of virtualenv

`python3.10 -m virtualenv env_ievad`
- activate environment

`source env_ievad/bin/activate`
- install depedenciew

`pip install -r requirements.txt`
- download the model checkpoint, from [here](https://storage.googleapis.com/audioset/vggish_model.ckpt)
- move model checkpoints from downloads to ./ievad/vggish (see download link above, either manually or using the following code)

`mv ~/Downloads/vggish_model.ckpt ievad/vggish`
- run program

`python run_pipeline.py`

## Usage

Inside the `ievad/files/raw` directory is where you can put sound files ending in `.wav` or `.aif` and they will then be used for the creation of the embeddings and the visualization of them.

Embeddings can also be computed without visualizing them using `python run_embed.py`.

Once Embeddings have been created you can just run `python run_plot.py` to prevent the embeddings from being calculated again.

Edit the ievad/config.yaml file to change the paths to your needs.