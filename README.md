# IEVAD - Interactive Embedding Visualizations of Acoustic Datasets
## Combine UMAP and interactive Visualizations to explore large acoustic datasets intuitively

ievad uses plotly's dash library to provide an interactive visualization for your acoustic dataset.

To use the app:
- create an environment and pip install the requirements.txt file
- download the model checkpoint, from [here](https://storage.googleapis.com/audioset/vggish_model.ckpt) and place it into the ievad/vggish folder (don't change its name)
- edit the ievad/config.yaml file to insert the paths to your files
- to condense sound files based on corresponding annotation files, run the run_file_condenser.py script
- run the run_embed.py script to generate embeddings from the files specified in the ievad/config.yaml file path
- once finished, run the run_plot.py file to start the browser app
    - note that depending on the number and size of files you include both run_embed.py and run_plot.py might take a while

<!-- ![example of visualization](docs/imgs/example.png) -->
![example of visualization](docs/imgs/example.gif)

UMAP code provided by [@avery-b](https://github.com/avery-b)

Production is still in early stage, ease of use will hopefully increase in the next months.