from ievad.umap_embed import get_embeddings
from ievad.dash_plot import plotUMAP_Continuous_plotly

# embed

# plot
embeddings, metadata_dict, divisions_array = get_embeddings()

plotUMAP_Continuous_plotly(embeddings, metadata_dict, divisions_array)
