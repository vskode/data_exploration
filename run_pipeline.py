from ievad.umap_embed import get_embeddings
from ievad.dash_plot import plotUMAP_Continuous_plotly

config = 4


if config == 1:
    embeddings, metadata_dict, divisions_array = get_embeddings()

    plotUMAP_Continuous_plotly(embeddings, metadata_dict, divisions_array)
elif config == 2:
    embeddings, metadata_dict, divisions_array = get_embeddings('2024-07-10_11-48___birdaves-wav', 
                                                                '2024-07-10_12-07___umap-wav-birdaves')

    plotUMAP_Continuous_plotly(embeddings, metadata_dict, divisions_array, title="UMAP Embedding - n_neighbors=15")
elif config == 3:
    embeddings, metadata_dict, divisions_array = get_embeddings('2024-07-10_11-48___birdaves-wav', 
                                                                '2024-07-10_14-21___umap-wav-birdaves')

    plotUMAP_Continuous_plotly(embeddings, metadata_dict, divisions_array, title="UMAP Embedding - n_neighbors=5")
elif config == 4:
    embeddings, metadata_dict, divisions_array = get_embeddings('2024-07-10_11-48___birdaves-wav', 
                                                                '2024-07-10_14-23___umap-wav-birdaves')

    plotUMAP_Continuous_plotly(embeddings, metadata_dict, divisions_array, title="UMAP Embedding - n_neighbors=50")
