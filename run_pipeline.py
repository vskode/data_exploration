from ievad.embed import main
from ievad.embed2d import get_embeddings
from ievad.plot import plotUMAP_Continuous_plotly

# embed
main()

# plot
acc_embeddings, folders, file_list, lenghts = get_embeddings(limit=6)
percentiles = 24
plotUMAP_Continuous_plotly(acc_embeddings, percentiles, 'plasma', 
                           file_list, lenghts)