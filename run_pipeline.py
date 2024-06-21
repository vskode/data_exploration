from ievad.generate_embeddings import generate_embeddings
from ievad.umap_embed import get_embeddings

# embed
generate_embeddings(model_name='vggish', ignore_check_if_combination_exists=False)
generate_embeddings(model_name='umap', ignore_check_if_combination_exists=True)

# plot
acc_embeddings, folders, file_list, lenghts = get_embeddings(limit=6)
