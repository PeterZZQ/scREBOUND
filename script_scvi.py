# In[]
# baseline, test scVI
import scvi
import numpy as np 
import anndata 
import pandas as pd
import scipy.sparse as sp
import scanpy as sc
import torch
import sys
sys.path.append("./src/")
import utils
import eval


import warnings
warnings.filterwarnings("ignore")

# In[]
n_mgene = 256
gene_embed_dict = torch.load(f"/project/zzhang834/llm_dataset/CellXGeneCZI/data_download/gene_embed_meta{n_mgene}.pt", weights_only = False)
feature_info = gene_embed_dict["labels"]

# only for the purpose of feature name and feature id translation
# adata_test = anndata.read_h5ad("/project/zzhang834/llm_dataset/CellXGeneCZI/data_download/blood/partition_10.h5ad")
# adata_test.layers["counts"] = adata_test.X.copy()

adata_test_meta1 = anndata.read_h5ad("dataset/scIB/Immune_ALL_human_meta.h5ad")
adata_test_meta2 = anndata.read_h5ad("dataset/scIB/human_pancreas_norm_complexBatch_meta.h5ad")
adata_test_meta3 = anndata.read_h5ad("dataset/scIB/Lung_atlas_public_meta.h5ad")
adata_test_meta1.X = adata_test_meta1.layers["counts"].copy()
adata_test_meta2.X = adata_test_meta2.layers["counts"].copy()
adata_test_meta3.X = adata_test_meta3.layers["counts"].copy()


# In[]
# Run scVI
adata_test = adata_test_meta2
BATCH_KEY = "tech"

scvi.settings.seed = 0
# NOTE: Step 1, Train scVI on reference scRNA-seq dataset
scvi.model.SCVI.setup_anndata(adata_test, batch_key = BATCH_KEY, layer = "counts", labels_key = None)

scvi_model = scvi.model.SCVI(adata_test, use_layer_norm = "both", use_batch_norm = "none", 
                           encode_covariates = True, dropout_rate = 0.2, n_layers = 2)
scvi_model.train()
# Save the trained model on the reference scRNA-seq data
# scvi_model.save("scvi_reference", overwrite=True)

# # Sanity check, visualize the trained reference result (scVI)
SCVI_LATENT_KEY = "X_scVI"
adata_test.obsm[SCVI_LATENT_KEY] = scvi_model.get_latent_representation()
sc.pp.neighbors(adata_test, use_rep = SCVI_LATENT_KEY)
sc.tl.umap(adata_test)

# adata_test.write_h5ad("results/scvi/adata_immune_all.h5ad")
adata_test.write_h5ad("results/scvi/adata_pancreas.h5ad")
# adata_test.write_h5ad("results/scvi/adata_lung.h5ad")


# In[]
import matplotlib.pyplot as plt
colormap =plt.cm.get_cmap("tab20")
adata_embed1 = anndata.read_h5ad("results/scvi/adata_immune_all.h5ad")
adata_embed2 = anndata.read_h5ad("results/scvi/adata_pancreas.h5ad")
adata_embed3 = anndata.read_h5ad("results/scvi/adata_lung.h5ad")

# immune cell
fig = utils.plot_embeds(embed = adata_embed1.obsm["X_umap"], annos = adata_embed1.obs[["final_annotation", "batch"]].astype("category"), markerscale = 15, figsize = (20, 17), s = 1, alpha = 0.4, colormap = colormap, label_inplace = False)
fig.tight_layout()
fig.savefig("results/scvi/embed_scIB_immune_all.png", bbox_inches = "tight")
# pancrease
fig = utils.plot_embeds(embed = adata_embed2.obsm["X_umap"], annos = adata_embed2.obs[["celltype", "tech"]].astype("category"), markerscale = 5, figsize = (20, 17), s = 3, alpha = 0.4, colormap = colormap, label_inplace = False)
fig.tight_layout()
fig.savefig("results/scvi/embed_scIB_pancreas.png", bbox_inches = "tight")
# lung atlas
fig = utils.plot_embeds(embed = adata_embed3.obsm["X_umap"], annos = adata_embed3.obs[["cell_type", "batch"]].astype("category"), markerscale = 5, figsize = (20, 17), s = 3, alpha = 0.7, colormap = colormap, label_inplace = False)
fig.tight_layout()
fig.savefig("results/scvi/embed_scIB_lung.png", bbox_inches = "tight")


# In[]
scores1 = eval.eval_batch_correction(adata = adata_embed1, embed_key = "X_scVI", label_key = "final_annotation", batch_key = "batch")
scores1["dataset"] = "Immune_ALL"
scores2 = eval.eval_batch_correction(adata = adata_embed2, embed_key = "X_scVI", label_key = "celltype", batch_key = "tech")
scores2["dataset"] = "Pancrease"
scores3 = eval.eval_batch_correction(adata = adata_embed3, embed_key = "X_scVI", label_key = "cell_type", batch_key = "dataset")
scores3["dataset"] = "Lung"

scores = pd.concat([scores1, scores2, scores3], axis = 0, ignore_index = True)
scores.to_csv("results/scvi/scores_scib.csv")

# %%
