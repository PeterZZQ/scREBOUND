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

adata_test_pancreas = anndata.read_h5ad("dataset/scIB/human_pancreas_norm_complexBatch_meta.h5ad")
adata_test_pancreas.X = adata_test_pancreas.layers["counts"].copy()

adata_train_pancreas = anndata.read_h5ad("/project/zzhang834/llm_dataset/CellXGeneCZI/data_download/pancreas/adata_meta256_4000hvg.h5ad")
adata_train_pancreas.layers["counts"] = adata_train_pancreas.X.copy()

# process celltype
adata_test_pancreas.obs.columns = ["batch", "celltype", "sizefactor", "libsize"]
adata_train_pancreas.obs = adata_train_pancreas.obs[["dataset_id", "cell_type"]]
adata_train_pancreas.obs.columns = ["batch", "celltype"]

# combine both datasets
adata_test = anndata.concat([adata_train_pancreas, adata_test_pancreas], axis = 0, join = "inner", label = "dataset", keys = ["train", "test"])
# combine the annotations
# align annots from test to train
adata_test.obs["celltype"] = adata_test.obs["celltype"].astype(object)
adata_test.obs.loc[adata_test.obs["celltype"] == "acinar", "celltype"] = "acinar cell"
adata_test.obs.loc[adata_test.obs["celltype"] == "pancreatic A cell", "celltype"] = "alpha"
adata_test.obs.loc[adata_test.obs["celltype"] == "pancreatic D cell", "celltype"] = "delta"
adata_test.obs.loc[adata_test.obs["celltype"] == "type B pancreatic cell", "celltype"] = "beta"
adata_test.obs.loc[adata_test.obs["celltype"] == "pancreatic PP cell", "celltype"] = "PP cell"
adata_test.obs.loc[adata_test.obs["celltype"] == "pancreatic acinar cell", "celltype"] = "acinar cell"
adata_test.obs.loc[adata_test.obs["celltype"] == "pancreatic ductal cell", "celltype"] = "ductal"
adata_test.obs.loc[adata_test.obs["celltype"] == "pancreatic endocrine cell", "celltype"] = "endocrine cell"
adata_test.obs.loc[adata_test.obs["celltype"] == "pancreatic epsilon cell", "celltype"] = "epsilon"
adata_test.obs.loc[adata_test.obs["celltype"] == "t_cell", "celltype"] = "T cell"
adata_test.obs.loc[adata_test.obs["celltype"] == "activated_stellate", "celltype"] = "pancreatic stellate cell"
adata_test.obs.loc[adata_test.obs["celltype"] == "quiescent_stellate", "celltype"] = "pancreatic stellate cell"
adata_test.obs.loc[adata_test.obs["celltype"] == "mast", "celltype"] = "mast cell"
adata_test.obs.loc[adata_test.obs["celltype"] == "endothelial", "celltype"] = "endothelial cell"

# additional merge of train
adata_test.obs.loc[adata_test.obs["celltype"] == "endothelial cell of lymphatic vessel", "celltype"] = "endothelial cell"
adata_test.obs.loc[adata_test.obs["celltype"] == "endothelial cell of vascular tree", "celltype"] = "endothelial cell"
adata_test.obs.loc[adata_test.obs["celltype"] == "epithelial cell of exocrine pancreas", "celltype"] = "epithelial cell"
adata_test.obs.loc[adata_test.obs["celltype"] == "epithelial cell of proximal tubule", "celltype"] = "epithelial cell"

adata_test.obs["celltype"] = adata_test.obs["celltype"].astype("category")

# In[]
# Run scVI
BATCH_KEY = "batch"

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
adata_test.write_h5ad("results/pancreas_finetune/scvi/adata_pancreas.h5ad")
# adata_test.write_h5ad("results/scvi/adata_lung.h5ad")


# In[]
import matplotlib.pyplot as plt
colormap =plt.cm.get_cmap("tab20")
adata_embed = anndata.read_h5ad("results/pancreas_finetune/scvi/adata_pancreas.h5ad")

# pancreas
fig = utils.plot_by_batch(x_rep = adata_embed.obsm["X_umap"], annos = np.array([x for x in adata_embed.obs["celltype"].values]), batches = np.array([x for x in adata_embed.obs["dataset"].values]), markerscale = 5, figsize = (40, 17), s = 3, alpha = 0.4, label_inplace = False)
fig.tight_layout()
fig.savefig("results/pancreas_finetune/scvi/embed_scIB_pancreas.png", bbox_inches = "tight")

# %%
