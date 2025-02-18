# In[]
import torch
from sklearn.neighbors import KNeighborsClassifier
import scipy.sparse as sp
import src.data_utils as data_utils
import anndata
import scanpy as sc
import numpy as np

from pathlib import Path
import sys, os

import numpy as np
import scipy.sparse as sparse
import tqdm
import anndata
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

# packages for distributed training
import torch
from torch.utils import data

sys.path.append("./src")

import data_utils
from transformer_model import TransformerModel, get_default_config
import trainer
import utils
import eval

import warnings
warnings.filterwarnings("ignore")


def evaluation(model, dataloader):
    # NOTE: training loop
    model.eval()
    with torch.no_grad():
        val_loss = 0.0
        val_loss_mlm = 0.0
        val_loss_sup = 0.0
        val_loss_kd = 0.0
        for data_sample in tqdm.tqdm(dataloader, desc=f"Evaluation"):
            loss, loss_mlm, loss_sup, loss_kd = trainer.infer_databatch(model, data_sample, multigpus = False)                            
            val_loss += loss.item()
            val_loss_mlm += loss_mlm.item()
            val_loss_sup += loss_sup.item()
            val_loss_kd += loss_kd.item()

        # log the values
        val_loss /= len(dataloader)
        val_loss_mlm /= len(dataloader)
        val_loss_sup /= len(dataloader)
        val_loss_kd /= len(dataloader)

        print(f"Val Loss (TOTAL): {val_loss:.4f}, Val Loss (MLM): {val_loss_mlm:.4f}, Val Loss (CLASS): {val_loss_sup:.4f}, Val Loss (KD): {val_loss_kd:.4f}")

    return val_loss, val_loss_mlm, val_loss_sup, val_loss_kd


# In[]
# NOTE: Given adata,
# 1. transform into meta-gene counts
# 3. transform meta-gene counts into cell sentence
def preprocess_anndata_knn(adata, meta_embed, gene_embed_key = "esm2"):
    """
    Group the genes into meta-genes in adata according to meta-gene embedding, 
    NOTE: genes are grouped into meta-genes according to the knn classifier, 
    however the result is very different from the hard-code assignment, could cause deterioration of the result 
    """
    # NOTE: kneighbors classifier is causing all the issue. Should use the exact hard-coded assignment as described
    knn = KNeighborsClassifier(n_neighbors = 5).fit(X = meta_embed.numpy(), y = np.arange(meta_embed.shape[0]))
    meta_labels = knn.predict(adata.varm[gene_embed_key])
    print(np.unique(meta_labels, return_counts = True))
    adata.var["meta_labels"] = meta_labels

    counts_meta = []
    for label in range(meta_embed.shape[0]):
        counts_meta.append(adata.layers["counts"][:, meta_labels == label].sum(axis = 1))
    counts_meta = sp.csr_matrix(np.hstack(counts_meta))

    adata_meta = anndata.AnnData(X = counts_meta, obs = adata.obs)
    sc.pp.normalize_total(adata_meta, target_sum = 10e4, key_added = "libsize")
    sc.pp.log1p(adata_meta)
    
    return adata_meta


# In[]
# read the gene protein embedding and gene & meta-gene assignment
n_mgene = 256
gene_embed_dict = torch.load(f"/project/zzhang834/llm_dataset/CellXGeneCZI/data_download/gene_embed_meta{n_mgene}.pt", weights_only = False)
feature_info = gene_embed_dict["labels"]

# only for the purpose of feature name and feature id translation
# adata_test = anndata.read_h5ad("/project/zzhang834/llm_dataset/CellXGeneCZI/data_download/blood/partition_10.h5ad")
# adata_test.layers["counts"] = adata_test.X.copy()

# adata_test1 = anndata.read_h5ad("dataset/scIB/Immune_ALL_human.h5ad")
# adata_test_meta1 = preprocess_anndata(adata_test1, feature_info, var_name = "gene_name")
# adata_test_meta1.write_h5ad("dataset/scIB/Immune_ALL_human_meta.h5ad")

# adata_test2 = anndata.read_h5ad("dataset/scIB/human_pancreas_norm_complexBatch.h5ad")
# adata_test_meta2 = preprocess_anndata(adata_test2, feature_info, var_name = "gene_name")
# adata_test_meta2.write_h5ad("dataset/scIB/human_pancreas_norm_complexBatch_meta.h5ad")

# adata_test3 = anndata.read_h5ad("dataset/scIB/Lung_atlas_public.h5ad")
# adata_test_meta3 = preprocess_anndata(adata_test3, feature_info, var_name = "gene_name")
# adata_test_meta3.write_h5ad("dataset/scIB/Lung_atlas_public_meta.h5ad")

adata_test_meta1 = anndata.read_h5ad("dataset/scIB/Immune_ALL_human_meta.h5ad")
adata_test_meta2 = anndata.read_h5ad("dataset/scIB/human_pancreas_norm_complexBatch_meta.h5ad")
adata_test_meta3 = anndata.read_h5ad("dataset/scIB/Lung_atlas_public_meta.h5ad")


# take out the pancreas dataset
adata_pancreas = anndata.read_h5ad("/project/zzhang834/llm_dataset/CellXGeneCZI/data_download/pancreas/adata_meta256_4000hvg.h5ad")
sc.pp.normalize_total(adata_pancreas, target_sum = 10e4, key_added = "libsize")
sc.pp.log1p(adata_pancreas)
adata_test_meta4 = adata_pancreas


# In[]
# NOTE: adjust the label of training pancreas dataset
adata_test_meta4.obs["cell_type"] = adata_test_meta4.obs["cell_type"].astype(object)
adata_test_meta4.obs.loc[adata_test_meta4.obs["cell_type"] == "acinar", "cell_type"] = "acinar cell"
adata_test_meta4.obs.loc[adata_test_meta4.obs["cell_type"] == "pancreatic A cell", "cell_type"] = "alpha"
adata_test_meta4.obs.loc[adata_test_meta4.obs["cell_type"] == "pancreatic D cell", "cell_type"] = "delta"
adata_test_meta4.obs.loc[adata_test_meta4.obs["cell_type"] == "type B pancreatic cell", "cell_type"] = "beta"
adata_test_meta4.obs.loc[adata_test_meta4.obs["cell_type"] == "pancreatic PP cell", "cell_type"] = "PP cell"
adata_test_meta4.obs.loc[adata_test_meta4.obs["cell_type"] == "pancreatic acinar cell", "cell_type"] = "acinar cell"
adata_test_meta4.obs.loc[adata_test_meta4.obs["cell_type"] == "pancreatic ductal cell", "cell_type"] = "ductal"
adata_test_meta4.obs.loc[adata_test_meta4.obs["cell_type"] == "pancreatic endocrine cell", "cell_type"] = "endocrine cell"
adata_test_meta4.obs.loc[adata_test_meta4.obs["cell_type"] == "pancreatic epsilon cell", "cell_type"] = "epsilon"
adata_test_meta4.obs.loc[adata_test_meta4.obs["cell_type"] == "t_cell", "cell_type"] = "T cell"
adata_test_meta4.obs.loc[adata_test_meta4.obs["cell_type"] == "mast", "cell_type"] = "mast cell"
adata_test_meta4.obs.loc[adata_test_meta4.obs["cell_type"] == "endothelial", "cell_type"] = "endothelial cell"

# additional merge of train
adata_test_meta4.obs.loc[adata_test_meta4.obs["cell_type"] == "endothelial cell of lymphatic vessel", "cell_type"] = "endothelial cell"
adata_test_meta4.obs.loc[adata_test_meta4.obs["cell_type"] == "endothelial cell of vascular tree", "cell_type"] = "endothelial cell"
adata_test_meta4.obs.loc[adata_test_meta4.obs["cell_type"] == "epithelial cell of exocrine pancreas", "cell_type"] = "epithelial cell"
adata_test_meta4.obs.loc[adata_test_meta4.obs["cell_type"] == "epithelial cell of proximal tubule", "cell_type"] = "epithelial cell"


# In[]
# # NOTE: Mapping is changed, which makes the result worse
# # gene embed full
# gene_embed_full = torch.load("/project/zzhang834/llm_dataset/proteome/embeddings/Homo_sapiens.GRCh38.gene_symbol_to_embedding_ESM2.pt", weights_only = False)
# # preprocessing, use gene embed full
# gene_overlap = np.intersect1d(adata_test.var["feature_name"].values, np.array([x for x in gene_embed_full.keys()]))
# print("number of overlapped genes: {:d}".format(len(gene_overlap)))
# adata_test = adata_test[:, adata_test.var["feature_name"].isin(gene_overlap)].copy()

# sc.pp.filter_genes(adata_test, min_cells = int(0.01 * adata_test.shape[0]))
# # sc.pp.normalize_total(adata_test)
# # sc.pp.log1p(adata_test)

# # sc.pp.highly_variable_genes(adata_test, n_top_genes = 4000)
# # adata_test = adata_test[:, adata_test.var["highly_variable"]]
# adata_test.varm["esm2"] = torch.vstack([gene_embed_full[x] for x in adata_test.var["feature_name"].values]).numpy()

# adata_test_meta = preprocess_anndata_knn(adata = adata_test, meta_embed = gene_embed_dict["meta_embed"], gene_embed_key = "esm2")


# In[]
# function
data_utils.set_seed(3)

# Define the device
device = torch.device("cuda:1")
print(f"GPU - Using device: {device}")
# Load the dataset
print(f"GPU - Loading dataset...")

# NOTE: save in localscratch for faster memory access
data_dir = Path(f"/project/zzhang834/LLM_KD/dataset/cellxgene")
# data_dir = Path(f"/localscratch/ziqi/localscratch_tempdata/cellxgene")
# load the token embedding
token_embed = torch.load(data_dir / f"token_embed_{n_mgene}.pt", weights_only = False)
# load the cell meta-info
meta_dict = torch.load(data_dir / f"meta_{n_mgene}.pt", weights_only = False)

# NOTE: 1. vanilla model with only mlm loss
# pretrained model
model_name = "checkpoint_0.98_dynmask_deepinj_2_12599"
# fine-tuned model
# model_name = "checkpoint_0.98_dynmask_deepinj_5"



# NOTE: 2. model with mlm and classifier/contrastive loss
# # compare with linear (no-activation) & non-deep injection
# model_name = "checkpoint_0.98_contr_rootannot_dynmask_1"
# model_name = "checkpoint_0.98_contrcb_rootannot_dynmask_1"

# # NOTE: for deep injection, both the injection part and the final activation part affect the final result
# model_name = "checkpoint_0.98_contrcb_rootannot_dynmask_deepinj_1"
# model_name = "checkpoint_0.98_contr_rootannot_dynmask_deepinj_1"

model_dir = f"/project/zzhang834/LLM_KD/checkpoint/{model_name}.pth"
# model_dir = f"/project/zzhang834/LLM_KD/checkpoint_pancreas/{model_name}.pth"

res_dir = f"results/{model_name}_56million/"
# res_dir = f"results/pancreas_finetune/{model_name}_56million/"
state = torch.load(model_dir, weights_only = False)
model_config = state["model_config"]
model_config.__dict__.update({"checkpoint_path": None, "checkpoint_prefix": None, "pretrain_path":  model_dir})

labels = None
batches = None
test_dataset1 = data_utils.sc_dataset_anndata(adata_meta = adata_test_meta1, labels = labels, batches = batches, njobs = 16)
test_dataset2 = data_utils.sc_dataset_anndata(adata_meta = adata_test_meta2, labels = labels, batches = batches, njobs = 16)
test_dataset3 = data_utils.sc_dataset_anndata(adata_meta = adata_test_meta3, labels = labels, batches = batches, njobs = 16)
test_dataset4 = data_utils.sc_dataset_anndata(adata_meta = adata_test_meta4, labels = labels, batches = batches, njobs = 16)

test_loader1 = data.DataLoader(test_dataset1, batch_size = 1, shuffle = False, pin_memory = True, num_workers = 8, prefetch_factor = 8)
test_loader2 = data.DataLoader(test_dataset2, batch_size = 1, shuffle = False, pin_memory = True, num_workers = 8, prefetch_factor = 8)
test_loader3 = data.DataLoader(test_dataset3, batch_size = 1, shuffle = False, pin_memory = True, num_workers = 8, prefetch_factor = 8)
test_loader4 = data.DataLoader(test_dataset4, batch_size = 1, shuffle = False, pin_memory = True, num_workers = 8, prefetch_factor = 8)

print(f"GPU - Done.")

fm_model = TransformerModel(model_config = model_config, token_embed = token_embed, n_batch = len(meta_dict["batch_code"]), n_label = len(meta_dict["label_code"]), device = device)

print(f"GPU - Preloading lastest model'")
# load parameter from last train
fm_model.load_state_dict(state["model_state_dict"])
print(f"GPU - Done.")


# In[]
# NOTE: calculate the embedding
import scvi
from umap import UMAP
# TODO: issue, for the classifier, should the masked input be used??
adata_embed1 = trainer.cell_embed(model = fm_model, dataloader = test_loader1, multi_gpus = False)
adata_embed1.obs = adata_test_meta1.obs.copy()
adata_embed1.obsm["latent"] = adata_embed1.X.copy()
# cluster structure is clearer with UMAP
# adata_embed1.obsm["X_umap"] = UMAP(n_components = 2).fit_transform(adata_embed1.X.toarray())
sc.pp.neighbors(adata_embed1, n_neighbors = 15, use_rep = "latent")
sc.tl.umap(adata_embed1, min_dist = 0.3)

adata_embed2 = trainer.cell_embed(model = fm_model, dataloader = test_loader2, multi_gpus = False)
adata_embed2.obs = adata_test_meta2.obs.copy()
adata_embed2.obsm["latent"] = adata_embed2.X.copy()
# adata_embed2.obsm["X_umap"] = UMAP(n_components = 2).fit_transform(adata_embed2.X.toarray())
sc.pp.neighbors(adata_embed2, n_neighbors = 15, use_rep = "latent")
sc.tl.umap(adata_embed2, min_dist = 0.3)

adata_embed3 = trainer.cell_embed(model = fm_model, dataloader = test_loader3, multi_gpus = False)
adata_embed3.obs = adata_test_meta3.obs.copy()
adata_embed3.obsm["latent"] = adata_embed3.X.copy()
# adata_embed3.obsm["X_umap"] = UMAP(n_components = 2).fit_transform(adata_embed3.X.toarray())
sc.pp.neighbors(adata_embed3, n_neighbors = 15, use_rep = "latent")
sc.tl.umap(adata_embed3, min_dist = 0.3)

adata_embed4 = trainer.cell_embed(model = fm_model, dataloader = test_loader4, multi_gpus = False)
adata_embed4.obs = adata_test_meta4.obs.copy()
adata_embed4.obsm["latent"] = adata_embed4.X.copy()
# adata_embed4.obsm["X_umap"] = UMAP(n_components = 2).fit_transform(adata_embed4.X.toarray())
sc.pp.neighbors(adata_embed4, n_neighbors = 15, use_rep = "latent")
sc.tl.umap(adata_embed4, min_dist = 0.3)


# In[]
# Visualize the latent embedding
if not os.path.exists(res_dir):
    os.makedirs(res_dir)

colormap =plt.cm.get_cmap("tab20")
# immune cell
fig = utils.plot_embeds(embed = adata_embed1.obsm["X_umap"], annos = adata_embed1.obs[["final_annotation", "batch"]].astype("category"), markerscale = 15, figsize = (20, 17), s = 1, alpha = 0.4, colormap = colormap, label_inplace = False)
fig.tight_layout()
fig.savefig(res_dir + "embed_scIB_immune_all.png", bbox_inches = "tight")
# pancreas
fig = utils.plot_embeds(embed = adata_embed2.obsm["X_umap"], annos = adata_embed2.obs[["celltype", "tech"]].astype("category"), markerscale = 5, figsize = (20, 17), s = 3, alpha = 0.4, colormap = colormap, label_inplace = False)
fig.tight_layout()
fig.savefig(res_dir + "embed_scIB_pancreas.png", bbox_inches = "tight")
# lung atlas
fig = utils.plot_embeds(embed = adata_embed3.obsm["X_umap"], annos = adata_embed3.obs[["cell_type", "batch"]].astype("category"), markerscale = 5, figsize = (20, 17), s = 3, alpha = 0.7, colormap = colormap, label_inplace = False)
fig.tight_layout()
fig.savefig(res_dir + "embed_scIB_lung.png", bbox_inches = "tight")

fig = utils.plot_embeds(embed = adata_embed4.obsm["X_umap"], annos = adata_embed4.obs[["cell_type", "dataset_id"]].astype("category"), markerscale = 5, figsize = (40, 17), s = 3, alpha = 0.7, label_inplace = False)
fig.tight_layout()
fig.savefig(res_dir + "embed_training_pancreas.png", bbox_inches = "tight")

# In[]
scores1 = eval.eval_batch_correction(adata = adata_embed1, embed_key = "latent", label_key = "final_annotation", batch_key = "batch")
scores1["dataset"] = "Immune_ALL"
scores2 = eval.eval_batch_correction(adata = adata_embed2, embed_key = "latent", label_key = "celltype", batch_key = "tech")
scores2["dataset"] = "Pancreas"
scores3 = eval.eval_batch_correction(adata = adata_embed3, embed_key = "latent", label_key = "cell_type", batch_key = "dataset")
scores3["dataset"] = "Lung"

scores = pd.concat([scores1, scores2, scores3], axis = 0, ignore_index = True)
scores.to_csv(res_dir + "scores_scib.csv")



# In[]
# # merge clusters
# coarse_ct = {}
# coarse_ct["monocyte"] = ["CD14-low, CD16-positive monocyte", "CD14-positive monocyte"]
# coarse_ct["nk"] = ["CD16-negative, CD56-bright natural killer cell, human", "natural killer cell"]
# coarse_ct["T"] = ["CD4-positive, alpha-beta T cell", "CD4-positive, alpha-beta cytotoxic T cell",
#                        "CD8-positive, alpha-beta T cell", "central memory CD4-positive, alpha-beta T cell",
#                        "central memory CD8-positive, alpha-beta T cell", "effector memory CD4-positive, alpha-beta T cell",
#                        "effector memory CD8-positive, alpha-beta T cell", "gamma-delta T cell", "mucosal invariant T cell",
#                        "naive thymus-derived CD4-positive, alpha-beta T cell", "naive thymus-derived CD8-positive, alpha-beta T cell",
#                        "regulatory T cell"]

# coarse_ct["CD4 T"] = ["CD4-positive, alpha-beta T cell", "CD4-positive, alpha-beta cytotoxic T cell",
#                        "central memory CD4-positive, alpha-beta T cell", "effector memory CD4-positive, alpha-beta T cell",
#                        "naive thymus-derived CD4-positive, alpha-beta T cell", "regulatory T cell"]

# coarse_ct["CD8 T"] = ["CD8-positive, alpha-beta T cell", "central memory CD8-positive, alpha-beta T cell",
#                        "effector memory CD8-positive, alpha-beta T cell", "naive thymus-derived CD8-positive, alpha-beta T cell"]

# coarse_ct["B"] = ["naive B cell", "transitional stage B cell", "memory B cell"]
# coarse_ct["dendritic"] = ["conventional dendritic cell", "dendritic cell", "plasmacytoid dendritic cell"]
# coarse_ct["thymocyte"] = ["double negative thymocyte"]
# coarse_ct["erythrocyte"] = ["erythrocyte"]

# adata_test.obs[["cell_type (coarse)"]] = adata_test.obs[["cell_type"]].values
# adata_test.obs.loc[adata_test.obs["cell_type"].isin(coarse_ct["monocyte"]), "cell_type (coarse)"] = "monocyte"
# adata_test.obs.loc[adata_test.obs["cell_type"].isin(coarse_ct["nk"]), "cell_type (coarse)"] = "nk"
# adata_test.obs.loc[adata_test.obs["cell_type"].isin(coarse_ct["T"]), "cell_type (coarse)"] = "T"
# adata_test.obs.loc[adata_test.obs["cell_type"].isin(coarse_ct["CD4 T"]), "cell_type (coarse)"] = "CD4 T"
# adata_test.obs.loc[adata_test.obs["cell_type"].isin(coarse_ct["CD8 T"]), "cell_type (coarse)"] = "CD8 T"
# adata_test.obs.loc[adata_test.obs["cell_type"].isin(coarse_ct["B"]), "cell_type (coarse)"] = "B"
# adata_test.obs.loc[adata_test.obs["cell_type"].isin(coarse_ct["dendritic"]), "cell_type (coarse)"] = "dendritic"
# adata_test.obs.loc[adata_test.obs["cell_type"].isin(coarse_ct["thymocyte"]), "cell_type (coarse)"] = "thymocyte"
# adata_test.obs.loc[adata_test.obs["cell_type"].isin(coarse_ct["erythrocyte"]), "cell_type (coarse)"] = "erythrocyte"

# fig = utils.plot_embeds(embed = adata_test2.obsm["latent_umap"], annos = adata_test.obs[["cell_type (coarse)"]].astype("category"), markerscale = 15, figsize = (25, 17), s = 1, alpha = 0.4, colormap = colormap, label_inplace = True)
# fig.tight_layout()
# # fig.savefig(res_dir + "")


# %%
