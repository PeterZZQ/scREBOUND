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

# packages for distributed training
import torch
from torch.utils import data

sys.path.append("./src")

import data_utils
from transformer_model import TransformerModel, get_default_config
import trainer
import utils




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


def preprocess_anndata(adata, feature_df, var_name = "gene_name"):
    """
    Group the genes into meta-genes in adata, according to the hard-coded grouping information
    """
    adata_s = adata.copy()
    
    if var_name == "gene_name":
        gene_ids = feature_df["feature_name"].values.squeeze()
        feature_df.index = gene_ids
    elif var_name == "ensembl_id":
        gene_ids = feature_df.index.values.squeeze()
    else:
        raise ValueError("Either gene_name or ensembl_id need to be provided in adata")

    gene_ids = np.intersect1d(adata_s.var.index.values, gene_ids)
    adata_s = adata_s[:, gene_ids]
    print(f"overlapping genes: {len(gene_ids)}")
    meta_labels = feature_df.loc[gene_ids, "labels"].values.squeeze()
    counts_meta = np.zeros((adata_s.shape[0], len(np.unique(feature_df["labels"].values))))
    for label in np.unique(meta_labels):
        counts_meta[:, label] = adata_s.layers["counts"][:, meta_labels == label].toarray().sum(axis = 1)
    counts_meta = sp.csr_matrix(counts_meta)
    
    adata_meta = anndata.AnnData(X = counts_meta, obs = adata_s.obs)

    # preprocess the meta-gene count
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

adata_test1 = anndata.read_h5ad("dataset/scIB/Immune_ALL_human.h5ad")
adata_test_meta1 = preprocess_anndata(adata_test1, feature_info, var_name = "gene_name")

adata_test2 = anndata.read_h5ad("dataset/scIB/human_pancreas_norm_complexBatch.h5ad")
adata_test_meta2 = preprocess_anndata(adata_test2, feature_info, var_name = "gene_name")

adata_test3 = anndata.read_h5ad("dataset/scIB/Lung_atlas_public.h5ad")
adata_test_meta3 = preprocess_anndata(adata_test3, feature_info, var_name = "gene_name")

# In[]
adata_test_meta = adata_test_meta3

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
# data_dir = Path(f"/project/zzhang834/LLM_KD/dataset/cellxgene")
data_dir = Path(f"/localscratch/ziqi/localscratch_tempdata/cellxgene")
# load the token embedding
token_embed = torch.load(data_dir / f"token_embed_{n_mgene}.pt")
# load the cell meta-info
meta_dict = torch.load(data_dir / f"meta_{n_mgene}.pt")

model_name = "checkpoint_0.98_contr_dynmask_1"
# model_name = "checkpoint_0.98_1"
model_dir = f"/project/zzhang834/LLM_KD/checkpoint/{model_name}.pth"
res_dir = f"results/{model_name}_55million/"
state = torch.load(model_dir)
model_config = state["model_config"]
model_config.__dict__.update({"checkpoint_path": None, "checkpoint_prefix": None, "pretrain_path":  model_dir})

labels = None
batches = None
test_dataset = data_utils.sc_dataset_anndata(adata_meta = adata_test_meta, labels = labels, batches = batches, njobs = 16)

test_loader = data.DataLoader(test_dataset, batch_size = 1, shuffle = False, pin_memory = True, num_workers = 8, prefetch_factor = 8)
print(f"GPU - Done.")

fm_model = TransformerModel(model_config = model_config, token_embed = token_embed, n_batch = len(meta_dict["batch_code"]), n_label = len(meta_dict["label_code"]), device = device)

print(f"GPU - Preloading lastest model'")
# load parameter from last train
fm_model.load_state_dict(state["model_state_dict"])
print(f"GPU - Done.")


# In[]
# visualization
import scvi
from umap import UMAP
# TODO: issue, for the classifier, should the masked input be used??
adata_embed = trainer.cell_embed(model = fm_model, dataloader = test_loader, multi_gpus = False)
adata_embed.obsm["latent_umap"] = UMAP(n_components = 2).fit_transform(adata_embed.X.toarray())
adata_embed.obs = adata_test_meta.obs.copy()


# In[]

if not os.path.exists(res_dir):
    os.makedirs(res_dir)

colormap =plt.cm.get_cmap("tab20")
# immune cell
fig = utils.plot_embeds(embed = adata_embed.obsm["latent_umap"], annos = adata_embed.obs[["final_annotation", "batch"]].astype("category"), markerscale = 15, figsize = (20, 17), s = 1, alpha = 0.4, colormap = colormap, label_inplace = False)
fig.tight_layout()
fig.savefig(res_dir + "embed_scIB_immune_all.png", bbox_inches = "tight")
# pancrease
fig = utils.plot_embeds(embed = adata_embed.obsm["latent_umap"], annos = adata_embed.obs[["celltype", "tech"]].astype("category"), markerscale = 5, figsize = (20, 17), s = 3, alpha = 0.4, colormap = colormap, label_inplace = False)
fig.tight_layout()
fig.savefig(res_dir + "embed_scIB_pancrease.png", bbox_inches = "tight")
# lung atlas
fig = utils.plot_embeds(embed = adata_embed.obsm["latent_umap"], annos = adata_embed.obs[["cell_type", "batch"]].astype("category"), markerscale = 5, figsize = (20, 17), s = 3, alpha = 0.7, colormap = colormap, label_inplace = False)
fig.tight_layout()
fig.savefig(res_dir + "embed_scIB_lung.png", bbox_inches = "tight")



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

# In[]


# %%
