# In[]
import torch
from sklearn.neighbors import KNeighborsClassifier
import scipy.sparse as sp
import data_utils
import anndata
import scanpy as sc
import numpy as np

# In[]
# TODO: Given adata,
# 1. transform into meta-gene counts
# 3. transform meta-gene counts into cell sentence


def preprocess_anndata(adata, meta_embed, gene_embed_key = "esm2"):
    """
    Group the genes into meta-genes in adata according to meta-gene embedding
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
# load the cell meta-info
meta_dict = torch.load(f"/localscratch/ziqi/localscratch_tempdata/cellxgene/meta_{n_mgene}_permu.pt")

# knn classifier meta-gene embedding
meta_embed = gene_embed_dict["meta_embed"]
# the gene embed here only include the preprocessed 4k genes in training data
gene_embed = gene_embed_dict["gene_embed"]
gene2meta_labels = gene_embed_dict["labels"]

# gene embed full
gene_embed_full = torch.load("/project/zzhang834/llm_dataset/proteome/embeddings/Homo_sapiens.GRCh38.gene_symbol_to_embedding_ESM2.pt", weights_only = False)

# In[]
adata_test = anndata.read_h5ad("/project/zzhang834/llm_dataset/CellXGeneCZI/data_download/blood/partition_10.h5ad")
adata_test.layers["counts"] = adata_test.X.copy()

adata_test.var.index = adata_test.var["feature_id"].values
adata_test = adata_test[:, [x for x in gene_embed_dict["gene_embed"].keys()]]

counts_meta = []
for label in range(n_mgene):
    counts_meta.append(adata_test.X[:, gene_embed_dict["labels"] == label].sum(axis = 1))
counts_meta = sp.csr_matrix(np.hstack(counts_meta))

adata_test_meta = anndata.AnnData(X = counts_meta, obs = adata_test.obs)
sc.pp.normalize_total(adata_test_meta, target_sum = 10e4, key_added = "libsize")
sc.pp.log1p(adata_test_meta)

# In[]
# Mapping is changed, which makes the result worse
# # preprocessing, use gene embed full
# gene_overlap = np.intersect1d(adata_test.var["feature_name"].values, np.array([x for x in gene_embed_full.keys()]))
# print("number of overlapped genes: {:d}".format(len(gene_overlap)))
# adata_test = adata_test[:, adata_test.var["feature_name"].isin(gene_overlap)].copy()

# sc.pp.filter_genes(adata_test, min_cells = int(0.01 * adata_test.shape[0]))
# sc.pp.normalize_total(adata_test)
# sc.pp.log1p(adata_test)

# sc.pp.highly_variable_genes(adata_test, n_top_genes = 4000)
# adata_test = adata_test[:, adata_test.var["highly_variable"]]
# adata_test.varm["esm2"] = torch.vstack([gene_embed_full[x] for x in adata_test.var["feature_name"].values]).numpy()

# adata_test_meta = preprocess_anndata(adata = adata_test, meta_embed = meta_embed, gene_embed_key = "esm2")

# In[]
expr_sent, feat_sent = data_utils.tokenize_expr_para(adata_test_meta.X, njobs = 16, nchunks = 16, npads = n_mgene + 1)

fp = np.memmap(f"temp_test_expr_sent_{n_mgene}.npz", dtype='float32', mode='w+', shape=(adata_test_meta.shape[0], n_mgene + 1))
fp[:] = expr_sent.toarray()[:]
fp.flush()
fp = np.memmap(f"temp_test_feat_sent_{n_mgene}.npz", dtype='int32', mode='w+', shape=(adata_test_meta.shape[0], n_mgene + 1))
fp[:] = feat_sent.toarray()[:]
fp.flush()

# In[]
# alternative dataset from anndata, TODO: need to deal with the case where no batch effect is needed
import pandas as pd
label_code = {val: idx for idx, val in enumerate(meta_dict["label_code"])}
labels = np.where(adata_test.obs["cell_type_ontology_term_id"].isin(label_code), adata_test.obs["cell_type_ontology_term_id"].map(label_code), -1)
batch_code = {val: idx for idx, val in enumerate(meta_dict["batch_code"])}
batchs = np.where(adata_test.obs["dataset_id"].isin(batch_code), adata_test.obs["dataset_id"].map(batch_code), -1)
test_dataset = data_utils.sc_dataset_chunk(expr_path = f"./temp_test_expr_sent_{n_mgene}.npz", gene_path = f"temp_test_feat_sent_{n_mgene}.npz", ncells = adata_test_meta.shape[0], npads = n_mgene + 1, labels = labels, batches = batchs)

# In[]
from pathlib import Path
import sys, os

import numpy as np
import tqdm

# packages for distributed training
import torch
from torch.utils import data


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
# function
data_utils.set_seed(3)

# Define the device
device = torch.device("cuda:1")
print(f"GPU - Using device: {device}")
# Load the dataset
print(f"GPU - Loading dataset...")

n_mgene = 256
# NOTE: save in localscratch for faster memory access
# data_dir = Path(f"/project/zzhang834/LLM_KD/dataset/cellxgene")
data_dir = Path(f"/localscratch/ziqi/localscratch_tempdata/cellxgene")
# load the token embedding
token_embed = torch.load(data_dir / f"token_embed_{n_mgene}.pt")
# load the cell meta-info
meta_dict = torch.load(data_dir / f"meta_{n_mgene}_permu.pt")


model_config = get_default_config()
batch_size = 512
lr = 0.5e-5 * (batch_size/32)
model_config.__dict__.update({"batch_size": batch_size,
                                "n_epoch": 1,
                                "lr": lr, # important for hyper-parameter tuning
                                "n_warmup_stp_lr": 4000, # important for hyper-parameter tuning, 5-10% of total steps
                                "d_embed": 512,
                                "n_head": 8,
                                "d_hidden": 2048, 
                                "n_layer": 4,
                                "d_output": 64,
                                "dropout": 0.05, # important for hyper-parameter tuning
                                "mask_prob": 0.4, # important for hyper-parameter tuning
                                "lamb_kd": 0.0,
                                "lamb_sup": 1.0,
                                "sup_type": "contrastive",
                                "sample_mlm": False,
                                "mlm_include_zero": False,
                                "pretrain_path":  "/project/zzhang834/LLM_KD/checkpoint_predfull/checkpoint_0.98_contr_1.pth",
                                "checkpoint_path": "/project/zzhang834/LLM_KD/checkpoint_predfull/",
                                "checkpoint_prefix": None
                                })

test_loader = data.DataLoader(test_dataset, batch_size = 1, shuffle = False, pin_memory = True, num_workers = 8, prefetch_factor = 8)
print(f"GPU - Done.")

# model hyper-parameters
fm_model = TransformerModel(model_config = model_config, token_embed = token_embed, n_batch = len(meta_dict["batch_code"]), n_label = len(meta_dict["label_code"]), device = device)
# wrap model into multi-gpus setting
# fm_model = DistributedDataParallel(fm_model, device_ids=[local_rank])

print(f"GPU - Preloading lastest model'")
# load parameter from last train
state = torch.load(model_config.pretrain_path)
fm_model.load_state_dict(state["model_state_dict"])


# In[]
# visualization
import scvi

# TODO: issue, for the classifier, should the masked input be used??
adata_test2 = trainer.cell_embed(model = fm_model, dataloader = test_loader, multi_gpus = False)
adata_test2.obsm["latent_umap"] = scvi.model.utils.mde(adata_test2.X.toarray(), accelerator = "gpu", seed = 0)

# In[]
import anndata
import seaborn as sns
import matplotlib.pyplot as plt
colormap =plt.cm.get_cmap("tab20b")
               
fig = utils.plot_embeds(embed = adata_test2.obsm["latent_umap"], annos = adata_test2.obs[["labels", "batchs"]].astype("category"), markerscale = 5, figsize = (20, 17), s = 1, alpha = 0.4, colormap = colormap, label_inplace = True)
fig.tight_layout()

# In[]


# %%
