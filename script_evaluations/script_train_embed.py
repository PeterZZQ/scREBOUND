# In[]
import torch
import anndata
import scanpy as sc
import numpy as np

from pathlib import Path
import sys, os

import numpy as np
import scipy.sparse as sp
import tqdm
import anndata
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

# packages for distributed training
import torch
import torch.nn as nn
from torch.utils import data

sys.path.append("/net/csefiles/xzhanglab/zzhang834/LLM_KD/src")
sys.path.append("/net/csefiles/xzhanglab/zzhang834/LLM_KD/batch_encoding")

import data_utils
# from transformer_stable import TransformerModel, get_default_config
# import trainer_stable as trainer_batch
from transformer_batch import TransformerModel, get_default_config
import trainer_batch as trainer_batch

import utils
# import eval
import batch_encode 
import warnings
warnings.filterwarnings("ignore")

def load_model(state, token_dict, label_dict, batch_dict, device, verbose = True):
    # create model configuration profile
    model_config = get_default_config()
    model_config.__dict__.update(state["model_config"])
    if verbose:
        for x, val in model_config.__dict__.items():
            print(x, end = ": ")
            print(val)

    # create model with profile
    model = TransformerModel(model_config = model_config, token_dict = token_dict, batch_dict = batch_dict, label_dict = label_dict, device = device).to(model_config.precision)

    # load model params
    if verbose:
        print("Load parameters...")
    filtered_state_dict = {k: v for k, v in state["model_state_dict"].items() if k in model.state_dict()}
    # Load the filtered state dictionary into the model
    model.load_state_dict(filtered_state_dict, strict=False)
    if verbose:
        print("Done.")

    return model

PROJECT_DIR = "/net/csefiles/xzhanglab/zzhang834/LLM_KD/"
data_dir = "/net/csefiles/xzhanglab/zzhang834/hs_download/"

# In[]
# --------------------------------------------------------------------------
#
# Loading the trained model
#
# --------------------------------------------------------------------------

data_utils.set_seed(0)

n_mgene = 256
device = torch.device("cuda")
print(f"GPU - Using device: {device}")

# NOTE: save in localscratch for faster memory access

batch_name = "level2"
# old model best
# model_name = f"cp_contrcb1_6_512_256_encbg_level2_1"
# model_dir = PROJECT_DIR + f"checkpoint/model_6_256/{model_name}.pth"

# new batch encoder model
# 1. batch encoder
# model_name = f"cp_6_512_256_concat_full_1"
# 2. batch encoder + contrastive
# model_name = f"cp_contrcb1_6_512_256_concat_full_1"
# 3. batch encoder + restart
# model_name = f"cp_6_512_256_concat_rawrestart_1"
# 4. batch encoder + contrastive + restart
# model_name = f"cp_contrcb1_mlm10_dyn_6_512_256_concat_rawrestart_1"
# model_dir = PROJECT_DIR + f"checkpoint/model_6_256_concat_full/{model_name}.pth"

# new vanilla model
# 1. vanilla
# model_name = f"cp_6_512_256_1"
# 2. vanilla + contrastive
# model_name = f"cp_contrcb1_mlm2_dyn_6_512_256_1"
# TODO: 3. vanilla + restart
model_name = f"cp_6_512_256_rawrestart_1"
# TODO: 4. vanilla + contrastive + restart
model_name = f"cp_contrcb1_mlm10_dyn_6_512_256_rawrestart_1"
model_dir = PROJECT_DIR + f"checkpoint/model_6_256_nobatch/{model_name}.pth"

# read in the key files
state = torch.load(model_dir, weights_only = False)
token_dict = torch.load(data_dir + f"meta_data/gene_embed_meta{n_mgene}_gpool.pt", weights_only = False)
label_dict = torch.load(data_dir + "meta_data/label_dict.pt", weights_only = False)
# ----------------------------------------------------------------------
# batch_dict = torch.load(data_dir + f"meta_data/batch_dict_batch_{batch_name}.pt")
# batch_dict["cats"] = batch_dict["cats"].drop(["prop_mito", "raw_mean_nnz", "nnz", "libsize"], axis = 1)
# batch_dict["n_cat_list"] = batch_dict["n_cat_list"][4:]

# # make value continuous
# batch_feats = pd.read_csv(data_dir + f"meta_data/feature_batch_level2_filter.csv", index_col = 0)
# batch_dict["cats"] = batch_feats[batch_dict["cats"].columns]

# new full list
batch_dict = torch.load(data_dir + f"meta_data/batch_dict_{batch_name}_10.pt", weights_only = False)

# # new adaptive
# batch_dict = torch.load(data_dir + f"meta_data/batch_dict_{batch_name}_expr10.pt", weights_only = False)
# ----------------------------------------------------------------------
batch_dict["cats"] = torch.tensor(batch_dict["cats"].values)

model_pretrain = load_model(state = state, token_dict = token_dict, label_dict = label_dict, batch_dict = batch_dict, device = device)

model_pretrain.model_config.batch_size = 512

# In[]
# ------------------------------------------------------------------------
#
# Extract embedding
# 
# ------------------------------------------------------------------------
# select 10 percent of cells, since the data are already permuted
num_partitions = 1
partition_size = 1000000
min_chunksize = 64

dataset_dict = {"DIR": data_dir + "permuted/",
                "num_partitions": num_partitions,
                "data_prefix": "counts",
                "meta_prefix": "obs",
                "batch_dict": batch_dict,
                "label_colname": "label_id",
                "batch_colname": "batch_" + batch_name + "_id"}


# In[]
# NOTE: our model embedding
res_dir = PROJECT_DIR + f"results/zs_annot/{model_name}/train_embed/"
if not os.path.exists(res_dir):
    os.makedirs(res_dir)

train_dataset = data_utils.sc_partition(data_path = dataset_dict["DIR"], batch_feats = dataset_dict["batch_dict"], min_chunksize = min_chunksize, normalize = model_pretrain.model_config.lognorm_data)
for partition_idx in range(dataset_dict["num_partitions"]):
    train_dataset.load_partition(idx = partition_idx, label_colname = dataset_dict["label_colname"], batch_colname = dataset_dict["batch_colname"], data_prefix = dataset_dict["data_prefix"], meta_prefix = dataset_dict["meta_prefix"])
    train_loader = data.DataLoader(train_dataset, batch_size = model_pretrain.model_config.batch_size//min_chunksize, shuffle = False, pin_memory = True, num_workers = 8, prefetch_factor = 8)
    adata_partition = trainer_batch.cell_embed(model = model_pretrain, dataloader = train_loader, multi_gpus = False)

    adata_partition.write_h5ad(res_dir + f"embed_model_{partition_idx}.h5ad")


# In[]
# NOTE: scGPT embedding
res_dir = PROJECT_DIR + f"results/zs_annot/scGPT/train_embed/"
if not os.path.exists(res_dir):
    os.makedirs(res_dir)
scgpt_dir = "/net/csefiles/xzhanglab/shared/foundation_evaluation/data/trainset_select/scGPT/"

for partition_idx in range(dataset_dict["num_partitions"]):
    meta_cell_idx = pd.read_parquet(dataset_dict["DIR"] + f"{dataset_dict["meta_prefix"]}_{partition_idx}_batchcode.parquet")
    adata = anndata.AnnData(X = np.load(scgpt_dir + f"partition_{partition_idx}.npy"))
    adata.obs.index = meta_cell_idx.index.values
    adata.obs["soma_joinid"] = meta_cell_idx["soma_joinid"].values.squeeze()
    adata.obs["label_id"] = meta_cell_idx[dataset_dict["label_colname"]].values.squeeze()
    adata.obs["batch_id"] = meta_cell_idx[dataset_dict["batch_colname"]].values.squeeze()

    adata.write_h5ad(res_dir + f"embed_model_{partition_idx}.h5ad")

# In[]
# NOTE: UCE embedding, already preprocessed
res_dir = PROJECT_DIR + f"results/zs_annot/UCE/train_embed/"
if not os.path.exists(res_dir):
    os.makedirs(res_dir)
uce_dir = "/net/csefiles/xzhanglab/shared/foundation_evaluation/data/trainset_select/UCE/"

for partition_idx in range(dataset_dict["num_partitions"]):
    adata = anndata.read_h5ad(uce_dir + f"partition_{partition_idx}.h5ad")
    adata_embed = anndata.AnnData(X = sp.csr_matrix(adata.obsm["X_uce"]))
    adata_embed.obs = adata.obs[["soma_joinid", dataset_dict["label_colname"], dataset_dict["batch_colname"]]]
    adata_embed.write_h5ad(res_dir + f"embed_model_{partition_idx}.h5ad")


# In[]
# NOTE: scmulan train embedding
res_dir = PROJECT_DIR + f"results/zs_annot/scMulan/train_embed/"
if not os.path.exists(res_dir):
    os.makedirs(res_dir)
scmulan_dir = "/net/csefiles/xzhanglab/shared/foundation_evaluation/data/trainset_select/scMulan/"

for partition_idx in range(dataset_dict["num_partitions"]):
    meta_cell_idx = pd.read_parquet(dataset_dict["DIR"] + f"{dataset_dict["meta_prefix"]}_{partition_idx}_batchcode.parquet")
    adata = anndata.AnnData(X = np.load(scmulan_dir + f"partition_{partition_idx}.npy"))
    adata.obs.index = meta_cell_idx.index.values
    adata.obs["soma_joinid"] = meta_cell_idx["soma_joinid"].values.squeeze()
    adata.obs["label_id"] = meta_cell_idx[dataset_dict["label_colname"]].values.squeeze()
    adata.obs["batch_id"] = meta_cell_idx[dataset_dict["batch_colname"]].values.squeeze()

    adata.write_h5ad(res_dir + f"embed_model_{partition_idx}.h5ad")


# In[]
# NOTE: scfoundation train embedding
res_dir = PROJECT_DIR + f"results/zs_annot/scFoundation/train_embed/"
if not os.path.exists(res_dir):
    os.makedirs(res_dir)
scfoundation_dir = "/net/csefiles/xzhanglab/shared/foundation_evaluation/data/trainset_select/scFoundation/"

for partition_idx in range(dataset_dict["num_partitions"]):
    meta_cell_idx = pd.read_parquet(dataset_dict["DIR"] + f"{dataset_dict["meta_prefix"]}_{partition_idx}_batchcode.parquet")
    adata = anndata.read_h5ad(scfoundation_dir + f"partition_{partition_idx}.h5ad")
    adata_embed = anndata.AnnData(X = adata.obsm["X_scFoundation"].toarray())
    adata_embed.obs = adata.obs
    adata_embed.write_h5ad(res_dir + f"embed_model_{partition_idx}.h5ad")

# In[]
# ------------------------------------------------------------------------
#
# Transform train partition file into anndata
# 
# ------------------------------------------------------------------------
# train_dataset = data_utils.sc_partition(data_path = dataset_dict["DIR"], batch_feats = dataset_dict["batch_dict"], min_chunksize = min_chunksize, normalize = model_pretrain.model_config.lognorm_data)
# for partition_idx in tqdm.tqdm(range(dataset_dict["num_partitions"])):
#     train_dataset.load_partition(idx = partition_idx, label_colname = dataset_dict["label_colname"], batch_colname = dataset_dict["batch_colname"], data_prefix = dataset_dict["data_prefix"], meta_prefix = dataset_dict["meta_prefix"])
#     counts_csr = sp.csr_matrix((train_dataset.expr_data, train_dataset.expr_indices, train_dataset.expr_indptr), shape=(train_dataset.ncells, train_dataset.ngenes))
#     meta_cells = pd.read_parquet(dataset_dict["DIR"] + f"{dataset_dict["meta_prefix"]}_{partition_idx}_batchcode.parquet")
#     vars = pd.read_csv(dataset_dict["DIR"] + "var.csv", index_col = 0)
#     vars.index = vars["feature_name"].values

#     adata = anndata.AnnData(X = counts_csr, obs = meta_cells, var = vars)
#     adata.write_h5ad(PROJECT_DIR + "dataset/trainset_select/" + f"anndata_partition_{partition_idx}.h5ad")
#     del counts_csr, adata

