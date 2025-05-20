# In[]
import torch
import anndata
import scanpy as sc
import numpy as np

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
from torch.utils import data
from torch.amp import autocast

sys.path.append("/net/csefiles/xzhanglab/zzhang834/LLM_KD/src")
sys.path.append("/net/csefiles/xzhanglab/zzhang834/LLM_KD/batch_encoding")

import data_utils
# from transformer_batch import TransformerModel, get_default_config
# import trainer_batch as trainer_batch

from screbound import TransformerModel, get_default_config
import trainer as trainer_batch

import utils
import batch_encode 
import warnings
warnings.filterwarnings("ignore")

# In[]
# -------------------------------------------------------------- 
#
# NOTE: Test embedding of baseline, one time computation
#
# --------------------------------------------------------------
PROJECT_DIR = "/net/csefiles/xzhanglab/zzhang834/LLM_KD/"
res_dir_scgpt = PROJECT_DIR + f"results/zs_annot/scGPT/test_embed/"
res_dir_scmulan = PROJECT_DIR + f"results/zs_annot/scMulan/test_embed/"
res_dir_uce = PROJECT_DIR + f"results/zs_annot/UCE/test_embed/"
res_dir_scfoundation = PROJECT_DIR + f"results/zs_annot/scFoundation/test_embed/"
res_dir_geneformer = PROJECT_DIR + f"results/zs_annot/geneformer/test_embed/"
if not os.path.exists(res_dir_scgpt):
    os.makedirs(res_dir_scgpt)
if not os.path.exists(res_dir_scmulan):
    os.makedirs(res_dir_scmulan)
if not os.path.exists(res_dir_uce):
    os.makedirs(res_dir_uce)
if not os.path.exists(res_dir_scfoundation):
    os.makedirs(res_dir_scfoundation)
if not os.path.exists(res_dir_geneformer):
    os.makedirs(res_dir_geneformer)

test_dir = "/net/csefiles/xzhanglab/shared/foundation_evaluation/data/testset_baseline_embed/"

for data_case in ["immune_all", "lung_atlas", "pancreas", "covid19"]:
    if data_case == "immune_all":
        adata_embed = anndata.read_h5ad(test_dir + "Immune_ALL_human_with_embeddings.h5ad")
        meta = adata_embed.obs[["batch", "final_annotation"]]
        meta.columns = ["batch", "label"]

    elif data_case == "lung_atlas":
        adata_embed = anndata.read_h5ad(test_dir + "Lung_atlas_public_with_embeddings.h5ad")
        meta = adata_embed.obs[["batch", "cell_type"]]
        meta.columns = ["batch", "label"]

    elif data_case == "pancreas":
        adata_embed = anndata.read_h5ad(test_dir + "human_pancreas_norm_complexBatch_with_embeddings.h5ad")
        meta = adata_embed.obs[["tech", "celltype"]]
        meta.columns = ["batch", "label"]

    elif data_case == "covid19":
        adata_embed = anndata.read_h5ad(test_dir + "covid19_with_embeddings.h5ad")
        meta = adata_embed.obs[["sample", "predicted.celltype.l1"]]
        meta.columns = ["batch", "label"]

    # if "X_scGPT" in adata_embed.obsm.keys():
    #     scgpt_embed = sp.csr_matrix(adata_embed.obsm["X_scGPT"])
    #     adata_embed_scgpt = anndata.AnnData(X = scgpt_embed, obs = meta)
    #     adata_embed_scgpt.write_h5ad(res_dir_scgpt + f"adata_embed_{data_case}.h5ad")

    # if "X_scMulan" in adata_embed.obsm.keys():
    #     scmulan_embed = sp.csr_matrix(adata_embed.obsm["X_scMulan"])
    #     adata_embed_scmulan = anndata.AnnData(X = scmulan_embed, obs = meta)
    #     adata_embed_scmulan.write_h5ad(res_dir_scmulan + f"adata_embed_{data_case}.h5ad")

    # if "X_uce" in adata_embed.obsm.keys():
    #     uce_embed = sp.csr_matrix(adata_embed.obsm["X_uce"])
    #     adata_embed_uce = anndata.AnnData(X = uce_embed, obs = meta)
    #     adata_embed_uce.write_h5ad(res_dir_uce + f"adata_embed_{data_case}.h5ad")

    # if "X_scFoundation" in adata_embed.obsm.keys():
    #     scfoundation_embed = sp.csr_matrix(adata_embed.obsm["X_scFoundation"])
    #     adata_embed_scfoundation = anndata.AnnData(X = scfoundation_embed, obs = meta)
    #     adata_embed_scfoundation.write_h5ad(res_dir_scfoundation + f"adata_embed_{data_case}.h5ad")

    if "X_geneformer" in adata_embed.obsm.keys():
        geneformer_embed = sp.csr_matrix(adata_embed.obsm["X_geneformer"])
        adata_embed_geneformer = anndata.AnnData(X = geneformer_embed, obs = meta)
        adata_embed_geneformer.write_h5ad(res_dir_geneformer + f"adata_embed_{data_case}.h5ad")

# %%
