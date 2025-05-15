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


# In[]
# function
data_utils.set_seed(0)
device = torch.device("cuda")
PROJECT_DIR = "/net/csefiles/xzhanglab/zzhang834/LLM_KD/"

# old data directory: for old model testing
# data_dir = "/net/csefiles/xzhanglab/zzhang834/hs_download/"
data_dir = "/data/zzhang834/hs_download/"
# stable model model
# model_name = "stable_4_512_level2"

# model_name = "cp_contrcb1_4_512_128_encbg_level2_1"
# model_dir = PROJECT_DIR + f"checkpoint/model_128/{model_name}.pth"

# model_name = "cp_contrcb1_4_512_256_encbg_level2_1"
# model_dir = PROJECT_DIR + f"checkpoint/model_4_256/{model_name}.pth"

# model_name = "cp_contrcb1_6_512_256_encbg_level2_1"
# model_dir = PROJECT_DIR + f"checkpoint/model_6_256/{model_name}.pth"

model_name = "cp_6_512_256_meta_1"
model_dir = PROJECT_DIR + f"screbound/{model_name}.pth"

batch_name = "level2"
state = torch.load(model_dir, weights_only = False)
token_dict = torch.load(data_dir + f"meta_data/gene_embed_meta256_gpool.pt", weights_only = False)
label_dict = torch.load(data_dir + f"meta_data/label_dict.pt", weights_only = False)
batch_dict = torch.load(data_dir + f"meta_data/batch_dict_{batch_name}_10.pt", weights_only = False)
batch_dict["cats"] = torch.tensor(batch_dict["cats"].values)

model_pretrain = load_model(state = state, token_dict = token_dict, label_dict = label_dict, batch_dict = batch_dict, device = device)
 
res_dir = PROJECT_DIR + f"results/checkpoint/{model_name}/"
if not os.path.exists(res_dir):
    os.makedirs(res_dir)

# In[]
# selection of test dataset
# data_case = "immune_all"
data_case = "pancreas"
# data_case = "lung_atlas"
# data_case = "covid19"
# data_case = "GBM"

if data_case != "covid19":
    EVAL_DATA_DIR = "/net/csefiles/xzhanglab/zzhang834/LLM_KD/dataset/scIB/"
else:
    EVAL_DATA_DIR = "/net/csefiles/xzhanglab/zzhang834/LLM_KD/dataset/evaluation_datasets/"

# gene_info = token_dict["labels"]
# gene_list = gene_info["feature_name"].values
# if data_case == "immune_all":
#     adata_test = anndata.read_h5ad(EVAL_DATA_DIR + "Immune_ALL_human.h5ad")
#     adata_test.X = adata_test.layers["counts"].copy() # checked: counts are raw
#     adata_test.obs["batch_id"], batch_code = pd.factorize(adata_test.obs["batch"])
#     adata_test.obs["label"] = adata_test.obs["final_annotation"]

# elif data_case == "pancreas":
#     adata_test = anndata.read_h5ad(EVAL_DATA_DIR + "human_pancreas_norm_complexBatch.h5ad")
#     adata_test.X = sp.csr_matrix(adata_test.layers["counts"])
#     adata_test.obs["batch_id"], batch_code = pd.factorize(adata_test.obs["tech"])
#     adata_test.obs["label"] = adata_test.obs["celltype"]

# elif data_case == "lung_atlas":
#     adata_test = anndata.read_h5ad(EVAL_DATA_DIR + "Lung_atlas_public.h5ad")
#     adata_test.X = adata_test.layers["counts"].copy()
#     adata_test.obs["batch_id"], batch_code = pd.factorize(adata_test.obs["batch"])
#     adata_test.obs["label"] = adata_test.obs["cell_type"]

# elif data_case == "covid":
#     adata_test = anndata.read_h5ad(EVAL_DATA_DIR2 + "covid19_aligned.h5ad")
#     adata_test.obs["batch_id"], batch_code = pd.factorize(adata_test.obs["sample"])
#     adata_test.obs["label"] = adata_test.obs["predicted.celltype.l1"]

# elif data_case == "GBM":
#     adata_test = anndata.read_h5ad(EVAL_DATA_DIR2 + "GBM_Fig4.h5ad")
#     adata_test.obs["batch_id"], batch_code = pd.factorize(adata_test.obs["sample_id"])
#     adata_test.obs["label"] = adata_test.obs["mstatus"]
#     adata_test.var_names_make_unique()

# adata_test = data_utils.align_genes(adata_test, gene_list)
# adata_test.layers["counts"] = adata_test.X.copy()
# adata_test.write_h5ad(EVAL_DATA_DIR + data_case + "_aligned.h5ad")

# In[]
# NOTE: aligned files are pre-calculated using align function in data_utils
adata_test = anndata.read_h5ad(EVAL_DATA_DIR + f"{data_case}_aligned.h5ad")

# NOTE: the batch features are not important for cell embedding
batch_features_digitize = None

print("create dataloader...")
label_colname = None
batch_colname = "batch_id"
test_dataset = data_utils.sc_dataset_anndata(adata = adata_test, gene_list = None, batch_feats = {"conts": None, "cats": batch_features_digitize},
                                             label_colname = label_colname, batch_colname = batch_colname, batch_size = 128, normalize = model_pretrain.model_config.lognorm_data)
test_loader = data.DataLoader(test_dataset, batch_size = 1, shuffle = False, pin_memory = True, num_workers = 8, prefetch_factor = 8)


# In[]
# --------------------------------------------------------------------------------------------------------------
#
# NOTE: calculate the embedding
#
# --------------------------------------------------------------------------------------------------------------

adata_embed = trainer_batch.cell_embed(model = model_pretrain, dataloader = test_loader, multi_gpus = False)
adata_embed.obs = adata_test.obs.copy()
adata_embed.obsm["latent"] = adata_embed.X.copy()

sc.pp.pca(adata_embed, n_comps = 30)
sc.pp.neighbors(adata_embed, n_neighbors = 15, use_rep = "latent")
sc.tl.umap(adata_embed, min_dist = 0.3)
adata_embed.obsm[f"X_umap_latent"] = adata_embed.obsm["X_umap"].copy()
del adata_embed.obsm["X_umap"]
if "contr" in adata_embed.obsm.keys():
    sc.pp.neighbors(adata_embed, n_neighbors = 15, use_rep = "contr")
    sc.tl.umap(adata_embed, min_dist = 0.3)
    adata_embed.obsm[f"X_umap_contr"] = adata_embed.obsm["X_umap"].copy()
    del adata_embed.obsm["X_umap"]

# In[]
# adata_embed = anndata.read_h5ad(res_dir + f"adata_embed_{data_case}.h5ad")
colormap =plt.cm.get_cmap("tab20")
use_rep = "latent"
if data_case == "lung_atlas":
    # Only lung_atlas has the patient group
    annos = adata_embed.obs[["label", "batch_id", "patientGroup"]].astype("category")
    fig = utils.plot_embeds(embed = adata_embed.obsm[f"X_umap_{use_rep}"], annos = annos, markerscale = 15, figsize = (12, 7), s = 1, alpha = 0.4, colormap = colormap, label_inplace = False)
    fig.tight_layout()
    # fig.savefig(res_dir + f"{use_rep}_embed_scIB_{data_case}.png", bbox_inches = "tight")

    fig = utils.plot_by_batch(adata_embed.obsm[f"X_umap_{use_rep}"], annos = np.array([x for x in adata_embed.obs["label"].values]),
                            batches = np.array([x for x in adata_embed.obs["patientGroup"].values]), markerscale = 15, figsize = (12, 5), s = 1, alpha = 0.4, colormap = colormap, label_inplace = False)
    fig.tight_layout()
    # fig.savefig(res_dir + f"{use_rep}_embed_scIB_{data_case}_patient.png", bbox_inches = "tight")

    adata_embed_ctrl = adata_embed[adata_embed.obs["patientGroup"] == "Ctrl"]
    adata_embed_parenchyma = adata_embed[adata_embed.obs["patientGroup"] == "Parenchyma"]
    adata_embed_nan = adata_embed[adata_embed.obs["patientGroup"] == "nan"]
    fig = utils.plot_embeds(embed = adata_embed_ctrl.obsm[f"X_umap_{use_rep}"], annos = adata_embed_ctrl.obs[["label", "batch_id"]].astype("category"), markerscale = 15, figsize = (12, 7), s = 1, alpha = 0.4, colormap = colormap, label_inplace = False)
    fig.tight_layout()
    # fig.savefig(res_dir + f"{use_rep}_embed_scIB_{data_case}_ctrl.png")
    fig = utils.plot_embeds(embed = adata_embed_parenchyma.obsm[f"X_umap_{use_rep}"], annos = adata_embed_parenchyma.obs[["label", "batch_id"]].astype("category"), markerscale = 15, figsize = (12, 7), s = 1, alpha = 0.4, colormap = colormap, label_inplace = False)
    fig.tight_layout()
    # fig.savefig(res_dir + f"{use_rep}_embed_scIB_{data_case}_parenchyma.png")
    fig = utils.plot_embeds(embed = adata_embed_nan.obsm[f"X_umap_{use_rep}"], annos = adata_embed_nan.obs[["label", "batch_id"]].astype("category"), markerscale = 15, figsize = (12, 7), s = 1, alpha = 0.4, colormap = colormap, label_inplace = False)
    fig.tight_layout()
    # fig.savefig(res_dir + f"{use_rep}_embed_scIB_{data_case}_nan.png")

else:
    if data_case == "immune_all":
        figsize = (15, 7)
        annos = adata_embed.obs[["label", "batch_id"]].astype("category")
    elif data_case == "pancreas":
        figsize = (12, 7)
        annos = adata_embed.obs[["label", "batch_id"]].astype("category")
    elif data_case == "covid19":
        figsize = (12, 7)
        annos = adata_embed.obs[["label", "batch_id", "dataset"]].astype("category")
        colormap = None
    elif data_case == "GBM":
        figsize = (12, 7)
        annos = adata_embed.obs[["label", "batch_id", "treatment"]].astype("category")
        colormap = None                
        
    fig = utils.plot_embeds(embed = adata_embed.obsm[f"X_umap_{use_rep}"], annos = annos, markerscale = 15, figsize = figsize, s = 1, alpha = 0.4, colormap = colormap, label_inplace = False)
    fig.tight_layout()
    # fig.savefig(res_dir + f"{use_rep}_embed_scIB_{data_case}.png", bbox_inches = "tight")

assert False

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
