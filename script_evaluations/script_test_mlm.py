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
from torch.utils import data
from torch.amp import autocast

sys.path.append("/net/csefiles/xzhanglab/zzhang834/LLM_KD/src")
sys.path.append("/net/csefiles/xzhanglab/zzhang834/LLM_KD/batch_encoding")

import data_utils
from transformer_batch import TransformerModel, get_default_config
# from transformer_stable import TransformerModel, get_default_config

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


def evaluate_mlm(model, dataloader, mask_prob):
    model.model_config.mask_prob = mask_prob
    # need to temperarily remove the discriminator loss, no label for mlm test
    sup_type = model.model_config.sup_type
    model.model_config.sup_type = None
    # need to remove the batch_factor_mask too
    mask_batchfactor = model.model_config.mask_batchfactor
    model.model_config.mask_batchfactor = False
    
    # NOTE: training loop
    model.eval()
    with torch.no_grad():
        val_loss, val_loss_mlm, val_loss_metric, val_loss_mincut, val_loss_ortho = 0.0, 0.0, 0.0, 0.0, 0.0
        for data_sample in tqdm.tqdm(dataloader, desc=f"Evaluation"):
            with autocast(device_type="cuda", dtype = torch.bfloat16, enabled = model.model_config.use_flashatten):
                if model.model_config.batch_enc == "onehot":
                    del data_sample["batch"]

                loss, loss_item = trainer_batch.infer_databatch(model, data_sample, multigpus = False)                            
            val_loss += loss.item()
            val_loss_mlm += loss_item["mlm"]
            val_loss_metric += loss_item["metric"]
            val_loss_mincut += loss_item["mincut"]
            val_loss_ortho += loss_item["ortho"]
        # log the values
        val_loss /= len(dataloader)
        val_loss_mlm /= len(dataloader)
        val_loss_metric /= len(dataloader)
        val_loss_mincut /= len(dataloader)
        val_loss_ortho /= len(dataloader)

        print(f"Val Loss (TOTAL): {val_loss:.4f}, Val Loss (MLM): {val_loss_mlm:.4f}, Val Loss (METRIC): {val_loss_metric:.4f}, Val Loss (MINCUT): {val_loss_mincut:.4f}, Val Loss (ORTHO): {val_loss_ortho:.4f}")

    # model.model_config.use_discriminator = use_discriminator
    model.model_config.sup_type = sup_type
    model.model_config.mask_batchfactor = mask_batchfactor

    return val_loss, val_loss_mlm, val_loss_metric, val_loss_mincut, val_loss_ortho


# In[]
# function
data_utils.set_seed(0)

device = torch.device("cuda")
print(f"GPU - Using device: {device}")

# NOTE: save in localscratch for faster memory access
PROJECT_DIR = "/net/csefiles/xzhanglab/zzhang834/LLM_KD/"
data_dir = "/net/csefiles/xzhanglab/zzhang834/hs_download/"
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
# model_name = f"cp_6_512_256_rawrestart_1"
# TODO: 4. vanilla + contrastive + restart
model_name = f"cp_contrcb1_mlm10_dyn_6_512_256_rawrestart_1"
model_dir = PROJECT_DIR + f"checkpoint/model_6_256_nobatch/{model_name}.pth"

state = torch.load(model_dir, weights_only = False)

token_dict = torch.load(data_dir + f"meta_data/gene_embed_meta256_gpool.pt", weights_only = False)
label_dict = torch.load(data_dir + f"meta_data/label_dict.pt", weights_only = False)
# ------------------------------------------------------------------------------------------------------------------------------------
# batch_dict = torch.load(data_dir + f"meta_data/batch_dict_batch_{batch_name}.pt", weights_only = False)
# # drop the stats features (not very useful)
# batch_dict["cats"] = batch_dict["cats"].drop(["prop_mito", "raw_mean_nnz", "nnz", "libsize"], axis = 1)
# batch_dict["n_cat_list"] = batch_dict["n_cat_list"][4:]

# make value continuous
# batch_feats = pd.read_csv(data_dir + f"meta_data/feature_batch_level2_filter.csv", index_col = 0)
# batch_dict["cats"] = batch_feats[batch_dict["cats"].columns]

# new full list
batch_dict = torch.load(data_dir + f"meta_data/batch_dict_{batch_name}_10.pt", weights_only = False)

# # new adaptive
# batch_dict = torch.load(data_dir + f"meta_data/batch_dict_{batch_name}_expr10.pt", weights_only = False)
# ------------------------------------------------------------------------------------------------------------------------------------
batch_dict["cats"] = torch.tensor(batch_dict["cats"].values)

model_pretrain = load_model(state = state, token_dict = token_dict, label_dict = label_dict, batch_dict = batch_dict, device = device)

res_dir = PROJECT_DIR + f"results/checkpoint/{model_name}/"
if not os.path.exists(res_dir):
    os.makedirs(res_dir)

# In[]
# selection of test dataset
data_case = "immune_all"
data_case = "pancreas"
data_case = "lung_atlas"
data_case = "covid19"
# data_case = "GBM"

if data_case != "covid19":
    EVAL_DATA_DIR = "/net/csefiles/xzhanglab/zzhang834/LLM_KD/dataset/scIB/"
else:
    EVAL_DATA_DIR = "/net/csefiles/xzhanglab/zzhang834/LLM_KD/dataset/evaluation_datasets/"

adata_test = anndata.read_h5ad(EVAL_DATA_DIR + f"{data_case}_aligned.h5ad")

# NOTE: Calculate the batch factor
if model_pretrain.model_config.batch_enc is not None:
    print("create batch factor...")
    batch_features = batch_encode.construct_batch_feats(adata = adata_test, use_mito = True, use_tech = False, use_nmeasure = False)
    
    # ------------------------------------------------------------------------------------------------------------------------------------
    # batch_features_digitize, num_bucket = batch_encode.tokenize_batch_feats(batch_features, use_mito = True, use_tech = False, use_nmeasure = False, expr_binsize = 1)
    # num_bucket = num_bucket[4:]
    # # drop the stats features (not very useful)
    # batch_features_digitize = batch_features_digitize.drop(["prop_mito", "raw_mean_nnz", "nnz", "libsize"], axis = 1)
    # # make value continuous
    # batch_features_digitize = batch_features.drop(["prop_mito", "raw_mean_nnz", "nnz", "libsize"], axis = 1)

    # NOTE: adaptive manner
    # batch_features_digitize, max_vals = batch_encode.tokenize_batch_feats(batch_features, max_vals = batch_dict["cat_maxvals"], nbins = 10, only_genes = True)    

    # NOTE: adaptive manner
    batch_features_digitize, max_vals = batch_encode.tokenize_batch_feats(batch_features, max_vals = batch_dict["cat_maxvals"], nbins = 10, only_genes = False)    
    # ------------------------------------------------------------------------------------------------------------------------------------
    batch_features_digitize = torch.tensor(batch_features_digitize.values, dtype = torch.float32)

else:
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

adata_embed.write_h5ad(res_dir + f"adata_embed_{data_case}.h5ad")
# In[]
adata_embed = anndata.read_h5ad(res_dir + f"adata_embed_{data_case}.h5ad")

sc.pp.neighbors(adata_embed, n_neighbors = 30, use_rep = "latent")
sc.tl.umap(adata_embed, min_dist = 0.5)
adata_embed.obsm[f"X_umap_latent"] = adata_embed.obsm["X_umap"].copy()
del adata_embed.obsm["X_umap"]

use_rep = "latent"
colormap =plt.cm.get_cmap("tab20")

if data_case == "lung_atlas":
    # Only lung_atlas has the patient group
    annos = adata_embed.obs[["label", "batch_id", "patientGroup"]].astype("category")
    fig = utils.plot_embeds(embed = adata_embed.obsm[f"X_umap_{use_rep}"], annos = annos, markerscale = 15, figsize = (12, 7), s = 1, alpha = 0.4, colormap = colormap, label_inplace = False)
    fig.tight_layout()
    fig.savefig(res_dir + f"{use_rep}_embed_scIB_{data_case}.png", bbox_inches = "tight")

    fig = utils.plot_by_batch(adata_embed.obsm[f"X_umap_{use_rep}"], annos = np.array([x for x in adata_embed.obs["label"].values]),
                            batches = np.array([x for x in adata_embed.obs["patientGroup"].values]), markerscale = 15, figsize = (12, 5), s = 1, alpha = 0.4, colormap = colormap, label_inplace = False)
    fig.tight_layout()
    fig.savefig(res_dir + f"{use_rep}_embed_scIB_{data_case}_patient.png", bbox_inches = "tight")

    adata_embed_ctrl = adata_embed[adata_embed.obs["patientGroup"] == "Ctrl"]
    adata_embed_parenchyma = adata_embed[adata_embed.obs["patientGroup"] == "Parenchyma"]
    adata_embed_nan = adata_embed[adata_embed.obs["patientGroup"] == "nan"]
    fig = utils.plot_embeds(embed = adata_embed_ctrl.obsm[f"X_umap_{use_rep}"], annos = adata_embed_ctrl.obs[["label", "batch_id"]].astype("category"), markerscale = 15, figsize = (12, 7), s = 1, alpha = 0.4, colormap = colormap, label_inplace = False)
    fig.tight_layout()
    fig.savefig(res_dir + f"{use_rep}_embed_scIB_{data_case}_ctrl.png")
    fig = utils.plot_embeds(embed = adata_embed_parenchyma.obsm[f"X_umap_{use_rep}"], annos = adata_embed_parenchyma.obs[["label", "batch_id"]].astype("category"), markerscale = 15, figsize = (12, 7), s = 1, alpha = 0.4, colormap = colormap, label_inplace = False)
    fig.tight_layout()
    fig.savefig(res_dir + f"{use_rep}_embed_scIB_{data_case}_parenchyma.png")
    fig = utils.plot_embeds(embed = adata_embed_nan.obsm[f"X_umap_{use_rep}"], annos = adata_embed_nan.obs[["label", "batch_id"]].astype("category"), markerscale = 15, figsize = (12, 7), s = 1, alpha = 0.4, colormap = colormap, label_inplace = False)
    fig.tight_layout()
    fig.savefig(res_dir + f"{use_rep}_embed_scIB_{data_case}_nan.png")

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
    fig.savefig(res_dir + f"{use_rep}_embed_scIB_{data_case}.png", bbox_inches = "tight")

# In[]
# --------------------------------------------------------------------------------------------------------------
#
# NOTE: Evaluation MLM task
#
# --------------------------------------------------------------------------------------------------------------
val_loss_list = []
val_loss_mlm_list = []
val_loss_metric_list = []
mask_prob_list = [0.1, 0.2, 0.3, 0.4]
for mask_prob in mask_prob_list:
    val_loss, val_loss_mlm, val_loss_metric,val_loss_mincut, val_loss_ortho = evaluate_mlm(model_pretrain, test_loader, mask_prob = mask_prob)
    val_loss_list.append(val_loss)
    val_loss_mlm_list.append(val_loss_mlm)
    val_loss_metric_list.append(val_loss_metric)

val_loss_df = pd.DataFrame(columns = ["mask prob", "val total", "val mlm", "val metric", "dataset"])
val_loss_df["mask prob"] = mask_prob_list
val_loss_df["val total"] = val_loss_list
val_loss_df["val mlm"] = val_loss_mlm_list
val_loss_df["val metric"] = val_loss_metric_list
val_loss_df["dataset"] = data_case
val_loss_df.to_csv(res_dir + f"{data_case}_valloss.csv")


# In[]
# for lung atlas, check the mlm on different patient cases
if data_case == "lung_atlas":
    adata_test_ctrl = adata_test[adata_test.obs["patientGroup"] == "Ctrl"]
    adata_test_parenchyma = adata_test[adata_test.obs["patientGroup"] == "Parenchyma"]
    adata_test_nan = adata_test[adata_test.obs["patientGroup"] == "nan"]

    test_dataset_ctrl = data_utils.sc_dataset_anndata(adata = adata_test_ctrl, gene_list = None, batch_feats = {"conts": None, "cats": batch_features_digitize}, label_colname = label_colname, batch_colname = batch_colname, batch_size = 128, normalize = model_pretrain.model_config.lognorm_data)
    test_loader_ctrl = data.DataLoader(test_dataset_ctrl, batch_size = 1, shuffle = False, pin_memory = True, num_workers = 8, prefetch_factor = 8)
    test_dataset_parenchyma = data_utils.sc_dataset_anndata(adata = adata_test_parenchyma, gene_list = None, batch_feats = {"conts": None, "cats": batch_features_digitize}, label_colname = label_colname, batch_colname = batch_colname, batch_size = 128, normalize = model_pretrain.model_config.lognorm_data)
    test_loader_parenchyma = data.DataLoader(test_dataset_parenchyma, batch_size = 1, shuffle = False, pin_memory = True, num_workers = 8, prefetch_factor = 8)
    test_dataset_nan = data_utils.sc_dataset_anndata(adata = adata_test_nan, gene_list = None, batch_feats = {"conts": None, "cats": batch_features_digitize}, label_colname = label_colname, batch_colname = batch_colname, batch_size = 128, normalize = model_pretrain.model_config.lognorm_data)
    test_loader_nan = data.DataLoader(test_dataset_nan, batch_size = 1, shuffle = False, pin_memory = True, num_workers = 8, prefetch_factor = 8)

    test_loaders = {"ctrl": test_loader_ctrl, "parenchyma": test_loader_parenchyma, "nan": test_loader_nan}

    mask_prob_list = [0.1, 0.2, 0.3, 0.4]
    val_loss_list = []
    val_loss_mlm_list = []
    val_loss_disc_list = []
    cases = []
    for case, test_loader in test_loaders.items():
        print(case)
        
        for mask_prob in mask_prob_list:
            val_loss, val_loss_mlm, val_loss_disc,val_loss_mincut, val_loss_ortho = evaluate_mlm(model_pretrain, test_loader, mask_prob = mask_prob)
            val_loss_list.append(val_loss)
            val_loss_mlm_list.append(val_loss_mlm)
            val_loss_disc_list.append(val_loss_disc)
            cases.append(case)

    val_loss_df = pd.DataFrame(columns = ["mask prob", "val total", "val mlm", "val disc", "dataset"])
    val_loss_df["mask prob"] = mask_prob_list * 3
    val_loss_df["val total"] = val_loss_list
    val_loss_df["val mlm"] = val_loss_mlm_list
    val_loss_df["val disc"] = val_loss_disc_list
    val_loss_df["dataset"] = data_case
    val_loss_df["patientGroup"] = cases

    val_loss_df.to_csv(res_dir + f"{data_case}_valloss_patient.csv")

# %%
