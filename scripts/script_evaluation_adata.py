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
# from transformer_vanilla import TransformerModel, get_default_config
from transformer_batch import TransformerModel, get_default_config
# import trainer_vanilla as trainer
import trainer_batch as trainer_batch
import utils
# import eval
import batch_encode 
import warnings
warnings.filterwarnings("ignore")

# In[]
EVAL_DATA_DIR = "/net/csefiles/xzhanglab/zzhang834/LLM_KD/dataset/scIB/"
# read the gene protein embedding and gene & meta-gene assignment
n_mgene = 256
use_meta_input = False
# shared across devices
gene_embed_dict = torch.load(f"/net/csefiles/xzhanglab/zzhang834/hs_download/gene_embed_meta{n_mgene}_gpool.pt", weights_only = False)
gene_info = gene_embed_dict["labels"]


adata_test1 = anndata.read_h5ad(EVAL_DATA_DIR + "Immune_ALL_human.h5ad")
# checked: counts are raw
adata_test1.X = adata_test1.layers["counts"].copy()
adata_test1.obs["assay"] = adata_test1.obs["chemistry"].cat.rename_categories({"10X": "10x 3' transcription profiling", 'smart-seq2': 'Smart-seq2', 'v2_10X': "10x 3' v2", "v3_10X": "10x 3' v3"})
adata_test1.obs["suspension_type"] = "cell"
adata_test1.obs["batch_id"], batch_code = pd.factorize(adata_test1.obs["batch"])
adata_test1.obs["label"] = adata_test1.obs["final_annotation"]

adata_test2 = anndata.read_h5ad(EVAL_DATA_DIR + "human_pancreas_norm_complexBatch.h5ad")
adata_test2.X = sp.csr_matrix(adata_test2.layers["counts"])
# # celseq, fluidigm c1, inDropx, and smarter are independent tech not related to any trained case
# adata_test2.obs["assay"] = adata_test2.obs["tech"].cat.rename_categories({"celseq": "CEL-seq", "celseq2": "CEL-seq2", "fluidigmc1": "Fluidigm C1 microfluidics platform",
#                                                                           "inDrop1": "inDrop", "inDrop2": "inDrop", "inDrop3": "inDrop", "inDrop4": "inDrop", "smarter": "Smart-like", "smartseq2": "Smart-seq2"})
adata_test2.obs["suspension_type"] = "cell"
adata_test2.obs["batch_id"], batch_code = pd.factorize(adata_test2.obs["tech"])
adata_test2.obs["label"] = adata_test2.obs["celltype"]

adata_test3 = anndata.read_h5ad(EVAL_DATA_DIR + "Lung_atlas_public.h5ad")
adata_test3.X = adata_test3.layers["counts"].copy()
adata_test3.obs["assay"] = adata_test3.obs["protocol"].cat.rename_categories({"10x v2": "10x 3' v2", "drop-seq": "Drop-seq"})
# both tech by default do cell instead of nucleus
adata_test3.obs["suspension_type"] = "cell"
adata_test3.obs["batch_id"], batch_code = pd.factorize(adata_test3.obs["batch"])
adata_test3.obs["label"] = adata_test3.obs["cell_type"]

# In[]
EVAL_DATA_DIR2 = "/net/csefiles/xzhanglab/zzhang834/LLM_KD/dataset/evaluation_datasets/"
adata_test4 = anndata.read_h5ad(EVAL_DATA_DIR2 + "covid19_aligned.h5ad")
adata_test4.obs["batch_id"], batch_code = pd.factorize(adata_test4.obs["sample"])
adata_test4.obs["label"] = adata_test4.obs["predicted.celltype.l1"]


adata_test5 = anndata.read_h5ad(EVAL_DATA_DIR2 + "GBM_Fig4.h5ad")
adata_test5.obs["batch_id"], batch_code = pd.factorize(adata_test5.obs["sample_id"])
adata_test5.obs["label"] = adata_test5.obs["mstatus"]
adata_test5.var_names_make_unique()

# In[]
# selection of test dataset
data_case = "immune_all"
data_case = "pancreas"
data_case = "lung_atlas"
data_case = "covid"
data_case = "GBM"
if data_case == "immune_all":
    adata_test = adata_test1.copy()
elif data_case == "pancreas":
    adata_test = adata_test2.copy()
elif data_case == "lung_atlas":
    adata_test = adata_test3.copy()
elif data_case == "covid":
    adata_test = adata_test4.copy()
elif data_case == "GBM":
    adata_test = adata_test5.copy()

gene_list = gene_info["feature_name"].values

# In[]
# function
data_utils.set_seed(0)

device = torch.device("cuda")
print(f"GPU - Using device: {device}")

# NOTE: save in localscratch for faster memory access
PROJECT_DIR = "/net/csefiles/xzhanglab/zzhang834/LLM_KD/"
data_dir = "/data/zzhang834/hs_download/"
# data_dir = "/project/zzhang834/hs_download/"

batch_name = "level2"

# ------------------------------Update for the model selected---------------------------------------------------
# NOTE: 1. vanilla model with only mlm loss
# # vanilla mlm model (no batch encoding, only mlm), best in test
# model_name = "cp_vanilla_4_512_meta_1"
# # best in train/val loss
# model_name = "cp_vanilla_4_512_meta_enc_trans_1"
# model_name = "cp_vanilla_4_512_meta_onehot_trans_1"
# model_name = "cp_vanilla_4_512_meta_enc_1"
# model_name = "cp_vanilla_4_512_meta_onehot_1"
# model_name = "cp_vanilla_4_512_meta_enc_trans_nofourier_1"
# model_name = "cp_vanilla_4_512_meta_onehot_trans_nofourier_1"

# Newer version of batch_encoding
# model_name = "cp_vanilla_4_512_meta_enc_trans_level0_1"
# model_name = "cp_vanilla_4_512_meta_enc_trans_wmask_level0_1"
# model_name = "cp_vanilla_4_512_meta_enc_trans_level2_1"
# model_name = "cp_vanilla_4_512_meta_enc_trans_wmask_level2_1"
# model_name = "cp_vanilla_4_512_meta_enc_level2_1"
# model_name = "cp_vanilla_4_512_meta_enc_wmask_level2_1"

# model_dir = PROJECT_DIR + f"checkpoint/{model_name}.pth"
# res_dir = PROJECT_DIR + f"results/checkpoint/{model_name}/"

# NOTE: 2. finetune model
# # contrastive model
# model_name = "cp_contrcb1_4_512_meta_nobatch_1"
# model_name = "cp_contrcbproj1_4_512_meta_nobatch_1"
# model_name = "cp_contrcbproj21_4_512_meta_nobatch_1"

# model_name = f"cp_contr1_4_512_meta_enc_trans_wmask_{batch_name}_1"
# model_name = f"cp_contrcb1_4_512_meta_enc_trans_wmask_{batch_name}_1"
# model_name = f"cp_contrcbproj1_4_512_meta_enc_trans_wmask_{batch_name}_1"
# model_name = f"cp_contrcb1_4_512_meta_enc_wmask_{batch_name}_1"
# model_name = f"cp_contrcbproj1_4_512_meta_enc_wmask_{batch_name}_1"

# model_dir = PROJECT_DIR + f"checkpoint_finetune/{model_name}.pth"
# res_dir = PROJECT_DIR + f"results/checkpoint_finetune/{model_name}/"

# NOTE: compression-model 
# single-layer
# model_name = "cp_nofrz_4_512_lognorm_1"
# model_name = "cp_frztrans_4_512_lognorm_1"
# model_name = "cp_frzall_4_512_lognorm_1"

# double layers
# model_name = "cp_frztrans_4_512_lognorm2_1"
# model_name = "cp_frzall_4_512_lognorm2_1"
# model_name = "cp_nofrz_4_512_lognorm2_1"
# model_name = "cp_hybrid_4_512_lognorm2_1"
# model_name = "cp_frztrans_4_512_raw2_1"
# model_name = "cp_nofrz_4_512_raw2_1"
# model_name = "cp_hybrid_4_512_raw2_1"
# model_name = "cp_hybrid_4_512_raw2_enc_wmask_level2_1"

# model_dir = PROJECT_DIR + f"checkpoint_compress/{model_name}.pth"
# res_dir = PROJECT_DIR + f"results/checkpoint_compress/{model_name}/"

# NOTE: compression-finetune
model_name = "cp_contrcbproj1_4_512_1"
model_name = f"cp_contrcb1_4_512_raw2_enc_wmask_{batch_name}_1"
model_dir = PROJECT_DIR + f"checkpoint_compress_finetune/{model_name}.pth"
res_dir = PROJECT_DIR + f"results/checkpoint_compress_finetune/{model_name}/"


if not os.path.exists(res_dir):
    os.makedirs(res_dir)

state = torch.load(model_dir, weights_only = False)
model_config = get_default_config()
model_config.__dict__.update(state["model_config"])
model_config.recon_layers = 2

for x, val in model_config.__dict__.items():
    print(x, end = ": ")
    print(val)
# ------------------------------------------------------------------------------------------------------------------

token_dict = torch.load(data_dir + f"gene_embed_meta{n_mgene}_gpool.pt", weights_only = False)
label_dict = torch.load(data_dir + "permuted/label_dict.pt", weights_only = False)
if model_config.batch_enc is not None:
    # batch_dict = torch.load(PROJECT_DIR + "batch_encoding/batch_enc_dict_nomito.pt", weights_only = False)
    batch_dict = torch.load(PROJECT_DIR + f"batch_encoding/batch_dict_batch_{batch_name}.pt")
else:
    batch_dict = None

fm_model = TransformerModel(model_config = model_config, token_dict = token_dict, batch_dict = batch_dict, label_dict = label_dict, device = device).to(model_config.precision)

print(f"GPU - Preloading lastest model")
# Get the common keys between the current model and the saved model
filtered_state_dict = {k: v for k, v in state["model_state_dict"].items() if k in fm_model.state_dict()}
# Load the filtered state dictionary into the model
fm_model.load_state_dict(filtered_state_dict, strict=False)

print(f"GPU - Done.")
# In[]
# align genes
if data_case != "covid":
    adata_test = data_utils.align_genes(adata_test, gene_list)
adata_test.layers["counts"] = adata_test.X.copy()

if model_config.batch_enc is not None:
    batch_features = batch_encode.construct_batch_feats(adata = adata_test, use_mito = True, use_tech = False, use_nmeasure = False)
    # batch_features.to_csv(f"../batch_encoding/feature_batch_{data_case}.csv")
    batch_features_digitize, num_bucket = batch_encode.tokenize_batch_feats(batch_features, use_mito = True, use_tech = False, use_nmeasure = False, expr_binsize = 1)
    batch_features_digitize = torch.tensor(batch_features_digitize.values, dtype = torch.float32)

else:
    batch_features_digitize = None

# In[]
# --------------------------------------------------------------------------------------------------------------
#
# NOTE: calculate the embedding
#
# --------------------------------------------------------------------------------------------------------------
# TODO: issue, for the classifier, should the masked input be used??
label_colname = None
batch_colname = "batch_id"
test_dataset = data_utils.sc_dataset_anndata(adata = adata_test, gene_list = None, batch_feats = {"conts": None, "cats": batch_features_digitize},
                                             label_colname = label_colname, batch_colname = batch_colname, batch_size = 128, normalize = model_config.lognorm_data)
test_loader = data.DataLoader(test_dataset, batch_size = 1, shuffle = False, pin_memory = True, num_workers = 8, prefetch_factor = 8)

adata_embed = trainer_batch.cell_embed(model = fm_model, dataloader = test_loader, multi_gpus = False)
adata_embed.obs = adata_test.obs.copy()
adata_embed.obsm["latent"] = adata_embed.X.copy()

sc.pp.neighbors(adata_embed, n_neighbors = 15, use_rep = "latent")
sc.tl.umap(adata_embed, min_dist = 0.3)
adata_embed.obsm[f"X_umap_latent"] = adata_embed.obsm["X_umap"].copy()
del adata_embed.obsm["X_umap"]
if "contr" in adata_embed.obsm.keys():
    sc.pp.neighbors(adata_embed, n_neighbors = 15, use_rep = "contr")
    sc.tl.umap(adata_embed, min_dist = 0.3)
    adata_embed.obsm[f"X_umap_contr"] = adata_embed.obsm["X_umap"].copy()
    del adata_embed.obsm["X_umap"]

adata_embed.write_h5ad(res_dir + f"adata_embed_{data_case}.h5ad")
# In[]
adata_embed = anndata.read_h5ad(res_dir + f"adata_embed_{data_case}.h5ad")

colormap =plt.cm.get_cmap("tab20")
for use_rep in ["contr", "latent"]:
    if f"X_umap_{use_rep}" in adata_embed.obsm.keys(): 
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
            elif data_case == "covid":
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
# # Check the compression model
# counts_norm_list = []
# for data_sample in test_loader:
#     expr_sample = data_sample["counts_norm"].reshape(-1, data_sample["counts_norm"].shape[-1]).to(fm_model.device, non_blocking = True)
#     # # first
#     # S = fm_model.gene_compression.get_score()
#     # expr_sample_norm = (expr_sample @ S).detach().cpu().numpy()
#     # expr_sample_norm = np.log1p(expr_sample_norm/(np.sum(expr_sample_norm, axis = 1, keepdims = True)  +1e-4) * 10e4)
#     # counts_norm_list.append(expr_sample_norm)

#     S, token_embed_meta, counts_norm_meta = fm_model.gene_compression(gene_embed = fm_model.token_embed, expr = expr_sample, log_norm = True)
#     counts_norm_list.append(counts_norm_meta.detach().cpu().numpy())

# counts_norm = np.vstack(counts_norm_list)
# # counts_stand = counts_norm/np.max(counts_norm, axis = 1, keepdims = True)

# adata_meta = anndata.AnnData(X = counts_norm, obs = adata_embed.obs)
# sc.pp.neighbors(adata_meta, n_neighbors = 15)
# sc.tl.umap(adata_meta, min_dist = 0.3)
# colormap =plt.cm.get_cmap("tab20")
# # immune cell
# fig = utils.plot_embeds(embed = adata_meta.obsm["X_umap"], annos = adata_meta.obs[["label", "batch_id"]].astype("category"), markerscale = 15, figsize = (20, 17), s = 1, alpha = 0.4, colormap = colormap, label_inplace = False)
# fig.tight_layout()


# In[]
# --------------------------------------------------------------------------------------------------------------
#
# NOTE: Evaluation MLM task
#
# --------------------------------------------------------------------------------------------------------------
from torch.amp import autocast
def evaluate_mlm(model, dataloader, mask_prob):
    model.model_config.mask_prob = mask_prob
    # need to temperarily remove the discriminator loss, no label for mlm test
    # use_discriminator = model.model_config.use_discriminator
    # model.model_config.use_discriminator = False
    sup_type = model.model_config.sup_type
    model.model_config.sup_type = None
    # need to remove the batch_factor_mask too
    mask_batchfactor = model.model_config.mask_batchfactor
    model.model_config.mask_batchfactor = False
    
    # NOTE: training loop
    model.eval()
    with torch.no_grad():
        val_loss, val_loss_mlm, val_loss_disc, val_loss_mincut, val_loss_ortho = 0.0, 0.0, 0.0, 0.0, 0.0
        for data_sample in tqdm.tqdm(dataloader, desc=f"Evaluation"):
            with autocast(device_type="cuda", dtype = torch.bfloat16, enabled = model.model_config.use_fastatten):
                if model.model_config.batch_enc == "onehot":
                    del data_sample["batch"]

                loss, loss_item = trainer_batch.infer_databatch(model, data_sample, multigpus = False)                            
            val_loss += loss.item()
            val_loss_mlm += loss_item["mlm"]
            val_loss_disc += loss_item["disc"]
            val_loss_mincut += loss_item["mincut"]
            val_loss_ortho += loss_item["ortho"]
        # log the values
        val_loss /= len(dataloader)
        val_loss_mlm /= len(dataloader)
        val_loss_disc /= len(dataloader)
        val_loss_mincut /= len(dataloader)
        val_loss_ortho /= len(dataloader)

        print(f"Val Loss (TOTAL): {val_loss:.4f}, Val Loss (MLM): {val_loss_mlm:.4f}, Val Loss (DISC): {val_loss_disc:.4f}, Val Loss (MINCUT): {val_loss_mincut:.4f}, Val Loss (ORTHO): {val_loss_ortho:.4f}")

    # model.model_config.use_discriminator = use_discriminator
    model.model_config.sup_type = sup_type
    model.model_config.mask_batchfactor = mask_batchfactor

    return val_loss, val_loss_mlm, val_loss_disc, val_loss_mincut, val_loss_ortho

val_loss_list = []
val_loss_mlm_list = []
val_loss_disc_list = []
mask_prob_list = [0.1, 0.2, 0.3, 0.4]
for mask_prob in mask_prob_list:
    val_loss, val_loss_mlm, val_loss_disc,val_loss_mincut, val_loss_ortho = evaluate_mlm(fm_model, test_loader, mask_prob = mask_prob)
    val_loss_list.append(val_loss)
    val_loss_mlm_list.append(val_loss_mlm)
    val_loss_disc_list.append(val_loss_disc)

val_loss_df = pd.DataFrame(columns = ["mask prob", "val total", "val mlm", "val disc", "dataset"])
val_loss_df["mask prob"] = mask_prob_list
val_loss_df["val total"] = val_loss_list
val_loss_df["val mlm"] = val_loss_mlm_list
val_loss_df["val disc"] = val_loss_disc_list
val_loss_df["dataset"] = data_case
val_loss_df.to_csv(res_dir + f"{data_case}_valloss.csv")


# In[]
# for lung atlas, check the mlm on different patient cases
if data_case == "lung_atlas":
    adata_test_ctrl = adata_test[adata_test.obs["patientGroup"] == "Ctrl"]
    adata_test_parenchyma = adata_test[adata_test.obs["patientGroup"] == "Parenchyma"]
    adata_test_nan = adata_test[adata_test.obs["patientGroup"] == "nan"]

    test_dataset_ctrl = data_utils.sc_dataset_anndata(adata = adata_test_ctrl, gene_list = None, batch_feats = {"conts": None, "cats": batch_features_digitize}, label_colname = label_colname, batch_colname = batch_colname, batch_size = 128, normalize = model_config.lognorm_data)
    test_loader_ctrl = data.DataLoader(test_dataset_ctrl, batch_size = 1, shuffle = False, pin_memory = True, num_workers = 8, prefetch_factor = 8)
    test_dataset_parenchyma = data_utils.sc_dataset_anndata(adata = adata_test_parenchyma, gene_list = None, batch_feats = {"conts": None, "cats": batch_features_digitize}, label_colname = label_colname, batch_colname = batch_colname, batch_size = 128, normalize = model_config.lognorm_data)
    test_loader_parenchyma = data.DataLoader(test_dataset_parenchyma, batch_size = 1, shuffle = False, pin_memory = True, num_workers = 8, prefetch_factor = 8)
    test_dataset_nan = data_utils.sc_dataset_anndata(adata = adata_test_nan, gene_list = None, batch_feats = {"conts": None, "cats": batch_features_digitize}, label_colname = label_colname, batch_colname = batch_colname, batch_size = 128, normalize = model_config.lognorm_data)
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
            val_loss, val_loss_mlm, val_loss_disc,val_loss_mincut, val_loss_ortho = evaluate_mlm(fm_model, test_loader, mask_prob = mask_prob)
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
