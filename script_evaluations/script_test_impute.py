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
from transformer_batch import TransformerModel, get_default_config
# from transformer_stable import TransformerModel, get_default_config
import trainer_batch as trainer_batch
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
PROJECT_DIR = "/net/csefiles/xzhanglab/zzhang834/LLM_KD/"

# old data directory: for old model testing
data_dir = "/net/csefiles/xzhanglab/zzhang834/hs_download/"
# data_dir = "/data/zzhang834/hs_download/"
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


batch_name = "level2"
state = torch.load(model_dir, weights_only = False)
token_dict = torch.load(data_dir + f"meta_data/gene_embed_meta256_gpool.pt", weights_only = False)
label_dict = torch.load(data_dir + f"meta_data/label_dict.pt", weights_only = False)
# ------------------------------------------------------------------------------------------------------------------------------------
# batch_dict = torch.load(data_dir + f"meta_data/batch_dict_batch_{batch_name}.pt", weights_only = False)
# # drop the stats features (not very useful)
# batch_dict["cats"] = batch_dict["cats"].drop(["prop_mito", "raw_mean_nnz", "nnz", "libsize"], axis = 1)
# batch_dict["n_cat_list"] = batch_dict["n_cat_list"][4:]
# # make continuous
# batch_feats = pd.read_csv(data_dir + f"meta_data/feature_batch_{batch_name}.csv", index_col = 0)
# batch_dict["cats"] = batch_feats[batch_dict["cats"].columns]

# new full list
batch_dict = torch.load(data_dir + f"meta_data/batch_dict_{batch_name}_10.pt", weights_only = False)

# # new adaptive
# batch_dict = torch.load(data_dir + f"meta_data/batch_dict_{batch_name}_expr10.pt", weights_only = False)
# ------------------------------------------------------------------------------------------------------------------------------------
batch_dict["cats"] = torch.tensor(batch_dict["cats"].values)

model_pretrain = load_model(state = state, token_dict = token_dict, label_dict = label_dict, batch_dict = batch_dict, device = device)
 
res_dir = PROJECT_DIR + f"results/zs_imputation/{model_name}/"
if not os.path.exists(res_dir):
    os.makedirs(res_dir)

# In[]
# selection of test dataset
for data_case in ["immune_all", "pancreas", "lung_atlas", "covid19"]:
    impute_data_dir = "/net/csefiles/xzhanglab/zzhang834/LLM_KD/dataset/imputation_test/"
    # genes already aligned 
    adata_test = anndata.read_h5ad(impute_data_dir + f"{data_case}_masked_2000.h5ad")

    mask_prob_list = [0.1, 0.2, 0.3, 0.4, 0.5]
    # mask_prob = mask_prob_list[4]
    for mask_prob in mask_prob_list:
        print(f"mask probability: {mask_prob}")
        counts_orig = adata_test.X.copy()
        adata_test.X = adata_test.layers[f"counts_mask_{mask_prob}"].copy()

        # Calculate the batch factor
        if model_pretrain.model_config.batch_enc is not None:
            print("create batch factor...")
            batch_features = batch_encode.construct_batch_feats(adata = adata_test, use_mito = True, use_tech = False, use_nmeasure = False)
            
            # batch_features_digitize, num_bucket = batch_encode.tokenize_batch_feats(batch_features, use_mito = True, use_tech = False, use_nmeasure = False, expr_binsize = 1)
            # batch_features_digitize = batch_features_digitize.drop(["prop_mito", "raw_mean_nnz", "nnz", "libsize"], axis = 1)
            # num_bucket = num_bucket[4:]
            # new
            batch_features_digitize, max_vals = batch_encode.tokenize_batch_feats(batch_features, max_vals = batch_dict["cat_maxvals"], nbins = 10, only_genes = False)    

            batch_features_digitize = torch.tensor(batch_features_digitize.values, dtype = torch.float32)

        else:
            batch_features_digitize = None

        print("create dataloader...")
        label_colname = None
        batch_colname = "batch_id"
        test_dataset = data_utils.sc_dataset_anndata(adata = adata_test, gene_list = None, batch_feats = {"conts": None, "cats": batch_features_digitize},
                                                    label_colname = label_colname, batch_colname = batch_colname, batch_size = 128, normalize = model_pretrain.model_config.lognorm_data)
        test_loader = data.DataLoader(test_dataset, batch_size = 1, shuffle = False, pin_memory = True, num_workers = 8, prefetch_factor = 8)


        print("calculate impute data...")
        adata_impute = trainer_batch.cell_impute(model = model_pretrain, dataloader = test_loader, multi_gpus = False, only_mean = True)
        adata_impute.obs = adata_test.obs.copy()
        adata_impute.X = counts_orig
        adata_impute.layers[f"mask_{mask_prob}"] = adata_test.layers[f"mask_{mask_prob}"]

        adata_impute.write_h5ad(res_dir + f"adata_impute_{data_case}_{mask_prob}.h5ad")

# %%
