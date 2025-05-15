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
from torch.amp import autocast, GradScaler

sys.path.append("/net/csefiles/xzhanglab/zzhang834/LLM_KD/src")

import data_utils
from transformer_stable import TransformerModel, get_default_config
import trainer_stable as trainer_batch
import warnings
warnings.filterwarnings("ignore")

import time

CAST_DTYPE = torch.bfloat16

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


def eval_speed(model, dataloader):
    """
    Description:
    ------------
        Obtain the model cell embedding for data in dataloader
    
    Parameters:
    ------------
        model: the transformer model
        dataloader: the dataloader for the input data
        mask_prob: the masking probability of data in the forward pass, default is 0

    """
    model_acc = model    
    # evaluation model
    model_acc.eval()
    # remove mask
    model_acc.model_config.mask_prob = 0.0

    if model_acc.model_config.use_flashatten:
        # because flashattention only accept 16bit model
        enable_casting = True
    else:
        enable_casting = False

    time_step_list = []
    cell_step_list = []
    with torch.no_grad():
        for data_sample in tqdm.tqdm(dataloader, desc=f"Calc embed"):
            start_time = time.time()
            with autocast(device_type="cuda", dtype = CAST_DTYPE, enabled = enable_casting):
                expr_sample = data_sample["counts_norm"].reshape(-1, data_sample["counts_norm"].shape[-1]).to(model_acc.device, non_blocking = True)
                batch_sample_id = data_sample["batch"].reshape(-1).to(model_acc.device, non_blocking = True) if "batch" in data_sample.keys() else None
                batch_sample_cat = data_sample["batch_cat"].reshape(-1, data_sample["batch_cat"].shape[-1]).to(model_acc.device, non_blocking = True) if "batch_cat" in data_sample.keys() else None
                batch_sample_cont = data_sample["batch_cont"].reshape(-1, data_sample["batch_cont"].shape[-1]).to(model_acc.device, non_blocking = True) if "batch_cont" in data_sample.keys() else None

                all_embed, cell_embed, mask_gene = model(counts_norm = expr_sample, batch_factors_cont = batch_sample_cont, batch_factors_cat = batch_sample_cat, batch_ids = None)                    
                expr_pred, expr_pred_meta = model_acc.predict_expr(cell_embed = cell_embed, batch_factors_cont = batch_sample_cont, batch_factors_cat = batch_sample_cat, batch_ids = batch_sample_id)
                
                assert model_acc.model_config.mlm_type == "raw"
                assert model_acc.model_config.lognorm_data == False

            end_time = time.time()
            time_step = end_time - start_time
            time_step_list.append(time_step)
            cell_step_list.append(cell_embed.shape[0])

    runtime = pd.DataFrame(columns = ["ncells", "runtime"], data = 0, index = np.arange(len(time_step_list)))
    runtime["ncells"] = cell_step_list
    runtime["runtime"] = time_step_list
    return runtime


# In[]
# --------------------------------------------------------------------------
#
# Loading the trained model
#
# --------------------------------------------------------------------------

torch.cuda.reset_peak_memory_stats()
torch.cuda.synchronize()  # Sync before measurement

data_utils.set_seed(0)
device = torch.device("cuda:1")
print(f"GPU - Using device: {device}")

# NOTE: save in localscratch for faster memory access
PROJECT_DIR = "/net/csefiles/xzhanglab/zzhang834/LLM_KD/"
# data_dir = "/data/zzhang834/hs_download/"
data_dir = "/net/csefiles/xzhanglab/zzhang834/hs_download/"

batch_name = "level2"
model_name = f"cp_contrcb1_4_512_128_encbg_level2_1"
model_dir = PROJECT_DIR + f"checkpoint/model_128/{model_name}.pth"
model_name = f"cp_contrcb1_6_512_256_encbg_level2_1"
model_dir = PROJECT_DIR + f"checkpoint/model_6_256/{model_name}.pth"


# read in the key files
state = torch.load(model_dir, weights_only = False)
token_dict = torch.load(data_dir + f"meta_data/gene_embed_meta256_gpool.pt", weights_only = False)
label_dict = torch.load(data_dir + "meta_data/label_dict.pt", weights_only = False)
batch_dict = torch.load(data_dir + f"meta_data/batch_dict_batch_{batch_name}.pt")

# drop the stats features (not very useful)
batch_dict["cats"] = batch_dict["cats"].drop(["prop_mito", "raw_mean_nnz", "nnz", "libsize"], axis = 1)
batch_dict["n_cat_list"] = batch_dict["n_cat_list"][4:]

batch_dict["cats"] = torch.tensor(batch_dict["cats"].values, dtype = torch.int32)

model_pretrain = load_model(state = state, token_dict = token_dict, label_dict = label_dict, batch_dict = batch_dict, device = device)

batch_size = 128

# select 10 percent of cells, since the data are already permuted
min_chunksize = 64

dataset_dict = {"DIR": data_dir + "permuted/",
                "data_prefix": "counts",
                "meta_prefix": "obs",
                "batch_dict": batch_dict,
                "label_colname": "label_id",
                "batch_colname": "batch_" + batch_name + "_id"}


# In[]
# NOTE: our model embedding
res_dir = PROJECT_DIR + f"results/runtime/"
if not os.path.exists(res_dir):
    os.makedirs(res_dir)

# read in the partition file
train_dataset = data_utils.sc_partition(data_path = dataset_dict["DIR"], batch_feats = dataset_dict["batch_dict"], min_chunksize = min_chunksize, normalize = model_pretrain.model_config.lognorm_data)
partition_idx = 0
train_dataset.load_partition(idx = partition_idx, label_colname = dataset_dict["label_colname"], batch_colname = dataset_dict["batch_colname"], data_prefix = dataset_dict["data_prefix"], meta_prefix = dataset_dict["meta_prefix"])
train_loader = data.DataLoader(train_dataset, batch_size = batch_size//min_chunksize, shuffle = False, pin_memory = True, num_workers = 8, prefetch_factor = 8)

# adata_test = anndata.read_h5ad(EVAL_DATA_DIR + f"{data_case}_aligned.h5ad")
# print("create dataloader...")
# label_colname = None
# batch_colname = "batch_id"
# test_dataset = data_utils.sc_dataset_anndata(adata = adata_test, gene_list = None, batch_feats = {"conts": None, "cats": batch_features_digitize},
#                                              label_colname = label_colname, batch_colname = batch_colname, batch_size = 128, normalize = model_pretrain.model_config.lognorm_data)
# test_loader = data.DataLoader(test_dataset, batch_size = 1, shuffle = False, pin_memory = True, num_workers = 8, prefetch_factor = 8)

runtime_df = eval_speed(model = model_pretrain, dataloader = train_loader)

# After step
peak_memory = torch.cuda.max_memory_allocated() / (1024**3)  # Convert to GB
print(f"Peak GPU memory used: {peak_memory:.2f} GB")
runtime_df["peak_memory"] = peak_memory

runtime_df.to_csv(res_dir + f"runtime_{model_name}.csv")



# %%
