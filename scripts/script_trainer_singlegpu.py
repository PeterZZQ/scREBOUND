# In[]
from pathlib import Path
import sys, os

import numpy as np
import pandas as pd

import torch
from torch.utils import data
# from transformers import AdamW
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR

# sys.path.append("/project/zzhang834/LLM_KD/src")
sys.path.append("/net/csefiles/xzhanglab/zzhang834/LLM_KD/src")

import data_utils
from transformer_batch import TransformerModel, get_default_config
import trainer_batch as trainer_batch

from torch.utils.tensorboard import SummaryWriter
from torch.nn.attention import sdpa_kernel, SDPBackend

def initialize_services(log_dir):
    writer = SummaryWriter(log_dir=log_dir)
    return writer

# In[]
data_utils.set_seed(0)
PROJECT_DIR = "/net/csefiles/xzhanglab/zzhang834/LLM_KD/"
# data_dir = "/net/csefiles/xzhanglab/zzhang834/hs_download/permuted/"
data_dir = "/data/zzhang834/hs_download/permuted/"
# Define the device
assert torch.cuda.is_available(), "Training on CPU is not supported"
device = torch.device("cuda")
print(f"GPU - Using device: {device}")

n_mgene = 256
model_config = get_default_config()
batch_size = 64 * 8 * 8
# batch_size = 128
# classifier for 4 gpus, 0.5e-5 too large for less than 4, slightly larger for bp16
lr = 0.3e-5 * (batch_size/32/8) # adjusted for 4 gpus

# accuracy almost the same as fp32
PRECISION = torch.float32
batch_name = "level2"

model_config.__dict__.update({"batch_size": batch_size,
                                "n_epoch": 1,
                                "lr": lr, # important for hyper-parameter tuning
                                "d_embed": 512,
                                "n_head": 8,  # TODO: make 12 head * 64 dimensions
                                "d_hidden": 2048, 
                                "n_layer": 4,
                                "d_output": 64,
                                "dropout": 0.1, # important for hyper-parameter tuning
                                "mask_prob": 0.4, # important for hyper-parameter tuning
                                "dynamic_maskprob": True, # mask_prob is dynamically updated from 0.1 to 0.7 during training
                                "mask_batchfactor": True,
                                "recon_meta": True,
                                "use_fourier": True,
                                "batch_enc": "encoder",
                                "insert_transformer": False,
                                "lamb_mincut": 1,
                                "use_fastatten": True,
                                "pretrain_path": None,
                                "precision": PRECISION,
                                "checkpoint_path": PROJECT_DIR + "checkpoint/",
                                "checkpoint_prefix": f"cp_vanilla_4_512_meta_enc_{batch_name}",
                                "lognorm_data": False
                                })


token_dict = torch.load(f"/data/zzhang834/hs_download/gene_embed_meta{n_mgene}_gpool.pt", weights_only = False)
label_dict = torch.load(data_dir + "label_dict.pt", weights_only = False)
batch_dict = torch.load(PROJECT_DIR + f"batch_encoding/batch_dict_batch_{batch_name}.pt", weights_only = False)
batch_dict["cats"] = torch.tensor(batch_dict["cats"].values, dtype = torch.int32)

fm_model = TransformerModel(model_config = model_config, token_dict = token_dict, batch_dict = batch_dict, label_dict = label_dict, device = device)

# data information
num_partitions = 55
partition_size = 1000000
# remove the last partition as it is for validation
steps_per_partition = np.ceil(partition_size/model_config.batch_size/1) # 489
# the partitions are complete
steps_per_epoch = int((num_partitions-1) * steps_per_partition)
print(f"total number of steps: {steps_per_epoch:d}")

# init optimizer and scheduler, learning rate scale with the batch_size, larger eps for more stability in bf16
fm_model.optimizer = AdamW(fm_model.parameters(), lr = model_config.lr, eps = 1e-6)
fm_model.scheduler = OneCycleLR(fm_model.optimizer, max_lr = model_config.lr, steps_per_epoch = steps_per_epoch, epochs = model_config.n_epoch, pct_start = 0.3)

# Load latest checkpoint
if model_config.pretrain_path is not None: 
    print(f"GPU - Preloading lastest model'")
    # load parameter from last train
    state = torch.load(model_config.pretrain_path, weights_only = False)
    # Get the common keys between the current model and the saved model
    filtered_state_dict = {k: v for k, v in state["model_state_dict"].items() if k in fm_model.state_dict()}
    # Load the filtered state dictionary into the model
    fm_model.load_state_dict(filtered_state_dict, strict = False)

    # NOTE: for continuous training, update optimizer and scheduler for consistent training
    fm_model.optimizer.load_state_dict(state['optimizer_state_dict'])
    fm_model.scheduler.load_state_dict(state["scheduler_state_dict"])
    initial_epoch = state['epoch']
    initial_step = state['step'] + 1
    del state
else:
    initial_epoch = 0
    initial_step = 0
    # If we couldn't find a model to preload, just start from scratch
    print(f'GPU - Could not find model to preload. Starting from scratch')


# calculate the dynamical value steps:
log_step = 100
if fm_model.model_config.dynamic_maskprob:
    mask_prob_init = 0.15
    mask_prob_end = 0.4
    fm_model.model_config.maskprob_step = (mask_prob_end - mask_prob_init) / steps_per_epoch / (fm_model.model_config.n_epoch - initial_epoch) * log_step
    fm_model.model_config.mask_prob = mask_prob_init

# if fm_model.model_config.use_discriminator:
#     lamb_reverse_init = 1.0
#     lamb_reverse_end = 1.0
#     fm_model.model_config.lamb_reverse_step = (lamb_reverse_end - lamb_reverse_init) / steps_per_epoch / (fm_model.model_config.n_epoch - initial_epoch) * log_step
#     fm_model.model_config.lamb_reverse = lamb_reverse_init
# else:
#     fm_model.model_config.lamb_reverse = 0.0

# Init logger process, only main thread
writer = initialize_services(model_config.checkpoint_path + model_config.checkpoint_prefix) 

# In[]
dataset_dict = {"DIR": data_dir, "num_partitions": num_partitions, "data_prefix": "counts", "meta_prefix": "obs", "batch_dict": batch_dict, "label_colname": "label_id", "batch_colname": "batch_" + batch_name + "_id"}

with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
    trainer_batch.train_singlegpu(model = fm_model, dataset_dict = dataset_dict, writer = writer, initial_epoch = initial_epoch, initial_step = initial_step, log_step = log_step)


                




# %%
