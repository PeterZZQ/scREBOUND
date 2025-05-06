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
# name of columns used when training
batch_name = "level2"

# vanilla model without batch
# model_name = "cp_vanilla_4_512_meta_1"

# batch-encoder version
model_name = f"cp_vanilla_4_512_meta_enc_{batch_name}_1"
# model_name = f"cp_vanilla_4_512_meta_enc_wmask_{batch_name}_1"
# model_name = f"cp_vanilla_4_512_meta_enc_trans_{batch_name}_1"
# model_name = f"cp_vanilla_4_512_meta_enc_trans_wmask_{batch_name}_1"

PRETRAIN_MODEL = PROJECT_DIR + "checkpoint/" + model_name + ".pth"

state = torch.load(PRETRAIN_MODEL, weights_only = False)
model_config.__dict__.update(state["model_config"])
# further update
model_config.__dict__.update({"batch_size": batch_size,
                              "lr": lr,
                              "mask_prob": 0.10, # important for hyper-parameter tuning
                              "dynamic_maskprob": False, # mask_prob is dynamically updated from 0.1 to 0.7 during training
                              "recon_meta": True, # train on orig data
                              "sup_type": "contrcb-proj",
                              "lamb_sup": 1,
                              "lamb_mincut": 0.1,
                              "precision": PRECISION,
                              "checkpoint_path": PROJECT_DIR + "checkpoint_finetune/",
                              "checkpoint_prefix": "cp_contrcbproj1_" + model_name.removeprefix("cp_vanilla_").removesuffix("_1"),
                              "lognorm_data": False
                            })


token_dict = torch.load(f"/data/zzhang834/hs_download/gene_embed_meta{n_mgene}_gpool.pt", weights_only = False)
label_dict = torch.load(data_dir + "label_dict.pt", weights_only = False)

# NOTE: batch dict is only used for setting the batch-encoder and provide batch-features for training
if model_config.batch_enc is not None:
    batch_dict = torch.load(PROJECT_DIR + f"batch_encoding/batch_dict_batch_{batch_name}.pt")
    batch_dict["cats"] = torch.tensor(batch_dict["cats"].values, dtype = torch.int32)
else:
    # NOTE: the batch_column significantly affect the contrcb, level0 is the old one,
    # can compare with level2, but level2 contrcb should be very similar to contrastive itself (same batch data not enough)
    batch_dict = None

fm_model = TransformerModel(model_config = model_config, token_dict = token_dict, batch_dict = batch_dict, label_dict = label_dict, device = device)

# freeze the transformer and only train the discriminator as baseline
# fm_model.freeze_fm_gradient(freeze_trans = True, freeze_predictor = True, freeze_batchenc = True, freeze_compression = True, freeze_discriminator = False)
# # finetune the compression
# fm_model.freeze_fm_gradient(freeze_trans = True, freeze_predictor = True, freeze_batchenc = True, freeze_compression = False, freeze_discriminator = False)

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
print(f"GPU - Preloading pretrained model")
# Get the common keys between the current model and the saved model
filtered_state_dict = {k: v for k, v in state["model_state_dict"].items() if k in fm_model.state_dict()}
# Load the filtered state dictionary into the model
fm_model.load_state_dict(filtered_state_dict, strict = False)

initial_epoch = 0
initial_step = 0

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
