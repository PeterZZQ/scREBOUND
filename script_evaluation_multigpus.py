# In[]
from pathlib import Path
import sys, os

import numpy as np
import tqdm
from datetime import timedelta

# packages for distributed training
import torch
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel
from torch.distributed import init_process_group, destroy_process_group
from torch.utils import data


import data_utils
from transformer_model import TransformerModel, get_default_config
import trainer


def sum_across_gpus(tensor):
    if dist.is_initialized():
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    return tensor

def evaluation(model, val_loader):
    # NOTE: training loop
    model.module.eval()
    with torch.no_grad():
        val_loss = 0.0
        val_loss_mlm = 0.0
        val_loss_sup = 0.0
        val_loss_kd = 0.0
        for data_sample in val_loader:
            loss, loss_mlm, loss_sup, loss_kd = trainer.infer_databatch(model, data_sample, multigpus = True)                            
            val_loss += loss.item()
            val_loss_mlm += loss_mlm.item()
            val_loss_sup += loss_sup.item()
            val_loss_kd += loss_kd.item()

        # log the values
        val_loss /= len(val_loader)
        val_loss_mlm /= len(val_loader)
        val_loss_sup /= len(val_loader)
        val_loss_kd /= len(val_loader)

        # sum across gpus
        if global_rank == 0:
            val_loss = sum_across_gpus(val_loss)/num_gpus
            val_loss_mlm = sum_across_gpus(val_loss_mlm)/num_gpus
            val_loss_sup = sum_across_gpus(val_loss_sup)/num_gpus
            val_loss_kd = sum_across_gpus(val_loss_kd)/num_gpus

            print(f"Val Loss (TOTAL): {val_loss:.4f}, Val Loss (MLM): {val_loss_mlm:.4f}, Val Loss (CLASS): {val_loss_sup:.4f}, Val Loss (KD): {val_loss_kd:.4f}")
        # sync
        dist.barrier()
    return val_loss, val_loss_mlm, val_loss_sup, val_loss_kd


# In[]
num_gpus = torch.cuda.device_count()
print(f"Number of available GPUs: {num_gpus}")
# environment variables generated when use torchrun
# local gpu id within the machine
local_rank = int(os.environ['LOCAL_RANK'])
# global gpu id across machines (uniq), same as local with one machine
global_rank = int(os.environ['RANK'])
print(f"local rank: {local_rank}")
print(f"global rank: {global_rank}")

# increase the time-out waiting time across gpus
init_process_group(backend='nccl', timeout=timedelta(minutes=60))
torch.cuda.set_device(local_rank) # Set the device to local rank


# In[]
# function
data_utils.set_seed(3)

# Define the device
device = torch.device("cuda")
print(f"GPU {local_rank} - Using device: {device}")
# Load the dataset
print(f"GPU {local_rank} - Loading dataset...")

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
lr = 1e-5 * (batch_size/32)
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
                                "sup_type": "classifier",
                                "sample_mlm": False,
                                "mlm_include_zero": False,
                                "pretrain_path":  "/project/zzhang834/LLM_KD/checkpoint_predfull/checkpoint_0.3_0_8999.pth",
                                "checkpoint_path": "/project/zzhang834/LLM_KD/checkpoint_predfull/",
                                "checkpoint_prefix": "checkpoint_0.3_wzero"
                                })

# construct dataset
scdataset = data_utils.sc_dataset_chunk(expr_path = data_dir / f"expr_sent_{n_mgene}_permu.npz", gene_path = data_dir / f"feat_sent_{n_mgene}_permu.npz",
                                        ncells = meta_dict["shape"][0], npads = meta_dict["shape"][1], labels = meta_dict["label"], batches = meta_dict["batch"], batch_size = model_config.batch_size)

# train test split
train_size = int(0.3 * len(scdataset))
val_size = int(0.001 * len(scdataset))

# the data is already pre-shuffled
train_dataset = data.Subset(scdataset, range(train_size))
# val_dataset = data.Subset(scdataset, range(train_size, train_size + val_size))
val_dataset = data.Subset(scdataset, range(train_size + 10 * val_size, train_size + 11 * val_size))
test_dataset = data.Subset(scdataset, range(train_size + val_size, len(scdataset)))

# obtain train/val/test loaders
# train_loader = data.DataLoader(train_dataset, batch_size = 1, shuffle = False, pin_memory = True, sampler = DistributedSampler(train_dataset, shuffle = True), num_workers = 8, prefetch_factor = 8)
val_loader = data.DataLoader(val_dataset, batch_size = 1, shuffle = False, pin_memory = True, sampler = DistributedSampler(val_dataset, shuffle = True), num_workers = 8, prefetch_factor = 8)
test_loader = data.DataLoader(test_dataset, batch_size = 1, shuffle = False, pin_memory = True, sampler = DistributedSampler(val_dataset, shuffle = True), num_workers = 8)
print(f"GPU {local_rank} - Done.")

# model hyper-parameters
fm_model = TransformerModel(model_config = model_config, token_embed = token_embed, n_batch = len(meta_dict["batch_code"]), n_label = len(meta_dict["label_code"]), device = device)
# wrap model into multi-gpus setting
fm_model = DistributedDataParallel(fm_model, device_ids=[local_rank])

print(f"GPU {local_rank} - Preloading lastest model'")
# load parameter from last train
state = torch.load(model_config.pretrain_path)
fm_model.module.load_state_dict(state["model_state_dict"])
del state

evaluation(model = fm_model, val_loader = val_loader)


destroy_process_group()

# %%
