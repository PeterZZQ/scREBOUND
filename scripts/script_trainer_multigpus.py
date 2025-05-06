# In[]
from pathlib import Path
import sys, os

import numpy as np
import pandas as pd
from datetime import timedelta

# packages for distributed training
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.distributed import init_process_group, destroy_process_group
# from transformers import AdamW, get_linear_schedule_with_warmup
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR

sys.path.append("/project/zzhang834/LLM_KD/src")

import data_utils
from transformer_batch import TransformerModel, get_default_config
import trainer_batch as trainer_batch

# TODO: update to wandb
from torch.utils.tensorboard import SummaryWriter

from torch.nn.attention import sdpa_kernel, SDPBackend


def initialize_services(log_dir):
    writer = SummaryWriter(log_dir=log_dir)
    return writer

# In[]
def main():
    data_utils.set_seed(0)
    PROJECT_DIR = "/project/zzhang834/LLM_KD/"
    data_dir = "/project/zzhang834/hs_download/permuted/"

    # Define the device
    assert torch.cuda.is_available(), "Training on CPU is not supported"
    device = torch.device("cuda")
    print(f"GPU {local_rank} - Using device: {device}")
    # Load the dataset
    print(f"GPU {local_rank} - Loading dataset...")

    n_mgene = 256
    model_config = get_default_config()
    batch_size = 64 * 8 * 2
    # classifier for 4 gpus, 0.5e-5 too large for less than 4, slightly larger for bp16
    lr = 0.3e-5 * (batch_size/32/2)

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
                                  "checkpoint_prefix": f"cp_vanilla_4_512_meta_enc_wmask_{batch_name}",
                                  "lognorm_data": False
                                  })

    token_dict = torch.load(f"/project/zzhang834/hs_download/gene_embed_meta{n_mgene}_gpool.pt", weights_only = False)
    label_dict = torch.load(data_dir + "label_dict.pt", weights_only = False)
    batch_dict = torch.load(PROJECT_DIR + f"batch_encoding/batch_dict_batch_{batch_name}.pt", weights_only = False)
    batch_dict["cats"] = torch.tensor(batch_dict["cats"].values, dtype = torch.int32)

    fm_model = TransformerModel(model_config = model_config, token_dict = token_dict, batch_dict = batch_dict, label_dict = label_dict, device = device)

    # wrap model into multi-gpus setting
    fm_model = DistributedDataParallel(fm_model, device_ids=[local_rank])

    # data information
    num_partitions = 55
    partition_size = 1000000
    # remove the last partition as it is for validation
    steps_per_partition = np.ceil(partition_size/model_config.batch_size/num_gpus) # 489
    # the partitions are complete
    steps_per_epoch = int((num_partitions-1) * steps_per_partition)
    if global_rank == 0:
        print(f"total number of steps: {steps_per_epoch:d}")

    # init optimizer and scheduler, learning rate scale with the batch_size, larger eps for more stability in bf16
    fm_model.module.optimizer = AdamW(fm_model.module.parameters(), lr = model_config.lr, eps = 1e-6)
    fm_model.module.scheduler = OneCycleLR(fm_model.module.optimizer, max_lr = model_config.lr, steps_per_epoch = steps_per_epoch, epochs = model_config.n_epoch, pct_start = 0.3)

    # if global_rank == 0:
    #     print(f"Linear scheduler with warmup, warmup steps: {model_config.n_warmup_stp_lr}, total steps: {model_config.n_epoch * len(train_loader)}, orig lr: {lr:.2e}")

    # Load latest checkpoint
    if model_config.pretrain_path is not None: 
        print(f"GPU {local_rank} - Preloading lastest model")
        # load parameter from last train
        state = torch.load(model_config.pretrain_path, weights_only = False)
        # Get the common keys between the current model and the saved model
        filtered_state_dict = {k: v for k, v in state["model_state_dict"].items() if k in fm_model.module.state_dict()}
        # Load the filtered state dictionary into the model
        fm_model.module.load_state_dict(filtered_state_dict, strict = False)

        # NOTE: for continuous training, update optimizer and scheduler for consistent training
        fm_model.module.optimizer.load_state_dict(state['optimizer_state_dict'])
        fm_model.module.scheduler.load_state_dict(state["scheduler_state_dict"])
        initial_epoch = state['epoch']
        # NOTE: prev step has done, save happens at the end
        initial_step = state['step'] + 1
        del state
    else:
        initial_epoch = 0
        initial_step = 0
        # If we couldn't find a model to preload, just start from scratch
        print(f'GPU {local_rank} - Could not find model to preload. Starting from scratch')


    # calculate the dynamical value steps:
    log_step = 100
    if fm_model.module.model_config.dynamic_maskprob:
        mask_prob_init = 0.15
        mask_prob_end = 0.4
        fm_model.module.model_config.maskprob_step = (mask_prob_end - mask_prob_init) / steps_per_epoch / (fm_model.module.model_config.n_epoch - initial_epoch) * log_step
        fm_model.module.model_config.mask_prob = mask_prob_init

    # if fm_model.module.model_config.use_discriminator:
    #     lamb_reverse_init = 1.0
    #     lamb_reverse_end = 1.0
    #     fm_model.module.model_config.lamb_reverse_step = (lamb_reverse_end - lamb_reverse_init) / steps_per_epoch / (fm_model.module.model_config.n_epoch - initial_epoch) * log_step
    #     fm_model.module.model_config.lamb_reverse = lamb_reverse_init
    # else:
    #     fm_model.module.model_config.lamb_reverse = 0.0

    # Init logger process, only main thread
    if global_rank == 0:
        writer = initialize_services(model_config.checkpoint_path + model_config.checkpoint_prefix) 
    else:
        writer = None

    # sync
    dist.barrier()

    dataset_dict = {"DIR": data_dir, "num_partitions": num_partitions, "data_prefix": "counts", "meta_prefix": "obs", "batch_dict": batch_dict, "label_colname": "label_id", "batch_colname": "batch_" + batch_name + "_id"}

    with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
        trainer_batch.train_multigpus(model = fm_model, global_rank = global_rank, dataset_dict = dataset_dict, writer = writer,
                                initial_epoch = initial_epoch, initial_step = initial_step, log_step = log_step)

                    
                    

# In[]
if __name__ == '__main__':

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

    main()
    
    destroy_process_group()

# %%
