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
from transformers import AdamW, get_linear_schedule_with_warmup

# for fp16 training
from torch.amp import autocast, GradScaler

sys.path.append("./src")

import data_utils
from transformer_model import TransformerModel, ModelConfig, get_default_config
import trainer

from torch.nn.utils import clip_grad_norm_

# TODO: update to wandb
from torch.utils.tensorboard import SummaryWriter


def initialize_services(log_dir):
    writer = SummaryWriter(log_dir=log_dir)
    return writer

# In[]
def main():
    data_utils.set_seed(1)
    
    # Define the device
    assert torch.cuda.is_available(), "Training on CPU is not supported"
    device = torch.device("cuda")
    print(f"GPU {local_rank} - Using device: {device}")
    # Load the dataset
    print(f"GPU {local_rank} - Loading dataset...")

    n_mgene = 256
    # NOTE: save in localscratch for faster memory access
    # no hvgs
    data_dir = Path(f"/project/zzhang834/LLM_KD/dataset/cellxgene")
    # top 4000 hvgs
    # data_dir = Path(f"/project/zzhang834/LLM_KD/dataset/cellxgene_4000hvg")
    # load the token embedding
    token_embed = torch.load(data_dir / f"token_embed_{n_mgene}.pt", weights_only = False)

    model_config = get_default_config()
    batch_size = 512
    # classifier for 4 gpus, 0.5e-5 too large for less than 4 
    lr = 0.3e-5 * (batch_size/32)

    model_config.__dict__.update({"batch_size": batch_size,
                                  "n_epoch": 1,
                                  "lr": lr, # important for hyper-parameter tuning
                                  "n_warmup_stp_lr": 4000, # important for hyper-parameter tuning, 5-10% of total steps
                                  "d_embed": 512,
                                  "n_head": 8,  # TODO: make 12 head * 64 dimensions
                                  "d_hidden": 2048, 
                                  "n_layer": 8,
                                  "d_output": 64,
                                  "dropout": 0.05, # important for hyper-parameter tuning
                                  "mask_prob": 0.4, # important for hyper-parameter tuning
                                  "dynamic_maskprob": True, # mask_prob is dynamically updated from 0.1 to 0.7 during training
                                  "lamb_kd": 0.0,
                                  "lamb_sup": 0.0,
                                  "sup_type": None,
                                  "mlm_include_zero": False,
                                  "deep_injection": True,
                                  "use_discriminator": False, # improve the cluster with discriminator, could be used for finetunning
                                  "lamb_disc": 0.0,
                                  "use_fastatten": True,
                                  "pretrain_path": None,
                                  "checkpoint_path":"/project/zzhang834/LLM_KD/checkpoint_fastatten/",
                                  "checkpoint_prefix": "checkpoint_8_512",
                                  })
    
    # construct dataset
    # load the cell meta-info
    meta_dict = torch.load(data_dir / f"meta_{n_mgene}_bincode.pt", weights_only = False)
    # total number of batches is 604, maximum batch size is 2million, minimum batch size is 843
    scdataset = data_utils.sc_dataset_chunk(expr_path = data_dir / f"expr_sent_{n_mgene}.npz", gene_path = data_dir / f"feat_sent_{n_mgene}.npz",
                                            ncells = meta_dict["shape"]["full"][0], npads = meta_dict["shape"]["full"][1], labels = meta_dict["label"],
                                            batches = meta_dict["batch"], batch_size = model_config.batch_size)



    # train test split
    train_size = int(0.98 * len(scdataset))
    val_size = int(0.004 * len(scdataset))

    # the data is already pre-shuffled
    train_dataset = data.Subset(scdataset, range(train_size))
    val_dataset = data.Subset(scdataset, range(train_size, train_size + val_size))
    test_dataset = data.Subset(scdataset, range(train_size + val_size, len(scdataset)))

    # obtain train/val/test loaders
    # NOTE: multi-gpus for only train_loader
    train_loader = data.DataLoader(train_dataset, batch_size = 1, shuffle = False, pin_memory = True, sampler = DistributedSampler(train_dataset, shuffle = True), num_workers = 8, prefetch_factor = 8)
    # val_loader = data.DataLoader(val_dataset, batch_size = 1, shuffle = False, pin_memory = True, num_workers = 8, prefetch_factor = 8)
    val_loader = data.DataLoader(val_dataset, batch_size = 1, shuffle = False, pin_memory = True, sampler = DistributedSampler(val_dataset, shuffle = False), num_workers = 8, prefetch_factor = 8)
    test_loader = data.DataLoader(test_dataset, batch_size = 1, shuffle = False, pin_memory = True, num_workers = 8)

    # update the warm-up steps to be 10% of total steps, make suret that model_config is consistent with fm_model configuration
    model_config.__dict__.update({"n_warmup_stp_lr": int(len(train_loader) * model_config.n_epoch * 0.2)})
    print(f"GPU {local_rank} - Done.")

    # model hyper-parameters
    fm_model = TransformerModel(model_config = model_config, token_embed = token_embed, n_batch = len(meta_dict["batch_code"]), n_label = len(meta_dict["label_code"]), device = device)

    # update the model parameters
    fm_model.label_bincode = torch.tensor(meta_dict["label_bincode"], dtype = torch.float32) #.to(device) label_bincode is moved to device in infer_databatch
    # NOTE: for the unknown labels, include the label mask, there are totally 898,317 cells with unknown labels (1.4% of total cell population)
    # the 677th dimension
    fm_model.label_mask = torch.tensor(meta_dict["label_bincode"][meta_dict["label_code"] == "unknown--unknown"].squeeze(), dtype = torch.float32).to(device)

    # wrap model into multi-gpus setting
    fm_model = DistributedDataParallel(fm_model, device_ids=[local_rank])
    # init optimizer and scheduler, learning rate scale with the batch_size
    optimizer = AdamW(fm_model.parameters(), lr = model_config.lr)

    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps = model_config.n_warmup_stp_lr, num_training_steps = model_config.n_epoch * len(train_loader))

    if global_rank == 0:
        print(f"Linear scheduler with warmup, warmup steps: {model_config.n_warmup_stp_lr}, total steps: {model_config.n_epoch * len(train_loader)}, orig lr: {lr:.2e}")

    # Load latest checkpoint
    if model_config.pretrain_path is not None: 
        print(f"GPU {local_rank} - Preloading lastest model'")
        # load parameter from last train
        state = torch.load(model_config.pretrain_path, weights_only = False)
        # Get the common keys between the current model and the saved model
        filtered_state_dict = {k: v for k, v in state["model_state_dict"].items() if k in fm_model.module.state_dict()}
        # Load the filtered state dictionary into the model
        fm_model.module.load_state_dict(filtered_state_dict, strict=False)

        # NOTE: for continuous training, update optimizer and scheduler for consistent training
        optimizer.load_state_dict(state['optimizer_state_dict'])
        scheduler.load_state_dict(state["scheduler_state_dict"])
        initial_epoch = state['epoch']
        initial_step = state['step']
        del state
    else:
        initial_epoch = 0
        initial_step = 0
        # If we couldn't find a model to preload, just start from scratch
        print(f'GPU {local_rank} - Could not find model to preload. Starting from scratch')

    if model_config.sup_type == "classifier-bincode":
        fm_model.module.label_bincode = torch.tensor(meta_dict["label_bincode"], dtype = torch.int32)

    # NOTE: for the unknown labels, include the label mask
    fm_model.module.label_mask = torch.tensor((meta_dict["label_code"] == "unknown"), dtype = torch.float32).to(device)

    # Init logger process, only main thread
    if global_rank == 0:
        writer = initialize_services(model_config.checkpoint_path + model_config.checkpoint_prefix) 
    else:
        writer = None

    # sync
    dist.barrier()

    trainer.train_multigpus_fastatten(model = fm_model, global_rank = global_rank, train_loader = train_loader, val_loader = val_loader, optimizer = optimizer, scheduler = scheduler, writer = writer,
                                      initial_epoch = initial_epoch, initial_step = initial_step, log_step = 100)

                    

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
