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


import data_utils
from transformer_model import TransformerModel, ModelConfig, get_default_config
import trainer

from torch.nn.utils import clip_grad_norm_

# TODO: update to wandb
from torch.utils.tensorboard import SummaryWriter

# Save model checkpoint
def save_checkpoint(epoch, step, model, optimizer, scheduler, loss, path):
    checkpoint = {
        'epoch': epoch,
        'step': step,
        'model_state_dict': model.module.state_dict(),
        'model_config': model.module.model_config, # save the model config for repeated training too
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'loss': loss
    }
    torch.save(checkpoint, path)
    print(f"Checkpoint saved at epoch {epoch}.")

def train(model, train_loader, val_loader, optimizer, scheduler, writer, initial_epoch, initial_step, log_step):
    # NOTE: training loop
    for epoch in range(initial_epoch, model.module.model_config.n_epoch):
        torch.cuda.empty_cache()
        # Disable tqdm on all nodes except the rank 0 GPU on each server
        batch_iterator = tqdm.tqdm(train_loader, desc=f"Processing Epoch {epoch:02d} on rank {global_rank}", disable=local_rank != 0)

        # NOTE: Training
        running_loss = 0.0
        running_loss_mlm = 0.0
        running_loss_sup = 0.0
        running_loss_kd = 0.0
        checkpoint_counter = 0        

        for step, data_sample in enumerate(batch_iterator):
            model.module.train()
            
            if step < initial_step:
                continue
            loss, loss_mlm, loss_sup, loss_kd = trainer.infer_databatch(model, data_sample, multigpus = True)

            optimizer.zero_grad()
            loss.backward()

            # clip gradient
            max_grad_norm = 1.0 
            clip_grad_norm_(model.module.parameters(), max_grad_norm)

            optimizer.step()
            scheduler.step()

            # NOTE: log the results
            running_loss += loss.item()
            running_loss_mlm += loss_mlm.item()
            running_loss_sup += loss_sup.item()
            running_loss_kd += loss_kd.item()

            if step % log_step == log_step - 1:
                interval = (step % log_step)
                running_loss /= interval
                running_loss_mlm /= interval
                running_loss_sup /= interval
                running_loss_kd /= interval
                
                # NOTE: writer is not None only when global_rank == 0, make sure only one thread write the result
                if writer is not None:
                    writer.add_scalar("Train Loss (TOTAL)", running_loss, epoch * len(train_loader) + step + 1)
                    writer.add_scalar("Train Loss (MLM)", running_loss_mlm, epoch * len(train_loader) + step + 1)
                    writer.add_scalar("Train Loss (CLASS)", running_loss_sup, epoch * len(train_loader) + step + 1)
                    writer.add_scalar("Train Loss (KD)", running_loss_kd, epoch * len(train_loader) + step + 1)
                    writer.add_scalar("Learning rate", scheduler.get_last_lr()[0], epoch * len(train_loader) + step + 1)

                    print(f"Epoch: {epoch}, Step: {step + 1}/{len(train_loader)}, Learning rate: {scheduler.get_last_lr()[0]:.2e}, Train Loss (TOTAL): {running_loss:.4f}, Train Loss (MLM): {running_loss_mlm:.4f}, Train Loss (CLASS): {running_loss_sup:.4f}, Train Loss (KD): {running_loss_kd:.4f}")
                
                running_loss = 0.0
                running_loss_mlm = 0.0
                running_loss_sup = 0.0
                running_loss_kd = 0.0

                checkpoint_counter += 1
                # model evaluation and checkpoint saving
                # only the first for evaluation
                # if (global_rank == 0) & (checkpoint_counter == 10):
                # all gpus for evalution
                if (checkpoint_counter == 10):
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
                        if writer is not None:
                            writer.add_scalar("Val Loss (TOTAL)", val_loss, epoch * len(train_loader) + step + 1)
                            writer.add_scalar("Val Loss (MLM)", val_loss_mlm, epoch * len(train_loader) + step + 1)
                            writer.add_scalar("Val Loss (CLASS)", val_loss_sup, epoch * len(train_loader) + step + 1)
                            writer.add_scalar("Val Loss (KD)", val_loss_kd, epoch * len(train_loader) + step + 1)

                            print(f"Epoch: {epoch}, Step: {step + 1}/{len(train_loader)}, Val Loss (TOTAL): {val_loss:.4f}, Val Loss (MLM): {val_loss_mlm:.4f}, Val Loss (CLASS): {val_loss_sup:.4f}, Val Loss (KD): {val_loss_kd:.4f}")

                            # save only for the writer gpus
                            save_checkpoint(epoch = epoch, step = step, model = model, optimizer = optimizer, scheduler = scheduler, loss = running_loss,
                                            path = f"{model.module.model_config.checkpoint_path}{model.module.model_config.checkpoint_prefix}_{epoch}_{step}.pth")
                    
                    checkpoint_counter = 0
                # sync all gpus after eval
                dist.barrier()
                
            initial_step = 0

    # save the final model
    save_checkpoint(epoch = model.module.model_config.n_epoch, step = 0, model = model, optimizer = optimizer, scheduler = scheduler, loss = running_loss,
                    path = f"{model.module.model_config.checkpoint_path}{model.module.model_config.checkpoint_prefix}_{model.module.model_config.n_epoch}.pth")

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
    # data_dir = Path(f"/project/zzhang834/LLM_KD/dataset/cellxgene")
    data_dir = Path(f"/localscratch/ziqi/localscratch_tempdata/cellxgene")
    # load the token embedding
    token_embed = torch.load(data_dir / f"token_embed_{n_mgene}.pt")
    # load the cell meta-info
    meta_dict = torch.load(data_dir / f"meta_{n_mgene}_permu.pt")


    model_config = get_default_config()
    batch_size = 512
    # classifier
    lr = 0.5e-5 * (batch_size/32)


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
                                  "dynamic_maskprob": False,
                                  "lamb_kd": 0.0,
                                  "lamb_sup": 1.0,
                                  "sup_type": "contrastive",
                                  "sample_mlm": False,
                                  "mlm_include_zero": False,
                                  "pretrain_path": None,
                                  "checkpoint_path": "/project/zzhang834/LLM_KD/checkpoint_predfull/",
                                  "checkpoint_prefix": "checkpoint_0.98_contr"
                                  })

    # construct dataset
    scdataset = data_utils.sc_dataset_chunk(expr_path = data_dir / f"expr_sent_{n_mgene}_permu.npz", gene_path = data_dir / f"feat_sent_{n_mgene}_permu.npz",
                                            ncells = meta_dict["shape"][0], npads = meta_dict["shape"][1], labels = meta_dict["label"], batches = meta_dict["batch"], batch_size = model_config.batch_size)

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
        state = torch.load(model_config.pretrain_path)
        fm_model.module.load_state_dict(state["model_state_dict"])
        # update optimizer and scheduler for consistent training
        optimizer.load_state_dict(state['optimizer_state_dict'])
        scheduler.load_state_dict(state["scheduler_state_dict"])

        if state['step'] == 0:
            initial_epoch = state['epoch'] + 1
            initial_step = state['step']
        else:
            initial_epoch = state['epoch']
            initial_step = state['step'] + 1
        del state
    else:
        initial_epoch = 0
        initial_step = 0
        # If we couldn't find a model to preload, just start from scratch
        print(f'GPU {local_rank} - Could not find model to preload. Starting from scratch')

    # Init logger process, only main thread
    if global_rank == 0:
        writer = initialize_services(model_config.checkpoint_path + model_config.checkpoint_prefix) 
    else:
        writer = None

    # sync
    dist.barrier()

    train(model = fm_model, train_loader = train_loader, val_loader = val_loader, optimizer = optimizer, scheduler = scheduler, writer = writer,
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
