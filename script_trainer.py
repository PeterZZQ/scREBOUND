# In[]
from pathlib import Path
import sys, os

import numpy as np
import tqdm
from datetime import timedelta

# packages for distributed training
import torch
import torch.distributed as dist
from torch.utils import data
import torch.nn as nn

sys.path.append("./src")

import data_utils

import trainer
from transformer_model import TransformerModel, get_default_config
from transformers import AdamW, get_linear_schedule_with_warmup

from torch.nn.utils import clip_grad_norm_

# TODO: update to wandb
from torch.utils.tensorboard import SummaryWriter

# Save model checkpoint
def save_checkpoint(epoch, step, model, optimizer, scheduler, loss, path):
    checkpoint = {
        'epoch': epoch,
        'step': step,
        'model_state_dict': model.state_dict(),
        'model_config': model.model_config, # save the model config for repeated training too
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'loss': loss
    }
    torch.save(checkpoint, path)
    print(f"Checkpoint saved at epoch {epoch}.")

def train(model, train_loader, val_loader, optimizer, scheduler, writer, initial_epoch, initial_step, log_step):
    # NOTE: dynamic masking probability, small to large
    if model.model_config.dynamic_maskprob:
        mask_prob_init = 0.2
        mask_prob_end = 0.5
        mask_prob_step = (mask_prob_end - mask_prob_init) / len(train_loader) / (model.model_config.n_epoch - initial_epoch) * log_step
        model.model_config.mask_prob = mask_prob_init
    
    if model.model_config.use_discriminator:
        lamb_reverse_init = 0.0
        lamb_reverse_end = 1.0
        lamb_reverse_step = (lamb_reverse_end - lamb_reverse_init) / len(train_loader) / (model.model_config.n_epoch - initial_epoch) * log_step
        model.model_config.lamb_reverse = lamb_reverse_init


    # NOTE: training loop
    checkpoint_counter = 0
    for epoch in range(initial_epoch, model.model_config.n_epoch):
        torch.cuda.empty_cache()
        # Disable tqdm on all nodes except the rank 0 GPU on each server
        batch_iterator = tqdm.tqdm(train_loader, desc=f"Processing Epoch {epoch:02d}")

        # NOTE: Training
        running_loss = 0.0
        running_loss_mlm = 0.0
        running_loss_sup = 0.0
        running_loss_kd = 0.0      

        for step, data_sample in enumerate(batch_iterator):
            model.train()
            if step < initial_step:
                continue

            # NOTE: need to process the label into bincode
            if model.model_config.sup_type is not None:
                data_sample["label"] = model.label_bincode[data_sample["label"],:]

            loss, loss_mlm, loss_sup, loss_kd = trainer.infer_databatch(model, data_sample, multigpus = False)

            optimizer.zero_grad()
            loss.backward()

            # clip gradient
            max_grad_norm = 1.0 
            clip_grad_norm_(model.parameters(), max_grad_norm)

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
                    writer.add_scalar("Train Loss (DISC)", running_loss_kd, epoch * len(train_loader) + step + 1)
                    writer.add_scalar("Learning rate", scheduler.get_last_lr()[0], epoch * len(train_loader) + step + 1)

                    print(f"Epoch: {epoch}, Step: {step + 1}/{len(train_loader)}, Learning rate: {scheduler.get_last_lr()[0]:.2e}, Train Loss (TOTAL): {running_loss:.4f}, Train Loss (MLM): {running_loss_mlm:.4f}, Train Loss (CLASS): {running_loss_sup:.4f}, Train Loss (DISC): {running_loss_kd:.4f}")
                
                running_loss = 0.0
                running_loss_mlm = 0.0
                running_loss_sup = 0.0
                running_loss_kd = 0.0

                checkpoint_counter += 1

                # update the mask_prob 
                if model.model_config.dynamic_maskprob:
                    model.model_config.mask_prob += mask_prob_step
                
                # update the reverse gradient weight 
                if model.model_config.use_discriminator:
                    model.model_config.lamb_reverse += lamb_reverse_step
             
                if (checkpoint_counter == 10):
                    model.eval()
                    with torch.no_grad():
                        val_loss = 0.0
                        val_loss_mlm = 0.0
                        val_loss_sup = 0.0
                        val_loss_kd = 0.0
                        for data_sample in val_loader:
                            # NOTE: need to process the label into bincode
                            if model.module.model_config.sup_type is not None:
                                label_sample = model.label_bincode[data_sample["label"],:]
                                data_sample["label"] = label_sample

                            loss, loss_mlm, loss_sup, loss_kd = trainer.infer_databatch(model, data_sample, multigpus = False)                            
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
                            writer.add_scalar("Val Loss (DISC)", val_loss_kd, epoch * len(train_loader) + step + 1)
                            writer.add_scalar("Mask prob", model.model_config.mask_prob, epoch * len(train_loader) + step + 1)
                            writer.add_scalar("Disc lamb", model.model_config.lamb_reverse, epoch * len(train_loader) + step + 1)
                            print(f"Epoch: {epoch}, Step: {step + 1}/{len(train_loader)}, Val Loss (TOTAL): {val_loss:.4f}, Val Loss (MLM): {val_loss_mlm:.4f}, Val Loss (CLASS): {val_loss_sup:.4f}, Val Loss (DISC): {val_loss_kd:.4f}")

                            # save only for the writer gpu
                            save_checkpoint(epoch = epoch, step = step, model = model, optimizer = optimizer, scheduler = scheduler, loss = running_loss,
                                            path = f"{model.model_config.checkpoint_path}{model.model_config.checkpoint_prefix}_{epoch}_{step}.pth")
                    
                    checkpoint_counter = 0                
               
            initial_step = 0

    # save the final model, also only for the writer gpu
    if writer is not None:
        save_checkpoint(epoch = model.model_config.n_epoch, step = 0, model = model, optimizer = optimizer, scheduler = scheduler, loss = running_loss,
                        path = f"{model.model_config.checkpoint_path}{model.model_config.checkpoint_prefix}_{model.model_config.n_epoch}.pth")
        


# In[]
def initialize_services(log_dir):
    writer = SummaryWriter(log_dir=log_dir)
    return writer

def main():
    data_utils.set_seed(0)
    
    # Define the device
    assert torch.cuda.is_available(), "Training on CPU is not supported"
    device = torch.device("cuda:0")
    # Load the dataset
    print(f"GPU - Loading dataset...")

    n_mgene = 256
    # NOTE: save in localscratch for faster memory access
    # data_dir = Path(f"/project/zzhang834/LLM_KD/dataset/cellxgene")
    data_dir = Path(f"/project/zzhang834/LLM_KD/dataset/cellxgene")
    # load the token embedding
    token_embed = torch.load(data_dir / f"token_embed_{n_mgene}.pt")

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
                                  "n_layer": 6,
                                  "d_output": 64,
                                  "dropout": 0.05, # important for hyper-parameter tuning
                                  "mask_prob": 0.5, # important for hyper-parameter tuning
                                  "lamb_kd": 0.0,
                                  "lamb_sup": 100.0,
                                  "sup_type": "contrastive",
                                  "mlm_include_zero": False,
                                  "deep_injection": True,
                                  "use_discriminator": False, # improve the cluster with discriminator, could be used for finetunning
                                  "lamb_disc": 0.0,
                                  "pretrain_path": None,
                                  "checkpoint_path": "/project/zzhang834/LLM_KD/singlegpu/",
                                  "checkpoint_prefix": "checkpoint_0.3"
                                  })

    # construct dataset
    meta_dict = torch.load(data_dir / f"meta_{n_mgene}_bincode.pt", weights_only = False)
    scdataset = data_utils.sc_dataset_chunk(expr_path = data_dir / f"expr_sent_{n_mgene}.npz", gene_path = data_dir / f"feat_sent_{n_mgene}.npz",
                                            ncells = meta_dict["shape"]["full"][0], npads = meta_dict["shape"]["full"][1], labels = meta_dict["label"],
                                            batches = meta_dict["batch"], batch_size = model_config.batch_size)

    # train test split
    train_size = int(0.3 * len(scdataset))
    # NOTE: reduce the train size for better hyper-parameter tunning
    # train_size = int(0.01 * len(scdataset))
    val_size = int(0.001 * len(scdataset))

    # the data is already pre-shuffled
    train_dataset = data.Subset(scdataset, range(train_size))
    val_dataset = data.Subset(scdataset, range(train_size, train_size + val_size))
    test_dataset = data.Subset(scdataset, range(train_size + val_size, len(scdataset)))

    # obtain train/val/test loaders
    # NOTE: multi-gpus for only train_loader
    train_loader = data.DataLoader(train_dataset, batch_size = 1, shuffle = True, pin_memory = True, num_workers = 8, prefetch_factor = 8)
    val_loader = data.DataLoader(val_dataset, batch_size = 1, shuffle = False, pin_memory = True, num_workers = 8, prefetch_factor = 8)
    test_loader = data.DataLoader(test_dataset, batch_size = 1, shuffle = False, pin_memory = True, num_workers = 8)

    model_config.__dict__.update({"n_warmup_stp_lr": int(len(train_loader) * model_config.n_epoch * 0.1)})
    print(f"GPU - Done.")

    # model hyper-parameters
    fm_model = TransformerModel(model_config = model_config, token_embed = token_embed, n_batch = len(meta_dict["batch_code"]), n_label = len(meta_dict["label_code"]), device = device)

    # update the model parameters
    fm_model.label_bincode = torch.tensor(meta_dict["label_bincode"], dtype = torch.float32) #.to(device) label_bincode is moved to device in infer_databatch
    # NOTE: for the unknown labels, include the label mask, there are totally 898,317 cells with unknown labels (1.4% of total cell population)
    # the 677th dimension
    fm_model.label_mask = torch.tensor(meta_dict["label_bincode"][meta_dict["label_code"] == "unknown--unknown"].squeeze(), dtype = torch.float32).to(device)
    
    # init optimizer and scheduler, learning rate scale with the batch_size
    optimizer = AdamW(fm_model.parameters(), lr = model_config.lr)
    # update the warm-up steps to be 10% of total steps
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps = model_config.n_warmup_stp_lr, num_training_steps = model_config.n_epoch * len(train_loader))

    print(f"Linear scheduler with warmup, warmup steps: {model_config.n_warmup_stp_lr}, total steps: {model_config.n_epoch * len(train_loader)}, orig lr: {lr:.2e}")

    # Load latest checkpoint
    if model_config.pretrain_path is not None: 
        print(f"GPU - Preloading lastest model'")
        # load parameter from last train
        state = torch.load(model_config.pretrain_path)
        fm_model.load_state_dict(state["model_state_dict"])
        if state['step'] == 0:
            initial_epoch = state['epoch'] + 1
            initial_step = state['step']
        else:
            initial_epoch = state['epoch']
            initial_step = state['step'] + 1
        optimizer.load_state_dict(state['optimizer_state_dict'])
        scheduler.load_state_dict(state["scheduler_state_dict"])
        del state
    else:
        initial_epoch = 0
        initial_step = 0
        # If we couldn't find a model to preload, just start from scratch
        print(f'GPU - Could not find model to preload. Starting from scratch')

    # Init logger process, only main thread
    writer = initialize_services(model_config.checkpoint_path + model_config.checkpoint_prefix) 
   
    train(model = fm_model, train_loader = train_loader, val_loader = val_loader, optimizer = optimizer, scheduler = scheduler, writer = writer,
          initial_epoch = initial_epoch, initial_step = initial_step, log_step = 10)

# In[]
if __name__ == '__main__':

    main()
    

# %%
