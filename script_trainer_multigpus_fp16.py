# In[]
from pathlib import Path
import sys, os

import numpy as np
import tqdm

# packages for distributed training
import torch
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel
from torch.distributed import init_process_group, destroy_process_group
from torch.utils import data
import torch.nn as nn
# for fp16 training
from torch.amp import autocast, GradScaler


import data_utils
from transformer_model import TransformerModel, ModelConfig, get_default_config
from transformers import AdamW, get_linear_schedule_with_warmup

# TODO: update to wandb
from torch.utils.tensorboard import SummaryWriter

# Train function
def train_epoch(model, train_loader, optimizer, scheduler, scaler, writer, epoch, init_step, log_step = 1000):
    model.module.train()
    running_loss = 0.0
    running_loss_mlm = 0.0
    running_loss_sup = 0.0
    running_loss_kd = 0.0

    for step, data_sample in enumerate(train_loader):
        if step < init_step:
            continue
        expr_sample, gene_sample, label_sample, batch_sample = data_sample
        expr_sample = expr_sample.to(model.device, non_blocking = True)
        gene_sample = gene_sample.to(model.device, non_blocking = True)
        label_sample = label_sample.to(model.device, non_blocking = True)
        batch_sample = batch_sample.to(model.device, non_blocking = True)
        
        optimizer.zero_grad()
        with autocast(device_type="cuda"):
            _, cell_embed, mask = model(gene_sent = gene_sample, expr_sent = expr_sample)
            
            # sample from the masked pool
            mask_gene_sample = []
            mask_expr_sample = []

            # loop through every single cell within the batch
            for i in range(cell_embed.shape[0]):
                # 1. target gene can be the gene with 0 expression (not included in the gene sentence)
                masked_genes = torch.tensor([x for x in model.module.gene_idx if x not in gene_sample[i]]).to(model.device)
                masked_exprs = torch.zeros_like(masked_genes)
                # 2. target gene can be the gene with mask
                masked_genes = torch.hstack([masked_genes, gene_sample[i, mask[i]]])
                masked_exprs = torch.hstack([masked_exprs, expr_sample[i, mask[i]]])
                # sample
                idx = np.random.choice(len(masked_genes), 1)
                mask_gene_sample.append(gene_sample[i, idx])
                mask_expr_sample.append(expr_sample[i, idx])

            mask_gene_sample = torch.cat(mask_gene_sample)
            mask_expr_sample = torch.cat(mask_expr_sample)
            
            expr_pred = model.module.predict_expr(cell_embed = cell_embed, gene_cond = mask_gene_sample, batch_cond = batch_sample)
            # cell type label
            label_pred = model.module.predict_label(cell_embed = cell_embed, batch_cond = batch_sample)

            # calculate the loss
            # 1. MSE loss between the predict expression and ground truth expression
            mse = nn.MSELoss()
            loss_mlm = mse(expr_pred, mask_expr_sample.unsqueeze(1))

            # 2. Classification loss between the predict label and the ground truth label
            ce = nn.CrossEntropyLoss()
            loss_sup = ce(label_pred, label_sample)

            # 3. KD loss from teacher model
            loss_kd = torch.tensor([0], device = model.device)
            loss = loss_mlm + loss_sup + model.model_config.lamb_kd * loss_kd
        
        # loss.backward()
        # optimizer.step()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        scheduler.step()

        # NOTE: log the results
        running_loss_mlm += loss_mlm.item()
        running_loss_sup += loss_sup.item()
        running_loss_kd += loss_kd.item()
        running_loss += loss.item()

        if (step % log_step == log_step - 1) or (step == len(train_loader) - 1):
            if step == len(train_loader) - 1:
                interval = step
            else:
                interval = log_step
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

                print(f"Epoch: {epoch}, Step: {step + 1}/{len(train_loader)}, Train Loss (TOTAL): {running_loss:.4f}, Train Loss (MLM): {running_loss_mlm:.4f}, Train Loss (CLASS): {running_loss_sup:.4f}, Train Loss (KD): {running_loss_kd:.4f}")

            if iter != len(train_loader) - 1:
                running_loss = 0.0
                running_loss_mlm = 0.0
                running_loss_sup = 0.0
                running_loss_kd = 0.0
            
            # save the current model, for only gpu 0
            if global_rank == 0:
                save_checkpoint(epoch = epoch, step = step, model = model, optimizer = optimizer,
                                scheduler = scheduler, loss = running_loss, path = f"checkpoint/checkpoint_multigpus_{epoch}_{step}.pth")

    return running_loss, running_loss_mlm, running_loss_sup, running_loss_kd

# Evaluation function
def evaluate_epoch(model, val_loader, writer, epoch):
    model.module.eval()
    val_loss = 0.0
    val_loss_mlm = 0.0
    val_loss_sup = 0.0
    val_loss_kd = 0.0
    with torch.no_grad():
        for data_sample in val_loader:
            expr_sample, gene_sample, label_sample, batch_sample = data_sample
            expr_sample = expr_sample.to(model.device, non_blocking = True)
            gene_sample = gene_sample.to(model.device, non_blocking = True)
            label_sample = label_sample.to(model.device, non_blocking = True)
            batch_sample = batch_sample.to(model.device, non_blocking = True)
            
            # Forward pass
            with autocast(device_type="cuda"):
                _, cell_embed, mask = model(gene_sent = gene_sample, expr_sent = expr_sample)
                
                # sample from the masked pool
                mask_gene_sample = []
                mask_expr_sample = []
                for i in range(cell_embed.shape[0]):
                    # 1. target gene can be the gene with 0 expression (not included in the gene sentence)
                    masked_genes = torch.tensor([x for x in model.module.gene_idx if x not in gene_sample[i]]).to(model.device)
                    masked_exprs = torch.zeros_like(masked_genes)
                    # 2. target gene can be the gene with mask
                    masked_genes = torch.hstack([masked_genes, gene_sample[i, mask[i]]])
                    masked_exprs = torch.hstack([masked_exprs, expr_sample[i, mask[i]]])
                    # sample
                    idx = np.random.choice(len(masked_genes), 1)
                    mask_gene_sample.append(gene_sample[i, idx])
                    mask_expr_sample.append(expr_sample[i, idx])           

                mask_gene_sample = torch.cat(mask_gene_sample)
                mask_expr_sample = torch.cat(mask_expr_sample)
                
                expr_pred = model.module.predict_expr(cell_embed = cell_embed, gene_cond = mask_gene_sample, batch_cond = batch_sample)
                # cell type label
                label_pred = model.module.predict_label(cell_embed = cell_embed, batch_cond = batch_sample)

                # calculate the loss
                # 1. MSE loss between the predict expression and ground truth expression
                mse = nn.MSELoss()
                loss_mlm = mse(expr_pred, mask_expr_sample.unsqueeze(1))

                # 2. Classification loss between the predict label and the ground truth label
                ce = nn.CrossEntropyLoss()
                loss_sup = ce(label_pred, label_sample)
                
                # 3. KD loss from teacher model
                loss_kd = torch.tensor([0], device = model.device)
                loss = loss_mlm + loss_sup + model.model_config.lamb_kd * loss_kd

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
            writer.add_scalar("Val Loss (TOTAL)", val_loss, epoch)
            writer.add_scalar("Val Loss (MLM)", val_loss_mlm, epoch)
            writer.add_scalar("Val Loss (CLASS)", val_loss_sup, epoch)
            writer.add_scalar("Val Loss (KD)", val_loss_kd, epoch)

            print(f"Epoch: {epoch}, Val Loss (TOTAL): {val_loss:.4f}, Val Loss (MLM): {val_loss_mlm:.4f}, Val Loss (CLASS): {val_loss_sup:.4f}, Val Loss (KD): {val_loss_kd:.4f}")

    return val_loss, val_loss_mlm, val_loss_sup, val_loss_sup, val_loss_kd

# Save model checkpoint
def save_checkpoint(epoch, step, model, optimizer, scheduler, loss, path):
    checkpoint = {
        'epoch': epoch,
        'step': step,
        'model_state_dict': model.module.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'loss': loss
    }
    torch.save(checkpoint, path)
    print(f"Checkpoint saved at epoch {epoch}.")

def initialize_services():
    writer = SummaryWriter(log_dir='./logs')
    return writer

# In[]
def train():
    data_utils.set_seed(0)
    
    # Define the device
    assert torch.cuda.is_available(), "Training on CPU is not supported"
    device = torch.device("cuda")
    print(f"GPU {local_rank} - Using device: {device}")
    # Load the dataset
    print(f"GPU {local_rank} - Loading dataset...")


    print("loading dataset...")
    n_mgene = 256
    data_dir = Path(f"/projects/zzhang834/LLM_KD/dataset/cellxgene")
    # load the token embedding
    token_embed = torch.load(data_dir / f"token_embed_{n_mgene}.pt")
    # load the cell meta-info
    meta_dict = torch.load(data_dir / f"meta_{n_mgene}.pt")
    # construct dataset
    scdataset = data_utils.sc_dataset(expr_path = data_dir / f"expr_sent_{n_mgene}.npz", gene_path = data_dir / f"feat_sent_{n_mgene}.npz",
                                    ncells = meta_dict["shape"][0], npads = meta_dict["shape"][1], labels = meta_dict["label"], batches = meta_dict["batch"])

    # train test split
    # train_size = int(0.95 * len(scdataset))
    # NOTE: reduce the train size for better hyper-parameter tunning
    train_size = int(0.1 * len(scdataset))
    val_size = int(0.01 * len(scdataset))
    test_size = len(scdataset) - train_size - val_size
    train_dataset, val_dataset, test_datset = data.random_split(scdataset, [train_size, val_size, test_size])

    model_config = get_default_config()
    batch_size = 1024
    model_config.__dict__.update({"batch_size": batch_size,
                                  "n_epoch": 3,
                                  "lr": 2e-5 * batch_size/32, # important for hyper-parameter tuning
                                  "n_warmup_stp_lr": 1000, # important for hyper-parameter tuning
                                  "d_token": token_embed.shape[1],
                                  "d_embed": 512,
                                  "n_head": 8,
                                  "d_hidden": 2048, 
                                  "n_layer": 4,
                                  "d_output": 256,
                                  "dropout": 0.05, # important for hyper-parameter tuning
                                  "mask_prob": 0.15, # important for hyper-parameter tuning
                                  "lamb_kd": 0.0,
                                  "pretrain_path": None # "checkpoint/checkpoint_multigpus.pth"
                                  })

    # init scaler for fp16 training
    scaler = GradScaler()
    # obtain train/val/test loaders
    # NOTE: multi-gpus for only train_loader
    train_loader = data.DataLoader(train_dataset, batch_size = model_config.batch_size, shuffle = False, pin_memory = True, sampler = DistributedSampler(train_dataset, shuffle=True))
    val_loader = data.DataLoader(val_dataset, batch_size = model_config.batch_size, shuffle = False, pin_memory = True)
    test_loader = data.DataLoader(test_datset, batch_size = model_config.batch_size, shuffle = False, pin_memory = True)
    print(f"GPU {local_rank} - Done.")

    # model hyper-parameters
    fm_model = TransformerModel(model_config = model_config, token_embed = token_embed, n_batch = len(meta_dict["batch_code"]), n_label = len(meta_dict["label_code"]), device = device)
    # wrap model into multi-gpus setting
    fm_model = DistributedDataParallel(fm_model, device_ids=[local_rank])
    # init optimizer and scheduler, learning rate scale with the batch_size
    optimizer = AdamW(fm_model.parameters(), lr = model_config.lr)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps = model_config.n_warmup_stp_lr, num_training_steps = model_config.n_epoch * len(train_loader))

    # Load latest checkpoint
    if model_config.pretrain_path is not None: 
        print(f"GPU {local_rank} - Preloading lastest model'")
        # load parameter from last train
        state = torch.load(model_config.pretrain_path)
        fm_model.module.load_state_dict(state["model_state_dict"])
        initial_epoch = state['epoch'] + 1
        initial_step = state['step'] + 1
        optimizer.load_state_dict(state['optimizer_state_dict'])
        scheduler.load_state_dict(state["scheduler_state_dict"])
        del state
    else:
        initial_epoch = 0
        initial_step = 0
        # If we couldn't find a model to preload, just start from scratch
        print(f'GPU {local_rank} - Could not find model to preload. Starting from scratch')

    # Init logger process, only main thread
    if global_rank == 0:
        writer = initialize_services() 
    else:
        writer = None

    # NOTE: training loop
    for epoch in range(initial_epoch, model_config.n_epoch):
        torch.cuda.empty_cache()

        # Disable tqdm on all nodes except the rank 0 GPU on each server
        batch_iterator = tqdm.tqdm(train_loader, desc=f"Processing Epoch {epoch:02d} on rank {global_rank}", disable=local_rank != 0)

        # distribute to multi-gpus, gradient sync at backward()
        train_loss, *_ = train_epoch(model = fm_model, train_loader = batch_iterator, optimizer = optimizer, scaler = scaler, scheduler = scheduler, writer = writer, epoch = epoch, init_step = initial_step)
        # only run evaluation on one gpu
        if global_rank == 0:
            val_loss, *_ = evaluate_epoch(model = fm_model, val_loader = val_loader, writer = writer, epoch = epoch)
            print(f"Epoch {epoch + 1}/{model_config.n_epoch}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

            # Save a checkpoint if validation loss improves, NOTE: only save for one thread, avoid conflicting
            save_checkpoint(epoch = epoch, step = 0, model = fm_model, optimizer = optimizer, scheduler = scheduler, loss = val_loss, path = f"checkpoint/checkpoint_multigpus_{epoch}_{0}.pth")

        # update step for each epoch
        initial_step = 0

        # [Opt] early stopping, not needed


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

    init_process_group(backend='nccl')
    torch.cuda.set_device(local_rank) # Set the device to local rank

    train()
    
    destroy_process_group()

# %%
