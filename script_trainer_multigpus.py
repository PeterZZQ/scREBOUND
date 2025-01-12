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

import data_utils
import transformer_model as trans
from transformers import AdamW, get_linear_schedule_with_warmup

# TODO: update to wandb
from torch.utils.tensorboard import SummaryWriter

# Train function
def train_epoch(model, train_loader, optimizer, scheduler, device, writer, epoch, lamb_kd = 1, log_iter = 100):
    model.train()
    running_loss = 0.0
    running_loss_mlm = 0.0
    running_loss_sup = 0.0
    running_loss_kd = 0.0

    for iter, data_sample in enumerate(train_loader):
        expr_sample, gene_sample, label_sample, batch_sample = data_sample
        expr_sample = expr_sample.to(device, non_blocking = True)
        gene_sample = gene_sample.to(device, non_blocking = True)
        label_sample = label_sample.to(device, non_blocking = True)
        batch_sample = batch_sample.to(device, non_blocking = True)
        
        optimizer.zero_grad()
        _, cell_embed, mask = model(gene_sent = gene_sample, expr_sent = expr_sample, mask_prob = 0.15)
        
        # sample from the masked pool
        mask_gene_sample = []
        mask_expr_sample = []

        # loop through every single cell within the batch
        for i in range(cell_embed.shape[0]):
            # 1. target gene can be the gene with 0 expression (not included in the gene sentence)
            masked_genes = torch.tensor([x for x in model.module.gene_idx if x not in gene_sample[i]]).to(device)
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
        loss_kd = torch.tensor([0], device = device)
        loss = loss_mlm + loss_sup + lamb_kd * loss_kd
    
        loss.backward()
        optimizer.step()
        scheduler.step()

        # NOTE: log the results
        running_loss_mlm += loss_mlm.item()
        running_loss_sup += loss_sup.item()
        running_loss_kd += loss_kd.item()
        running_loss += loss.item()

        if (iter % log_iter == log_iter - 1) or (iter == len(train_loader) - 1):
            if iter == len(train_loader) - 1:
                interval = iter
            else:
                interval = log_iter
            running_loss /= interval
            running_loss_mlm /= interval
            running_loss_sup /= interval
            running_loss_kd /= interval
            
            # NOTE: writer is not None only when global_rank == 0, make sure only one thread write the result
            if writer is not None:
                writer.add_scalar("Train Loss (TOTAL)", running_loss, epoch * len(train_loader) + iter)
                writer.add_scalar("Train Loss (MLM)", running_loss_mlm, epoch * len(train_loader) + iter)
                writer.add_scalar("Train Loss (CLASS)", running_loss_sup, epoch * len(train_loader) + iter)
                writer.add_scalar("Train Loss (KD)", running_loss_kd, epoch * len(train_loader) + iter)

                print(f"Epoch: {epoch}, Step: {iter}/{len(train_loader)}, Train Loss (TOTAL): {running_loss:.4f}, Train Loss (MLM): {running_loss_mlm:.4f}, Train Loss (CLASS): {running_loss_sup:.4f}, Train Loss (KD): {running_loss_kd:.4f}")

            if iter != len(train_loader) - 1:
                running_loss = 0.0
                running_loss_mlm = 0.0
                running_loss_sup = 0.0
                running_loss_kd = 0.0
            
    return running_loss, running_loss_mlm, running_loss_sup, running_loss_kd

# Evaluation function
def evaluate_epoch(model, val_loader, device, writer, epoch, lamb_kd = 1):
    model.eval()
    val_loss = 0.0
    val_loss_mlm = 0.0
    val_loss_sup = 0.0
    val_loss_kd = 0.0
    with torch.no_grad():
        for data_sample in val_loader:
            expr_sample, gene_sample, label_sample, batch_sample = data_sample
            expr_sample = expr_sample.to(device, non_blocking = True)
            gene_sample = gene_sample.to(device, non_blocking = True)
            label_sample = label_sample.to(device, non_blocking = True)
            batch_sample = batch_sample.to(device, non_blocking = True)
            
            # Forward pass
            _, cell_embed, mask = model(gene_sent = gene_sample, expr_sent = expr_sample, mask_prob = 0.15)
            
            # sample from the masked pool
            mask_gene_sample = []
            mask_expr_sample = []
            for i in range(cell_embed.shape[0]):
                # 1. target gene can be the gene with 0 expression (not included in the gene sentence)
                masked_genes = torch.tensor([x for x in model.module.gene_idx if x not in gene_sample[i]]).to(device)
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
            loss_kd = torch.tensor([0], device = device)
            loss = loss_mlm + loss_sup + lamb_kd * loss_kd

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
def save_checkpoint(epoch, model, optimizer, scheduler, loss, path):
    checkpoint = {
        'epoch': epoch,
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

# Dummy variables to make Pylance happy :D
train_dataset = None
local_rank = -1
global_rank = -1
n_epochs = 5
step_number = 0
last_step = False

def train():

    data_utils.set_seed(0)
    n_mgene = 256
    res_dir = Path(f"/project/zzhang834/LLM_KD/dataset/cellxgene")

    # Define the device
    assert torch.cuda.is_available(), "Training on CPU is not supported"
    device = torch.device("cuda")
    print(f"GPU {local_rank} - Using device: {device}")

    # Load the dataset
    print(f"GPU {local_rank} - Loading dataset...")


    print("loading dataset...")
    # load the token embedding
    token_embed = torch.load(res_dir / f"token_embed_{n_mgene}.pt")
    # load the cell meta-info
    meta_dict = torch.load(res_dir / f"meta_{n_mgene}.pt")
    # construct dataset
    scdataset = data_utils.sc_dataset(expr_path = res_dir / f"expr_sent_{n_mgene}.npz", gene_path = res_dir / f"feat_sent_{n_mgene}.npz",
                                    ncells = meta_dict["shape"][0], npads = meta_dict["shape"][1], labels = meta_dict["label"], batches = meta_dict["batch"])

    # train test split
    train_size = int(0.95 * len(scdataset))
    val_size = int(0.01 * len(scdataset))
    test_size = len(scdataset) - train_size - val_size
    train_dataset, val_dataset, test_datset = data.random_split(scdataset, [train_size, val_size, test_size])


    # obtain train/val/test loaders
    # NOTE: multi-gpus for only train_loader
    batch_size = 512
    train_loader = data.DataLoader(train_dataset, batch_size = batch_size, shuffle = False, pin_memory = True, sampler = DistributedSampler(train_dataset, shuffle=True))
    val_loader = data.DataLoader(val_dataset, batch_size = batch_size, shuffle = False, pin_memory = True)
    test_loader = data.DataLoader(test_datset, batch_size = batch_size, shuffle = False, pin_memory = True)
    print(f"GPU {local_rank} - Done.")

    # training function
    fm_model = trans.TransformerModel(token_embed = token_embed, n_batch = len(meta_dict["batch_code"]), n_label = len(meta_dict["label_code"]),
                                    n_embed = 512, n_head = 8, n_hidden = 2048, n_layers = 4, output_dim = 256, dropout = 0.05, device = device)
    # wrap model into multi-gpus setting
    fm_model = DistributedDataParallel(fm_model, device_ids=[local_rank])
    # init optimizer and scheduler, learning rate scale with the batch_size
    optimizer = AdamW(fm_model.parameters(), lr=2e-5 * batch_size/32)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps = 1000, num_training_steps = n_epochs * len(train_loader))

    # Load latest checkpoint
    if os.path.exists("checkpoint/checkpoint_multigpus.pth"): 
        print(f"GPU {local_rank} - Preloading lastest model'")
        # load parameter from last train
        state = torch.load("checkpoint/checkpoint_multigpus.pth")
        fm_model.module.load_state_dict(state["model_state_dict"])
        initial_epoch = state['epoch'] + 1
        optimizer.load_state_dict(state['optimizer_state_dict'])
        scheduler.load_state_dict(state["scheduler_state_dict"])
        del state
    else:
        initial_epoch = 0
        # If we couldn't find a model to preload, just start from scratch
        print(f'GPU {local_rank} - Could not find model to preload. Starting from scratch')

    # Init logger process, only main thread
    if global_rank == 0:
        writer = initialize_services() 
    else:
        writer = None

    # NOTE: training loop
    best_val_loss = float('inf')

    for epoch in range(initial_epoch, n_epochs):
        torch.cuda.empty_cache()

        # Disable tqdm on all nodes except the rank 0 GPU on each server
        # batch_iterator = tqdm.tqdm(train_loader, desc=f"Processing Epoch {epoch:02d} on rank {global_rank}", disable=local_rank != 0)

        # distribute to multi-gpus, gradient sync at backward()
        train_loss, *_ = train_epoch(model = fm_model, train_loader = train_loader, optimizer = optimizer, scheduler = scheduler, device = device, writer = writer, epoch = epoch)
        # only run evaluation on one gpu
        if global_rank == 0:
            val_loss, *_ = evaluate_epoch(model = fm_model, val_loader = val_loader, device = device, writer = writer, epoch = epoch)
            print(f"Epoch {epoch + 1}/{n_epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

            # Save a checkpoint if validation loss improves, NOTE: only save for one thread, avoid conflicting
            if val_loss < best_val_loss:
                save_checkpoint(epoch, fm_model, optimizer, scheduler, val_loss, path = "checkpoint/checkpoint_multigpus.pth")
                best_val_loss = val_loss

        # [Opt] write the early stopping, not necessary for pretraining


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
