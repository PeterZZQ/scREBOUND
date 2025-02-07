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

import src.data_utils as data_utils
from torch.utils.data.distributed import DistributedSampler

from src.transformer_model import TransformerModel, ModelConfig, get_default_config, SupervisedContrastiveLoss
from transformers import AdamW, get_linear_schedule_with_warmup

# TODO: update to wandb
from torch.utils.tensorboard import SummaryWriter


def infer_databatch(model, data_sample):
        expr_sample, gene_sample, label_sample, batch_sample = data_sample
        expr_sample = expr_sample.squeeze(0).to(model.device, non_blocking = True)
        gene_sample = gene_sample.squeeze(0).to(model.device, non_blocking = True)
        label_sample = label_sample.squeeze(0).to(model.device, non_blocking = True)
        batch_sample = batch_sample.squeeze(0).to(model.device, non_blocking = True)
        
        _, cell_embed, mask = model(gene_sent = gene_sample, expr_sent = expr_sample)

        # calculate the loss
        # 1. MSE loss between the predict expression and ground truth expression
        # NOTE: the expr_sent is log-normalized in advance
        mse_mlm = nn.MSELoss()
        if model.model_config.sample_mlm:
            # sample from the masked pool
            mask_gene_sample = []
            mask_expr_sample = []
            # loop through every single cell within the batch
            for i in range(cell_embed.shape[0]):
                # 1. target gene can be the gene with 0 expression (not included in the gene sentence)
                masked_genes = torch.tensor([x for x in model.gene_idx if x not in gene_sample[i]]).to(model.device)
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

            # prediction of the target genes with shape (batch_size, 1)
            expr_pred = model.predict_expr(cell_embed = cell_embed, gene_cond = mask_gene_sample, batch_cond = batch_sample)
            loss_mlm = mse_mlm(expr_pred, mask_expr_sample.unsqueeze(1))
        
        else:
            # predict the gene expression (batch_size, n_mgenes)
            expr_pred = model.predict_expr(cell_embed = cell_embed, gene_cond = None, batch_cond = batch_sample)

            recon_expr_sample = torch.vstack([expr_pred[x,y] for x,y in enumerate(gene_sample)]) 
            loss_mlm = ((recon_expr_sample - expr_sample) * mask).pow(2).sum(1).mean()
            if model.model_config.mlm_include_zero:
                # 0-expression gene
                loss_mlm_zeroexpr = [expr_pred[x, model.gene_idx[~torch.isin(model.gene_idx, y)]].pow(2).sum() for x,y in enumerate(gene_sample)]
                loss_mlm += sum(loss_mlm_zeroexpr)/len(loss_mlm_zeroexpr)

        # 2. Classification loss between the predict label and the ground truth label
        if model.model_config.sup_type == "classifier":
            # cell type label
            label_pred = model.predict_label(cell_embed = cell_embed, batch_cond = batch_sample)
            ce = nn.CrossEntropyLoss()
            loss_sup = ce(label_pred, label_sample)
        
        elif model.model_config.sup_type == "contrastive":
            contr = SupervisedContrastiveLoss(temperature = 0.07)
            loss_sup = contr(features = cell_embed, labels = label_sample)

        else:
            raise ValueError("`sub_type' can only be classifier or contrastive")

        # 3. KD loss from teacher model
        loss_kd = torch.tensor([0.0], device = model.device)
        loss = loss_mlm + model.model_config.lamb_sup * loss_sup + model.model_config.lamb_kd * loss_kd

        return loss, loss_mlm, loss_sup, loss_kd

# Save model checkpoint
def save_checkpoint(epoch, step, model, optimizer, scheduler, loss, path):
    checkpoint = {
        'epoch': epoch,
        'step': step,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'loss': loss
    }
    torch.save(checkpoint, path)
    print(f"Checkpoint saved at epoch {epoch}.")

def train(model, model_config, train_loader, val_loader, optimizer, scheduler, writer, initial_epoch, initial_step, log_step):
    # NOTE: training loop
    for epoch in range(initial_epoch, model_config.n_epoch):
        torch.cuda.empty_cache()
        # Disable tqdm on all nodes except the rank 0 GPU on each server
        batch_iterator = tqdm.tqdm(train_loader, desc=f"Processing Epoch {epoch:02d}")

        # NOTE: Training
        running_loss = 0.0
        running_loss_mlm = 0.0
        running_loss_sup = 0.0
        running_loss_kd = 0.0
        checkpoint_counter = 0        

        for step, data_sample in enumerate(batch_iterator):
            model.train()
            
            if step < initial_step:
                continue
            loss, loss_mlm, loss_sup, loss_kd = infer_databatch(model, data_sample)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()

            # NOTE: log the results
            running_loss_mlm += loss_mlm.item()
            running_loss_sup += loss_sup.item()
            running_loss_kd += loss_kd.item()
            running_loss += loss.item()

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

                    print(f"Epoch: {epoch}, Step: {step + 1}/{len(train_loader)}, Train Loss (TOTAL): {running_loss:.4f}, Train Loss (MLM): {running_loss_mlm:.4f}, Train Loss (CLASS): {running_loss_sup:.4f}, Train Loss (KD): {running_loss_kd:.4f}")
                    # print the current learning rate
                    print(f"Learning rate: {scheduler.get_last_lr()[0]:.2e}")    
                
                running_loss = 0.0
                running_loss_mlm = 0.0
                running_loss_sup = 0.0
                running_loss_kd = 0.0

                checkpoint_counter += 1
                # model evaluation and checkpoint saving
                if (checkpoint_counter == 10):
                    model.eval()
                    with torch.no_grad():
                        val_loss = 0.0
                        val_loss_mlm = 0.0
                        val_loss_sup = 0.0
                        val_loss_kd = 0.0
                        for data_sample in val_loader:
                            loss, loss_mlm, loss_sup, loss_kd = infer_databatch(model, data_sample)                            
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

                    save_checkpoint(epoch = epoch, step = step, model = model, optimizer = optimizer,
                                    scheduler = scheduler, loss = running_loss, path = f"{model.model_config.checkpoint_path}{model.model_config.checkpoint_prefix}_{epoch}_{step}.pth")
                    
                    checkpoint_counter = 0
                    dist.barrier()
                
            initial_step = 0
            

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
                                  "pretrain_path": None,
                                  "checkpoint_path": "/project/zzhang834/LLM_KD/checkpoint_predfull/",
                                  "checkpoint_prefix": "checkpoint_0.3"
                                  })

    # construct dataset
    scdataset = data_utils.sc_dataset_chunk(expr_path = data_dir / f"expr_sent_{n_mgene}_permu.npz", gene_path = data_dir / f"feat_sent_{n_mgene}_permu.npz",
                                            ncells = meta_dict["shape"][0], npads = meta_dict["shape"][1], labels = meta_dict["label"], batches = meta_dict["batch"], batch_size = model_config.batch_size)

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
    print(f"GPU - Done.")

    # model hyper-parameters
    fm_model = TransformerModel(model_config = model_config, token_embed = token_embed, n_batch = len(meta_dict["batch_code"]), n_label = len(meta_dict["label_code"]), device = device)
    # init optimizer and scheduler, learning rate scale with the batch_size
    optimizer = AdamW(fm_model.parameters(), lr = model_config.lr)
    # update the warm-up steps to be 10% of total steps
    model_config.__dict__.update({"n_warmup_stp_lr": int(len(train_loader) * model_config.n_epoch * 0.1)})
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
   
    train(model = fm_model, model_config = model_config, train_loader = train_loader, val_loader = val_loader,
          optimizer = optimizer, scheduler = scheduler, writer = writer, initial_epoch = initial_epoch, initial_step = initial_step, log_step = 100)


# In[]
if __name__ == '__main__':

    main()
    

# %%
