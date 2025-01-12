# In[]
from pathlib import Path
from torch.utils import data
import torch.nn as nn

import torch
import data_utils
import transformer_model as trans
import numpy as np
from transformers import AdamW, get_linear_schedule_with_warmup
from torch.utils.tensorboard import SummaryWriter


# generate the special token embedding
data_utils.set_seed(0)
n_mgene = 256
res_dir = Path(f"/project/zzhang834/LLM_KD/dataset/cellxgene")
# In[]
# token embeddings: include meta-gene embedding and the special token embedding
# gene_embed_meta_dict = torch.load(f"/project/zzhang834/llm_dataset/CellXGeneCZI/data_download/gene_embed_meta{n_mgene}.pt")
# gene_embed_meta = gene_embed_meta_dict["meta_embed"]

# # NOTE: generate special token includes cls, padding, mask
# special_token_embed = torch.normal(mean = 0, std = 1, size = (3, gene_embed_meta.shape[1]))
# token_embed = torch.vstack([gene_embed_meta, special_token_embed])
# torch.save(token_embed, f = res_dir / f"token_embed_{n_mgene}.pt")

# In[]
print("loading dataset...")
# load the token embedding
token_embed = torch.load(res_dir / f"token_embed_{n_mgene}.pt")
# load the cell meta-info
meta_dict = torch.load(res_dir / f"meta_{n_mgene}.pt")
# construct dataset
scdataset = data_utils.sc_dataset(expr_path = res_dir / f"expr_sent_{n_mgene}.npz", gene_path = res_dir / f"feat_sent_{n_mgene}.npz",
                                  ncells = meta_dict["shape"][0], npads = meta_dict["shape"][1], labels = meta_dict["label"], batches = meta_dict["batch"])

# train test split
train_size = int(0.001 * len(scdataset))
val_size = int(0.001 * len(scdataset))
test_size = len(scdataset) - train_size - val_size
train_dataset, val_dataset, test_datset = data.random_split(scdataset, [train_size, val_size, test_size])

# obtain train/val/test loaders
train_loader = data.DataLoader(train_dataset, batch_size = 32, shuffle = True, pin_memory = True)
val_loader = data.DataLoader(val_dataset, batch_size = 32, shuffle = False, pin_memory = True)
test_loader = data.DataLoader(test_datset, batch_size = 32, shuffle = False, pin_memory = True)
print("Done.")

# In[]
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
        embeds, cell_embed, mask = model(gene_sent = gene_sample, expr_sent = expr_sample, mask_prob = 0.15)
        
        # sample from the masked pool
        mask_gene_sample = []
        mask_expr_sample = []
        for i in range(cell_embed.shape[0]):
            # 1. include both the masked gene and the zero expression
            masked_genes = torch.tensor([x for x in model.gene_idx if x not in gene_sample[i]]).to(device)
            masked_exprs = torch.zeros_like(masked_genes)
            masked_genes = torch.hstack([masked_genes, gene_sample[i, mask[i]]])
            masked_exprs = torch.hstack([masked_exprs, expr_sample[i, mask[i]]])
            # 2. include only the masked gene
            masked_genes = torch.hstack([masked_genes, gene_sample[i, mask[i]]])
            masked_exprs = torch.hstack([masked_exprs, expr_sample[i, mask[i]]])
            # sample
            idx = np.random.choice(len(masked_genes), 1)
            mask_gene_sample.append(gene_sample[i, idx])
            mask_expr_sample.append(expr_sample[i, idx])

        mask_gene_sample = torch.cat(mask_gene_sample)
        mask_expr_sample = torch.cat(mask_expr_sample)
        
        expr_pred = model.predict_expr(cell_embed = cell_embed, gene_cond = mask_gene_sample, batch_cond = batch_sample)
        # cell type label
        label_pred = model.predict_label(cell_embed = cell_embed, batch_cond = batch_sample)

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
            writer.add_scalar("Train Loss (TOTAL)", running_loss, epoch * len(train_loader) + iter)
            writer.add_scalar("Train Loss (MLM)", running_loss_mlm, epoch * len(train_loader) + iter)
            writer.add_scalar("Train Loss (CLASS)", running_loss_sup, epoch * len(train_loader) + iter)
            writer.add_scalar("Train Loss (KD)", running_loss_kd, epoch * len(train_loader) + iter)

            print(f"Epoch: {epoch}, Step: {epoch * len(train_loader) + iter}, Train Loss (TOTAL): {running_loss:.4f}, Train Loss (MLM): {running_loss_mlm:.4f}, Train Loss (CLASS): {running_loss_sup:.4f}, Train Loss (KD): {running_loss_kd:.4f}")

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
            embeds, cell_embed, mask = fm_model(gene_sent = gene_sample, expr_sent = expr_sample, mask_prob = 0.15)
            
            # sample from the masked pool
            mask_gene_sample = []
            mask_expr_sample = []
            for i in range(cell_embed.shape[0]):
                # 1. include both the masked gene and the zero expression
                masked_genes = torch.tensor([x for x in model.gene_idx if x not in gene_sample[i]]).to(device)
                masked_exprs = torch.zeros_like(masked_genes)
                masked_genes = torch.hstack([masked_genes, gene_sample[i, mask[i]]])
                masked_exprs = torch.hstack([masked_exprs, expr_sample[i, mask[i]]])
                # 2. include only the masked gene
                masked_genes = torch.hstack([masked_genes, gene_sample[i, mask[i]]])
                masked_exprs = torch.hstack([masked_exprs, expr_sample[i, mask[i]]])
                # sample
                idx = np.random.choice(len(masked_genes), 1)
                mask_gene_sample.append(gene_sample[i, idx])
                mask_expr_sample.append(expr_sample[i, idx])           

            mask_gene_sample = torch.cat(mask_gene_sample)
            mask_expr_sample = torch.cat(mask_expr_sample)
            
            expr_pred = fm_model.predict_expr(cell_embed = cell_embed, gene_cond = mask_gene_sample, batch_cond = batch_sample)
            # cell type label
            label_pred = fm_model.predict_label(cell_embed = cell_embed, batch_cond = batch_sample)

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
        writer.add_scalar("Val Loss (TOTAL)", val_loss, epoch)
        writer.add_scalar("Val Loss (MLM)", val_loss_mlm, epoch)
        writer.add_scalar("Val Loss (CLASS)", val_loss_sup, epoch)
        writer.add_scalar("Val Loss (KD)", val_loss_kd, epoch)

        print(f"Epoch: {epoch}, Val Loss (TOTAL): {val_loss:.4f}, Val Loss (MLM): {val_loss_mlm:.4f}, Val Loss (CLASS): {val_loss_sup:.4f}, Val Loss (KD): {val_loss_kd:.4f}")

    return val_loss, val_loss_mlm, val_loss_sup, val_loss_sup, val_loss_kd

# Save model checkpoint
def save_checkpoint(epoch, model, optimizer, scheduler, loss, path = 'checkpoint/checkpoint.pth'):
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'loss': loss
    }
    torch.save(checkpoint, path)
    print(f"Checkpoint saved at epoch {epoch}.")

# Main Trainer Function
def train_transformer(model, train_dataloader, val_dataloader, num_epochs = 5, checkpoint_path = "checkpoint/checkpoint.pth"):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    writer = SummaryWriter(log_dir='./logs')

    optimizer = AdamW(model.parameters(), lr=2e-5)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps = 1000, num_training_steps = num_epochs * len(train_loader))

    best_val_loss = float('inf')

    for epoch in range(num_epochs):
        train_loss, *_ = train_epoch(model, train_dataloader, optimizer, scheduler, device, writer, epoch)
        val_loss, *_ = evaluate_epoch(model, val_dataloader, device, writer, epoch)

        print(f"Epoch {epoch + 1}/{num_epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

        # Save a checkpoint if validation loss improves
        if val_loss < best_val_loss:
            save_checkpoint(epoch, model, optimizer, scheduler, val_loss, path=checkpoint_path)
            best_val_loss = val_loss

        # [Opt] write the early stopping, not necessary for pretraining


# In[]
# training function

device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
fm_model = trans.TransformerModel(token_embed = token_embed, n_batch = len(meta_dict["batch_code"]), n_label = len(meta_dict["label_code"]),
                                  n_embed = 512, n_head = 8, n_hidden = 2048, n_layers = 4, output_dim = 256, dropout = 0.05)


# with large dataset, 1-5 epochs is enough, more data rather than more epochs
train_transformer(fm_model, train_loader, val_loader, num_epochs=5, checkpoint_path="checkpoint/checkpoint.pth")



# %%
