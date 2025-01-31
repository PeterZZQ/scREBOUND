# In[]
from pathlib import Path
import sys, os

import numpy as np
import tqdm

# packages for distributed training
import torch
from torch.utils import data


import data_utils
from transformer_model import TransformerModel, get_default_config
import trainer
import utils

def evaluation(model, dataloader):
    # NOTE: training loop
    model.eval()
    with torch.no_grad():
        val_loss = 0.0
        val_loss_mlm = 0.0
        val_loss_sup = 0.0
        val_loss_kd = 0.0
        for data_sample in tqdm.tqdm(dataloader, desc=f"Evaluation"):
            loss, loss_mlm, loss_sup, loss_kd = trainer.infer_databatch(model, data_sample, multigpus = False)                            
            val_loss += loss.item()
            val_loss_mlm += loss_mlm.item()
            val_loss_sup += loss_sup.item()
            val_loss_kd += loss_kd.item()

        # log the values
        val_loss /= len(dataloader)
        val_loss_mlm /= len(dataloader)
        val_loss_sup /= len(dataloader)
        val_loss_kd /= len(dataloader)

        print(f"Val Loss (TOTAL): {val_loss:.4f}, Val Loss (MLM): {val_loss_mlm:.4f}, Val Loss (CLASS): {val_loss_sup:.4f}, Val Loss (KD): {val_loss_kd:.4f}")

    return val_loss, val_loss_mlm, val_loss_sup, val_loss_kd

# In[]
# function
data_utils.set_seed(3)

# Define the device
device = torch.device("cuda:1")
print(f"GPU - Using device: {device}")
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
                                "sup_type": "contrastive",
                                "sample_mlm": False,
                                "mlm_include_zero": False,
                                "pretrain_path":  "/project/zzhang834/LLM_KD/checkpoint_predfull/checkpoint_0.98_contr_1.pth",
                                "checkpoint_path": "/project/zzhang834/LLM_KD/checkpoint_predfull/",
                                "checkpoint_prefix": None
                                })

# construct dataset
scdataset = data_utils.sc_dataset_chunk(expr_path = data_dir / f"expr_sent_{n_mgene}_permu.npz", gene_path = data_dir / f"feat_sent_{n_mgene}_permu.npz",
                                        ncells = meta_dict["shape"][0], npads = meta_dict["shape"][1], labels = meta_dict["label"], batches = meta_dict["batch"], batch_size = model_config.batch_size)

# train test split
train_size = int(0.98 * len(scdataset))
val_size = int(0.00 * len(scdataset))
test_size = int(0.02 * len(scdataset))
# the data is already pre-shuffled
train_dataset = data.Subset(scdataset, range(train_size))
# val_dataset = data.Subset(scdataset, range(train_size, train_size + val_size))
# val_dataset = data.Subset(scdataset, range(train_size + 10 * val_size, train_size + 11 * val_size))
test_dataset = data.Subset(scdataset, range(train_size + val_size, min(train_size + val_size + test_size, len(scdataset))))

# obtain train/val/test loaders
# NOTE: multi-gpus for only train_loader
# val_loader = data.DataLoader(val_dataset, batch_size = 1, shuffle = False, pin_memory = True, num_workers = 8, prefetch_factor = 8)
test_loader = data.DataLoader(test_dataset, batch_size = 1, shuffle = False, pin_memory = True, num_workers = 8, prefetch_factor = 8)
print(f"GPU - Done.")

# model hyper-parameters
fm_model = TransformerModel(model_config = model_config, token_embed = token_embed, n_batch = len(meta_dict["batch_code"]), n_label = len(meta_dict["label_code"]), device = device)
# wrap model into multi-gpus setting
# fm_model = DistributedDataParallel(fm_model, device_ids=[local_rank])

print(f"GPU - Preloading lastest model'")
# load parameter from last train
state = torch.load(model_config.pretrain_path)
fm_model.load_state_dict(state["model_state_dict"])

# In[]
fm_model.model_config.sup_type = "contrastive"
# val_loss, val_loss_mlm, val_loss_sup, val_loss_kd = evaluation(model = fm_model, dataloader = val_loader)
test_loss, test_loss_mlm, test_loss_sup, test_loss_kd = evaluation(model = fm_model, dataloader = test_loader)

fm_model.model_config.mask_prob = 0.0
test_loss2, test_loss_mlm2, test_loss_sup2, test_loss_kd2 = evaluation(model = fm_model, dataloader = test_loader)

fm_model.model_config.mask_prob = 0.6
test_loss3, test_loss_mlm3, test_loss_sup3, test_loss_kd3 = evaluation(model = fm_model, dataloader = test_loader)

fm_model.model_config.mask_prob = 0.8
test_loss4, test_loss_mlm4, test_loss_sup4, test_loss_kd4 = evaluation(model = fm_model, dataloader = test_loader)

fm_model.model_config.mask_prob = 1.0
test_loss5, test_loss_mlm5, test_loss_sup5, test_loss_kd5 = evaluation(model = fm_model, dataloader = test_loader)

# In[]
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
sns.set_theme()
loss = pd.DataFrame(data = np.array([[0.4, test_loss_mlm, test_loss_sup], [0.0, test_loss_mlm2, test_loss_sup2], [0.6, test_loss_mlm3, test_loss_sup3], [0.8, test_loss_mlm4, test_loss_sup4], [1.0, test_loss_mlm5, test_loss_sup5]]), columns = ["mask_prob", "loss (mlm)", "loss (sup)"])
fig = plt.figure(figsize = (20, 7))
ax = fig.subplots(nrows = 1, ncols = 2)
sns.barplot(data = loss, x = "mask_prob", y = "loss (mlm)", hue = "mask_prob", ax = ax[0], width = 0.6)
sns.barplot(data = loss, x = "mask_prob", y = "loss (sup)", hue = "mask_prob", ax = ax[1], width = 0.6)
for i in ax[0].containers:
    ax[0].bar_label(i,)
for i in ax[1].containers:
    ax[1].bar_label(i,)

ax[0].legend(loc='upper left', bbox_to_anchor=(1.05, 1), borderaxespad=0)
ax[1].legend(loc='upper left', bbox_to_anchor=(1.05, 1), borderaxespad=0)

fig.savefig("test_loss.png", bbox_inches = "tight")

# In[]
# visualization
import scvi

# TODO: issue, for the classifier, should the masked input be used??
adata_test = trainer.cell_embed(model = fm_model, dataloader = test_loader, multi_gpus = False)

adata_test.obsm["latent_umap"] = scvi.model.utils.mde(adata_test.X.toarray(), accelerator = "gpu", seed = 0)
# adata_test.write_h5ad("res_testloader_0.3.h5ad")

# In[]
import anndata
import seaborn as sns
# transform the code name back
adata_test.obs["labels_name"] = [meta_dict["label_code"][x] for x in adata_test.obs["labels"]]
adata_test.obs["batchs_name"] = [meta_dict["batch_code"][x] for x in adata_test.obs["batchs"]]


adata_test = adata_test[adata_test.obs["batchs"].isin([72, 54, 8]), :]
# adata_test = anndata.read_h5ad("res_testloader_0.3.h5ad")
fig = utils.plot_embeds(embed = adata_test.obsm["latent_umap"], annos = adata_test.obs[["labels", "batchs"]].astype("category"), markerscale = 5, figsize = (20, 17), s = 1, alpha = 0.4)
fig.tight_layout()
fig.savefig("temp_result.png", bbox_inches = "tight")

spectral_cmap = sns.color_palette("Spectral", as_cmap=True, n_colors = 400)

fig = utils.plot_embeds(embed = adata_test.obsm["latent_umap"], annos = adata_test.obs[["labels"]].astype("category"), markerscale = 5, figsize = (20, 17), s = 1, alpha = 0.4, colormap = spectral_cmap)
fig.savefig("temp_result_label.png", bbox_inches = "tight")
fig = utils.plot_embeds(embed = adata_test.obsm["latent_umap"], annos = adata_test.obs[["batchs"]].astype("category"), markerscale = 5, figsize = (20, 17), s = 1, alpha = 0.4, colormap = spectral_cmap)
fig.savefig("temp_result_batch.png", bbox_inches = "tight")

# In[]


# %%
