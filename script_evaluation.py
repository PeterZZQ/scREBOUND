# In[]
from pathlib import Path
import sys, os

import numpy as np
import tqdm

# packages for distributed training
import torch
from torch.utils import data
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
sns.set_theme()

sys.path.append("src/")
import data_utils
from transformer import TransformerModel, get_default_config
import trainer_meta as trainer_meta
import utils as utils

def evaluation(model, dataloader):
    # NOTE: training loop
    model.eval()
    with torch.no_grad():
        val_loss = 0.0
        val_loss_mlm = 0.0
        val_loss_sup = 0.0
        val_loss_kd = 0.0
        for data_sample in tqdm.tqdm(dataloader, desc=f"Evaluation"):
            loss, loss_mlm, loss_sup, loss_kd = trainer_meta.infer_databatch(model, data_sample, multigpus = False)                            
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
# data_dir = Path(f"/project/zzhang834/LLM_KD/dataset/cellxgene_old")
data_dir = Path(f"/localscratch/ziqi/localscratch_tempdata/cellxgene")
# data_dir = Path(f"/project/zzhang834/LLM_KD/dataset/cellxgene")
# load the token embedding
token_embed = torch.load(data_dir / f"token_embed_{n_mgene}.pt")
# load the cell meta-info
meta_dict = torch.load(data_dir / f"meta_{n_mgene}.pt")


model_dir = "/project/zzhang834/LLM_KD/checkpoint/checkpoint_0.98_contr_dynmask_1.pth"
state = torch.load(model_dir)
model_config = state["model_config"]
model_config.__dict__.update({"checkpoint_path": None, "checkpoint_prefix": None, "pretrain_path":  model_dir})

# construct dataset
scdataset = data_utils.sc_dataset_chunk(expr_path = data_dir / f"expr_sent_{n_mgene}.npz", gene_path = data_dir / f"feat_sent_{n_mgene}.npz",
                                        ncells = meta_dict["shape"]["full"][0], npads = meta_dict["shape"]["full"][1], labels = meta_dict["label"], batches = meta_dict["batch"], batch_size = model_config.batch_size)

# train test split
train_size = int(0.98 * len(scdataset))
val_size = int(0.00 * len(scdataset))
test_size = int(0.02 * len(scdataset))
# the data is already pre-shuffled
train_dataset = data.Subset(scdataset, range(train_size))
test_dataset = data.Subset(scdataset, range(train_size + val_size, min(train_size + val_size + test_size, len(scdataset))))

# obtain train/val/test loaders
test_loader = data.DataLoader(test_dataset, batch_size = 1, shuffle = False, pin_memory = True, num_workers = 8, prefetch_factor = 8)
print(f"GPU - Done.")

# model hyper-parameters
fm_model = TransformerModel(model_config = model_config, token_embed = token_embed, n_batch = len(meta_dict["batch_code"]), n_label = len(meta_dict["label_code"]), device = device)

print(f"GPU - Preloading lastest model'")
# load parameter from last train
# fm_model.load_state_dict(state["model_state_dict"])
print(f"GPU - Done.")

# In[]
# NOTE: the dynamic masking mode allows for the adjustment of mask probability
fm_model.model_config.sup_type = "contrastive"

losses_mlm = []
losses_sup = []
mask_probs = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
for mask_prob in mask_probs:
    fm_model.model_config.mask_prob = mask_prob
    test_loss, test_loss_mlm, test_loss_sup, test_loss_kd = evaluation(model = fm_model, dataloader = test_loader)
    losses_mlm.append(test_loss_mlm)
    losses_sup.append(test_loss_sup)

# In[]
res_dir = "results/checkpoint_0.98_contr_dynmask2/"
if not os.path.exists(res_dir):
    os.makedirs(res_dir)

loss_df = pd.DataFrame()
loss_df["mask_prob"] = mask_probs
loss_df["loss (mlm)"] = losses_mlm
loss_df["loss (sup)"] = losses_sup

fig = plt.figure(figsize = (20, 7))
ax = fig.subplots(nrows = 1, ncols = 2)
sns.barplot(data = loss_df, x = "mask_prob", y = "loss (mlm)", hue = "mask_prob", ax = ax[0], width = 0.6)
sns.barplot(data = loss_df, x = "mask_prob", y = "loss (sup)", hue = "mask_prob", ax = ax[1], width = 0.6)
for i in ax[0].containers:
    ax[0].bar_label(i,)
for i in ax[1].containers:
    ax[1].bar_label(i,)

ax[0].legend(loc='upper left', bbox_to_anchor=(1.05, 1), borderaxespad=0)
ax[1].legend(loc='upper left', bbox_to_anchor=(1.05, 1), borderaxespad=0)

fig.savefig(res_dir + "test_loss.png", bbox_inches = "tight", dpi = 200)

# In[]
# visualization
import scvi

# TODO: issue, for the classifier, should the masked input be used??
adata_test = trainer_meta.cell_embed(model = fm_model, dataloader = test_loader, multi_gpus = False)

adata_test.obsm["latent_umap"] = scvi.model.utils.mde(adata_test.X.toarray(), accelerator = "gpu", seed = 0)
# adata_test.write_h5ad("res_testloader_0.3.h5ad")

# In[]
import anndata
import seaborn as sns
# transform the code name back
adata_test.obs["labels_name"] = [meta_dict["label_code"][x] for x in adata_test.obs["labels"]]
adata_test.obs["batchs_name"] = [meta_dict["batch_code"][x] for x in adata_test.obs["batchs"]]

# select three batches
adata_test2 = adata_test[adata_test.obs["batchs"].isin([72, 54, 8]), :]
# adata_test2 = adata_test[adata_test.obs["labels"].isin([0,1,2,3,4,5,6]), :]
# adata_test = anndata.read_h5ad("res_testloader_0.3.h5ad")
fig = utils.plot_embeds(embed = adata_test2.obsm["latent_umap"], annos = adata_test2.obs[["labels", "batchs"]].astype("category"), markerscale = 15, figsize = (20, 17), s = 1, alpha = 0.4)
fig.tight_layout()
fig.savefig(res_dir + "test_embed.png", bbox_inches = "tight")

spectral_cmap = sns.color_palette("Spectral", as_cmap=True, n_colors = 40)

fig = utils.plot_embeds(embed = adata_test2.obsm["latent_umap"], annos = adata_test2.obs[["labels_name"]].astype("category"), markerscale = 15, figsize = (20, 17), s = 1, alpha = 0.4)


# fig = utils.plot_embeds(embed = adata_test.obsm["latent_umap"], annos = adata_test.obs[["labels"]].astype("category"), markerscale = 15, figsize = (20, 17), s = 1, alpha = 0.4, colormap = spectral_cmap)
# # fig.savefig("temp_result_label.png", bbox_inches = "tight")
# fig = utils.plot_embeds(embed = adata_test.obsm["latent_umap"], annos = adata_test.obs[["batchs"]].astype("category"), markerscale = 15, figsize = (20, 17), s = 1, alpha = 0.4, colormap = spectral_cmap)
# # fig.savefig("temp_result_batch.png", bbox_inches = "tight")


# %%
