# In[]
import torch
import torch.nn as nn
import torch.optim as optim

import tqdm

import pandas as pd
import numpy as np
import torch.utils.data as data
import tqdm
from itertools import chain
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

import batch_encode
import copy

import sys
sys.path.append("/project/zzhang834/LLM_KD/src/")
import base_model
import data_utils


# create dataset
class batch_dataset(data.Dataset):
    def __init__(self, batchdata_cont, batchdata_cat):
        super(batch_dataset, self).__init__()
        # standardize the cont data
        if batchdata_cont is not None:
            self.batchdata_cont = StandardScaler().fit_transform(batchdata_cont)
        else:
            self.batchdata_cont = None

        self.batchdata_cat = batchdata_cat

    def __len__(self):
        # Return the number of batches
        return self.batchdata_cat.shape[0]

    def __getitem__(self, index):
        batchsample_cat = self.batchdata_cat[index, :]
        
        if self.batchdata_cont is not None:
            batchsample_cont = self.batchdata_cont[index, :]
            return {"cont": batchsample_cont, "cat": batchsample_cat}

        else:
            return {"cat": batchsample_cat}

# In[]
# Load the digitized batch feature for easy learning
digitize = True
drop_mito = True
expr_binsize = 1

# In[]
data_utils.set_seed(0)
batch_data_cat = pd.read_csv("batch_feature_digitized_nomito.csv", index_col = 0)
# batch_data_cat = pd.read_csv("batch_feature_digitized_nomito_bin5.csv", index_col = 0)
# batch_data_cat = pd.read_csv("batch_feature_digitized.csv", index_col = 0)
device = torch.device("cuda:0")

n_cat_list = [int(x) for x in batch_data_cat.loc["nbuckets", :].values]
batch_data_cat = batch_data_cat.iloc[:-1, :].values
batch_data_cont = None
n_conts = 0

dataset = batch_dataset(batchdata_cont = batch_data_cont, batchdata_cat = batch_data_cat.astype(np.float32))
# split the dataset
train_size = int(0.9 * len(dataset))
val_size = int(0.1 * len(dataset))

# the data is already pre-shuffled
train_dataset = data.Subset(dataset, range(train_size))
val_dataset = data.Subset(dataset, range(train_size, len(dataset)))

train_dataloader = data.DataLoader(train_dataset, batch_size = 8, shuffle = True, pin_memory = True)
val_dataloader = data.DataLoader(val_dataset, batch_size = 8, shuffle = False, pin_memory = True)

batch_enc = base_model.encoder_batchfactor(n_cont_feat = n_conts, n_cat_list = n_cat_list, n_embed = 128).to(device)
batch_dec = base_model.decoder_batchfactor(n_cont_feat = n_conts, n_cat_list = n_cat_list, n_embed = 128).to(device)

optimizer = optim.Adam(params = chain(batch_enc.parameters(), batch_dec.parameters()), lr = 5e-4)

scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='min', factor=0.5, patience=3, min_lr=1e-5, verbose=True
)

# In[]
# Augment the data, drop-out on the batch-related genes
# add drop-out on the gene level representation: reason, mis-measurement
def augment_expr_dropout(data_batch, column_mask = None, dropout_probs = [0.1]):
    
    data_batch_augs = []
    for dropout_prob in dropout_probs:
        data_batch_aug = data_batch.clone()

        # mask
        mask_prob = torch.full(data_batch_aug.shape, dropout_prob).to(data_batch_aug.device)
        mask = torch.bernoulli(mask_prob).bool()
        # remove mask on certain positions
        if column_mask is not None:
            mask[:, column_mask] = 0
        data_batch_aug[mask] = -1

        # make the gene 0
        data_batch_aug[:, 1:] = torch.where(data_batch_aug[:, 1:] == -1, 0, data_batch_aug[:, 1:])
        data_batch_augs.append(data_batch_aug)

    data_batch_augs = torch.cat(data_batch_augs, dim = 0)

    return data_batch_augs


# In[]
dropout_probs = [0.05, 0.1, 0.15, 0.2, 0.25]
nepochs = 60
best_val = 1e5
for epoch in range(nepochs):
    batch_iterator = tqdm.tqdm(train_dataloader, desc=f"Processing Epoch {epoch:02d}")

    running_loss = 0
    running_loss_cont = []
    running_loss_cat = []
    for step, data_sample in enumerate(batch_iterator):
        data_sample_aug = {}
        if digitize:
            data_sample_aug["cat"] = augment_expr_dropout(data_sample["cat"], column_mask = torch.arange(1, 4), dropout_probs = dropout_probs).to(device)
            data_sample["cat"] = data_sample["cat"].to(device)

        else:
            data_sample_aug["cat"] = data_sample["cat"].to(device) 
            data_sample_aug["cont"] = augment_expr_dropout(data_sample["cont"], column_mask = torch.arange(1, 2), dropout_probs = dropout_probs).to(device)
            data_sample["cat"] = data_sample["cat"].to(device)
        
        optimizer.zero_grad()
        if digitize:
            embed, weight = batch_enc(batch_factors_cont = None, batch_factors_cat = data_sample["cat"])
            embed_aug, weight = batch_enc(batch_factors_cont = None, batch_factors_cat = data_sample_aug["cat"])
        else:
            embed, weight = batch_enc(batch_factors_cont = data_sample["cont"], batch_factors_cat = data_sample["cat"])
            embed_aug, weight = batch_enc(batch_factors_cont = data_sample_aug["cont"], batch_factors_cat = data_sample_aug["cat"])
            
        recon_batch_cont, recon_batch_cat = batch_dec(embed)
        recon_batch_cont_aug, recon_batch_cat_aug = batch_dec(embed_aug)

        # mse for continuous variable
        loss_conts = []
        for id in range(batch_enc.n_cont_feat):
            loss_cont = (torch.cat([recon_batch_cont[id], recon_batch_cont_aug[id]], dim = 0) - torch.cat([data_sample["cont"][:, id]] * (len(dropout_probs) + 1), dim = 0)).pow(2).mean() 
            loss_conts.append(loss_cont) 
        
        # ce for categorical
        loss_cats = []
        for id in range(batch_enc.n_cat_feat):
            loss_ce = nn.CrossEntropyLoss()
            loss_cat = loss_ce(torch.cat([recon_batch_cat[id], recon_batch_cat_aug[id]], dim = 0), torch.cat([data_sample["cat"][:, id].long()] * (len(dropout_probs) + 1), dim = 0))
            loss_cats.append(loss_cat)

        loss = sum(loss_conts) + sum(loss_cats)
        
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        running_loss_cont.append([x.item() for x in loss_conts]) 
        running_loss_cat.append([x.item() for x in loss_cats])

    running_loss /= step
    running_loss_cont = np.array(running_loss_cont).sum(axis = 1).mean(axis = 0)
    running_loss_cat = np.array(running_loss_cat).sum(axis = 1).mean(axis = 0)
    
    print(f"Epoch: {epoch}, Step: {step + 1}/{len(train_dataloader)}, Loss (TOTAL): {running_loss:.4f}, Loss (Cont): {running_loss_cont}, Loss (Cat): {running_loss_cat}")

    # evaluation
    val_loss = 0
    val_loss_cont = []
    val_loss_cat = []

    with torch.no_grad():
        for step, data_sample in enumerate(val_dataloader):
            data_sample["cat"] = data_sample["cat"].to(device)
            if digitize:
                embed, weight = batch_enc(batch_factors_cont = None, batch_factors_cat = data_sample["cat"])
            else:
                data_sample["cont"] = data_sample["cont"].to(device)
                embed, weight = batch_enc(batch_factors_cont = data_sample["cont"], batch_factors_cat = data_sample["cat"])

            recon_batch_cont, recon_batch_cat = batch_dec(embed)

            loss_conts = []
            for id in range(batch_enc.n_cont_feat):
                loss_cont = (recon_batch_cont[id] - data_sample["cont"][:, id]).pow(2).mean() 
                loss_conts.append(loss_cont) 
            
            # ce for categorical
            loss_cats = []
            for id in range(batch_enc.n_cat_feat):
                loss_ce = nn.CrossEntropyLoss()
                loss_cat = loss_ce(recon_batch_cat[id], data_sample["cat"][:, id].long())
                loss_cats.append(loss_cat)

            loss_val = sum(loss_conts) + sum(loss_cats)

            val_loss += loss_val.item()
            val_loss_cont.append([x.item() for x in loss_conts]) 
            val_loss_cat.append([x.item() for x in loss_cats])

        val_loss /= step
        val_loss_cont = np.array(val_loss_cont).sum(axis = 1).mean(axis = 0)
        val_loss_cat = np.array(val_loss_cat).sum(axis = 1).mean(axis = 0)
        print(f"Val Loss (TOTAL): {val_loss:.4f}, Val Loss (Cont): {val_loss_cont}, Val Loss (Cat): {val_loss_cat}")
        if val_loss < best_val:
            best_val = val_loss
            best_enc_states = copy.deepcopy(batch_enc.state_dict())
            best_dec_states = copy.deepcopy(batch_dec.state_dict())
    scheduler.step(val_loss)
    print(f"learning rate: {scheduler.get_last_lr()[0]:.2e}")

print("load best model")
batch_enc.load_state_dict(best_enc_states)
batch_dec.load_state_dict(best_dec_states)

# In[]
# # Print importance of each feature
# print(batch_data_cat.columns[np.argsort(weight.numpy())[::-1]])

# # Print the reconstruction accuracy of each feature
# print(batch_data_cat.columns[np.argsort(val_loss_cat)])

# In[]
# creat batch dict
batch_dict = {"state_dict_enc": best_enc_states,
              "state_dict_dec": best_dec_states,
              "n_cat_list": n_cat_list,
              "n_cont_feats": n_conts, 
              "cats": torch.tensor(batch_data_cat, dtype = torch.int32), 
              "conts": batch_data_cont
              }

# Extract the batch_encoder
torch.save(batch_dict, "batch_enc_dict_mito.pt")


# In[]
# Evaluate on the test dataset
import anndata

EVAL_DATA_DIR = "/project/zzhang834/LLM_KD/dataset/scIB/"
n_mgene = 256
gene_embed_dict = torch.load(f"/localscratch/ziqi/hs_download/gene_embed_meta{n_mgene}_kmeans.pt", weights_only = False)
gene_info = gene_embed_dict["labels"]

adata_test1 = anndata.read_h5ad(EVAL_DATA_DIR + "Immune_ALL_human.h5ad")
# checked: counts are raw
adata_test1.X = adata_test1.layers["counts"].copy()
gene_list = gene_info["feature_name"].values

adata_test1 = data_utils.align_genes(adata_test1, gene_list)
adata_test1.layers["counts"] = adata_test1.X.copy()
adata_test1.obs["assay"] = adata_test1.obs["chemistry"].cat.rename_categories({"10X": "10x 3' transcription profiling", 'smart-seq2': 'Smart-seq2', 'v2_10X': "10x 3' v2", "v3_10X": "10x 3' v3"})
adata_test1.obs["suspension_type"] = "cell"
adata_test1.obs["batch_id"], batch_code = pd.factorize(adata_test1.obs["batch"])

batch_features = batch_encode.construct_batch_feats(adata_test1, use_mito = (not drop_mito))
batch_features_digitize, num_bucket = batch_encode.tokenize_batch_feats(batch_features, use_mito = (not drop_mito), expr_binsize = expr_binsize)
batch_features_digitize = torch.tensor(batch_features_digitize.values, dtype = torch.float32)

# In[]
data_sample = {"cat": batch_features_digitize.to(device)}

with torch.no_grad():
    embed, weight = batch_enc(batch_factors_cont = None, batch_factors_cat = data_sample["cat"])
    recon_batch_cont, recon_batch_cat = batch_dec(embed)

    loss_cats = pd.DataFrame(columns = batch_features.columns, data = np.zeros((1, batch_features.shape[1])))
    for id in range(batch_enc.n_cat_feat):
        loss_ce = nn.CrossEntropyLoss()
        loss_cat = loss_ce(recon_batch_cat[id], data_sample["cat"][:, id].long())
        loss_cats.iloc[0, id] = loss_cat.item()
    
loss_test = np.sum(loss_cats.values)
loss_cats["total"] = loss_test
loss_cats["model"] = "trained"
loss_cats["dataset"] = "immune"
print(loss_test)


# In[]
# randomly initialized model for baseline
batch_enc_rand = base_model.encoder_batchfactor(n_cont_feat = n_conts, n_cat_list = n_cat_list, n_embed = 128).to(device)
batch_dec_rand = base_model.decoder_batchfactor(n_cont_feat = n_conts, n_cat_list = n_cat_list, n_embed = 128).to(device)

with torch.no_grad():
    embed, weight = batch_enc_rand(batch_factors_cont = None, batch_factors_cat = data_sample["cat"])
    recon_batch_cont, recon_batch_cat = batch_dec_rand(embed)

    loss_cats_rand = pd.DataFrame(columns = batch_features.columns, data = np.zeros((1, batch_features.shape[1])))
    for id in range(batch_enc.n_cat_feat):
        loss_ce = nn.CrossEntropyLoss()
        loss_cat = loss_ce(recon_batch_cat[id], data_sample["cat"][:, id].long())
        loss_cats_rand.iloc[0, id] = loss_cat.item()

loss_test_rand = np.sum(loss_cats_rand.values)
loss_cats_rand["total"] = loss_test_rand
loss_cats_rand["model"] = "random"
loss_cats_rand["dataset"] = "immune"

print(loss_test_rand)

# In[]
loss_eval = pd.concat([loss_cats, loss_cats_rand], axis =0, ignore_index = True)
loss_eval = loss_eval.drop(labels = ["dataset"], axis = 1)
import seaborn as sns
sns.set_theme()
fig = plt.figure(figsize = (40, 10))
ax = fig.subplots(nrows = 1, ncols = 1)
sns.barplot(data = loss_eval.melt(id_vars='model', var_name='cat', value_name='loss'),
            hue = "model", x = "cat", y = "loss")
ax.set_yscale("log")
ax.set_xticklabels(ax.get_xticklabels(), rotation = 90)
ax.set_xlabel(None)
for container in ax.containers:
    ax.bar_label(container, fmt = "%.4f")

fig.tight_layout()
fig.savefig("loss_eval.png", bbox_inches = "tight")

# %%
