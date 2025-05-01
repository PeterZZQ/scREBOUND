
# In[]
# --------------------------------------------------------------------------------------------------------------
#
# Evaluate different version of vanilla model
#
# --------------------------------------------------------------------------------------------------------------
import seaborn as sns
import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
import sys
sys.path.append("/net/csefiles/xzhanglab/zzhang834/LLM_KD/src")

import eval
import anndata

sns.set_theme()

# In[]
# --------------------------------------------------------------------------------------------------------------
#
# NOTE: Evaluation batch effect removal
#
# --------------------------------------------------------------------------------------------------------------
PROJECT_DIR = "/net/csefiles/xzhanglab/zzhang834/LLM_KD/"
# vanilla mlm model
# model_name = "cp_vanilla_4_512_meta_1"
# res_dir = PROJECT_DIR + f"results/checkpoint/{model_name}/"

# # finetune model
# model_name = "cp_contrcb1_4_512_meta_nobatch_1"
model_name = "cp_contrcbproj1_4_512_meta_nobatch_1"
res_dir = PROJECT_DIR + f"results/checkpoint_finetune/{model_name}/"

# In[]
# data_case = "immune_all"
# data_case = "pancreas"
# data_case = "lung_atlas"

use_rep = "latent"
use_rep = "contr"

for data_case in ["immune_all", "pancreas", "lung_atlas"]:
    adata_embed = anndata.read_h5ad(res_dir + f"adata_embed_{data_case}.h5ad")

    if data_case != "lung_atlas":
        scores = eval.eval_batch_correction(adata_embed, embed_key = use_rep, batch_key = "batch_id", label_key = "label")
        
    else:
        adata_embed_ctrl = adata_embed[adata_embed.obs["patientGroup"] == "Ctrl"]
        adata_embed_parenchyma = adata_embed[adata_embed.obs["patientGroup"] == "Parenchyma"]
        adata_embed_nan = adata_embed[adata_embed.obs["patientGroup"] == "nan"]
        scores = eval.eval_batch_correction(adata_embed, embed_key = use_rep, batch_key = "batch_id", label_key = "label")
        scores["case"] = "total"
        scores_ctrl = eval.eval_batch_correction(adata_embed_ctrl, embed_key = use_rep, batch_key = "batch_id", label_key = "label")
        scores_ctrl["case"] = "ctrl"
        scores_parenchyma = eval.eval_batch_correction(adata_embed_parenchyma, embed_key = use_rep, batch_key = "batch_id", label_key = "label")
        scores_parenchyma["case"] = "parenchyma"
        scores_nan = eval.eval_batch_correction(adata_embed_nan, embed_key = use_rep, batch_key = "batch_id", label_key = "label")
        scores_nan["case"] = "nan"
        scores = pd.concat([scores, scores_ctrl, scores_parenchyma, scores_nan], axis = 0)
        
    scores.to_csv(res_dir + f"scores_br_{data_case}_{use_rep}.csv")

# In[]
# choice 1: insert batch into transformer or not
# choice 2: use onehot encoding or use batch encoding
model1 = "cp_vanilla_4_512_meta_enc_trans_1"
model2 = "cp_vanilla_4_512_meta_enc_1"
model3 = "cp_vanilla_4_512_meta_onehot_trans_1"
model4 = "cp_vanilla_4_512_meta_onehot_1"
model5 = "cp_vanilla_4_512_meta_nobatch_1"
PROJECT_DIR = "/net/csefiles/xzhanglab/zzhang834/LLM_KD/"
res_dir = PROJECT_DIR + f"results/checkpoint/"

# data_case = "immune_all"
data_case = "lung_atlas"
val_loss_dict = {}
val_loss_dict["enc_trans"] = pd.read_csv(res_dir + model1 + f"/{data_case}_valloss.csv", index_col = 0)
val_loss_dict["enc"] = pd.read_csv(res_dir + model2 + f"/{data_case}_valloss.csv", index_col = 0)
val_loss_dict["onehot_trans"] = pd.read_csv(res_dir + model3 + f"/{data_case}_valloss.csv", index_col = 0)
val_loss_dict["onehot"] = pd.read_csv(res_dir + model4 + f"/{data_case}_valloss.csv", index_col = 0)
val_loss_dict["nobatch"] = pd.read_csv(res_dir + model5 + f"/{data_case}_valloss.csv", index_col = 0)

val_loss_dict["enc_trans"]["batch_encode"] = "encoder"
val_loss_dict["enc_trans"]["insert_trans"] = True
val_loss_dict["enc"]["batch_encode"] = "encoder"
val_loss_dict["enc"]["insert_trans"] = False
val_loss_dict["onehot_trans"]["batch_encode"] = "onehot"
val_loss_dict["onehot_trans"]["insert_trans"] = True
val_loss_dict["onehot"]["batch_encode"] = "onehot"
val_loss_dict["onehot"]["insert_trans"] = False
val_loss_dict["nobatch"]["batch_encode"] = "no batch"
val_loss_dict["nobatch"]["insert_trans"] = False

val_loss = pd.concat([x for x in val_loss_dict.values()], axis = 0)

fig = plt.figure(figsize = (8, 10))
ax = fig.subplots(ncols = 1, nrows = 4)

for i, mask_prob in enumerate([0.1, 0.2, 0.3, 0.4]):
    sns.barplot(data = val_loss[val_loss["mask prob"] == mask_prob], x = "batch_encode", y = "val mlm", hue = "insert_trans", ax = ax[i])
    ax[i].set_xlabel(None)
    for container in ax[i].containers:
        ax[i].bar_label(container, fmt = "%.4f")
        leg = ax[i].legend(loc='upper left', prop={'size': 15}, frameon = False, bbox_to_anchor=(1.04, 1), title = "insert transformer")

fig.suptitle(data_case, fontsize = 20)
fig.savefig(res_dir + f"compare_batchencoding_{data_case}", bbox_inches = "tight")

# choice 3: expr encoding, fourier encoding v.s. learnable encoding
model1 = "cp_vanilla_4_512_meta_enc_trans_1"
model2 = "cp_vanilla_4_512_meta_enc_trans_nofourier_1"
model3 = "cp_vanilla_4_512_meta_onehot_trans_1"
model2 = "cp_vanilla_4_512_meta_onehot_trans_nofourier_1"

# data_case = "immune_all"
data_case = "lung_atlas"
val_loss_dict = {}
val_loss_dict["enc"] = pd.read_csv(res_dir + model1 + f"/{data_case}_valloss.csv", index_col = 0)
val_loss_dict["enc_nofourier"] = pd.read_csv(res_dir + model2 + f"/{data_case}_valloss.csv", index_col = 0)
val_loss_dict["onehot"] = pd.read_csv(res_dir + model3 + f"/{data_case}_valloss.csv", index_col = 0)
val_loss_dict["onehot_nofourier"] = pd.read_csv(res_dir + model4 + f"/{data_case}_valloss.csv", index_col = 0)

val_loss_dict["enc"]["batch_encode"] = "encoder"
val_loss_dict["enc"]["expr_embed"] = "fourier"
val_loss_dict["enc_nofourier"]["batch_encode"] = "encoder"
val_loss_dict["enc_nofourier"]["expr_embed"] = "mlp"
val_loss_dict["onehot"]["batch_encode"] = "onehot"
val_loss_dict["onehot"]["expr_embed"] = "fourier"
val_loss_dict["onehot_nofourier"]["batch_encode"] = "onehot"
val_loss_dict["onehot_nofourier"]["expr_embed"] = "mlp"

val_loss = pd.concat([x for x in val_loss_dict.values()], axis = 0)

fig = plt.figure(figsize = (8, 10))
ax = fig.subplots(ncols = 1, nrows = 4)

for i, mask_prob in enumerate([0.1, 0.2, 0.3, 0.4]):
    sns.barplot(data = val_loss[val_loss["mask prob"] == mask_prob], x = "batch_encode", y = "val mlm", hue = "expr_embed", ax = ax[i])
    ax[i].set_xlabel(None)
    for container in ax[i].containers:
        ax[i].bar_label(container, fmt = "%.4f")
        leg = ax[i].legend(loc='upper left', prop={'size': 15}, frameon = False, bbox_to_anchor=(1.04, 1), title = "expr embedding")

fig.suptitle(data_case, fontsize = 20)
fig.savefig(res_dir + f"compare_exprcoding_{data_case}", bbox_inches = "tight")


# --------------------------------------------------------------------------------------------------------------
# Summary:
# If we consistently evaluate the model performance using the mask prediction accuracy (should be a comprehensive metric compared to embedding alignment)
# We evaluate the model performance on stand alone dataset: immune_all, lung_atlas
# 1. using fourier encoding is better than using mlp for continuous expr encoding
# 2. use the onehot batch encoding is the worst (Possible Reason: the batch encoding is all-zero on test)
# 3. use the batch encoding in better than one-hot, which shows that the batch statistics is possibly proving some information
# 4. however, the best performing model is the one without using any batch encoding information (Possible Reason: the batch statistics is not generalizable, more regularization might also be needed).
# 5. Potential regularization: discriminator, metric learning, etc
# --------------------------------------------------------------------------------------------------------------

# In[]
# Compare discriminator fine-tunning
# NOTE: mask_prob should not be too large for fine-tunning
model1 = "cp_vanilla_4_512_meta_enc_trans_1"
model2 = "cp_disc1_4_512_meta_maskprob0.15_1"
PROJECT_DIR = "/net/csefiles/xzhanglab/zzhang834/LLM_KD/"
res_dir1 = PROJECT_DIR + f"results/checkpoint/"
res_dir2 = PROJECT_DIR + f"results/checkpoint_finetune/"

data_case = "immune_all"
# data_case = "lung_atlas"
val_loss_dict = {}
val_loss_dict["orig"] = pd.read_csv(res_dir1 + model1 + f"/{data_case}_valloss.csv", index_col = 0)
val_loss_dict["disc"] = pd.read_csv(res_dir2 + model2 + f"/{data_case}_valloss.csv", index_col = 0)
val_loss_dict["orig"]["model"] = "orig"
val_loss_dict["disc"]["model"] = "disc-maskprob0.15"

val_loss = pd.concat([x for x in val_loss_dict.values()], axis = 0)

fig = plt.figure(figsize = (5, 15))
ax = fig.subplots(ncols = 1, nrows = 4)

for i, mask_prob in enumerate([0.1, 0.2, 0.3, 0.4]):
    sns.barplot(data = val_loss[val_loss["mask prob"] == mask_prob], x = "model", y = "val mlm", ax = ax[i])
    ax[i].set_xlabel(None)
    for container in ax[i].containers:
        ax[i].bar_label(container, fmt = "%.4f")
        # leg = ax[i].legend(loc='upper left', prop={'size': 15}, frameon = False, bbox_to_anchor=(1.04, 1), title = "model")

fig.suptitle(data_case, fontsize = 20)
fig.savefig(res_dir2 + f"compare_disc_{data_case}", bbox_inches = "tight")

# In[]
# Compare contrcb fine-tunning
# NOTE: mask_prob should not be too large for fine-tunning
model1 = "cp_vanilla_4_512_meta_enc_trans_1"
model2 = "cp_contrcb1_4_512_meta_1"
PROJECT_DIR = "/net/csefiles/xzhanglab/zzhang834/LLM_KD/"
res_dir1 = PROJECT_DIR + f"results/checkpoint/"
res_dir2 = PROJECT_DIR + f"results/checkpoint_finetune/"

data_case = "immune_all"
data_case = "lung_atlas"
val_loss_dict = {}
val_loss_dict["orig"] = pd.read_csv(res_dir1 + model1 + f"/{data_case}_valloss.csv", index_col = 0)
val_loss_dict["disc"] = pd.read_csv(res_dir2 + model2 + f"/{data_case}_valloss.csv", index_col = 0)
val_loss_dict["orig"]["model"] = "orig"
val_loss_dict["disc"]["model"] = "contrcb-maskprob0.1"

val_loss = pd.concat([x for x in val_loss_dict.values()], axis = 0)

fig = plt.figure(figsize = (5, 15))
ax = fig.subplots(ncols = 1, nrows = 4)

for i, mask_prob in enumerate([0.1, 0.2, 0.3, 0.4]):
    sns.barplot(data = val_loss[val_loss["mask prob"] == mask_prob], x = "model", y = "val mlm", ax = ax[i])
    ax[i].set_xlabel(None)
    for container in ax[i].containers:
        ax[i].bar_label(container, fmt = "%.4f")
        # leg = ax[i].legend(loc='upper left', prop={'size': 15}, frameon = False, bbox_to_anchor=(1.04, 1), title = "model")

fig.suptitle(data_case, fontsize = 20)
fig.savefig(res_dir2 + f"compare_contrcb_{data_case}", bbox_inches = "tight")

# In[]
# NOTE: mask_prob should not be too large for fine-tunning
model1 = "cp_vanilla_4_512_meta_nobatch_1"
model2 = "cp_contrcb1_4_512_meta_nobatch_1"
model3 = "cp_contrcbproj1_4_512_meta_nobatch_1"
PROJECT_DIR = "/net/csefiles/xzhanglab/zzhang834/LLM_KD/"
res_dir1 = PROJECT_DIR + f"results/checkpoint/"
res_dir2 = PROJECT_DIR + f"results/checkpoint_finetune/"

data_case = "immune_all"
data_case = "lung_atlas"
data_case = "pancreas"
val_loss_dict = {}
val_loss_dict["orig"] = pd.read_csv(res_dir1 + model1 + f"/{data_case}_valloss.csv", index_col = 0)
val_loss_dict["contrcb"] = pd.read_csv(res_dir2 + model2 + f"/{data_case}_valloss.csv", index_col = 0)
val_loss_dict["contrcb-proj"] = pd.read_csv(res_dir2 + model3 + f"/{data_case}_valloss.csv", index_col = 0)
val_loss_dict["orig"]["model"] = "orig"
val_loss_dict["contrcb"]["model"] = "contrcb-maskprob0.1"
val_loss_dict["contrcb-proj"]["model"] = "contrcb-proj-maskprob0.1"

val_loss = pd.concat([x for x in val_loss_dict.values()], axis = 0)

fig = plt.figure(figsize = (7, 15))
ax = fig.subplots(ncols = 1, nrows = 4)

for i, mask_prob in enumerate([0.1, 0.2, 0.3, 0.4]):
    sns.barplot(data = val_loss[val_loss["mask prob"] == mask_prob], x = "model", y = "val mlm", ax = ax[i])
    ax[i].set_xlabel(None)
    for container in ax[i].containers:
        ax[i].bar_label(container, fmt = "%.4f")
        # leg = ax[i].legend(loc='upper left', prop={'size': 15}, frameon = False, bbox_to_anchor=(1.04, 1), title = "model")

fig.suptitle(data_case, fontsize = 20)
fig.savefig(res_dir2 + f"compare_contrcb_{data_case}_nobatch", bbox_inches = "tight")

