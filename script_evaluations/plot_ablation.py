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
import matplotlib
import sys
sys.path.append("/net/csefiles/xzhanglab/zzhang834/LLM_KD/src")

import eval
import anndata

sns.set_theme()

PROJECT_DIR = "/net/csefiles/xzhanglab/zzhang834/LLM_KD/"
res_dir = PROJECT_DIR + f"results/ablation/"

# ablation dict
# model_dict = {"cp_6_512_256_1": "Vanilla",
#               "cp_contrcb1_mlm2_dyn_6_512_256_1": "Contr",
#               "cp_6_512_256_concat_full_1": "Batch Enc",
#               "cp_contrcb1_6_512_256_concat_full_1": "Contr & Batch Enc",
#               "cp_6_512_256_concat_rawrestart_1": "Batch Enc2",
#               "cp_contrcb1_mlm10_dyn_6_512_256_concat_rawrestart_1": "Contr & Batch Enc2", 
#               "cp_6_512_256_rawrestart_1": "Vanilla2", 
#               "cp_contrcb1_mlm10_dyn_6_512_256_rawrestart_1": "Contr2"}

model_dict_impute = {"cp_6_512_256_rawrestart_1": "Vanilla",
                    #  "cp_contrcb1_mlm10_dyn_6_512_256_rawrestart_1": "Contr",
                     "cp_6_512_256_concat_rawrestart_1": "Batch Enc",
                     "cp_contrcb1_mlm10_dyn_6_512_256_concat_rawrestart_1": "Batch Enc & Contr"}

# model_dict_contr = {"cp_6_512_256_concat_rawrestart_1": "scREBOUND w/o contr",
#                     "cp_contrcb1_mlm10_dyn_6_512_256_concat_rawrestart_1": "scREBOUND w/ contr"}
model_dict_contr = {"cp_6_512_256_concat_rawrestart_1": "Batch Enc",
                    "cp_contrcb1_mlm10_dyn_6_512_256_concat_rawrestart_1": "Batch Enc & Contr"}

# In[]
# --------------------------------------------------------------------------------------------------------------
#
# NOTE: Compare MLM
#
# --------------------------------------------------------------------------------------------------------------

loss_dir = PROJECT_DIR + f"results/checkpoint/"
val_loss_list = []
for data_case in ["immune_all", "pancreas", "lung_atlas", "covid19"]:
    for model_name in model_dict_impute.keys():
        val_loss = pd.read_csv(loss_dir + model_name + f"/{data_case}_valloss.csv", index_col = 0)
        val_loss["method"] = model_dict_impute[model_name]
        val_loss["data_case"] = data_case
        val_loss_list.append(val_loss)

val_loss = pd.concat(val_loss_list, axis = 0, ignore_index = True)

fig = plt.figure(figsize = (15, 10))
ax = fig.subplots(ncols = 1, nrows = 4)

for i, mask_prob in enumerate([0.1, 0.2, 0.3, 0.4]):
    sns.barplot(data = val_loss[val_loss["mask prob"] == mask_prob], x = "data_case", y = "val mlm", hue = "method", ax = ax[i])
    ax[i].set_xlabel(None)
    ax[i].set_ylabel(f"Mask Prob {mask_prob}", fontsize = 17)
    for container in ax[i].containers:
        ax[i].bar_label(container, fmt = "%.3f", color = "blue")
    leg = ax[i].legend(loc='upper left', prop={'size': 10}, frameon = False, bbox_to_anchor=(1.04, 1), title = "Batch Label")
fig.savefig(res_dir + "ablation_mlm.png", bbox_inches = "tight", dpi = 250)
plt.tight_layout()


# In[]
# --------------------------------------------------------------------------------------------------------------
#
# NOTE: Compare Batch removal
#
# --------------------------------------------------------------------------------------------------------------
sns.set_theme()
use_pca = True
scores_list = []
for data_case in ["immune_all", "pancreas", "lung_atlas", "covid19"]:
    for model_name in model_dict_contr.keys():
        br_score_dir = PROJECT_DIR + f"results/zs_br/{model_name}/"
        if use_pca:
            scores = pd.read_csv(br_score_dir + f"scores_br_{data_case}_pca.csv", index_col = 0)
        else:
            scores = pd.read_csv(br_score_dir + f"scores_br_{data_case}.csv", index_col = 0)
        if data_case == "lung_atlas":
            scores = scores[scores["case"] == "total"]
            # assert False
        
        scores["model"] = model_dict_contr[model_name]
        scores["data_case"] = data_case
        scores_list.append(scores)

scores = pd.concat(scores_list, axis = 0, ignore_index = True)

fig = plt.figure(figsize = (18, 4))
ax = fig.subplots(ncols = 3, nrows = 1)

# for i, score in enumerate(["ari", "nmi", "asw", "asw (batch)"]):
for i, score in enumerate(["ari", "nmi", "asw"]):
    sns.barplot(data = scores, x = "data_case", y = score, hue = "model", ax = ax[i])
    ax[i].set_xlabel(None)
    ax[i].set_ylabel(score.upper(), fontsize = 15)
    for container in ax[i].containers:
        ax[i].bar_label(container, fmt = "%.2f", color = "blue")
    if i == 2:
        leg = ax[i].legend(loc='upper left', fontsize = 15, title_fontsize = 15, frameon = False, bbox_to_anchor=(1.04, 1), title = "Model")
    else:
        ax[i].legend().remove()
    ax[i].set_xticklabels(["Immune", "Pancreas", "Lung", "Covid19"], fontsize = 15)
plt.tight_layout()
if use_pca:
    fig.savefig(res_dir + f"ablation_br_pca100.png", bbox_inches = "tight")
else:
    fig.savefig(res_dir + f"ablation_br.png", bbox_inches = "tight")


# In[]
# --------------------------------------------------------------------------------------------------------------
#
# NOTE: Compare Annotation
#
# --------------------------------------------------------------------------------------------------------------
scores = []
data_cases = ["immune_all", "pancreas", "lung_atlas", "covid19"]
for data_case in data_cases:
    # for model_name in model_dict.keys():
    for model_name in model_dict_contr.keys():
        score_dir = PROJECT_DIR + f"results/zs_annot/{model_name}/"
        score = pd.read_csv(score_dir + f"class_{data_case}_pca100.csv", index_col = 0)

        score["model"] = model_dict_contr[model_name]
        score["data_case"] = data_case
        scores.append(score)

scores = pd.concat(scores, axis = 0, ignore_index = True)

sns.set_theme()
fig = plt.figure(figsize = (13, 5))
axs = fig.subplots(nrows = 1, ncols = 2)
scores_sub = scores[scores["classifier"] == "knn_5"]
for idx, metric in enumerate(["F1-score (weighted)", "accuracy"]):
    sns.barplot(data = scores_sub, x = "data_case", y = metric, hue = "model", ax = axs[idx], width = 0.8)
    axs[idx].set_xlabel(None)
    axs[idx].set_ylabel(metric.upper(), fontsize = 15)
    for container in axs[idx].containers:
        axs[idx].bar_label(container, fmt = "%.2f", color = "blue")
    if idx == 1:
        leg = axs[idx].legend(loc='upper left', fontsize = 15, title_fontsize = 15, frameon = False, bbox_to_anchor=(1.04, 1), title = "Model")
    else:
        axs[idx].legend().remove()
    axs[idx].set_xticklabels(["Immune", "Pancreas", "Lung", "Covid19"], fontsize = 15)

fig.suptitle("Ablation: zero-shot cell type annotation (kNN)", fontsize = 20)
plt.tight_layout() 
fig.savefig(PROJECT_DIR + f"results/ablation/ablation_annot_knn_pca.png", bbox_inches = "tight")

fig = plt.figure(figsize = (13, 5))
axs = fig.subplots(nrows = 1, ncols = 2)
scores_sub = scores[scores["classifier"] == "svm"]
for idx, metric in enumerate(["F1-score (weighted)", "accuracy"]):
    sns.barplot(data = scores_sub, x = "data_case", y = metric, hue = "model", ax = axs[idx], width = 0.8)
    axs[idx].set_xlabel(None)
    axs[idx].set_ylabel(metric.upper(), fontsize = 15)
    for container in axs[idx].containers:
        axs[idx].bar_label(container, fmt = "%.2f", color = "blue")
    if idx == 1:
        leg = axs[idx].legend(loc='upper left', fontsize = 15, title_fontsize = 15, frameon = False, bbox_to_anchor=(1.04, 1), title = "Model")
    else:
        axs[idx].legend().remove()
    axs[idx].set_xticklabels(["Immune", "Pancreas", "Lung", "Covid19"], fontsize = 15)

fig.suptitle("Ablation: zero-shot cell type annotation (SVM)", fontsize = 20)
plt.tight_layout()  
fig.savefig(res_dir + f"ablation_annot_svm_pca.png", bbox_inches = "tight")

matplotlib.rc_file_defaults()

# In[]
sns.set_theme(font_scale = 1.2)

for data_case in ["immune_all", "lung_atlas", "pancreas", "covid19"]:
    scores = []
    for model_name in model_dict_contr.keys():
        score_dir = PROJECT_DIR + f"results/zs_annot/{model_name}/"
        score_final = pd.read_csv(score_dir + f"class_crossbatch_{data_case}_pca100.csv", index_col = 0)
        score_final["method"] = model_dict_contr[model_name]
        score_final.loc[score_final["classifier"] == "knn_5", "classifier"] = "KNN"
        score_final.loc[score_final["classifier"] == "svm", "classifier"] = "SVM"
        scores.append(score_final)
    scores = pd.concat(scores, axis = 0, ignore_index = True)

    fig = plt.figure(figsize = (25, 5))
    axs = fig.subplots(nrows = 1, ncols = 2)
    for idx, metric in enumerate(["F1-score (weighted)", "accuracy"]):
        sns.barplot(data = scores, x = "classifier", y = metric, hue = "method", ax = axs[idx], capsize = 0.2, width = 0.6, saturation = 0.8)
        axs[idx].set_xlabel(None)
        axs[idx].set_ylabel(metric.upper(), fontsize = 15)
        # for container in axs[idx].containers:
            # axs[idx].bar_label(container, fmt = "%.2f", color = "blue")
        leg = axs[idx].legend(loc='upper left', fontsize = 15, title_fontsize = 15, frameon = False, bbox_to_anchor=(1.04, 1), title = "Method")

    fig.suptitle(f"Zero-shot cell annotation ({data_case}-PCA100)", fontsize = 20)
    plt.tight_layout()  

    fig.savefig(res_dir + f"ablation_annot_cb_{data_case}_pca100.png", bbox_inches = "tight")

matplotlib.rc_file_defaults()

# In[]
# --------------------------------------------------------------------------------------------------------------
#
# NOTE: Compare imputation
#
# --------------------------------------------------------------------------------------------------------------

sns.set_theme(font_scale = 1.2)

scores_list = []
for model_name in model_dict_impute.keys():
    score_dir = PROJECT_DIR + f"results/zs_imputation/{model_name}/"
    scores_list.append(pd.read_csv(score_dir + f"imputation_acc.csv", index_col = 0))
    scores_list[-1]["method"] = model_dict_impute[model_name]

scores = pd.concat(scores_list, axis = 0, ignore_index = True)
scores = scores[scores["mask_prob"].isin([0.1, 0.2])]

fig = plt.figure(figsize = (15, 20))
axs = fig.subplots(nrows = 4, ncols = 2)
for idx, metric in enumerate(["Pearson", "Spearman"]):
    for dataset_id, data_case in enumerate(data_cases):
        scores_sub = scores[scores["data_case"] == data_case]
        sns.barplot(data = scores_sub, x = "mask_prob", y = metric, hue = "method", ax = axs[dataset_id, idx])
        axs[dataset_id, idx].set_xlabel(None)
        axs[dataset_id, idx].set_ylabel(metric.upper(), fontsize = 15)
        for container in axs[dataset_id, idx].containers:
            axs[dataset_id, idx].bar_label(container, fmt = "%.3f", color = "blue", fontsize = 12)
        
        if idx != 1:
            axs[dataset_id, idx].legend().remove()
        else:
            leg = axs[dataset_id, idx].legend(loc='upper left', fontsize = 15, title_fontsize = 15, frameon = False, bbox_to_anchor=(1.04, 1), title = "Method")
        axs[dataset_id, idx].set_title(f"{data_case}: {metric}", fontsize = 20)

        if metric == "Spearman":
            axs[dataset_id, idx].set_ylim([0.25, 0.45])

plt.tight_layout()

fig.savefig(res_dir + f"ablation_imputate_acc.png", bbox_inches = "tight")
matplotlib.rc_file_defaults()

# %%
