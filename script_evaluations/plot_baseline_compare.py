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

# ablation dict
model_dict_embed = {"cp_contrcb1_mlm2_dyn_6_512_256_1": "Contr",
                    "cp_contrcb1_6_512_256_concat_full_1": "Contr & Batch Enc",
                    "cp_contrcb1_mlm10_dyn_6_512_256_concat_rawrestart_1": "Contr & Batch Enc2",
                    "scFoundation": "scFoundation",
                    "scGPT": "scGPT", 
                    "UCE": "UCE",
                    "scMulan": "scMulan",
                    "geneformer": "geneformer",
                    }

model_dict_impute = {"cp_contrcb1_mlm2_dyn_6_512_256_1": "Contr",
                    "cp_contrcb1_6_512_256_concat_full_1": "Contr & Batch Enc",
                    "cp_contrcb1_mlm10_dyn_6_512_256_concat_rawrestart_1": "Contr & Batch Enc2",
                    "scFoundation": "scFoundation",
                    "scVI": "scVI", 
                    "scGPT_human": "scGPT-human",
                    "scGPT_zeroshot": "scGPT-zeroshot"}

data_cases = ["immune_all", "pancreas", "lung_atlas", "covid19"]

# In[]
# --------------------------------------------------------------------------------------------------------------
#
# NOTE: Compare Batch removal
#
# --------------------------------------------------------------------------------------------------------------
use_pca = True
scores_list = []
for data_case in ["immune_all", "pancreas", "lung_atlas", "covid19"]:
    for model_name in model_dict_embed.keys():
        br_score_dir = PROJECT_DIR + f"results/zs_br/{model_name}/"
        if use_pca:
            scores = pd.read_csv(br_score_dir + f"scores_br_{data_case}_pca.csv", index_col = 0)
        else:
            scores = pd.read_csv(br_score_dir + f"scores_br_{data_case}.csv", index_col = 0)
        if data_case == "lung_atlas":
            scores = scores[scores["case"] == "total"]
            # assert False
        
        scores["model"] = model_dict_embed[model_name]
        scores["data_case"] = data_case
        scores_list.append(scores)

scores = pd.concat(scores_list, axis = 0, ignore_index = True)

fig = plt.figure(figsize = (20, 12))
ax = fig.subplots(ncols = 1, nrows = 4)

for i, score in enumerate(["ari", "nmi", "asw", "asw (batch)"]):
    sns.barplot(data = scores, x = "data_case", y = score, hue = "model", ax = ax[i])
    ax[i].set_xlabel(None)
    ax[i].set_ylabel(score.upper(), fontsize = 15)
    for container in ax[i].containers:
        ax[i].bar_label(container, fmt = "%.2f", color = "blue")
    leg = ax[i].legend(loc='upper left', prop={'size': 10}, frameon = False, bbox_to_anchor=(1.04, 1), title = "batch annos")
plt.tight_layout()
if use_pca:
    fig.savefig(PROJECT_DIR + "results/zs_br/" + f"compare_br_pca100.png", bbox_inches = "tight")
else:
    fig.savefig(PROJECT_DIR + "results/zs_br/" + f"ablation_br.png", bbox_inches = "tight")

# In[]
# --------------------------------------------------------------------------
#
# Plot cell type annotation cross-batch
#
# -------------------------------------------------------------------------
sns.set_theme(font_scale = 1.2)

for data_case in ["immune_all", "lung_atlas", "pancreas", "covid19"]:
    scores = []
    for model_name in model_dict_embed.keys():
        score_dir = PROJECT_DIR + f"results/zs_annot/{model_name}/"
        score_final = pd.read_csv(score_dir + f"class_crossbatch_{data_case}_pca100.csv", index_col = 0)
        score_final["method"] = model_dict_embed[model_name]

        score_final.loc[score_final["classifier"] == "knn_5", "classifier"] = "KNN"
        score_final.loc[score_final["classifier"] == "svm", "classifier"] = "SVM"

        scores.append(score_final)

    scores = pd.concat(scores, axis = 0, ignore_index = True)
    # scores = scores[scores["batch_test"] == 0]

    fig = plt.figure(figsize = (25, 5))
    axs = fig.subplots(nrows = 1, ncols = 2)
    # drop "F1-score (unweighted)" as the cluster sizes are not even
    for idx, metric in enumerate(["F1-score (weighted)", "accuracy"]):
        sns.barplot(data = scores, x = "classifier", y = metric, hue = "method", ax = axs[idx], capsize = 0.2, width = 0.6, saturation = 0.8)
        axs[idx].set_xlabel(None)
        axs[idx].set_ylabel(metric.upper(), fontsize = 15)
        # for container in axs[idx].containers:
            # axs[idx].bar_label(container, fmt = "%.2f", color = "blue")
        leg = axs[idx].legend(loc='upper left', fontsize = 15, title_fontsize = 15, frameon = False, bbox_to_anchor=(1.04, 1), title = "Method")
        axs[idx].set_ylim([0.6, 1])

    fig.suptitle(f"Zero-shot cell annotation ({data_case}-PCA100)", fontsize = 20)
    plt.tight_layout()  

    fig.savefig(PROJECT_DIR + "results/zs_annot/" + f"benchmark_annot_cb_{data_case}_pca100.png", bbox_inches = "tight")
    
matplotlib.rc_file_defaults()


# In[]
# --------------------------------------------------------------------------
#
# Plot cell type annotation
#
# -------------------------------------------------------------------------
scores = []
data_cases = ["immune_all", "lung_atlas", "pancreas", "covid19"]
for data_case in data_cases:
    for model_name in model_dict_embed.keys():
        if model_name == "geneformer":
            continue
        score_dir = PROJECT_DIR + f"results/zs_annot/{model_name}/"
        score = pd.read_csv(score_dir + f"class_{data_case}_pca100.csv", index_col = 0)

        score["model"] = model_name
        score["data_case"] = data_case
        scores.append(score)

scores = pd.concat(scores, axis = 0, ignore_index = True)

sns.set_theme()
fig = plt.figure(figsize = (35, 5))
axs = fig.subplots(nrows = 1, ncols = 2)
scores_sub = scores[scores["classifier"] == "knn_5"]
for idx, metric in enumerate(["F1-score (weighted)", "accuracy"]): # drop unweighted F1
    sns.barplot(data = scores_sub, x = "data_case", y = metric, hue = "model", ax = axs[idx])
    axs[idx].set_xlabel(None)
    axs[idx].set_ylabel(metric.upper(), fontsize = 15)
    for container in axs[idx].containers:
        axs[idx].bar_label(container, fmt = "%.2f", color = "blue")
    leg = axs[idx].legend(loc='upper left', prop={'size': 10}, frameon = False, bbox_to_anchor=(1.04, 1), title = "Method")

fig.suptitle("Benchmark: zero-shot cell type annotation", fontsize = 20)
plt.tight_layout() 
fig.savefig(PROJECT_DIR + "results/zs_annot/" + f"benchmark_annot_knn_pca.png", bbox_inches = "tight")

fig = plt.figure(figsize = (35, 5))
axs = fig.subplots(nrows = 1, ncols = 2)
scores_sub = scores[scores["classifier"] == "svm"]
for idx, metric in enumerate(["F1-score (weighted)", "accuracy"]):
    sns.barplot(data = scores_sub, x = "data_case", y = metric, hue = "model", ax = axs[idx])
    axs[idx].set_xlabel(None)
    axs[idx].set_ylabel(metric.upper(), fontsize = 15)
    for container in axs[idx].containers:
        axs[idx].bar_label(container, fmt = "%.2f", color = "blue")
    leg = axs[idx].legend(loc='upper left', prop={'size': 10}, frameon = False, bbox_to_anchor=(1.04, 1), title = "Method")

fig.suptitle("Benchmark: zero-shot cell type annotation", fontsize = 20)
plt.tight_layout()  
fig.savefig(PROJECT_DIR + "results/zs_annot/" + f"benchmark_annot_svm_pca.png", bbox_inches = "tight")

matplotlib.rc_file_defaults()

# In[]
# --------------------------------------------------------------------------
#
# Plot imputation
#
# -------------------------------------------------------------------------
sns.set_theme(font_scale = 1.2)

scores_list = []
for model_name in model_dict_impute.keys():
    score_dir = PROJECT_DIR + f"results/zs_imputation/{model_name}/"
    scores_list.append(pd.read_csv(score_dir + f"imputation_acc.csv", index_col = 0))
    scores_list[-1]["method"] = model_dict_impute[model_name]
scores = pd.concat(scores_list, axis = 0, ignore_index = True)

fig = plt.figure(figsize = (45, 20))
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

        # if metric == "Spearman":
            # axs[dataset_id, idx].set_ylim([0.25, 0.45])
            
plt.tight_layout()

fig.savefig(PROJECT_DIR + f"results/zs_imputation/imputate_acc.png", bbox_inches = "tight")
matplotlib.rc_file_defaults()
# %%
