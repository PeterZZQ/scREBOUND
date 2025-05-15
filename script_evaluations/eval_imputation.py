# In[]
import anndata
import numpy as np 
import scipy.sparse as sp
import os
import torch
from torch import nn
from sklearn.metrics import mean_squared_error, mean_absolute_error
from scipy.stats import pearsonr, spearmanr
import pandas as pd
from tqdm import tqdm
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib

def compute_metrics(counts_gt, counts_impute, mask):
    mse = []
    mae = []
    pearson = []
    spearman = []
    for cell in tqdm(range(counts_gt.shape[0])):
        cell_true = counts_gt[cell, :][mask[cell, :]]
        cell_pred = counts_impute[cell, :][mask[cell, :]] 
        mse.append(mean_squared_error(cell_true, cell_pred))
        mae.append(mean_absolute_error(cell_true, cell_pred))
        pearson.append(pearsonr(cell_true, cell_pred)[0])
        spearman.append(spearmanr(cell_true, cell_pred)[0])
    mse = np.mean(np.array(mse))
    mae = np.mean(np.array(mae))
    pearson = np.mean(np.array(pearson))
    spearman = np.mean(np.array(spearman))
    return mse, mae, pearson, spearman

# In[]
# ---------------------------------------------------------------------------
#
# [NOTE: one-time] Generate testing datasets
#
# ---------------------------------------------------------------------------
# # Load dataset
# impute_data_dir = "/net/csefiles/xzhanglab/zzhang834/LLM_KD/dataset/imputation_test/"
# if not os.path.exists(impute_data_dir):
#     os.makedirs(impute_data_dir)

# data_case = "immune_all"
# data_case = "pancreas"
# data_case = "lung_atlas"
# data_case = "covid19"

# if data_case == "covid19":
#     EVAL_DATA_DIR = "/net/csefiles/xzhanglab/zzhang834/LLM_KD/dataset/evaluation_datasets/"
# else:
#     EVAL_DATA_DIR = "/net/csefiles/xzhanglab/zzhang834/LLM_KD/dataset/scIB/"

# adata_test = anndata.read_h5ad(EVAL_DATA_DIR + f"{data_case}_aligned.h5ad")

# # NOTE: add mask
# def add_mask(counts, mask_prob):
#     mask = (np.random.rand(counts.shape[0], counts.shape[1]) > mask_prob)
#     # keep the value if mask is 1, set the value 0 if mask is 0
#     counts_mask = counts * mask
#     return counts_mask, mask

# # Add random masking
# np.random.seed(0)
# # 1. select the cells that need to be impute
# num_cells = 2000

# select_idx = np.random.choice(adata_test.shape[0], num_cells, replace = False)
# adata_test_select = adata_test[select_idx, :]
# # should be raw count here
# counts_select = adata_test_select.X.toarray()

# for mask_prob in [0.1, 0.2, 0.3, 0.4, 0.5]:
#     counts_mask, mask = add_mask(counts_select, mask_prob = mask_prob)
#     # construct anndata
#     adata_test_select.layers[f"counts_mask_{mask_prob}"] = sp.csr_matrix(counts_mask)
#     adata_test_select.layers[f"mask_{mask_prob}"] = sp.csr_matrix(1 - mask)

# adata_test_select.write_h5ad(impute_data_dir + f"{data_case}_masked_{num_cells}.h5ad")


# In[]
# ---------------------------------------------------------------------------
#
# Test on models
#
# ---------------------------------------------------------------------------

# Evaluation for scGPT zero shot
PROJECT_DIR = "/net/csefiles/xzhanglab/zzhang834/LLM_KD/"
model_name = "scGPT_zeroshot"

impute_data_dir = f"/net/csefiles/xzhanglab/shared/foundation_evaluation/data/imputation_test/{model_name}/"
res_dir = PROJECT_DIR + f"results/zs_imputation/{model_name}/"
if not os.path.exists(res_dir):
    os.makedirs(res_dir)

scores_list = []
data_case_list = ["covid19", "immune_all", "lung_atlas", "pancreas"]
for data_case in data_case_list:
    for mask in [0.1, 0.2, 0.3, 0.4, 0.5]:
        print("Data case:", data_case)
        print("Mask probability:", mask)

        adata_impute = anndata.read_h5ad(impute_data_dir + f"{data_case}_masked_2000_{mask}.h5ad")
        mask_mtx = adata_impute.layers[f"mask_{mask}"].astype(bool)
        mask_mtx = mask_mtx if isinstance(mask_mtx, np.ndarray) else mask_mtx.toarray()
        counts_impute = adata_impute.layers["scGPT_predictions"]
        counts_impute = counts_impute if isinstance(counts_impute, np.ndarray) else counts_impute.toarray()
        counts_gt = adata_impute.X if isinstance(adata_impute.X, np.ndarray) else adata_impute.X.toarray()
        
        # normalize the count_gt by libsize 1 before comparison
        counts_impute = counts_impute/(np.sum(counts_impute, axis = 1, keepdims = True) + 1e-6)
        counts_gt = counts_gt/(np.sum(counts_gt, axis = 1, keepdims = True) + 1e-6)

        mse, mae, pearson, spearman = compute_metrics(counts_gt, counts_impute, mask_mtx)
        scores = pd.DataFrame.from_dict({"data_case": [data_case],
                                        "mask_prob": [mask], 
                                        "method": [model_name],
                                        "MSE": [mse],
                                        "MAE": [mae],
                                        "Pearson": [pearson],
                                        "Spearman": [spearman]})

        scores_list.append(scores)
scores = pd.concat(scores_list, axis = 0, ignore_index = True)
scores.to_csv(res_dir + f"imputation_acc.csv")

# In[]
# Evaluation for scGPT Human
PROJECT_DIR = "/net/csefiles/xzhanglab/zzhang834/LLM_KD/"
model_name = "scGPT_human"

impute_data_dir = f"/net/csefiles/xzhanglab/shared/foundation_evaluation/data/imputation_test/{model_name}/"
res_dir = PROJECT_DIR + f"results/zs_imputation/{model_name}/"
if not os.path.exists(res_dir):
    os.makedirs(res_dir)

scores_list = []
data_case_list = ["covid19", "immune_all", "lung_atlas", "pancreas"]
for data_case in data_case_list:
    for mask in [0.1, 0.2, 0.3, 0.4, 0.5]:
        print("Data case:", data_case)
        print("Mask probability:", mask)

        adata_impute = anndata.read_h5ad(impute_data_dir + f"{data_case}_masked_2000_{mask}.h5ad")
        mask_mtx = adata_impute.layers[f"mask_{mask}"].astype(bool)
        mask_mtx = mask_mtx if isinstance(mask_mtx, np.ndarray) else mask_mtx.toarray()
        counts_impute = adata_impute.layers["scGPT_predictions"]
        counts_impute = counts_impute if isinstance(counts_impute, np.ndarray) else counts_impute.toarray()
        print(f"Impute number of genes: {counts_impute.shape[1]}")

        counts_gt = adata_impute.X if isinstance(adata_impute.X, np.ndarray) else adata_impute.X.toarray()
        
        # normalize the count_gt by libsize 1 before comparison
        counts_impute = counts_impute/(np.sum(counts_impute, axis = 1, keepdims = True) + 1e-6)
        counts_gt = counts_gt/(np.sum(counts_gt, axis = 1, keepdims = True) + 1e-6)

        mse, mae, pearson, spearman = compute_metrics(counts_gt, counts_impute, mask_mtx)
        scores = pd.DataFrame.from_dict({"data_case": [data_case],
                                        "mask_prob": [mask], 
                                        "method": [model_name],
                                        "MSE": [mse],
                                        "MAE": [mae],
                                        "Pearson": [pearson],
                                        "Spearman": [spearman]})

        scores_list.append(scores)
scores = pd.concat(scores_list, axis = 0, ignore_index = True)
scores.to_csv(res_dir + f"imputation_acc.csv")

# In[]
# Evaluation for scFoundation
PROJECT_DIR = "/net/csefiles/xzhanglab/zzhang834/LLM_KD/"
model_name = "scFoundation"

input_data_dir = PROJECT_DIR + "dataset/imputation_test/"

impute_data_dir = f"/net/csefiles/xzhanglab/shared/foundation_evaluation/data/imputation_test/{model_name}/"
res_dir = PROJECT_DIR + f"results/zs_imputation/{model_name}/"
if not os.path.exists(res_dir):
    os.makedirs(res_dir)

scores_list = []
data_case_list = ["covid19", "immune_all", "lung_atlas", "pancreas"]
for data_case in data_case_list:
    for mask in [0.1, 0.2, 0.3, 0.4, 0.5]:
        print("Data case:", data_case)
        print("Mask probability:", mask)
        adata_input = anndata.read_h5ad(input_data_dir + f"{data_case}_masked_2000.h5ad")
        adata_impute = anndata.read_h5ad(impute_data_dir + f"{data_case}_masked_2000_{mask}.h5ad")
        gene_overlap = np.intersect1d(adata_input.var.index.values, adata_impute.var.index.values)
        
        adata_impute = adata_impute[:, gene_overlap]
        adata_input = adata_input[:, gene_overlap]

        mask_mtx = adata_impute.layers[f"mask_{mask}"].astype(bool)
        mask_mtx = mask_mtx if isinstance(mask_mtx, np.ndarray) else mask_mtx.toarray()

        counts_impute = adata_impute.layers[f"scFoundation_prediction"]
        counts_impute = counts_impute if isinstance(counts_impute, np.ndarray) else counts_impute.toarray()
        print(f"Impute number of genes: {counts_impute.shape[1]}")

        # counts_gt = adata_impute.X if isinstance(adata_impute.X, np.ndarray) else adata_impute.X.toarray()
        counts_gt = adata_input.X if isinstance(adata_input.X, np.ndarray) else adata_input.X.toarray()
        
        # normalize the count_gt by libsize 1 before comparison
        counts_impute = counts_impute/(np.sum(counts_impute, axis = 1, keepdims = True) + 1e-6)
        counts_gt = counts_gt/(np.sum(counts_gt, axis = 1, keepdims = True) + 1e-6)

        mse, mae, pearson, spearman = compute_metrics(counts_gt, counts_impute, mask_mtx)
        scores = pd.DataFrame.from_dict({"data_case": [data_case],
                                        "mask_prob": [mask], 
                                        "method": [model_name],
                                        "MSE": [mse],
                                        "MAE": [mae],
                                        "Pearson": [pearson],
                                        "Spearman": [spearman]})

        scores_list.append(scores)
scores = pd.concat(scores_list, axis = 0, ignore_index = True)
scores.to_csv(res_dir + f"imputation_acc.csv")


# In[]
PROJECT_DIR = "/net/csefiles/xzhanglab/zzhang834/LLM_KD/"
# old model best
# model_name = f"cp_contrcb1_6_512_256_encbg_level2_1"
# model_dir = PROJECT_DIR + f"checkpoint/model_6_256/{model_name}.pth"

# new batch encoder model
# 1. batch encoder
# model_name = f"cp_6_512_256_concat_full_1"
# 2. batch encoder + contrastive
# model_name = f"cp_contrcb1_6_512_256_concat_full_1"
# 3. batch encoder + restart
# model_name = f"cp_6_512_256_concat_rawrestart_1"
# 4. batch encoder + contrastive + restart
# model_name = f"cp_contrcb1_mlm10_dyn_6_512_256_concat_rawrestart_1"

# new vanilla model
# 1. vanilla
# model_name = f"cp_6_512_256_1"
# 2. vanilla + contrastive
# model_name = f"cp_contrcb1_mlm2_dyn_6_512_256_1"
# TODO: 3. vanilla + restart
model_name = f"cp_6_512_256_rawrestart_1"
# TODO: 4. vanilla + contrastive + restart
model_name = f"cp_contrcb1_mlm10_dyn_6_512_256_rawrestart_1"

impute_data_dir = PROJECT_DIR + f"results/zs_imputation/{model_name}/"
res_dir = PROJECT_DIR + f"results/zs_imputation/{model_name}/"
if not os.path.exists(res_dir):
    os.makedirs(res_dir)

scores_list = []
data_case_list = ["covid19", "immune_all", "lung_atlas", "pancreas"]
for data_case in data_case_list:
    for mask in [0.1, 0.2, 0.3, 0.4, 0.5]:
        print("Data case:", data_case)
        print("Mask probability:", mask)

        adata_impute = anndata.read_h5ad(impute_data_dir + f"adata_impute_{data_case}_{mask}.h5ad")
        mask_mtx = adata_impute.layers[f"mask_{mask}"].astype(bool)
        mask_mtx = mask_mtx if isinstance(mask_mtx, np.ndarray) else mask_mtx.toarray()
        counts_impute = adata_impute.layers["mean"]
        counts_impute = counts_impute if isinstance(counts_impute, np.ndarray) else counts_impute.toarray()
        print(f"Impute number of genes: {counts_impute.shape[1]}")
        counts_gt = adata_impute.X if isinstance(adata_impute.X, np.ndarray) else adata_impute.X.toarray()

        # normalize the count_gt by libsize 1 before comparison
        # counts_impute = counts_impute/(np.sum(counts_impute, axis = 1, keepdims = True) + 1e-6)
        counts_gt = counts_gt/(np.sum(counts_gt, axis = 1, keepdims = True) + 1e-6)

        mse, mae, pearson, spearman = compute_metrics(counts_gt, counts_impute, mask_mtx)
        scores = pd.DataFrame.from_dict({"data_case": [data_case],
                                        "mask_prob": [mask], 
                                        "method": [model_name],
                                        "MSE": [mse],
                                        "MAE": [mae],
                                        "Pearson": [pearson],
                                        "Spearman": [spearman]})
        scores_list.append(scores)
scores = pd.concat(scores_list, axis = 0, ignore_index = True)
scores.to_csv(res_dir + f"imputation_acc.csv")

# In[]
# scVI
PROJECT_DIR = "/net/csefiles/xzhanglab/zzhang834/LLM_KD/"
model_name = "scVI"

impute_data_dir = f"/net/csefiles/xzhanglab/shared/foundation_evaluation/data/imputation_test/{model_name}/"
res_dir = PROJECT_DIR + f"results/zs_imputation/{model_name}/"
if not os.path.exists(res_dir):
    os.makedirs(res_dir)

scores_list = []
data_case_list = ["covid19", "immune_all", "lung_atlas", "pancreas"]
# data_case_list = ["lung_atlas"]
for data_case in data_case_list:
    for mask in [0.1, 0.2, 0.3, 0.4, 0.5]:
        print("Data case:", data_case)
        print("Mask probability:", mask)

        adata_impute = anndata.read_h5ad(impute_data_dir + f"{data_case}_masked_2000_{mask}.h5ad")
        print(f"Impute number of genes: {adata_impute.shape[1]}")

        mask_mtx = adata_impute.layers[f"mask_{mask}"].astype(bool)
        mask_mtx = mask_mtx if isinstance(mask_mtx, np.ndarray) else mask_mtx.toarray()
        counts_impute = adata_impute.layers["scVI_predicted"]
        counts_impute = counts_impute if isinstance(counts_impute, np.ndarray) else counts_impute.toarray()        
        counts_gt = adata_impute.X if isinstance(adata_impute.X, np.ndarray) else adata_impute.X.toarray()
        
        # normalize the count_gt by libsize 1 before comparison
        counts_impute = counts_impute/(np.sum(counts_impute, axis = 1, keepdims = True) + 1e-6)
        counts_gt = counts_gt/(np.sum(counts_gt, axis = 1, keepdims = True) + 1e-6)


        mse, mae, pearson, spearman = compute_metrics(counts_gt, counts_impute, mask_mtx)
        scores = pd.DataFrame.from_dict({"data_case": [data_case],
                                        "mask_prob": [mask], 
                                        "method": [model_name],
                                        "MSE": [mse],
                                        "MAE": [mae],
                                        "Pearson": [pearson],
                                        "Spearman": [spearman]})

        scores_list.append(scores)
scores = pd.concat(scores_list, axis = 0, ignore_index = True)
scores.to_csv(res_dir + f"imputation_acc.csv")

# %%
