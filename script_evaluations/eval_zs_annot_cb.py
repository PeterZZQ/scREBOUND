# In[]
import torch 
import anndata
import scipy.sparse as sp
import numpy as np
import pandas as pd

import tqdm
from sklearn.neighbors import KNeighborsClassifier
from sklearn.decomposition import PCA
from sklearn.metrics import classification_report

import seaborn as sns
import matplotlib

import scanpy as sc
import sys, os 
sys.path.append("/net/csefiles/xzhanglab/zzhang834/LLM_KD/src")
import utils
import matplotlib.pyplot as plt

def knn_classification(embed_train, label_train, embed_test, batch_size = 512, n_neighbors = 5):
    knn_classifier = KNeighborsClassifier(n_neighbors=n_neighbors)
    knn_classifier.fit(embed_train, label_train)

    # Predict labels for test data
    label_test_pred = []
    for i in tqdm.tqdm(range(0, embed_test.shape[0], batch_size)):
        if isinstance(embed_test, np.ndarray):
            embed_batch = embed_test[i:i + batch_size]
        else:
            embed_batch = embed_test[i:i + batch_size].toarray()
        label_pred_batch = knn_classifier.predict(embed_batch)
        label_test_pred.append(label_pred_batch)
    label_test_pred = np.concatenate(label_test_pred)

    return label_test_pred


def svm_classification(embed_train, label_train, embed_test, batch_size = 512):
    from sklearn.svm import SVC
    svm = SVC(kernel='rbf', probability=False, class_weight = 'balanced')
    svm.fit(embed_train, label_train)

    # Predict labels for test data
    label_test_pred = []
    for i in tqdm.tqdm(range(0, embed_test.shape[0], batch_size)):
        if isinstance(embed_test, np.ndarray):
            embed_batch = embed_test[i:i + batch_size]
        else:
            embed_batch = embed_test[i:i + batch_size].toarray()
        label_pred_batch = svm.predict(embed_batch)
        label_test_pred.append(label_pred_batch)
    label_test_pred = np.concatenate(label_test_pred)

    return label_test_pred



# In[]
# -------------------------------------------------------
#
# Benchmark cell type annotation: Train on one batch, eval on another batch
# 
# -------------------------------------------------------

data_dir = "/net/csefiles/xzhanglab/zzhang834/hs_download/"
PROJECT_DIR = "/net/csefiles/xzhanglab/zzhang834/LLM_KD/"

# NOTE: our model
# model_name = "stable_4_512_level2"

# old model best
# model_name = "cp_contrcb1_6_512_256_encbg_level2_1"

# new batch encoder model
# 1. batch encoder
# model_name = "cp_6_512_256_concat_full_1"
# 2. batch encoder + contrastive
# model_name = "cp_contrcb1_6_512_256_concat_full_1"
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
# model_name = f"cp_6_512_256_rawrestart_1"
# TODO: 4. vanilla + contrastive + restart
model_name = f"cp_contrcb1_mlm10_dyn_6_512_256_rawrestart_1"

# Baseline model
# model_name = "scGPT"
# model_name = "scMulan"
# model_name = "UCE"
# model_name = "scFoundation"
# model_name = "geneformer"

res_dir = PROJECT_DIR + f"results/zs_annot/{model_name}/"
if not os.path.exists(res_dir):
    os.makedirs(res_dir)

if model_name in ["scGPT", "scMulan", "UCE", "scFoundation", "geneformer"]:
    test_embed_dir = res_dir + f"test_embed/"
else:
    test_embed_dir = PROJECT_DIR + f"results/checkpoint/{model_name}/"

# Construct training dataset
for data_case in ["immune_all", "lung_atlas", "pancreas", "covid19"]:
# for data_case in ["covid19"]:
    adata_test = anndata.read_h5ad(test_embed_dir + f"adata_embed_{data_case}.h5ad")
    if "batch_id" not in adata_test.obs.columns:
        adata_test.obs["batch_id"], batch_code = pd.factorize(adata_test.obs["batch"], sort = True)
    adata_test.obs["label_id"], label_code = pd.factorize(adata_test.obs["label"], sort = True)

    score_final_list = []

    uniq_batches = np.unique(adata_test.obs["batch_id"].values)
    if data_case == "covid19":
        uniq_batches = np.random.choice(uniq_batches, 10)

    for test_batch in uniq_batches:
        print(f"Test batch: {test_batch}")
        embed_test = adata_test[adata_test.obs["batch_id"] == test_batch].X.toarray()
        label_id_test_gt = adata_test[adata_test.obs["batch_id"] == test_batch].obs["label_id"].values.squeeze()
        label_test_gt = label_code[label_id_test_gt]

        embed_train = adata_test[adata_test.obs["batch_id"] != test_batch].X.toarray()
        label_id_train = adata_test[adata_test.obs["batch_id"] != test_batch].obs["label_id"].values.squeeze()
        label_train = label_code[label_id_train]

        print(f"training set: {len(embed_train)}, testing set: {len(embed_test)}")

        n_neighbors = 5
        score_list = []
        # NOTE: some feature dimensions are too high, reduce to pca for all comparison
        for reduce_dim in [True]:
            if reduce_dim:
                print("calculate pca")
                pca_op = PCA(n_components = 100)
                embed_train_input = pca_op.fit_transform(embed_train)
                embed_test_input = pca_op.transform(embed_test)
            else:
                embed_train_input = embed_train
                embed_test_input = embed_test

            print("knn classifier")
            label_id_test = knn_classification(embed_train = embed_train_input, label_train = label_id_train, embed_test = embed_test_input, batch_size = 512, n_neighbors = n_neighbors)
            label_pred_test_knn = label_code[label_id_test]

            print("svm classifier")
            label_id_test = svm_classification(embed_train = embed_train_input, label_train = label_id_train, embed_test = embed_test_input, batch_size = 512)
            label_pred_test_svm = label_code[label_id_test]

            # drop the no-mapping labels
            print(classification_report(label_test_gt, label_pred_test_knn, output_dict = False, zero_division=0))
            score_dict_knn = classification_report(label_test_gt, label_pred_test_knn, output_dict = True, zero_division=0)
            score_knn = pd.DataFrame.from_dict(score_dict_knn)

            print(classification_report(label_test_gt, label_pred_test_svm, output_dict = False, zero_division=0))
            score_dict_svm = classification_report(label_test_gt, label_pred_test_svm, output_dict = True, zero_division=0)
            score_svm = pd.DataFrame.from_dict(score_dict_svm)

            score = pd.DataFrame(columns = [["classifier", "accuracy", "F1-score (unweighted)", "F1-score (weighted)", "use_pca"]])
            score["classifier"] = [f"knn_{n_neighbors}", "svm"]
            score["accuracy"] = [score_knn.loc["f1-score", "accuracy"], score_svm.loc["f1-score", "accuracy"]]
            score["F1-score (unweighted)"] = [score_knn.loc["f1-score", "macro avg"], score_svm.loc["f1-score", "macro avg"]]
            score["F1-score (weighted)"] = [score_knn.loc["f1-score", "weighted avg"], score_svm.loc["f1-score", "weighted avg"]]
            score["use_pca"] = [reduce_dim, reduce_dim]
            score_list.append(score)

        score_final = pd.concat(score_list, axis = 0, ignore_index = True)
        score_final["batch_test"] = test_batch

        score_final_list.append(score_final)

    score_final = pd.concat(score_final_list, axis = 0, ignore_index = True)
    score_final.to_csv(res_dir + f"class_crossbatch_{data_case}_pca100.csv")
    print(f"Done: {data_case}")


# %%
