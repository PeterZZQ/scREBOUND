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
import sys 
sys.path.append("/net/csefiles/xzhanglab/zzhang834/LLM_KD/src")
import utils
import matplotlib.pyplot as plt

def knn_classification(embed_train, label_train, embed_test, batch_size = 512, n_neighbors = 5, select_label = None):
    knn_classifier = KNeighborsClassifier(n_neighbors=n_neighbors)

    if select_label is not None:
        select_idx = []
        for idx, label in enumerate(label_train):
            if label in select_label:
                select_idx.append(idx)
        select_idx = np.array(select_idx)
        if isinstance(embed_train, np.ndarray):
            embed_train_select = embed_train[select_idx, :]
        else:
            embed_train_select = embed_train[select_idx, :].toarray()
        label_id_train_select = label_id_train[select_idx]
    
    else:
        if isinstance(embed_train, np.ndarray):
            embed_train_select = embed_train
        else:
            embed_train_select = embed_train.toarray()
        label_id_train_select = label_id_train

    knn_classifier.fit(embed_train_select, label_id_train_select)

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

    return label_test_pred, {"embed_train": embed_train_select, "label_train": label_id_train_select}


def svm_classification(embed_train, label_train, embed_test, batch_size = 512, select_label = None):
    from sklearn.svm import SVC
    svm = SVC(kernel='rbf', probability = False, class_weight = 'balanced')

    if select_label is not None:
        select_idx = []
        for idx, label in enumerate(label_train):
            if label in select_label:
                select_idx.append(idx)
        select_idx = np.array(select_idx)
        if isinstance(embed_train, np.ndarray):
            embed_train_select = embed_train[select_idx, :]
        else:
            embed_train_select = embed_train[select_idx, :].toarray()
        label_id_train_select = label_id_train[select_idx]
    
    else:
        if isinstance(embed_train, np.ndarray):
            embed_train_select = embed_train
        else:
            embed_train_select = embed_train.toarray()
        label_id_train_select = label_id_train

    svm.fit(embed_train_select, label_id_train_select)

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

    return label_test_pred, {"embed_train": embed_train_select, "label_train": label_id_train_select}

def find_label_mapping(data_case):
    # [NOTE: OPT] filtering based on the label name, select data with only related labels 
    # HSPCs, Megakaryocyte progenitors in gt annotation is missing, 
    #'CL:0000037--hematopoietic stem cell', all cells are classified as hsc
    if data_case == "immune_all":
        ct2onto = {'CD10+ B cells': 'CL:0000785--mature B cell',
                    'CD20+ B cells': 'CL:0000785--mature B cell',
                    # 'CD14+ Monocytes': 'CL:0001054--CD14-positive monocyte', 
                    'CD16+ Monocytes': 'CL:0000576--monocyte',
                    'CD14+ Monocytes': 'CL:0000576--monocyte',
                    'Monocyte progenitors': 'CL:0000576--monocyte',
                    'CD4+ T cells': 'CL:0000492--CD4-positive helper T cell',
                    'CD8+ T cells': 'CL:0000625--CD8-positive, alpha-beta T cell',
                    'Erythrocytes': 'CL:0000232--erythrocyte',
                    'Erythroid progenitors': 'CL:0000232--erythrocyte', 
                    'Monocyte-derived dendritic cells': 'CL:0011031--monocyte-derived dendritic cell',
                    'NK cells': 'CL:0000623--natural killer cell',
                    'NKT cells': 'CL:0000814--mature NK T cell',
                    'Plasma cells': 'CL:0000786--plasma cell',
                    'Plasmacytoid dendritic cells': 'CL:0001058--plasmacytoid dendritic cell, human',
                    'HSPCs': 'Other',
                    'Megakaryocyte progenitors': 'Other'} 

        # select the relevant labels
        select_label = np.array([x for x in ct2onto.values() if x != 'Other'])
        select_label = np.unique(select_label)

    elif data_case == "pancreas":
        ct2onto = {'acinar': 'CL:0002064--pancreatic acinar cell',
                    'activated_stellate': 'CL:0002410--pancreatic stellate cell',
                    'alpha': 'CL:0000171--pancreatic A cell',
                    'beta': 'CL:0000169--type B pancreatic cell',
                    'delta': 'CL:0000173--pancreatic D cell', 
                    'ductal': 'CL:0002079--pancreatic ductal cell',
                    'endothelial': 'CL:0000115--endothelial cell',
                    'epsilon': 'CL:0005019--pancreatic epsilon cell',
                    'gamma': 'CL:0002275--pancreatic PP cell',
                    'macrophage': 'CL:0000235--macrophage',
                    'mast': 'CL:0000097--mast cell',
                    'quiescent_stellate': 'CL:0002410--pancreatic stellate cell',
                    'schwann': 'CL:0002573--Schwann cell',
                    't_cell': 'CL:0000084--T cell'}

        select_label = np.array([x for x in ct2onto.values()])
        select_label = np.unique(select_label)

    elif data_case == "lung_atlas":
        ct2onto = {'B cell': 'CL:0000785--mature B cell',
                    'Basal 1': 'CL:0000646--basal cell',
                    'Basal 2': 'CL:0000646--basal cell',
                    'Ciliated': 'CL:0000064--ciliated cell',
                    'Dendritic cell': 'CL:0000451--dendritic cell',
                    'Endothelium': 'CL:0000115--endothelial cell',
                    'Fibroblast': 'CL:0000057--fibroblast', 
                    'Ionocytes': 'CL:0005006--ionocyte',
                    'Lymphatic': 'Other', # issue: if use lymphoid, then all T cells would also be classified as lymphoid
                    'Type 1': 'CL:4028004--alveolar type 1 fibroblast cell',
                    'Type 2': 'CL:4028006--alveolar type 2 fibroblast cell',
                    'Macrophage': 'CL:0000235--macrophage',
                    'Mast cell': 'CL:0000097--mast cell',
                    'Neutrophil_CD14_high': 'CL:0000775--neutrophil',
                    'Neutrophils_IL1R2': 'CL:0000775--neutrophil',
                    'Secretory': 'CL:0000151--secretory cell',
                    'T/NK cell': 'T/NK cell',
                    'T/NK cell': 'T/NK cell'}

        # NOTE: Here drop the T and NK cells as there is no ground truth correspond to those two cell types T/NK are mixed
        select_label = np.array([x for x in ct2onto.values() if (x != 'Other') and (x != 'T/NK cell')]) # + ['CL:0000084--T cell', 'CL:0000623--natural killer cell']) 
        select_label = np.unique(select_label)

    elif data_case == "covid19":
        ct2onto = {'B': 'CL:0000785--mature B cell',
                'CD4 T': 'CL:0000492--CD4-positive helper T cell',
                'CD8 T': 'CL:0000625--CD8-positive, alpha-beta T cell',
                'DC': 'CL:0000451--dendritic cell',
                'Mono': 'CL:0000576--monocyte',
                'NK': 'CL:0000623--natural killer cell',
                'other': 'Other',
                'other T': 'Other'}  # 'CL:0000084--T cell' should be dropped instead
        
        select_label = np.array([x for x in ct2onto.values() if (x != 'Other')])
        select_label = np.unique(select_label)

    return ct2onto, select_label

# In[]
# -------------------------------------------------------
#
# Benchmark cell type annotation: train with training data, eval in test data
# 
# -------------------------------------------------------

data_dir = "/net/csefiles/xzhanglab/zzhang834/hs_download/"
PROJECT_DIR = "/net/csefiles/xzhanglab/zzhang834/LLM_KD/"
# label name:id dictionary
label_dict = torch.load(data_dir + "meta_data/label_dict.pt", weights_only = False)

# Methods
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
# model_name = f"cp_6_512_256_rawrestart_1"
# TODO: 4. vanilla + contrastive + restart
# model_name = f"cp_contrcb1_mlm10_dyn_6_512_256_rawrestart_1"

# Baseline methods
# model_name = "scGPT"
# model_name = "scMulan"
# model_name = "UCE"
model_name = "scFoundation"
res_dir = PROJECT_DIR + f"results/zs_annot/{model_name}/"
train_embed_dir = res_dir + f"train_embed/"

if model_name in ["scGPT", "UCE", "scMulan", "scFoundation"]:
    test_embed_dir = res_dir + f"test_embed/"
else:
    test_embed_dir = PROJECT_DIR + f"results/checkpoint/{model_name}/"

# Construct training dataset
embed_train = []
label_id_train = []
num_partitions = 1
if num_partitions > 1:
    for idx in range(num_partitions):
        adata_embed_idx = anndata.read_h5ad(train_embed_dir + f"embed_model_{idx}.h5ad")
        embed_train.append(adata_embed_idx.X.copy())
        label_id_train.append(np.array([x for x in adata_embed_idx.obs["label_id"].values]))
    if isinstance(embed_train[0], np.ndarray):
        embed_train = np.vstack(embed_train) 
    else:
        embed_train = sp.vstack(embed_train) 
    label_id_train = np.concatenate(label_id_train, axis = 0)

else:
    adata_embed_idx = anndata.read_h5ad(train_embed_dir + f"embed_model_0.h5ad")
    embed_train = adata_embed_idx.X.copy()
    label_id_train = np.array([x for x in adata_embed_idx.obs["label_id"].values]) 

# find label name
label_train = label_dict["label_bincode"].index.values[label_id_train]

# In[]
# NOTE: Load test embedding

n_neighbors = 5
reduce_dim = True
n_pcs = 100

for data_case in ["immune_all", "lung_atlas", "pancreas", "covid19"]:
    adata_test = anndata.read_h5ad(test_embed_dir + f"adata_embed_{data_case}.h5ad")
    embed_test = adata_test.X

    ct2onto, select_label = find_label_mapping(data_case)

    select_label_id = []
    for label in select_label:
        select_label_id.append(np.where(label_dict["label_bincode"].index.values == label)[0][0])
    select_label_id = np.array(select_label_id)

    # Ground truth label
    label_test_gt = np.array([x for x in adata_test.obs["label"].values]).astype(object)
    label_test_gt_mapped = []
    label_test_gt_id = []
    for x in label_test_gt:
        label_test_gt_mapped.append(ct2onto[x])
        
        if len(np.where(label_dict["label_bincode"].index.values == ct2onto[x])[0]) == 1:
            label_test_gt_id.append(np.where(label_dict["label_bincode"].index.values == ct2onto[x])[0][0])
        else:
            # other, no mapping label 
            label_test_gt_id.append(-1)
    label_gt_mapped = np.array(label_test_gt_mapped).astype(object)
    label_id_gt = np.array(label_test_gt_id)

    score_list = []
    if reduce_dim:
        print("calculate pca")
        pca_op = PCA(n_components = n_pcs)
        embed_train_input = pca_op.fit_transform(embed_train)
        embed_test_input = pca_op.transform(embed_test)
    else:
        embed_train_input = embed_train
        embed_test_input = embed_test

    print("knn classifier")
    label_id_test, train_data_select = knn_classification(embed_train = embed_train_input, label_train = label_id_train, embed_test = embed_test_input, batch_size = 512, n_neighbors = n_neighbors, select_label = select_label_id)
    label_pred_test_knn = label_dict["label_bincode"].index.values[label_id_test]

    print("svm classifier")
    label_id_test, train_data_select = svm_classification(embed_train = embed_train_input, label_train = label_id_train, embed_test = embed_test_input, batch_size = 512, select_label = select_label_id)
    label_pred_test_svm = label_dict["label_bincode"].index.values[label_id_test]

    # drop the no-mapping labels
    print(classification_report(label_gt_mapped[label_id_gt != -1], label_pred_test_knn[label_id_gt != -1], output_dict = False, zero_division=0))
    score_dict_knn = classification_report(label_gt_mapped[label_id_gt != -1], label_pred_test_knn[label_id_gt != -1], output_dict = True, zero_division=0)
    score_knn = pd.DataFrame.from_dict(score_dict_knn)

    print(classification_report(label_gt_mapped[label_id_gt != -1], label_pred_test_svm[label_id_gt != -1], output_dict = False, zero_division=0))
    score_dict_svm = classification_report(label_gt_mapped[label_id_gt != -1], label_pred_test_svm[label_id_gt != -1], output_dict = True, zero_division=0)
    score_svm = pd.DataFrame.from_dict(score_dict_svm)

    score = pd.DataFrame(columns = [["classifier", "accuracy", "F1-score (unweighted)", "F1-score (weighted)", "use_pca"]])
    score["classifier"] = [f"knn_{n_neighbors}", "svm"]
    score["accuracy"] = [score_knn.loc["f1-score", "accuracy"], score_svm.loc["f1-score", "accuracy"]]
    score["F1-score (unweighted)"] = [score_knn.loc["f1-score", "macro avg"], score_svm.loc["f1-score", "macro avg"]]
    score["F1-score (weighted)"] = [score_knn.loc["f1-score", "weighted avg"], score_svm.loc["f1-score", "weighted avg"]]
    score["use_pca"] = [reduce_dim, reduce_dim]
    score_list.append(score)

    score_final = pd.concat(score_list, axis = 0, ignore_index = True)
    score_final.to_csv(res_dir + f"class_{data_case}_pca{n_pcs}.csv")


# In[]
# Check the embedding of the model
# adata_train_embed = anndata.AnnData(X = train_data_select["embed_train"],
#                                     obs = pd.DataFrame(index = [f"train_{x}" for x in range(train_data_select["embed_train"].shape[0])],
#                                                        data = np.concatenate([train_data_select["label_train"][:,None], label_dict["label_bincode"].index.values[train_data_select["label_train"], None]], axis = 1),
#                                                        columns = ["label_id", "label"])
#                                     )
# adata_train_embed.obs["label_predict"] = adata_train_embed.obs["label"]

# # latent by default
# adata_test_embed = anndata.AnnData(X = embed_test if not reduce_dim else embed_test_pca,
#                                    obs = pd.DataFrame(index = [f"test_{x}" for x in range(label_id_test.shape[0])],
#                                                       data = np.concatenate([label_id_test[:,None], label_gt_mapped[:, None], label_pred_test[:, None]], axis = 1),
#                                                       columns = ["label_id", "label", "label_predict"])
#                                    )

# adata_embed = anndata.concat([adata_train_embed, adata_test_embed], axis = 0, join = "inner", label = "source", keys = ["train", "test"])

# sc.pp.neighbors(adata_embed, n_neighbors = 15)
# sc.tl.umap(adata_embed, min_dist = 0.3)

# adata_test_embed.obsm["X_umap"] = adata_embed[adata_embed.obs["source"] == "test", :].obsm["X_umap"]
    
# colormap =plt.cm.get_cmap("tab20")
# annos = adata_embed.obs[["label", "source", "label_predict"]].astype("category")
# fig = utils.plot_embeds(embed = adata_embed.obsm[f"X_umap"], annos = annos, markerscale = 15, figsize = (20, 7), s = 1, alpha = 0.4, colormap = colormap, label_inplace = False)
# fig.tight_layout()
# fig.savefig(res_dir + f"embed_train_test_{data_case}.png", bbox_inches = "tight")

# fig = utils.plot_by_batch(adata_embed.obsm[f"X_umap"], annos = np.array([x for x in adata_embed.obs["label"].values]),
#                         batches = np.array([x for x in adata_embed.obs["source"].values]), markerscale = 15, figsize = (15, 5) if data_case == "lung_atlas" else (12, 5), s = 1, alpha = 0.4, colormap = colormap, label_inplace = False)
# fig.tight_layout()
# fig.savefig(res_dir + f"embed_train_test_{data_case}_sep.png", bbox_inches = "tight")

# fig = utils.plot_embeds(embed = adata_test_embed.obsm[f"X_umap"], annos = adata_test_embed.obs[["label", "label_predict"]].astype("category"), markerscale = 15, figsize = (20, 7), s = 1, alpha = 0.4, colormap = colormap, label_inplace = False)
# fig.tight_layout()
# fig.savefig(res_dir + f"embed_predict_{data_case}.png", bbox_inches = "tight")



# %%
