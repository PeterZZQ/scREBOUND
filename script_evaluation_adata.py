# In[]
import torch
from sklearn.neighbors import KNeighborsClassifier
import scipy.sparse as sp
import src.data_utils as data_utils
import anndata
import scanpy as sc
import numpy as np

from pathlib import Path
import sys, os

import numpy as np
import tqdm
import anndata
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

# packages for distributed training
import torch
import torch.nn as nn
from torch.utils import data

sys.path.append("./src")

import data_utils
from transformer_model import TransformerModel, get_default_config
import trainer
import utils
import eval

import warnings
warnings.filterwarnings("ignore")


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

# Test cell type prediction, considering ct and its parents for stability
def label_prediction(cell_embed, select_labels = None, use_hierclass = False):
    """\
    Description:
    ------------
        Predict the cell type label using the cell embedding.

    Parameters:
    ------------
        cell_embed: the cell embedding tensor (ncells, ndims)
        select_labels: the labels of interest (Cell Ontology form) in the cell population, narrow the searching scope
        use_hierclass: set True to consider the parents prediction for stability
    """
    with torch.no_grad():
        output = nn.functional.sigmoid(fm_model.classifier(torch.FloatTensor(cell_embed).to(fm_model.device)))
        if use_hierclass:
            class_scores = []
            for label_bincode in meta_dict["label_bincode"]:
                class_scores.append(output[:, label_bincode == 1].mean(dim = 1))
            class_scores = torch.vstack(class_scores).cpu().numpy().T
        else:
            class_scores = output.cpu().numpy()
        
    if select_labels is not None:
        idx = np.array([False if (label in select_labels) else True for label in meta_dict["label_code"]], dtype = bool)
        class_scores[:, idx] = -100 
    class_pred = meta_dict["label_code"][np.argmax(class_scores, axis = 1)]
    return class_pred
# In[]
# NOTE: Given adata,
# 1. transform into meta-gene counts
# 3. transform meta-gene counts into cell sentence
def preprocess_anndata_knn(adata, meta_embed, gene_embed_key = "esm2"):
    """
    Group the genes into meta-genes in adata according to meta-gene embedding, 
    NOTE: genes are grouped into meta-genes according to the knn classifier, 
    however the result is very different from the hard-code assignment, could cause deterioration of the result 
    """
    # NOTE: kneighbors classifier is causing all the issue. Should use the exact hard-coded assignment as described
    knn = KNeighborsClassifier(n_neighbors = 5).fit(X = meta_embed.numpy(), y = np.arange(meta_embed.shape[0]))
    meta_labels = knn.predict(adata.varm[gene_embed_key])
    print(np.unique(meta_labels, return_counts = True))
    adata.var["meta_labels"] = meta_labels

    counts_meta = []
    for label in range(meta_embed.shape[0]):
        counts_meta.append(adata.layers["counts"][:, meta_labels == label].sum(axis = 1))
    counts_meta = sp.csr_matrix(np.hstack(counts_meta))

    adata_meta = anndata.AnnData(X = counts_meta, obs = adata.obs)
    sc.pp.normalize_total(adata_meta, target_sum = 10e4, key_added = "libsize")
    sc.pp.log1p(adata_meta)
    
    return adata_meta


# In[]
# read the gene protein embedding and gene & meta-gene assignment
n_mgene = 256
gene_embed_dict = torch.load(f"/project/zzhang834/llm_dataset/CellXGeneCZI/data_download/gene_embed_meta{n_mgene}.pt", weights_only = False)
feature_info = gene_embed_dict["labels"]

# only for the purpose of feature name and feature id translation
# adata_test = anndata.read_h5ad("/project/zzhang834/llm_dataset/CellXGeneCZI/data_download/blood/partition_10.h5ad")
# adata_test.layers["counts"] = adata_test.X.copy()

adata_test1 = anndata.read_h5ad("dataset/scIB/Immune_ALL_human.h5ad")
adata_test_meta1 = data_utils.preprocess_anndata(adata_test1, feature_info, var_name = "gene_name")
# adata_test_meta1.write_h5ad("dataset/scIB/Immune_ALL_human_meta.h5ad")

adata_test2 = anndata.read_h5ad("dataset/scIB/human_pancreas_norm_complexBatch.h5ad")
adata_test_meta2 = data_utils.preprocess_anndata(adata_test2, feature_info, var_name = "gene_name")
# adata_test_meta2.write_h5ad("dataset/scIB/human_pancreas_norm_complexBatch_meta.h5ad")

adata_test3 = anndata.read_h5ad("dataset/scIB/Lung_atlas_public.h5ad")
adata_test_meta3 = data_utils.preprocess_anndata(adata_test3, feature_info, var_name = "gene_name")
# adata_test_meta3.write_h5ad("dataset/scIB/Lung_atlas_public_meta.h5ad")

# adata_test_meta1 = anndata.read_h5ad("dataset/scIB/Immune_ALL_human_meta.h5ad")
# adata_test_meta2 = anndata.read_h5ad("dataset/scIB/human_pancreas_norm_complexBatch_meta.h5ad")
# adata_test_meta3 = anndata.read_h5ad("dataset/scIB/Lung_atlas_public_meta.h5ad")


# take out the pancreas dataset
adata_pancreas = anndata.read_h5ad("/project/zzhang834/llm_dataset/CellXGeneCZI/data_download/pancreas/adata_meta256.h5ad")
sc.pp.normalize_total(adata_pancreas, target_sum = 10e4, key_added = "libsize")
sc.pp.log1p(adata_pancreas)
adata_test_meta4 = adata_pancreas


# In[]
# NOTE: adjust the label of training pancreas dataset
adata_test_meta4.obs["cell_type"] = adata_test_meta4.obs["cell_type"].astype(object)
adata_test_meta4.obs.loc[adata_test_meta4.obs["cell_type"] == "acinar", "cell_type"] = "acinar cell"
adata_test_meta4.obs.loc[adata_test_meta4.obs["cell_type"] == "pancreatic A cell", "cell_type"] = "alpha"
adata_test_meta4.obs.loc[adata_test_meta4.obs["cell_type"] == "pancreatic D cell", "cell_type"] = "delta"
adata_test_meta4.obs.loc[adata_test_meta4.obs["cell_type"] == "type B pancreatic cell", "cell_type"] = "beta"
adata_test_meta4.obs.loc[adata_test_meta4.obs["cell_type"] == "pancreatic PP cell", "cell_type"] = "PP cell"
adata_test_meta4.obs.loc[adata_test_meta4.obs["cell_type"] == "pancreatic acinar cell", "cell_type"] = "acinar cell"
adata_test_meta4.obs.loc[adata_test_meta4.obs["cell_type"] == "pancreatic ductal cell", "cell_type"] = "ductal"
adata_test_meta4.obs.loc[adata_test_meta4.obs["cell_type"] == "pancreatic endocrine cell", "cell_type"] = "endocrine cell"
adata_test_meta4.obs.loc[adata_test_meta4.obs["cell_type"] == "pancreatic epsilon cell", "cell_type"] = "epsilon"
adata_test_meta4.obs.loc[adata_test_meta4.obs["cell_type"] == "t_cell", "cell_type"] = "T cell"
adata_test_meta4.obs.loc[adata_test_meta4.obs["cell_type"] == "mast", "cell_type"] = "mast cell"
adata_test_meta4.obs.loc[adata_test_meta4.obs["cell_type"] == "endothelial", "cell_type"] = "endothelial cell"

# additional merge of train
adata_test_meta4.obs.loc[adata_test_meta4.obs["cell_type"] == "endothelial cell of lymphatic vessel", "cell_type"] = "endothelial cell"
adata_test_meta4.obs.loc[adata_test_meta4.obs["cell_type"] == "endothelial cell of vascular tree", "cell_type"] = "endothelial cell"
adata_test_meta4.obs.loc[adata_test_meta4.obs["cell_type"] == "epithelial cell of exocrine pancreas", "cell_type"] = "epithelial cell"
adata_test_meta4.obs.loc[adata_test_meta4.obs["cell_type"] == "epithelial cell of proximal tubule", "cell_type"] = "epithelial cell"


# In[]
# # NOTE: Mapping is changed, which makes the result worse
# # gene embed full
# gene_embed_full = torch.load("/project/zzhang834/llm_dataset/proteome/embeddings/Homo_sapiens.GRCh38.gene_symbol_to_embedding_ESM2.pt", weights_only = False)
# # preprocessing, use gene embed full
# gene_overlap = np.intersect1d(adata_test.var["feature_name"].values, np.array([x for x in gene_embed_full.keys()]))
# print("number of overlapped genes: {:d}".format(len(gene_overlap)))
# adata_test = adata_test[:, adata_test.var["feature_name"].isin(gene_overlap)].copy()

# sc.pp.filter_genes(adata_test, min_cells = int(0.01 * adata_test.shape[0]))
# # sc.pp.normalize_total(adata_test)
# # sc.pp.log1p(adata_test)

# # sc.pp.highly_variable_genes(adata_test, n_top_genes = 4000)
# # adata_test = adata_test[:, adata_test.var["highly_variable"]]
# adata_test.varm["esm2"] = torch.vstack([gene_embed_full[x] for x in adata_test.var["feature_name"].values]).numpy()

# adata_test_meta = preprocess_anndata_knn(adata = adata_test, meta_embed = gene_embed_dict["meta_embed"], gene_embed_key = "esm2")


# In[]
# function
data_utils.set_seed(3)

# Define the device
device = torch.device("cuda:1")
print(f"GPU - Using device: {device}")
# Load the dataset
print(f"GPU - Loading dataset...")

# NOTE: save in localscratch for faster memory access
data_dir = Path(f"/project/zzhang834/LLM_KD/dataset/cellxgene")
# data_dir = Path(f"/localscratch/ziqi/localscratch_tempdata/cellxgene")
# load the token embedding
token_embed = torch.load(data_dir / f"token_embed_{n_mgene}.pt", weights_only = False)
# load the cell meta-info
meta_dict = torch.load(data_dir / f"meta_{n_mgene}_bincode.pt", weights_only = False)

# ------------------------------Update for the model selected---------------------------------------------------
# NOTE: 1. vanilla model with only mlm loss
# vanilla mlm model
model_name = "checkpoint_8_512_1"
# pretrained classification model
# model_name = "checkpoint_6_512_classi100_1"
# model_name = "checkpoint_6_512_classiunweight100_1"
model_dir = f"/project/zzhang834/LLM_KD/checkpoint_fastatten/{model_name}.pth"
res_dir = f"results_fastatten/{model_name}/"

# # fine-tuned model
# model_name = "checkpoint_6_512_classiunweight100_disc10_1"
# model_dir = f"/project/zzhang834/LLM_KD/checkpoint_disc/{model_name}.pth"
# res_dir = f"results/finetune_disc/{model_name}/"

# model_name = "checkpoint_6_512_contr0_1"
# model_name = "checkpoint_6_512_contr10_1"
# model_name = "checkpoint_6_512_contrcb10_1"
# model_dir = f"/project/zzhang834/LLM_KD/checkpoint_contr/{model_name}.pth"

# res_dir = f"results/finetune_contr/{model_name}/"

state = torch.load(model_dir, weights_only = False)
model_config = state["model_config"]
# model_config.__dict__.update({"checkpoint_path": None, "checkpoint_prefix": None, "pretrain_path":  model_dir, "use_discriminator": False, "lamb_disc": 0.0})

model_config.__dict__.update({"checkpoint_path": None, "checkpoint_prefix": None, "pretrain_path":  model_dir})
# ------------------------------------------------------------------------------------------------------------------


labels = None
batches = None
test_dataset1 = data_utils.sc_dataset_anndata(adata_meta = adata_test_meta1, labels = labels, batches = batches, njobs = 16)
test_dataset2 = data_utils.sc_dataset_anndata(adata_meta = adata_test_meta2, labels = labels, batches = batches, njobs = 16)
test_dataset3 = data_utils.sc_dataset_anndata(adata_meta = adata_test_meta3, labels = labels, batches = batches, njobs = 16)
test_dataset4 = data_utils.sc_dataset_anndata(adata_meta = adata_test_meta4, labels = labels, batches = batches, njobs = 16)

test_loader1 = data.DataLoader(test_dataset1, batch_size = 1, shuffle = False, pin_memory = True, num_workers = 8, prefetch_factor = 8)
test_loader2 = data.DataLoader(test_dataset2, batch_size = 1, shuffle = False, pin_memory = True, num_workers = 8, prefetch_factor = 8)
test_loader3 = data.DataLoader(test_dataset3, batch_size = 1, shuffle = False, pin_memory = True, num_workers = 8, prefetch_factor = 8)
test_loader4 = data.DataLoader(test_dataset4, batch_size = 1, shuffle = False, pin_memory = True, num_workers = 8, prefetch_factor = 8)

print(f"GPU - Done.")

fm_model = TransformerModel(model_config = model_config, token_embed = token_embed, n_batch = len(meta_dict["batch_code"]), n_label = len(meta_dict["label_code"]), device = device)

print(f"GPU - Preloading lastest model'")
# Get the common keys between the current model and the saved model
filtered_state_dict = {k: v for k, v in state["model_state_dict"].items() if k in fm_model.state_dict()}
# Load the filtered state dictionary into the model
fm_model.load_state_dict(filtered_state_dict, strict=False)

print(f"GPU - Done.")


# In[]
# ------------------------------------------------------------------------------------------------------------------------
#
# Visualize the embedding and measure the batch effect removal
#
# ------------------------------------------------------------------------------------------------------------------------
# NOTE: calculate the embedding
# TODO: issue, for the classifier, should the masked input be used??
adata_embed1 = trainer.cell_embed(model = fm_model, dataloader = test_loader1, multi_gpus = False)
adata_embed1.obs = adata_test_meta1.obs.copy()
adata_embed1.obsm["latent"] = adata_embed1.X.copy()
# cluster structure is clearer with UMAP
# adata_embed1.obsm["X_umap"] = UMAP(n_components = 2).fit_transform(adata_embed1.X.toarray())
sc.pp.neighbors(adata_embed1, n_neighbors = 15, use_rep = "latent")
sc.tl.umap(adata_embed1, min_dist = 0.3)

adata_embed2 = trainer.cell_embed(model = fm_model, dataloader = test_loader2, multi_gpus = False)
adata_embed2.obs = adata_test_meta2.obs.copy()
adata_embed2.obsm["latent"] = adata_embed2.X.copy()
# adata_embed2.obsm["X_umap"] = UMAP(n_components = 2).fit_transform(adata_embed2.X.toarray())
sc.pp.neighbors(adata_embed2, n_neighbors = 15, use_rep = "latent")
sc.tl.umap(adata_embed2, min_dist = 0.3)

adata_embed3 = trainer.cell_embed(model = fm_model, dataloader = test_loader3, multi_gpus = False)
adata_embed3.obs = adata_test_meta3.obs.copy()
adata_embed3.obsm["latent"] = adata_embed3.X.copy()
# adata_embed3.obsm["X_umap"] = UMAP(n_components = 2).fit_transform(adata_embed3.X.toarray())
sc.pp.neighbors(adata_embed3, n_neighbors = 15, use_rep = "latent")
sc.tl.umap(adata_embed3, min_dist = 0.3)

adata_embed4 = trainer.cell_embed(model = fm_model, dataloader = test_loader4, multi_gpus = False)
adata_embed4.obs = adata_test_meta4.obs.copy()
adata_embed4.obsm["latent"] = adata_embed4.X.copy()
# adata_embed4.obsm["X_umap"] = UMAP(n_components = 2).fit_transform(adata_embed4.X.toarray())
sc.pp.neighbors(adata_embed4, n_neighbors = 15, use_rep = "latent")
sc.tl.umap(adata_embed4, min_dist = 0.3)


# In[]
# Visualize the latent embedding
if not os.path.exists(res_dir):
    os.makedirs(res_dir)

colormap =plt.cm.get_cmap("tab20")
# immune cell
fig = utils.plot_embeds(embed = adata_embed1.obsm["X_umap"], annos = adata_embed1.obs[["final_annotation", "batch"]].astype("category"), markerscale = 15, figsize = (20, 17), s = 1, alpha = 0.4, colormap = colormap, label_inplace = False)
fig.tight_layout()
fig.savefig(res_dir + "embed_scIB_immune_all.png", bbox_inches = "tight")
# pancreas
fig = utils.plot_embeds(embed = adata_embed2.obsm["X_umap"], annos = adata_embed2.obs[["celltype", "tech"]].astype("category"), markerscale = 5, figsize = (20, 17), s = 3, alpha = 0.4, colormap = colormap, label_inplace = False)
fig.tight_layout()
fig.savefig(res_dir + "embed_scIB_pancreas.png", bbox_inches = "tight")
# lung atlas
fig = utils.plot_embeds(embed = adata_embed3.obsm["X_umap"], annos = adata_embed3.obs[["cell_type", "batch"]].astype("category"), markerscale = 5, figsize = (20, 17), s = 3, alpha = 0.7, colormap = colormap, label_inplace = False)
fig.tight_layout()
fig.savefig(res_dir + "embed_scIB_lung.png", bbox_inches = "tight")

fig = utils.plot_embeds(embed = adata_embed4.obsm["X_umap"], annos = adata_embed4.obs[["cell_type", "dataset_id"]].astype("category"), markerscale = 5, figsize = (40, 17), s = 3, alpha = 0.7, label_inplace = False)
fig.tight_layout()
fig.savefig(res_dir + "embed_training_pancreas.png", bbox_inches = "tight")

# In[]
scores1 = eval.eval_batch_correction(adata = adata_embed1, embed_key = "latent", label_key = "final_annotation", batch_key = "batch")
scores1["dataset"] = "Immune_ALL"
scores2 = eval.eval_batch_correction(adata = adata_embed2, embed_key = "latent", label_key = "celltype", batch_key = "tech")
scores2["dataset"] = "Pancreas"
scores3 = eval.eval_batch_correction(adata = adata_embed3, embed_key = "latent", label_key = "cell_type", batch_key = "dataset")
scores3["dataset"] = "Lung"
scores4 = eval.eval_batch_correction(adata = adata_embed4, embed_key = "latent", label_key = "cell_type", batch_key = "dataset_id")
scores4["dataset"] = "Pancreas-training"

scores = pd.concat([scores1, scores2, scores3, scores4], axis = 0, ignore_index = True)
scores.to_csv(res_dir + "scores_scib.csv")

# if fm_model.model_config.sup_type is None:
#     # no further test needed
assert False

# In[]
# ------------------------------------------------------------------------------------------------------------------------
#
# Measure the prediction accuracy of the classifier
#
# ------------------------------------------------------------------------------------------------------------------------
# HSPCs, Megakaryocyte progenitors in gt annotation is missing, 
#'CL:0000037--hematopoietic stem cell', all cells are classified as hsc
ct2onto1 = {'CD10+ B cells': 'CL:0000785--mature B cell',
            'CD20+ B cells': 'CL:0000785--mature B cell',
            # 'CD14+ Monocytes': 'CL:0001054--CD14-positive monocyte', 
            'CD16+ Monocytes': 'CL:0000576--monocyte',
            'CD14+ Monocytes': 'CL:0000576--monocyte',
            'Monocyte progenitors': 'CL:0000576--monocyte',
            'CD4+ T cells': 'CL:0000492--CD4-positive helper T cell',
            'CD8+ T cells': 'CL:0000625--CD8-positive, alpha-beta T cell',
            'Erythrocytes': 'CL:0000232--erythrocyte',
            'Erythrocytes progenitors': 'CL:0000232--erythrocyte', 
            'Monocyte-derived dendritic cells': 'CL:0011031--monocyte-derived dendritic cell',
            'NK cells': 'CL:0000623--natural killer cell',
            'NKT cells': 'CL:0000814--mature NK T cell',
            'Plasma cells': 'CL:0000786--plasma cell',
            'Plasmacytoid dendritic cells': 'CL:0001058--plasmacytoid dendritic cell, human',
            'HSPCs': 'Other',
            'Megakaryocyte progenitors': 'Other'} 

# ct2onto1 = pd.DataFrame(ct2onto1, index = ["cell ontology"]).T
# ct2onto1["ground truth label"] = ct2onto1.index.values

select_label1 = np.array([x for x in ct2onto1.values() if x != 'Other'])
select_label1 = np.unique(select_label1)

# replace the ground truth label with cell ontology term for accuracy measurement
adata_embed1.obs["cell ontology (gt)"] = adata_embed1.obs["final_annotation"].astype(object)
for ct, ct_ontology in ct2onto1.items():
    adata_embed1.obs.loc[adata_embed1.obs["cell ontology (gt)"] == ct, "cell ontology (gt)"] = ct_ontology
adata_embed1.obs["cell ontology (gt)"] = adata_embed1.obs["cell ontology (gt)"].astype("category")

ct2onto2 = {'acinar': 'CL:0002064--pancreatic acinar cell',
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

# ct2onto2 = pd.DataFrame(ct2onto2, index = ["cell ontology"]).T
# ct2onto2["ground truth label"] = ct2onto2.index.values

select_label2 = np.array([x for x in ct2onto2.values()])
select_label2 = np.unique(select_label2)

# replace the ground truth label with cell ontology term for accuracy measurement
adata_embed2.obs["cell ontology (gt)"] = adata_embed2.obs["celltype"].astype(object)
for ct, ct_ontology in ct2onto2.items():
    adata_embed2.obs.loc[adata_embed2.obs["cell ontology (gt)"] == ct, "cell ontology (gt)"] = ct_ontology
adata_embed2.obs["cell ontology (gt)"] = adata_embed2.obs["cell ontology (gt)"].astype("category")


ct2onto3 = {'B cell': 'CL:0000785--mature B cell',
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

# ct2onto3 = pd.DataFrame(ct2onto3, index = ["cell ontology"]).T
# ct2onto3["ground truth label"] = ct2onto3.index.values

select_label3 = np.array([x for x in ct2onto3.values() if (x != 'Other') and (x != 'T/NK cell')] + ['CL:0000084--T cell', 'CL:0000623--natural killer cell'])
select_label3 = np.unique(select_label3)

adata_embed3.obs["cell ontology (gt)"] = adata_embed3.obs["cell_type"].astype(object)
for ct, ct_ontology in ct2onto3.items():
    adata_embed3.obs.loc[adata_embed3.obs["cell ontology (gt)"] == ct, "cell ontology (gt)"] = ct_ontology
adata_embed3.obs["cell ontology (gt)"] = adata_embed3.obs["cell ontology (gt)"].astype("category")

# transform the label for dataset 4
adata_embed4.obs["cell ontology (gt)"] = [x + "--" + y for x,y in zip(adata_embed4.obs["cell_type_ontology_term_id"], adata_embed4.obs["cell_type"])]
adata_embed4.obs["cell ontology (gt)"] = adata_embed4.obs["cell ontology (gt)"].astype("category")
select_label4 = np.unique(adata_embed4.obs["cell ontology (gt)"].values)

adata_embed1.obs["label_predict (use_hier)"] = label_prediction(adata_embed1.X.toarray(), select_label1, use_hierclass = True)
adata_embed1.obs["label_predict"] = label_prediction(adata_embed1.X.toarray(), select_label1, use_hierclass = False)
# transform the ground truth labels
adata_embed2.obs["label_predict (use_hier)"] = label_prediction(adata_embed2.X.toarray(), select_label2, use_hierclass = True)
adata_embed2.obs["label_predict"] = label_prediction(adata_embed2.X.toarray(), select_label2, use_hierclass = False)
adata_embed3.obs["label_predict (use_hier)"] = label_prediction(adata_embed3.X.toarray(), select_label3, use_hierclass = True)
adata_embed3.obs["label_predict"] = label_prediction(adata_embed3.X.toarray(), select_label3, use_hierclass = False)

adata_embed4.obs["label_predict (use_hier)"] = label_prediction(adata_embed4.X.toarray(), select_label4, use_hierclass = True)
adata_embed4.obs["label_predict"] = label_prediction(adata_embed4.X.toarray(), select_label4, use_hierclass = False)

# NOTE: need to process embed3 as both T cells and NK cells are TNK cells for accuracy calculation
adata_embed3.obs.loc[adata_embed3.obs["label_predict"] == 'CL:0000084--T cell', "label_predict"] = 'T/NK cell'
adata_embed3.obs.loc[adata_embed3.obs["label_predict"] == 'CL:0000623--natural killer cell', "label_predict"] = 'T/NK cell'
adata_embed3.obs.loc[adata_embed3.obs["label_predict (use_hier)"] == 'CL:0000084--T cell', "label_predict (use_hier)"] = 'T/NK cell'
adata_embed3.obs.loc[adata_embed3.obs["label_predict (use_hier)"] == 'CL:0000623--natural killer cell', "label_predict (use_hier)"] = 'T/NK cell'

adata_embed1.obs["label_predict (use_hier)"] = pd.Categorical(adata_embed1.obs["label_predict (use_hier)"], categories=adata_embed1.obs["cell ontology (gt)"].cat.categories)
adata_embed1.obs["label_predict"] = pd.Categorical(adata_embed1.obs["label_predict"], categories=adata_embed1.obs["cell ontology (gt)"].cat.categories)
adata_embed2.obs["label_predict (use_hier)"] = pd.Categorical(adata_embed2.obs["label_predict (use_hier)"], categories=adata_embed2.obs["cell ontology (gt)"].cat.categories)
adata_embed2.obs["label_predict"] = pd.Categorical(adata_embed2.obs["label_predict"], categories=adata_embed2.obs["cell ontology (gt)"].cat.categories)
adata_embed3.obs["label_predict (use_hier)"] = pd.Categorical(adata_embed3.obs["label_predict (use_hier)"], categories=adata_embed3.obs["cell ontology (gt)"].cat.categories)
adata_embed3.obs["label_predict"] = pd.Categorical(adata_embed3.obs["label_predict"], categories=adata_embed3.obs["cell ontology (gt)"].cat.categories)

adata_embed4.obs["label_predict (use_hier)"] = pd.Categorical(adata_embed4.obs["label_predict (use_hier)"], categories=adata_embed4.obs["cell ontology (gt)"].cat.categories)
adata_embed4.obs["label_predict"] = pd.Categorical(adata_embed4.obs["label_predict"], categories=adata_embed4.obs["cell ontology (gt)"].cat.categories)


# In[]
# Measure the prediction accuracy, drop the Other cells
from sklearn.metrics import f1_score
f1_score1 = f1_score(y_true = np.array([x for x in adata_embed1.obs.loc[adata_embed1.obs["cell ontology (gt)"] != "Other", "cell ontology (gt)"]]),
                   y_pred = np.array([x for x in adata_embed1.obs.loc[adata_embed1.obs["cell ontology (gt)"] != "Other", "label_predict (use_hier)"]]),
                   average = "micro")
f1_score2 = f1_score(y_true = np.array([x for x in adata_embed2.obs.loc[adata_embed2.obs["cell ontology (gt)"] != "Other", "cell ontology (gt)"]]),
                   y_pred = np.array([x for x in adata_embed2.obs.loc[adata_embed2.obs["cell ontology (gt)"] != "Other", "label_predict (use_hier)"]]),
                   average = "micro")
f1_score3 = f1_score(y_true = np.array([x for x in adata_embed3.obs.loc[adata_embed3.obs["cell ontology (gt)"] != "Other", "cell ontology (gt)"]]),
                   y_pred = np.array([x for x in adata_embed3.obs.loc[adata_embed3.obs["cell ontology (gt)"] != "Other", "label_predict (use_hier)"]]),
                   average = "micro")
f1_score4 = f1_score(y_true = np.array([x for x in adata_embed4.obs.loc[adata_embed4.obs["cell ontology (gt)"] != "Other", "cell ontology (gt)"]]),
                   y_pred = np.array([x for x in adata_embed4.obs.loc[adata_embed4.obs["cell ontology (gt)"] != "Other", "label_predict (use_hier)"]]),
                   average = "micro")

f1_score5 = f1_score(y_true = np.array([x for x in adata_embed1.obs.loc[adata_embed1.obs["cell ontology (gt)"] != "Other", "cell ontology (gt)"]]),
                   y_pred = np.array([x for x in adata_embed1.obs.loc[adata_embed1.obs["cell ontology (gt)"] != "Other", "label_predict"]]),
                   average = "micro")
f1_score6 = f1_score(y_true = np.array([x for x in adata_embed2.obs.loc[adata_embed2.obs["cell ontology (gt)"] != "Other", "cell ontology (gt)"]]),
                   y_pred = np.array([x for x in adata_embed2.obs.loc[adata_embed2.obs["cell ontology (gt)"] != "Other", "label_predict"]]),
                   average = "micro")
f1_score7 = f1_score(y_true = np.array([x for x in adata_embed3.obs.loc[adata_embed3.obs["cell ontology (gt)"] != "Other", "cell ontology (gt)"]]),
                   y_pred = np.array([x for x in adata_embed3.obs.loc[adata_embed3.obs["cell ontology (gt)"] != "Other", "label_predict"]]),
                   average = "micro")
f1_score8 = f1_score(y_true = np.array([x for x in adata_embed4.obs.loc[adata_embed4.obs["cell ontology (gt)"] != "Other", "cell ontology (gt)"]]),
                   y_pred = np.array([x for x in adata_embed4.obs.loc[adata_embed4.obs["cell ontology (gt)"] != "Other", "label_predict"]]),
                   average = "micro")

scores = pd.DataFrame(columns = ["F1 score (micro)", "dataset", "method"])
scores["F1 score (micro)"] = [f1_score1, f1_score2, f1_score3, f1_score4, f1_score5, f1_score6, f1_score7, f1_score8]
scores["dataset"] = ["Immune All", "Pancreas", "Lung Atlas", "Pancreas-training", "Immune All", "Pancreas", "Lung Atlas", "Pancreas-training"]
scores["method"] = ["class (use hier)", "class (use hier)", "class (use hier)", "class (use hier)", "class", "class", "class", "class"]
scores.to_csv(res_dir + "scores_prediction.csv")

fig = utils.plot_embeds(embed = adata_embed1.obsm["X_umap"], annos = adata_embed1.obs[["final_annotation", "cell ontology (gt)", "label_predict (use_hier)", "label_predict"]].astype("category"), markerscale = 5, figsize = (25, 17), s = 3, alpha = 0.4, colormap = colormap, label_inplace = False, ncols = 2)
fig.tight_layout()
fig.savefig(res_dir + "prediction_scIB_immune_all.png", bbox_inches = "tight")

fig = utils.plot_embeds(embed = adata_embed2.obsm["X_umap"], annos = adata_embed2.obs[["celltype", "cell ontology (gt)", "label_predict (use_hier)", "label_predict"]].astype("category"), markerscale = 5, figsize = (30, 17), s = 3, alpha = 0.4, colormap = colormap, label_inplace = False, ncols = 2)
fig.tight_layout()
fig.savefig(res_dir + "prediction_scIB_pancreas.png", bbox_inches = "tight")

fig = utils.plot_embeds(embed = adata_embed3.obsm["X_umap"], annos = adata_embed3.obs[["cell_type", "cell ontology (gt)", "label_predict (use_hier)", "label_predict"]].astype("category"), markerscale = 5, figsize = (30, 17), s = 3, alpha = 0.4, colormap = colormap, label_inplace = False, ncols = 2)
fig.tight_layout()
fig.savefig(res_dir + "prediction_scIB_lung.png", bbox_inches = "tight")

fig = utils.plot_embeds(embed = adata_embed4.obsm["X_umap"], annos = adata_embed4.obs[["cell_type", "cell ontology (gt)", "label_predict (use_hier)", "label_predict"]].astype("category"), markerscale = 5, figsize = (30, 17), s = 3, alpha = 0.4, colormap = colormap, label_inplace = False, ncols = 2)
fig.tight_layout()
fig.savefig(res_dir + "prediction_scIB_training_pancreas.png", bbox_inches = "tight")


# In[]
assert False
# NOTE: Sanity check, check the classifier weight, only for weighted classifier
import torch.nn as nn
classifier_weight = nn.functional.softmax(fm_model.classifier_weight.data, dim = 0)
classifier_weight = classifier_weight.detach().cpu().numpy()

fig = plt.figure(figsize = (10, 7))
ax = fig.add_subplot()
sns.histplot(classifier_weight)

# average weight
baseline_weight = 1/fm_model.n_label
print(np.sum(classifier_weight > baseline_weight))
print(np.sum(classifier_weight <= baseline_weight))


print(meta_dict["label_code"][np.argsort(classifier_weight)[::-1]])


# In[]
# NOTE: check pancreas dataset distribution across batchs/techs
tech_group1 = ["fluidigmc1", "smarter", "smartseq2"]
adata_techgroup1 = adata_test_meta2[adata_test_meta2.obs["tech"].isin(tech_group1),:]
adata_techgroup2 = adata_test_meta2[~adata_test_meta2.obs["tech"].isin(tech_group1),:]

# only happens in certain cell types including: 
ct_interest = ["alpha", "beta", "delta", "gamma"]
adata_techgroup1 = adata_techgroup1[adata_techgroup1.obs["celltype"].isin(ct_interest), :]
adata_techgroup2 = adata_techgroup2[adata_techgroup2.obs["celltype"].isin(ct_interest), :]

libsize_techgroup1 = adata_techgroup1.X.toarray().sum(axis = 1)
libsize_techgroup2 = adata_techgroup2.X.toarray().sum(axis = 1)

fig = plt.figure(figsize = (25, 7))
ax = fig.subplots(nrows = 1, ncols = 2)
sns.kdeplot(libsize_techgroup1, ax = ax[0], color = "blue")
sns.kdeplot(libsize_techgroup2, ax = ax[0], color = "red")

# wilcoxon test to check which gene has more distinct distribution
adata_merge = anndata.concat([adata_techgroup1, adata_techgroup2], axis = 0, label = "group", keys = ["smartseq etc.", "rest"])
sc.tl.rank_genes_groups(adata_merge, groupby = "group", method = "wilcoxon")
adata_merge.uns["rank_genes_groups"]

sc.pl.rank_genes_groups_dotplot(adata_merge, groupby="group", standard_scale="var", n_genes = 10, ax = ax[1])

fig.suptitle("fluidigmc1, smarter, smartseq2 (blue) v.s. rest")
fig.savefig("results/test_pancreas_4000hvg_stats.png", bbox_inches = "tight")

# In[]
# # merge clusters
# coarse_ct = {}
# coarse_ct["monocyte"] = ["CD14-low, CD16-positive monocyte", "CD14-positive monocyte"]
# coarse_ct["nk"] = ["CD16-negative, CD56-bright natural killer cell, human", "natural killer cell"]
# coarse_ct["T"] = ["CD4-positive, alpha-beta T cell", "CD4-positive, alpha-beta cytotoxic T cell",
#                        "CD8-positive, alpha-beta T cell", "central memory CD4-positive, alpha-beta T cell",
#                        "central memory CD8-positive, alpha-beta T cell", "effector memory CD4-positive, alpha-beta T cell",
#                        "effector memory CD8-positive, alpha-beta T cell", "gamma-delta T cell", "mucosal invariant T cell",
#                        "naive thymus-derived CD4-positive, alpha-beta T cell", "naive thymus-derived CD8-positive, alpha-beta T cell",
#                        "regulatory T cell"]

# coarse_ct["CD4 T"] = ["CD4-positive, alpha-beta T cell", "CD4-positive, alpha-beta cytotoxic T cell",
#                        "central memory CD4-positive, alpha-beta T cell", "effector memory CD4-positive, alpha-beta T cell",
#                        "naive thymus-derived CD4-positive, alpha-beta T cell", "regulatory T cell"]

# coarse_ct["CD8 T"] = ["CD8-positive, alpha-beta T cell", "central memory CD8-positive, alpha-beta T cell",
#                        "effector memory CD8-positive, alpha-beta T cell", "naive thymus-derived CD8-positive, alpha-beta T cell"]

# coarse_ct["B"] = ["naive B cell", "transitional stage B cell", "memory B cell"]
# coarse_ct["dendritic"] = ["conventional dendritic cell", "dendritic cell", "plasmacytoid dendritic cell"]
# coarse_ct["thymocyte"] = ["double negative thymocyte"]
# coarse_ct["erythrocyte"] = ["erythrocyte"]

# adata_test.obs[["cell_type (coarse)"]] = adata_test.obs[["cell_type"]].values
# adata_test.obs.loc[adata_test.obs["cell_type"].isin(coarse_ct["monocyte"]), "cell_type (coarse)"] = "monocyte"
# adata_test.obs.loc[adata_test.obs["cell_type"].isin(coarse_ct["nk"]), "cell_type (coarse)"] = "nk"
# adata_test.obs.loc[adata_test.obs["cell_type"].isin(coarse_ct["T"]), "cell_type (coarse)"] = "T"
# adata_test.obs.loc[adata_test.obs["cell_type"].isin(coarse_ct["CD4 T"]), "cell_type (coarse)"] = "CD4 T"
# adata_test.obs.loc[adata_test.obs["cell_type"].isin(coarse_ct["CD8 T"]), "cell_type (coarse)"] = "CD8 T"
# adata_test.obs.loc[adata_test.obs["cell_type"].isin(coarse_ct["B"]), "cell_type (coarse)"] = "B"
# adata_test.obs.loc[adata_test.obs["cell_type"].isin(coarse_ct["dendritic"]), "cell_type (coarse)"] = "dendritic"
# adata_test.obs.loc[adata_test.obs["cell_type"].isin(coarse_ct["thymocyte"]), "cell_type (coarse)"] = "thymocyte"
# adata_test.obs.loc[adata_test.obs["cell_type"].isin(coarse_ct["erythrocyte"]), "cell_type (coarse)"] = "erythrocyte"

# fig = utils.plot_embeds(embed = adata_test2.obsm["latent_umap"], annos = adata_test.obs[["cell_type (coarse)"]].astype("category"), markerscale = 15, figsize = (25, 17), s = 1, alpha = 0.4, colormap = colormap, label_inplace = True)
# fig.tight_layout()
# # fig.savefig(res_dir + "")


# %%
