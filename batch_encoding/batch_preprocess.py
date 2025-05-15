# In[]
import pandas as pd
import numpy as np
import torch
import os
from tqdm import tqdm
import scipy.sparse as sp
from multiprocessing import Pool

import sys
sys.path.append("/net/csefiles/xzhanglab/zzhang834/LLM_KD/src/")

import batch_encode

# In[]
# ---------------------------------------------------------------------------------------------------------------------------------------
#
# Obtain batch information, construct batch features
#
# ---------------------------------------------------------------------------------------------------------------------------------------
# hm_housekeeping = pd.read_csv("/net/csefiles/xzhanglab/zzhang834/LLM_KD/dataset/MostStable.csv", sep = ";")
# n_mgene = 256
# gene_embed_dict = torch.load(f"/net/csefiles/xzhanglab/zzhang834/hs_download/meta_data/gene_embed_meta{n_mgene}_gpool.pt", weights_only = False)

# # TODO: the hk and mito genes are selected from only the protein embedding genes, there are more genes in the original data
# hk_gene_name = np.intersect1d(hm_housekeeping["Gene name"].values.squeeze(), 
#                               gene_embed_dict["labels"]["feature_name"].values.squeeze())
# mito_gene_name = [gene for gene in gene_embed_dict["labels"]["feature_name"].values.squeeze() if gene.startswith("MT-")]

# # ribosomal genes
# ribo_gene_df = pd.read_csv("/net/csefiles/xzhanglab/zzhang834/LLM_KD/batch_encoding/gene_ribosomal_go0005840.csv", index_col = 0)
# ribo_gene_name = [x for x in ribo_gene_df["external_gene_name"].values.squeeze() if not pd.isna(x)]
# ribo_gene_name = np.intersect1d(ribo_gene_name, full_gene)
# # mainly MRPL, MRPS, RPL, RPS, first two are nuclear mito-ribosomal genes, droped
# # there are 91 ribosomal genes, further filter-out the low-detection genes or calculate average
# ribo_gene_name = [x for x in ribo_gene_name if (x[:3] == "RPL")|(x[:3] == "RPS")]

# # stress-response gene
# stress_gene_df = pd.read_csv("/net/csefiles/xzhanglab/zzhang834/LLM_KD/batch_encoding/gene_stress_go0006950.csv", index_col = 0)
# stress_gene_name = [x for x in stress_gene_df["external_gene_name"].values.squeeze() if not pd.isna(x)]
# # there are 28 genes, further filter-out the low-dection genes or calculate average
# stress_gene_name = [x for x in np.intersect1d(stress_gene_name, full_gene)]

# In[]
n_mgene = 256
gene_embed_dict = torch.load(f"/net/csefiles/xzhanglab/zzhang834/hs_download/meta_data/gene_embed_meta{n_mgene}_gpool.pt", weights_only = False)
full_gene = gene_embed_dict["labels"]["feature_name"].values

# NOTE: human hk/mito already extracted and hard-coded in the batch_encode file
# selected ribo gene and stress gene are already hard-coded in the batch_encode file
# hk_gene_name = batch_encode.hk_gene_name
# mito_gene_name = batch_encode.mito_gene_name
# ribo_gene_name = batch_encode.ribo_gene_name
# stress_gene_name = batch_encode.stress_gene_name
# include all genes above, hard-coded
batch_gene_name = batch_encode.batch_gene_name

# In[]
# read in the batch_meta_information
meta_cells = []
# Old dataset
data_dir = "/net/csefiles/xzhanglab/zzhang834/hs_download/permuted/"
res_dir = "/net/csefiles/xzhanglab/zzhang834/hs_download/meta_data/"
# New dataset
# data_dir = "/data/zzhang834/hs_healthy_2025_01_30/permuted/"
# res_dir = "/data/zzhang834/hs_healthy_2025_01_30/meta_data/"

sizes = np.loadtxt(data_dir + "sizes.txt")
chunk_sizes = []
for partition_idx in tqdm(range(0, len(sizes))):
    meta_cells_chunk = pd.read_parquet(os.path.join(data_dir, f"obs_{partition_idx}.parquet"))
    chunk_sizes.append(meta_cells_chunk.shape[0])
    meta_cells.append(meta_cells_chunk)
meta_cells = pd.concat(meta_cells, axis = 0)

# Within the meta-data, except for the tech-factors
# Biological factors: developmental-stage, disease, sex
# Other relevant: tissue, tissue_general, donor_id

# factors that decide batches
# NOTE: tissue_type only has tissue, 238 tissue, 
batch_bio_factors = ["development_stage", "sex", "tissue", "tissue_general"]
batch_tech_factors = ["dataset_id", "assay", "suspension_type"]
# unique dataset: 611, but unique batch: 873
meta_cells["batch_level0"] = ["-".join(x) for x in zip(*[meta_cells[x].values for x in batch_tech_factors])]
meta_cells["batch_level1"] = ["-".join(x) for x in zip(*[meta_cells[x].values for x in ["dataset_id", "assay", "suspension_type", "development_stage", "sex", "tissue_general"]])]
meta_cells["batch_level2"] = ["-".join(x) for x in zip(*[meta_cells[x].values for x in batch_bio_factors + batch_tech_factors])]

# In[]
print("Generate the batch id values...")
batch_id1, batch_code1 = pd.factorize(meta_cells["batch_level0"].values, sort = True)
meta_cells["batch_level0_id"] = batch_id1
batch_id2, batch_code2 = pd.factorize(meta_cells["batch_level1"].values, sort = True)
meta_cells["batch_level1_id"] = batch_id2
batch_id3, batch_code3 = pd.factorize(meta_cells["batch_level2"].values, sort = True)
meta_cells["batch_level2_id"] = batch_id3
batch_code = {"batch_level0": batch_code1, "batch_level1": batch_code2, "batch_level2": batch_code3}

print(f"number of batches: batch_level0: {len(batch_code1)}, batch_level1: {len(batch_code2)}, batch_level2: {len(batch_code3)}")

# ptr = 0
# for cum_idx, chunk_size in enumerate(chunk_sizes):
#     meta_cells_chunk = meta_cells.iloc[ptr:(chunk_size + ptr), :]
#     meta_cells_chunk.to_parquet(data_dir + f"obs_{cum_idx}_batchcode.parquet")
#     ptr += chunk_size
# print("Done.")

# In[]
# Each dataset has both batch & condition,
# batch is technical factors including technology, suspension type, etc that are not important for biological information of study
# condition also exist at the same time (sex, developmental stage, tissue type, etc), the model should be generalizable towards all conditions

def calc_tech_features(batch_id):
    print(f"process batch {batch_id}...")
    meta_tech_batch = meta_cells.loc[meta_cells[batch_name] == batch_id, :]
    batch_tech = pd.DataFrame(index = [batch_id], columns = ["assay", "suspension_type"])

    assay_info = np.unique(meta_tech_batch["assay"].values)
    suspension_type = np.unique(meta_tech_batch["suspension_type"].values)
    assert len(assay_info) == 1
    assert len(suspension_type) == 1

    batch_tech.loc[batch_id, "assay"] = assay_info[0]
    batch_tech.loc[batch_id, "suspension_type"] = suspension_type[0]
    return batch_tech

def calc_expr_features(partition_idx):
    print(f"processing partition {partition_idx}.")

    # construct batch_data
    batch_data = pd.DataFrame(data = 0.0, index = batch_code[batch_name], columns = batch_info + batch_gene_name)
    
    # extract libsize for normalization, more accurate: calculate the libsize from the data directly    
    meta_cells_chunk = pd.read_parquet(os.path.join(data_dir, f"obs_{partition_idx}_batchcode.parquet"))
    # read in the raw count matrix for the corresponding partition
    data_chunk = np.memmap(os.path.join(data_dir, f"counts_data_{partition_idx}.npz"), dtype = 'float32', mode = 'r', shape = (int(sizes[partition_idx, 0]),))
    indices_chunk = np.memmap(os.path.join(data_dir, f"counts_indices_{partition_idx}.npz"), dtype = 'int16', mode = 'r', shape = (int(sizes[partition_idx, 1]),))
    indptr_chunk = np.memmap(os.path.join(data_dir, f"counts_indptr_{partition_idx}.npz"), dtype = 'uint64', mode = 'r', shape = (int(sizes[partition_idx, 2]),))
    # counts_chunk = sp.csr_matrix((data_chunk, indices_chunk, indptr_chunk), shape=(meta_cells_chunk.shape[0], len(full_gene)))
    counts_chunk = {"data": data_chunk, "indices": indices_chunk, "indptr": indptr_chunk, "row": meta_cells_chunk.shape[0], "col": len(full_gene)}
    
    # NOTE: maybe calculate libsize, nnz, raw_mean_nnz all directly from data, summed features across all cells within the batch, need to divide by number of cells in the end
    expr_dict = batch_encode.construct_expr_feats(counts_raw = counts_chunk, batch_labels = np.array([x for x in meta_cells_chunk[batch_name].values]),
                                                  batch_list = batch_code[batch_name], gene_name = full_gene, gene_select = batch_gene_name)

    # the returned value is sum
    batch_data.loc[expr_dict["batch_stats"].index, ["libsize", "nnz", "raw_mean_nnz", "num_cells"]] += expr_dict["batch_stats"][["libsize", "nnz", "raw_mean_nnz", "ncells"]].values
    batch_data.loc[expr_dict["prop_mito"].index, ["prop_mito"]] += expr_dict["prop_mito"].values
    batch_data.loc[expr_dict["expr"].index, batch_gene_name] += expr_dict["expr"][batch_gene_name].values

    print(f"Done {partition_idx}.")
    return batch_data    


# In[]
print("generate batch features....")
batch_name = "batch_level2" # level0 for ribo and stress selection
tech_info = ["assay", "suspension_type"]
batch_info = ["libsize", "nnz", "raw_mean_nnz", "prop_mito", "num_cells"]

print(f"Use the batch ver: {batch_name}, total number of batches: {len(batch_code[batch_name])}")
batch_data_full = pd.DataFrame(index = batch_code[batch_name], columns = tech_info + batch_info + batch_gene_name)
# In[]
print("1. calculate batch-tech features....")
# only select the useful information
meta_cells = meta_cells[tech_info + [batch_name]]

args = [(x,) for x in batch_code[batch_name]]
with Pool(processes = 32) as pool:
    batch_tech_list = pool.starmap(calc_tech_features, args)
batch_tech = pd.concat(batch_tech_list, axis = 0)
# set the final value
batch_data_full.loc[batch_tech.index, batch_tech.columns] = batch_tech.values

# In[]
# read in the count matrix, of the same order as meta-cells
print("2. calculate batch-expr features...")
# init
batch_data_full.loc[:, batch_info + batch_gene_name] = 0.0

n_processes = 2
results = []
if n_processes == 1:
    for partition in range(len(sizes)):
        results.append(calc_expr_features(partition))
else:
    with Pool(processes = n_processes) as pool:
        results = pool.starmap(calc_expr_features, [(x,) for x in range(len(sizes))])

for batch_expr in results:
    batch_data_full.loc[batch_expr.index, batch_expr.columns] += batch_expr.values 
# average by number of cells
batch_data_full[batch_info + batch_gene_name] = batch_data_full[batch_info + batch_gene_name].values/(batch_data_full["num_cells"].values.squeeze()[:,None] + 1e-4)

# batch_data_full.to_csv(res_dir + f"feature_{batch_name}.csv")
batch_data_full.to_csv(f"feature_{batch_name}_validation.csv")

# assert False
# In[]
# ---------------------------------------------------------------------------------------------
#
# Digitize the batch features
#
# ---------------------------------------------------------------------------------------------
use_mito = True
use_tech = False
use_nmeasure = False
batch_name = "batch_level2"
expr_binsize = 1

# need to provide full category or max continuous values
# batch_data = pd.read_csv(f"feature_{batch_name}_filter.csv", index_col = 0)
# key feature hard-coded already
batch_data = pd.read_csv(res_dir + f"feature_{batch_name}.csv", index_col = 0)
# ncells is irrelevant
batch_data = batch_data.drop(["num_cells"], axis = 1)

# old
batch_data_cat, n_cat_list = batch_encode.tokenize_batch_feats(batch_data, use_mito = use_mito, use_tech = use_tech, use_nmeasure = use_nmeasure, expr_binsize = expr_binsize)


batch_dict = {"state_dict_enc": None,
              "state_dict_dec": None,
              "n_cat_list": n_cat_list,
              "n_cont_feats": 0, 
              "cats": batch_data_cat, 
              "conts": None
              }

file_name = f"batch_dict_{batch_name}"
# if use_mito:
#     file_name += "_mito"
if use_tech:
    file_name += "_tech"
torch.save(batch_dict, data_dir + file_name + ".pt")


# In[]
# ---------------------------------------------------------------------------------------------
#
# NOTE: New adaptive binning
#
# ---------------------------------------------------------------------------------------------
# read in the batch dict 
data_dir = "/net/csefiles/xzhanglab/zzhang834/hs_download/"
data_dir = "/data/zzhang834/hs_download/"
batch_name = "level2"
batch_feats = pd.read_csv(data_dir + f"meta_data/feature_batch_{batch_name}_filter.csv", index_col = 0)
nbins = 10
batch_feats_digitize_full, max_vals = batch_encode.tokenize_batch_feats(batch_feats = batch_feats, max_vals = None, nbins = nbins, normalize = True, margin = 0.0)
# libsize already normalized
batch_feats_digitize_full = batch_feats_digitize_full.drop(["libsize"], axis = 1)
max_vals = max_vals[1:]
for col in batch_feats_digitize_full.columns:
    print(col)
    uniq_val, uniq_count = np.unique(batch_feats_digitize_full[col].values.squeeze(), return_counts = True)
    print(uniq_val)
    # print(uniq_count)

batch_dict_expr = {"state_dict_enc": None,
                   "state_dict_dec": None,
                   "n_cat_list": np.array([nbins] * batch_feats_digitize_full.shape[1]),
                   "cat_maxvals": max_vals,
                   "n_cont_feats": 0,
                   "cats": batch_feats_digitize_full,
                   "conts": None}
torch.save(batch_dict_expr, data_dir + f"meta_data/batch_dict_{batch_name}_10.pt")

# In[]
# drop features related to genes
batch_feats = batch_feats.drop(["assay", "suspension_type", "prop_mito", "raw_mean_nnz", "nnz", "libsize", "num_cells"], axis = 1)
# only consider genes
batch_feats_digitize, max_vals = batch_encode.tokenize_batch_gene(batch_feats = batch_feats, max_vals = None, margin = 0.1, nbins = nbins, normalize = True)


for col in batch_feats_digitize.columns:
    uniq_val, uniq_count = np.unique(batch_feats_digitize[col].values.squeeze(), return_counts = True)
    print(uniq_val)
    # print(uniq_count)

batch_dict_expr = {"state_dict_enc": None,
                   "state_dict_dec": None,
                   "n_cat_list": np.array([nbins] * batch_feats_digitize.shape[1]),
                   "cat_maxvals": max_vals,
                   "n_cont_feats": 0,
                   "cats": batch_feats_digitize,
                   "conts": None}
torch.save(batch_dict_expr, data_dir + f"meta_data/batch_dict_{batch_name}_expr10.pt")


assert False
# In[]
# ---------------------------------------------------------------------------------------------
#
# Evaluate the feature importance [NOTE: analysis only run once]
#
# ---------------------------------------------------------------------------------------------
# check the mean expression across batches, and the variance of expression across batches
# Sort according to the variance across batches.
import matplotlib.pyplot as plt

batch_name = "batch_level2"
batch_data = pd.read_csv(f"feature_{batch_name}.csv", index_col = 0)
batch_data_expr = batch_data[hk_gene_name + ribo_gene_name + stress_gene_name]

batch_data_expr_digitize = batch_data_expr.copy()
expr_bucket = np.arange(0, 10.0, 1)
batch_data_expr_digitize[:] = np.digitize(batch_data_expr_digitize.values, expr_bucket, right = True)
gene_mean = batch_data_expr_digitize.mean()
gene_var = batch_data_expr_digitize.var()

# In[]
# filtering ribosomal genes by variance across batches
ribo_gene_df = pd.DataFrame(index = ribo_gene_name, columns = ["mean", "var"], data = 0.0)
ribo_gene_df["mean"] = gene_mean[ribo_gene_name].values
ribo_gene_df["var"] = gene_var[ribo_gene_name].values

fig = plt.figure(figsize = (10, 5))
ax = fig.subplots(nrows = 1, ncols = 2)
_ = ax[0].hist(ribo_gene_df["mean"].values, bins = 30)
ax[0].set_title("Mean Ribo")
_ = ax[1].hist(ribo_gene_df["var"].values, bins = 30)
ax[1].set_title("Var Ribo")

ribo_gene_select_var = ribo_gene_df.loc[(ribo_gene_df["var"] > 1), :].index
# sort by variance (large to small)
ribo_gene_select_var = ribo_gene_select_var[np.argsort(ribo_gene_df.loc[ribo_gene_select_var, "var"].values)[::-1]]

# In[]
# select ribosomal genes that are not highly correlated, greedy algorithm

# 1. calculate pairwise correlation to decide the cut-off
batch_data_ribo = batch_data_expr_digitize[ribo_gene_name]
corr_matrix = batch_data_ribo.corr().abs()
batch_data_ribo_select = batch_data_expr_digitize[ribo_gene_select_var]
corr_matrix_select = batch_data_ribo_select.corr().abs()
# check the distribution of correlation values (select only the upper triangle)
fig = plt.figure(figsize = (10, 5))
ax = fig.subplots(nrows = 1, ncols = 2)
_ = ax[0].hist(corr_matrix.values[np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)], bins = 30)
_ = ax[1].hist(corr_matrix_select.values[np.triu(np.ones(corr_matrix_select.shape), k=1).astype(bool)], bins = 30)

# In[]
# select those with correlation smaller than 0.9
cut_off = 0.9
ribo_select_pool = []
ribo_remain_pool = ribo_gene_select_var
# sort the remaining ribo gene by the variance (large to small)
while(len(ribo_remain_pool) != 0):
    ribo_gene_x = ribo_remain_pool[0]
    ribo_select_pool.append(ribo_gene_x)

    ribo_remain_pool = ribo_remain_pool[1:]
    high_corr_idx = [idx_y for idx_y, ribo_gene_y in enumerate(ribo_remain_pool) if corr_matrix_select.loc[ribo_gene_x, ribo_gene_y] > cut_off]
    # if large than cut_off, drop ribo_gene_y    
    ribo_remain_pool = np.delete(ribo_remain_pool, high_corr_idx)

ribo_select_final = ribo_select_pool

# In[]
# check the correlation of hk genes (keep all hk genes)
batch_data_hk_select = batch_data_expr_digitize[hk_gene_name]
corr_matrix = batch_data_hk_select.corr().abs()
_ = plt.hist(corr_matrix.values[np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)], bins = 30)

hk_gene_df = pd.DataFrame(index = hk_gene_name, columns = ["mean", "var"], data = 0.0)
hk_gene_df["mean"] = gene_mean[hk_gene_name].values
hk_gene_df["var"] = gene_var[hk_gene_name].values

fig = plt.figure(figsize = (10, 5))
ax = fig.subplots(nrows = 1, ncols = 2)
_ = ax[0].hist(hk_gene_df["mean"].values, bins = 30)
ax[0].set_title("Mean HK")
_ = ax[1].hist(hk_gene_df["var"].values, bins = 30)
ax[1].set_title("Var HK")

# In[]
# filter the stress genes by variance
stress_gene_df = pd.DataFrame(index = stress_gene_name, columns = ["mean", "var"], data = 0.0)
stress_gene_df["mean"] = gene_mean[stress_gene_name].values
stress_gene_df["var"] = gene_var[stress_gene_name].values

fig = plt.figure(figsize = (10, 5))
ax = fig.subplots(nrows = 1, ncols = 2)
_ = ax[0].hist(stress_gene_df["mean"].values, bins = 30)
ax[0].set_title("Mean Stress")
_ = ax[1].hist(stress_gene_df["var"].values, bins = 30)
ax[1].set_title("Var Stress")

# var > 3, use mean cut-off > 1 doesn't affect the overall number of genes
# almost the same
stress_gene_select_var = stress_gene_df.loc[(stress_gene_df["var"] > 0.6), :].index

# In[]
batch_data_stress = batch_data_expr_digitize[stress_gene_name]
corr_matrix = batch_data_stress.corr().abs()
batch_data_stress_select = batch_data_expr_digitize[stress_gene_select_var]
corr_matrix_select = batch_data_stress_select.corr().abs()
# check the distribution of correlation values (select only the upper triangle)
fig = plt.figure(figsize = (10, 5))
ax = fig.subplots(nrows = 1, ncols = 2)
_ = ax[0].hist(corr_matrix.values[np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)], bins = 30)
_ = ax[1].hist(corr_matrix_select.values[np.triu(np.ones(corr_matrix_select.shape), k=1).astype(bool)], bins = 30)
# all correlation smaller than 0.9
stress_select_final = [x for x in stress_gene_select_var]


# In[]
# filter the batch feature data
batch_name = "batch_level0"
batch_data = pd.read_csv(f"feature_{batch_name}.csv", index_col = 0)
batch_data_filter = batch_data[["assay", "suspension_type", "libsize", "nnz", "raw_mean_nnz", "prop_mito", "num_cells"] + hk_gene_name + ribo_select_final + stress_select_final]
batch_data_filter.to_csv(f"feature_{batch_name}_filter.csv")




# In[]
# ---------------------------------------------------------------------------------------------
#
# Calculate the cell type in each batch
#
# ---------------------------------------------------------------------------------------------
# # Obtain the possible batches for each cell type
# meta_cells = []
# data_dir = "/localscratch/ziqi/hs_download/permuted/"
# sizes = np.loadtxt(data_dir + "sizes.txt")
# chunk_sizes = []
# for partition_idx in range(0, len(sizes)):
#     meta_cells_chunk = pd.read_parquet(os.path.join(data_dir, f"obs_{partition_idx}_batchcode.parquet"))
#     chunk_sizes.append(meta_cells_chunk.shape[0])
#     meta_cells.append(meta_cells_chunk)
# meta_cells = pd.concat(meta_cells, axis = 0)

# # load the bincode of the labels
# code_book = torch.load(data_dir + f"codebook.pt", weights_only = False)
# label_dict = {"label_bincode": code_book["label_bincode"], "label_code": code_book["label_code"]}
# # load the batch_dict
# batch_dict = torch.load("batch_enc_dict.pt", weights_only = False)
# # number of batches
# n_batches = batch_dict["cats"].shape[0]
# n_labels = label_dict["label_code"].shape[0]
# cell_batch = np.zeros((n_labels, n_batches))
# for label_id, label in enumerate(code_book["label_code"]):
#     batch_label = np.unique(meta_cells.loc[meta_cells["label_id"] == label_id, "batch_id"].values)
#     cell_batch[label_id, batch_label] = 1

# label_dict["label_batch"] = cell_batch
# torch.save(label_dict, data_dir + "label_dict.pt")


# In[]
# ---------------------------------------------------------------------------------------------
#
# Process assay ontology
#
# ---------------------------------------------------------------------------------------------
# # 
# # TODO: improve the technology annotation with bincode
# from owlready2 import *

# # Load ontology from a local .owl file or a URL
# onto = get_ontology("/project/zzhang834/LLM_KD/dataset/efo.owl").load()
# assay_codebook = {"10x 3' v3": 0, "Smart-seq2": 1, "10x 3' v2": 2, "10x 5' v1": 3, "10x 5' transcription profiling": 4, "Seq-Well": 5, 
#                     "10x 3' v1": 6, "10x 5' v2": 7, "Seq-Well S3": 8, "Drop-seq": 9, "microwell-seq": 10, "Smart-seq v4": 11,
#                     "ScaleBio single cell RNA sequencing": 12, "10x 3' transcription profiling": 13, "TruDrop": 14, "MARS-seq": 15,
#                     "CEL-seq2": 16, "SPLiT-seq": 17, "BD Rhapsody Whole Transcriptome Analysis": 18, "BD Rhapsody Targeted mRNA": 19,
#                     "sci-RNA-seq": 20}

# # assay bincode
# assay_onto_bincode = pd.DataFrame(index = assay_codebook.keys(), columns = assay_codebook.keys(), data = np.eye(len(assay_codebook.keys())).astype(int))
# for assay in assay_onto_bincode.index:
#     cls = onto.search_one(label=assay)
#     ancestors = [x for x in cls.ancestors() if x.name[:3] == "EFO"]
#     for x in ancestors:
#         for label in x.label:
#             if label in assay_onto_bincode.columns:
#                 assay_onto_bincode.loc[assay, label] = 1


# #print all classes
# with open("/project/zzhang834/LLM_KD/assay_onto.txt", "w") as f:
#     for x in onto.classes():
#         if x.name[:3] == "EFO":
#             f.write(str(x.name) + ": " + str(x.label) +"\n")
#             print()

# for assay in assay_codebook.keys():
#     cls = onto.search_one(label=assay)
#     ancestors = [x for x in cls.ancestors() if x.name[:3] == "EFO"]
#     print(assay)
#     print(cls)
#     print(ancestors)
#     print([x.label for x in ancestors])
#     print()
#     # break


# In[]
# def slice_csr(data, indices, indptr, start_row, end_row, num_cols):
#     """Slice a CSR matrix by manipulating the internal data"""
#     # Create a new indptr for the slice
#     new_indptr = indptr[start_row:end_row+1] - indptr[start_row]
    
#     # Slice the data and indices corresponding to the rows
#     data_start = indptr[start_row]
#     data_end = indptr[end_row]
    
#     new_data = data[data_start:data_end]
#     new_indices = indices[data_start:data_end]
    
#     # Number of rows in the new matrix
#     num_rows = end_row - start_row
    
#     # Create the new sliced CSR matrix
#     try:
#         sliced_csr = sp.csr_matrix((new_data, new_indices, new_indptr), shape=(num_rows, num_cols))
#     except:
#         raise ValueError("issue with index")

#     return sliced_csr

# too slow
# def calc_expr_batch_features(partition_idx):
#     print(f"processing partition {partition_idx}.")
#     expr_batch_norm = pd.DataFrame(data = 0.0, index = batch_code[batch_name], columns = gene_embed_dict["labels"]["feature_name"].values)
#     prop_mito = pd.DataFrame(data = 0.0, index = batch_code[batch_name], columns = ["prop_mito"])
#     ngenes = expr_batch_norm.shape[1]

#     # extract libsize    
#     meta_cells_chunk = pd.read_parquet(os.path.join(data_dir, f"obs_{partition_idx}_batchcode.parquet"))
#     libsize = meta_cells_chunk["raw_sum"].values.reshape(-1)

#     # read in the count matrix for the corresponding partition
#     data_chunk = np.memmap(os.path.join(data_dir, f"counts_data_{partition_idx}.npz"), dtype = 'float32', mode = 'r', shape = (int(sizes[partition_idx, 0]),))
#     indices_chunk = np.memmap(os.path.join(data_dir, f"counts_indices_{partition_idx}.npz"), dtype = 'int16', mode = 'r', shape = (int(sizes[partition_idx, 1]),))
#     indptr_chunk = np.memmap(os.path.join(data_dir, f"counts_indptr_{partition_idx}.npz"), dtype = 'uint64', mode = 'r', shape = (int(sizes[partition_idx, 2]),))

#     for uniq_batch in batch_code[batch_name]:
#         batch_idx = np.where((meta_cells_chunk[batch_name] == uniq_batch).values)[0]
#         if len(batch_idx) == 0:
#             continue
        
#         for idx in batch_idx:
#             expr_idx = slice_csr(data_chunk, indices_chunk, indptr_chunk, start_row = idx, end_row = idx+1, num_cols = ngenes).astype(np.float32).toarray().squeeze()

#             # select mito genes, calculate the mito-proportion within the cell
#             expr_idx_df = pd.DataFrame(data = expr_idx[None,:], columns = expr_batch_norm.columns)
#             mito_prop = np.sum(expr_idx_df[mito_gene_name].values)/(np.sum(expr_idx_df.values) + 1e-6)
#             prop_mito.loc[uniq_batch, "prop_mito"] += mito_prop

#             # raw expr, normalize by libsize, and log1p
#             expr_idx = expr_idx / (libsize[idx] + 1e-4) * 10e4
#             expr_idx = np.log1p(expr_idx)
#             assert np.min(expr_idx) >= 0
#             expr_batch_norm.loc[uniq_batch, :] += expr_idx

#     print("Done.")
#     return {"expr": expr_batch_norm, "prop_mito": prop_mito}


# %%
