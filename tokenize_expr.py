# In[]
import torch
import anndata
from pathlib import Path
import scanpy as sc
import numpy as np
import scipy.sparse as sp
import src.data_utils as data_utils
import pandas as pd

import sys, os
sys.path.append("src")

from data_utils import set_seed

# In[]
# ----------------------------------------------------------------------------
#
# Tokenize the gene expression data into sentences
#
# ----------------------------------------------------------------------------
n_mgene = 256
# whether use the binary count or not
COUNT_TYPE = "mean"
DROP_ZEROS = False

res_dir = Path(f"/project/zzhang834/LLM_KD/dataset/cellxgene_{COUNT_TYPE}_{int(DROP_ZEROS)}")
if not os.path.exists(res_dir):
    os.makedirs(res_dir)

gene_embed_dict = torch.load(f"/project/zzhang834/llm_dataset/CellXGeneCZI/data_download/gene_embed_meta{n_mgene}.pt", weights_only = False)

meta_genes, meta_gene_sizes = np.unique(gene_embed_dict["labels"]["labels"].values, return_counts = True)
meta_gene_sizes = meta_gene_sizes[np.argsort(meta_genes)]

# In[]
# original esm is half precision, need to modify to full precision for training
token_embed = gene_embed_dict["meta_embed"].to(torch.float32)
# add the cls, mask, and padding tokens
set_seed(0)
special_tokens = torch.randn(3, token_embed.shape[1])
token_embed = torch.concat([token_embed, special_tokens], dim = 0)
torch.save(token_embed, res_dir / f"token_embed_{n_mgene}.pt")

# In[]
# for supervision
labels = []
batches = []

tissue_list = ["all_remain", "pancreas", "blood", "brain", "heart", "intestine", "kidney", "lung"]
# tissue_list = ["pancreas"]
print("read gene expression...")
# read in the anndata
counts_norm = []
for tissue in tissue_list:
    print(tissue)
    data_dir = Path(f"/project/zzhang834/llm_dataset/CellXGeneCZI/data_download/{tissue}")
    adata = anndata.read_h5ad(data_dir / f"adata_meta{n_mgene}.h5ad")
    # use bin data
    if COUNT_TYPE == "binary":
        adata.X = adata.layers["counts_bin"].copy()
    elif COUNT_TYPE == "mean":
        # average by meta-gene size
        adata.X = sp.csr_matrix(adata.X.toarray()/meta_gene_sizes[None, :])

    sc.pp.normalize_total(adata, target_sum = 10e4, key_added = "libsize")
    sc.pp.log1p(adata)

    counts_norm.append(adata.X.copy())
    meta = adata.obs.copy()
    
    labels.append(np.array([x + "--" + y for x,y in zip(meta["cell_type_ontology_term_id"].values, meta["cell_type"].values)]))
    batches.append(meta["dataset_id"].astype(object).values.squeeze())
    del adata, meta

# after aggregate all counts
counts_norm = sp.vstack(counts_norm)
# combine the data together
labels = np.concatenate(labels, axis = 0)
batches = np.concatenate(batches, axis = 0)
label_ids, label_codes = pd.factorize(labels)
batch_ids, batch_codes = pd.factorize(batches)

ncells = len(label_ids)
meta_dict = {"label": label_ids, "batch": batch_ids, "label_code": label_codes, "batch_code": batch_codes, 
             "shape": {"full": (ncells, n_mgene + 1)}, "cls_idx": n_mgene, "pad_idx": n_mgene + 1, "mask_idx": n_mgene + 2}



# In[]
# permute the cells
permute_cells = True
if permute_cells:
    print("permute cells...")
    np.random.seed(0)
    permuted_idx = np.random.permutation(meta_dict["shape"]["full"][0])
    meta_dict["label"] = meta_dict["label"][permuted_idx]
    meta_dict["batch"] = meta_dict["batch"][permuted_idx]
    counts_norm = counts_norm[permuted_idx,:]

# Tokenize counts_norm
print("tokenize the data")
n_chunks = 4
counts_norm_list = data_utils.divide_chunks(counts = counts_norm, nchunks = n_chunks)
del counts_norm

for chunk, counts_norm in enumerate(counts_norm_list):
    print(f"save sentence for data chunk: {chunk}/{n_chunks}")
    ncells_chunk = counts_norm.shape[0]
    meta_dict["shape"][f"chunk_{chunk}"] = (ncells_chunk, n_mgene + 1)
    # NOTE: two versions: directly use the continuous expr values, or bin the expr values
    expr_sent, feat_sent = data_utils.tokenize_expr_para(counts_norm, dropzeros = DROP_ZEROS, nbins = None, njobs = 16, nchunks = 16, npads = n_mgene + 1)

    # copy from UCE, issue with memmap, only work with dense matrix
    fp = np.memmap(res_dir / f"expr_sent_{n_mgene}_{chunk}.npz", dtype='float32', mode='w+', shape=(ncells_chunk, n_mgene + 1))
    fp[:] = expr_sent.toarray()[:]
    fp.flush()
    fp = np.memmap(res_dir / f"feat_sent_{n_mgene}_{chunk}.npz", dtype='int32', mode='w+', shape=(ncells_chunk, n_mgene + 1))
    fp[:] = feat_sent.toarray()[:]
    fp.flush()

    del expr_sent, feat_sent

torch.save(meta_dict, f = res_dir / f"meta_{n_mgene}.pt")

    
expr_sents = []
for chunk in range(n_chunks):
    ncells_chunk = counts_norm_list[chunk].shape[0]
    expr_sent = np.memmap(res_dir / f"expr_sent_{n_mgene}_{chunk}.npz", dtype = "float32", mode = "r", shape = (ncells_chunk, n_mgene + 1))
    expr_sents.append(expr_sent)
expr_sents = np.vstack(expr_sents)
fp = np.memmap(res_dir / f"expr_sent_{n_mgene}.npz", dtype='float32', mode='w+', shape=(ncells, n_mgene + 1))
fp[:] = expr_sents[:]
fp.flush()
del expr_sents

feat_sents = []
for chunk in range(n_chunks):
    ncells_chunk = counts_norm_list[chunk].shape[0]
    feat_sent = np.memmap(res_dir / f"feat_sent_{n_mgene}_{chunk}.npz", dtype = "int32", mode = "r", shape = (ncells_chunk, n_mgene + 1))
    feat_sents.append(feat_sent)
feat_sents = np.vstack(feat_sents)
fp = np.memmap(res_dir / f"feat_sent_{n_mgene}.npz", dtype='int32', mode='w+', shape=(ncells, n_mgene + 1))
fp[:] = feat_sents[:]
fp.flush()
del feat_sents

print("Done.")

# In[]
# ----------------------------------------------------------------------------
#
# Create dataset from the tokenized data using huggingface, NOTE: too costly
#
# ----------------------------------------------------------------------------
# # 1. Huggingface dataset, too costly
# print("loading dataset...")
# n_mgene = 256
# res_dir = Path(f"/project/zzhang834/LLM_KD/dataset/cellxgene")
# meta_dict = torch.load(res_dir / f"meta_{n_mgene}.pt")
# expr_sents = np.memmap(res_dir / f"expr_sent_{n_mgene}.npz", dtype = "float32", mode = "r", shape = meta_dict["shape"])
# feat_sents = np.memmap(res_dir / f"expr_sent_{n_mgene}.npz", dtype = "float32", mode = "r", shape = meta_dict["shape"])

# from datasets import Dataset, Features, Value, Sequence
# # huggingface dataset save data as a dataframe structure, columns are data, label, etc, and rows are observation/samples
# # for sequence data, each item in the data column should be saved as a list

# print("building HF dataset...")
# data_dict = {
#     "expr": expr_sents.astype("float32").tolist(),
#     "gene": feat_sents.astype("int32").tolist(),
#     "label": meta_dict["label"].astype("int32"),
#     "batch": meta_dict["batch"].astype("int32"),
# }

# features = Features({
#     "expr": Sequence(Value("float32"), length = expr_sents.shape[1]),
#     "gene": Sequence(Value("int32"), length = feat_sents.shape[1]),
#     "label": Value('int32'), # alternatively use ClassLabel, here use value for consistency between two methods
#     "batch": Value('int32')
#     })
# # only saved in the memory
# sc_dataset = Dataset.from_dict(data_dict, features = features)

# print("save to disk...")
# # store to the disk, constrain the shard size
# sc_dataset.save_to_disk(res_dir / f"cellxgene_{n_mgene}", max_shard_size="1GB")

# print("Done.")


