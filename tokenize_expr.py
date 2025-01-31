# In[]
import torch
import anndata
from pathlib import Path
import scanpy as sc
import numpy as np
import scipy.sparse as sp
import data_utils
import pandas as pd

# In[]
# ----------------------------------------------------------------------------
#
# Tokenize the gene expression data into sentences
#
# ----------------------------------------------------------------------------
expr_sents = []
feat_sents = []

# for supervision
labels = []
batches = []

n_mgene = 256
res_dir = Path(f"/project/zzhang834/LLM_KD/dataset/cellxgene")
print("tokenize gene expression...")
# read in the anndata
for tissue in ["blood", "brain", "heart", "intestine", "kidney", "lung"]:
# for tissue in ["blood"]:
    print(tissue)
    data_dir = Path(f"/project/zzhang834/llm_dataset/CellXGeneCZI/data_download/{tissue}")
    adata = anndata.read_h5ad(data_dir / f"adata_meta{n_mgene}.h5ad")
    adata.layers["counts"] = adata.X.copy()
    sc.pp.normalize_total(adata, target_sum = 10e4, key_added = "libsize")
    sc.pp.log1p(adata)

    counts_norm = adata.X.copy()
    meta = adata.obs.copy()
    del adata

    # NOTE: two versions: directly use the continuous expr values, or bin the expr values
    expr_sent, feat_sent = data_utils.tokenize_expr_para(counts_norm, njobs = 16, nchunks = 16, npads = n_mgene + 1)

    expr_sents.append(expr_sent)
    feat_sents.append(feat_sent)

    labels.append(meta["cell_type_ontology_term_id"].astype(object).values.squeeze())
    batches.append(meta["dataset_id"].astype(object).values.squeeze())

print("save tokenized results...")
# combine the data together
labels = np.concatenate(labels, axis = 0)
batches = np.concatenate(batches, axis = 0)
label_ids, label_codes = pd.factorize(labels)
batch_ids, batch_codes = pd.factorize(batches)

ncells = len(label_ids)
meta_dict = {"label": label_ids, "batch": batch_ids, "label_code": label_codes, "batch_code": batch_codes, 
             "shape": (ncells, n_mgene + 1), "cls_idx": n_mgene, "pad_idx": n_mgene + 1, "mask_idx": n_mgene + 2}
torch.save(meta_dict, f = res_dir / f"meta_{n_mgene}.pt")

expr_sents = sp.vstack(expr_sents)
feat_sents = sp.vstack(feat_sents)

# copy from UCE, issue with memmap, only work with dense matrix
fp = np.memmap(res_dir / f"expr_sent_{n_mgene}.npz", dtype='float32', mode='w+', shape=(ncells, n_mgene + 1))
fp[:] = expr_sents.toarray()[:]
fp.flush()
fp = np.memmap(res_dir / f"feat_sent_{n_mgene}.npz", dtype='int32', mode='w+', shape=(ncells, n_mgene + 1))
fp[:] = feat_sents.toarray()[:]
fp.flush()

print("Done.")


# In[]
# ----------------------------------------------------------------------------
#
# shuffle the dataset first, prevent the overhead of shuffling the data while training
#
# ----------------------------------------------------------------------------

np.random.seed(0)
res_dir = Path(f"/project/zzhang834/LLM_KD/dataset/cellxgene")
n_mgene = 256
meta_dict = torch.load(res_dir / f"meta_{n_mgene}.pt")
expr_sents = np.memmap(res_dir / f"expr_sent_{n_mgene}.npz", dtype = "float32", mode = "r", shape = meta_dict["shape"])
feat_sents = np.memmap(res_dir / f"feat_sent_{n_mgene}.npz", dtype = "int32", mode = "r", shape = meta_dict["shape"])

# generate the permuted index
permuted_idx = np.random.permutation(meta_dict["shape"][0])
meta_dict["label"] = meta_dict["label"][permuted_idx]
meta_dict["batch"] = meta_dict["batch"][permuted_idx]
expr_sents = expr_sents[permuted_idx, :]
feat_sents = feat_sents[permuted_idx, :]

fp = np.memmap(res_dir / f"expr_sent_{n_mgene}_permu.npz", dtype='float32', mode='w+', shape=meta_dict["shape"])
fp[:] = expr_sents[:]
fp.flush()
fp = np.memmap(res_dir / f"feat_sent_{n_mgene}_permu.npz", dtype='int32', mode='w+', shape=meta_dict["shape"])
fp[:] = feat_sents[:]
fp.flush()

torch.save(meta_dict, f = res_dir / f"meta_{n_mgene}_permu.pt")

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



# %%
