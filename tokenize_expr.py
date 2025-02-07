# In[]
import torch
import anndata
from pathlib import Path
import scanpy as sc
import numpy as np
import scipy.sparse as sp
import src.data_utils as data_utils
import pandas as pd

# In[]
# ----------------------------------------------------------------------------
#
# Tokenize the gene expression data into sentences
#
# ----------------------------------------------------------------------------
# for supervision
labels = []
batches = []

n_mgene = 256
res_dir = Path(f"/project/zzhang834/LLM_KD/dataset/cellxgene")
print("read gene expression...")
# read in the anndata
counts_norm = []
for tissue in ["all_remain", "blood", "brain", "heart", "intestine", "kidney", "lung"]:
    print(tissue)
    data_dir = Path(f"/project/zzhang834/llm_dataset/CellXGeneCZI/data_download/{tissue}")
    adata = anndata.read_h5ad(data_dir / f"adata_meta{n_mgene}.h5ad")
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
    expr_sent, feat_sent = data_utils.tokenize_expr_para(counts_norm, njobs = 16, nchunks = 16, npads = n_mgene + 1)

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


# In[]
# ----------------------------------------------------------------------------
#
# Process the labels
#
# ----------------------------------------------------------------------------
import pandas as pd
from ontobio.ontol_factory import OntologyFactory
n_mgene = 256
meta_dict = torch.load(f"/localscratch/ziqi/localscratch_tempdata/cellxgene/meta_{n_mgene}.pt")
ont = OntologyFactory().create("/project/zzhang834/LLM_KD/dataset/cl.json")
# Step 2: Function to get ancestors in Cell Ontology
def get_ancestors(cl_id):
    """
    Get all ancestors of a cell type based on its Cell Ontology ID.
    """
    asso_list = ont.ancestors(cl_id)

    return [x for x in asso_list if x[:2] == "CL"]

def get_parents(cl_id):
    """
    Get all ancestors of a cell type based on its Cell Ontology ID.
    """
    asso_list = ont.parents(cl_id)

    return [x for x in asso_list if x[:2] == "CL"]


label_ids, label_id_counts = np.unique(meta_dict["label"], return_counts = True)
label_ids = label_ids[np.argsort(label_id_counts)]
label_ids_code = meta_dict["label_code"][label_ids]
label_code_clid = np.array([x.split("--")[0] for x in label_ids_code])
label_id_counts = np.sort(label_id_counts)

# create the dataframe
label_code_df = pd.DataFrame(index = label_code_clid, columns = ["cell_type"], data = np.array([x.split("--")[1] for x in label_ids_code])[:,None])



# filter for small cell types,
n_cells_filter = 500
label_id_counts_f = label_id_counts[label_id_counts < n_cells_filter]
label_code_clid_f = label_code_clid[label_id_counts < n_cells_filter]

for clid in label_code_clid_f:
    ancesters = get_ancestors(clid)
    # ancesters = get_parents(clid)
    for ancester in ancesters:
        if ancester in label_code_clid:
            print(clid)
            print(label_code_df.loc[clid, "cell_type"])
            print(ancester)
            print(label_code_df.loc[ancester, "cell_type"])
    break

def check_ancester(ancesters, ancesters_list):
    selected_ancesters = []
    for ancester in ancesters:
        if ancester in ancesters_list:
            selected_ancesters.append(selected_ancesters)
    
    if len(selected_ancesters) == 0:
        return None
    else:
        return selected_ancesters


for clid in label_code_clid_f:
    # find all parents and sort them by ascending orders
    while(selected_):
        ancesters = get_parents(clid)



    selected_ancesters = check_ancester(ancesters, label_code_clid)

    if selected_ancesters is None:
        ancesters_level2 = []
        for ancester in ancesters:
            selected_ancesters = check_ancester(ancesters_level2, label_code_clid)
            if selected_ancesters is not None: 
                ancesters_level2.extend(selected_ancesters)
        if len(ancesters_level2) == 0:
            print("no ancester:" + str(label_code_df.loc[clid, "cell_type"]))
        else:
            print("ancester:" + str(label_code_df.loc[clid, "cell_type"]))
    else:
        print("ancester:" + str(label_code_df.loc[clid, "cell_type"]))
            
    break




# %%
