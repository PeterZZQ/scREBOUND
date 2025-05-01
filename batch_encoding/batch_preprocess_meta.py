# In[]
import pandas as pd
import numpy as np
import torch
import os
from tqdm import tqdm
import scipy.sparse as sp


def slice_csr(data, indices, indptr, start_row, end_row, num_cols):
    """Slice a CSR matrix by manipulating the internal data"""
    # Create a new indptr for the slice
    new_indptr = indptr[start_row:end_row+1] - indptr[start_row]
    
    # Slice the data and indices corresponding to the rows
    data_start = indptr[start_row]
    data_end = indptr[end_row]
    
    new_data = data[data_start:data_end]
    new_indices = indices[data_start:data_end]
    
    # Number of rows in the new matrix
    num_rows = end_row - start_row
    
    # Create the new sliced CSR matrix
    try:
        sliced_csr = sp.csr_matrix((new_data, new_indices, new_indptr), shape=(num_rows, num_cols))
    except:
        raise ValueError("issue with index")

    return sliced_csr

# In[]
hm_housekeeping = pd.read_csv("/project/zzhang834/LLM_KD/dataset/MostStable.csv", sep = ";")
n_mgene = 256
gene_embed_dict = torch.load(f"/localscratch/ziqi/hs_download/gene_embed_meta{n_mgene}_fuzzy.pt", weights_only = False)

# TODO: the hk and mito genes are selected from only the protein embedding genes, there are more genes in the original data
hk_gene_name = np.intersect1d(hm_housekeeping["Gene name"].values.squeeze(), 
                              gene_embed_dict["labels"]["feature_name"].values.squeeze())
mito_gene_name = [gene for gene in gene_embed_dict["labels"]["feature_name"].values.squeeze() if gene.startswith("MT-")]


# In[]
# read in the batch_meta_information
meta_cells = []
data_dir = "/project/zzhang834/LLM_KD/dataset/cellxgene_sum_new/"
sizes = np.loadtxt(data_dir + "sizes.txt")
chunk_sizes = []
for partition_idx in range(0, len(sizes)):
    meta_cells_chunk = pd.read_parquet(os.path.join(data_dir, f"obs_{partition_idx}.parquet"))
    chunk_sizes.append(meta_cells_chunk.shape[0])
    meta_cells.append(meta_cells_chunk)
meta_cells = pd.concat(meta_cells, axis = 0)

# factors that decide batches
# NOTE: tissue_type only has tissue, 238 tissue, 
batch_bio_factors = ["development_stage", "sex", "tissue", "tissue_general"]
batch_tech_factors = ["dataset_id", "assay", "suspension_type"]
# unique dataset: 611, but unique batch: 873
meta_cells["batch_tech"] = ["-".join(x) for x in zip(*[meta_cells[x].values for x in batch_tech_factors])]
meta_cells["batch"] = ["-".join(x) for x in zip(*[meta_cells[x].values for x in ["dataset_id", "assay", "suspension_type", "development_stage", "sex", "tissue_general"]])]
meta_cells["batch_detail"] = ["-".join(x) for x in zip(*[meta_cells[x].values for x in batch_bio_factors + batch_tech_factors])]

# NOTE: for meta, no need to re-calculate the batch factor 
# load the batch_factors learned from single-cell version
batch_name = "batch_tech"
batch_factors = pd.read_csv("batch_feature.csv", index_col = 0)
batch_code = batch_factors.index.values.squeeze()

cat = pd.Categorical(meta_cells[batch_name].values, categories = batch_code, ordered = True)
meta_cells["batch_id"] = cat.codes

# In[]
# save the updated meta-cells
ptr = 0
for cum_idx, chunk_size in enumerate(chunk_sizes):
    meta_cells_chunk = meta_cells.iloc[ptr:(chunk_size + ptr), :]
    meta_cells_chunk.to_parquet(data_dir + f"obs_{cum_idx}_batchcode.parquet")
    ptr += chunk_size



# %%
