"""
Dataloaders

"""
import warnings
warnings.filterwarnings("ignore")
import torch
import numpy as np
import torch.utils.data as data
import scipy.sparse as sp
# import gc
import random
# from math import ceil, floor
import anndata
# import scanpy as sc

import os
import pandas as pd
from tqdm import tqdm

def set_seed(seed):
    # Set seed for PyTorch
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)           # If using GPU
    torch.cuda.manual_seed_all(seed)       # If using multi-GPU
    # Set seed for NumPy
    np.random.seed(seed)
    # Set seed for Python random
    random.seed(seed)
    # Ensure deterministic behavior in some operations
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
# ----------------------------------------------------------------------------------------------------------
#
# Ver. 1. Training data are saved in a distributed format, more flexible for training 
# 
# ----------------------------------------------------------------------------------------------------------

def slice_csr(data, indices, indptr, start_row, end_row, num_cols):
    """\
    Description:
    -------------
        Slicing csr matrix by row, used when the csr matrix are saved by (data, indices, indptr) separately
        memory efficient when data, indices, and indptr are saved in memmap format, do not need to load all
    
    Parameters:
    -------------
        data: 1D array (memmaped) that save the data of csr matrix
        indices: 1D array that save the indices of csr matrix
        indptr: 1D array that save the indptr of csr matrix
        start_row: start row of the slicing
        end_row: end row of the slicing
        num_cols: number of columns in csr matrix

    Returns:
    ------------
        csr matrix slice
    """
    
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



class sc_partition(data.Dataset):

    def __init__(self, data_path, min_chunksize, normalize, batch_feats):
        """
        Description:
        -------------
            Create training dataset from distributed training data. Training data are saved by partitions in csr format.
            To save computational resources, the sampling of dataset is by contiguous chunk (of size min_chunksize), so the training data need to be permuted in advance
            If min_chunksize == 1, the sampling of dataset is completely random with more cost.

            NOTE: the current version permute the order of mini-batch within the training chunks (by batch_size), but not permuting the data that consist of each mini-batch
            It is fine with 1 epoch since we already permute the data in advance, but for more epochs, more detailed permutation is necessary to change the composition of each mini-batch.
            Currently the training ordering only change but the composition of each mini-batch is still the same

        Parameters:
        -------------
            data_path: stores the path to the training dataset. Under data_path, there should be count partition files, and meta-data partition files,
            batch_size: the size of each loading data chunks/mini-batch (by number of cells)
            normalize: boolean vector indicating whether the data need to be log-normalized (raw) or not (normalized)

        """
        super(sc_partition, self).__init__()
        
        self.data_path = data_path
        # NOTE: Load the data size file if it exists, or calculate from file size
        if os.path.exists(self.data_path + "sizes.txt"):
            self.sizes = np.loadtxt(self.data_path + "sizes.txt")
        else:
            self.sizes = None
        self.min_chunksize = min_chunksize

        self.normalize = normalize

        # load the batch features
        self.batch_feats_cont = batch_feats["conts"] if batch_feats is not None else None
        self.batch_feats_cat = batch_feats["cats"] if batch_feats is not None else None



    def load_partition(self, idx, label_colname = None, batch_colname = None, data_prefix = "counts", meta_prefix = "obs"):
        """\
        Description:
        -------------
            loading training data partition given partition index

        Parameters:
        -------------
            idx: the index of the training data partition
            label_colname: the column name of cell type label (NOTE: factorized) within the meta-data
            batch_colname: the column name of batch label (NOTE: factorized) within the meta-data
            data_prefix: the prefix of the dataset partition name; for partition {idx}, the csr file name follows `{data_prefix}_data_{idx}.npz', `{data_prefix}_indices_{idx}.npz', `{data_prefix}_indptr_{idx}.npz'
            meta_prefix: the prefix of the meta-data; for partition {idx}, the csr file name follows `{meta_prefix}_{idx}.parquet'
             
        """
        meta_cells = pd.read_parquet(self.data_path + f"{meta_prefix}_{idx}_batchcode.parquet")
        vars = pd.read_csv(self.data_path + "var.csv", index_col = 0)

        self.ncells = meta_cells.shape[0]
        self.ngenes = vars.shape[0]

        fname_expr_data = self.data_path + f"{data_prefix}_data_{idx}.npz"
        fname_expr_indices = self.data_path + f"{data_prefix}_indices_{idx}.npz"
        fname_expr_indptr = self.data_path + f"{data_prefix}_indptr_{idx}.npz"
        # loading the sizes of data, indices, and indptr of each partition
        if self.sizes is not None:
            data_size = self.sizes[idx, 0]
            indices_size = self.sizes[idx, 1]
            indptr_size = self.sizes[idx, 2]
        else:
            data_size = os.path.getsize(fname_expr_data) // np.dtype(np.float32).itemsize
            indices_size = os.path.getsize(fname_expr_indices) // np.dtype(np.int16).itemsize
            indptr_size = os.path.getsize(fname_expr_indptr) // np.dtype(np.uint64).itemsize

        self.expr_data = np.memmap(fname_expr_data, dtype = "float32", mode = "r", shape = (int(data_size), ))
        self.expr_indices = np.memmap(fname_expr_indices, dtype = "int16", mode = "r", shape = (int(indices_size), ))
        self.expr_indptr = np.memmap(fname_expr_indptr, dtype = "uint64", mode = "r", shape = (int(indptr_size), ))

        self.batch_ids = meta_cells[batch_colname].values.squeeze()

        if label_colname is not None:
            self.labels = meta_cells[label_colname].values.squeeze()
        else:
            self.labels = None
            

    def __len__(self):
        # Return the number of batches
        return (self.ncells + self.min_chunksize - 1) // self.min_chunksize
    
    def __getitem__(self, idx):
        
        # NOTE: obtain the data mini-batch (start_idx:end_idx) from the training chunk on disk and load it into the memory
        start_idx = int(idx * self.min_chunksize)
        end_idx = int(min((idx + 1) * self.min_chunksize, self.ncells))
        counts = torch.tensor(slice_csr(data = self.expr_data, indices = self.expr_indices, indptr = self.expr_indptr,
                                   start_row = start_idx, end_row = end_idx, num_cols = self.ngenes).astype(np.float32).toarray())
    
        if self.normalize:
            # normalize the raw count and log-transform
            counts_norm = counts/(counts.sum(dim = 1, keepdim = True) + 1e-4) * 10e4
            counts_norm = torch.log1p(counts_norm)
        else:
            counts_norm = counts

        sample = {"counts_norm": counts_norm}

        sample["batch"] = self.batch_ids[start_idx:end_idx]
        if self.batch_feats_cat is not None:
            sample["batch_cat"] = torch.tensor(self.batch_feats_cat[sample["batch"], :], dtype = torch.float32)
        if self.batch_feats_cont is not None:
            sample["batch_cont"] = torch.tensor(self.batch_feats_cont[sample["batch"], :], dtype = torch.float32)
        sample["batch"] = torch.tensor(sample["batch"], dtype = torch.int32)

        if self.labels is not None:
            label = self.labels[start_idx:end_idx]
            sample["label"] = torch.tensor(label, dtype = torch.int32)

        return sample


# ------------------------------------------------------------------------------------------------------------
#
# Evaluation of training result
#
# ------------------------------------------------------------------------------------------------------------

def align_genes(adata, gene_list):
    gene_list_common = np.intersect1d(gene_list, adata.var.index.values.squeeze())
    X = pd.DataFrame(np.zeros((adata.shape[0], len(gene_list))), index = adata.obs.index.values, columns = gene_list)
    X.loc[:, gene_list_common] = adata[:, gene_list_common].X.toarray()

    adata_align = anndata.AnnData(sp.csr_matrix(X.values))
    adata_align.var = pd.DataFrame(index = X.columns)
    adata_align.obs = adata.obs
    return adata_align


def align_genes_memeff(adata, gene_align):
    """
    Memory efficient version: might be slower
    """
    gene_orig = adata.var.index.values.squeeze()
    gene_common = np.intersect1d(gene_align, gene_orig)
    gene_orig_common_position = np.array([np.where(gene_orig == x)[0][0] for x in gene_common])
    gene_align_common_position = np.array([np.where(gene_align == x)[0][0] for x in gene_common])

    counts_align = sp.lil_matrix((adata.shape[0], len(gene_align)))
    for idx in tqdm(range(len(gene_common))):
        counts_align[:, gene_align_common_position[idx]] = adata.X[:, gene_orig_common_position[idx]]

    adata_align = anndata.AnnData(X = counts_align.tocsr())
    adata_align.obs = adata.obs.copy()
    adata_align.var.index = gene_align
    
    return adata_align

class sc_dataset_anndata(data.Dataset):
    """
    construct scdataset from anndata
    """
    def __init__(self, adata, gene_list, batch_feats = None, label_colname = None, batch_colname = None, batch_size = 128, normalize = True):
        """
        expr_path: stores the path to the expr data on disk
        gene_path: stores the path to the gene name of cells on disk 
        meta_path: stores the path to the meta data of cells on disk
        """
        super(sc_dataset_anndata, self).__init__()

        self.ncells = adata.shape[0]

        # check if the count matrix in adata in compressed format
        if isinstance(adata.X, np.ndarray):
            adata.X = sp.csr_matrix(adata.X)

        # find overlapping genes
        if gene_list is not None:
            adata = align_genes(adata, gene_list)
        X = adata.X.toarray()

        # normalize the count
        if normalize:
            libsize = X.sum(axis = 1)
            self.X_norm = X/(libsize[:, None] + 1e-4) * 10e4
            self.X_norm = np.log1p(self.X_norm).astype(np.float32)  
        else:
            self.X_norm = X.astype(np.float32)   

        if batch_feats is not None:
            # load the batch features 
            self.batch_feats_cont = batch_feats["conts"]
            self.batch_feats_cat = batch_feats["cats"]

        self.batch_ids = adata.obs[batch_colname].values.squeeze()

        # note, should be integer values
        if label_colname is not None:
            self.labels = adata.obs[label_colname].values.squeeze()
        else:
            self.labels = None

        self.batch_size = batch_size


    def __len__(self):
        # Return the number of batches
        return (self.ncells + self.batch_size - 1) // self.batch_size
    
    def __getitem__(self, idx):
        start_idx = int(idx * self.batch_size)
        end_idx = int(min((idx + 1) * self.batch_size, self.ncells))
        # to be decided
        counts_norm = torch.tensor(self.X_norm[start_idx:end_idx,:])

        sample = {"counts_norm": counts_norm}
        if self.labels is not None:
            label = self.labels[start_idx:end_idx]
            sample["label"] = torch.tensor(label, dtype = torch.int32)

        sample["batch"] = self.batch_ids[start_idx:end_idx]
        if self.batch_feats_cat is not None:
            sample["batch_cat"] = torch.tensor(self.batch_feats_cat[sample["batch"], :], dtype = torch.float32)
        if self.batch_feats_cont is not None:
            sample["batch_cont"] = torch.tensor(self.batch_feats_cont[sample["batch"], :], dtype = torch.float32)
        sample["batch"] = torch.tensor(sample["batch"], dtype = torch.int32)

        return sample

def adata2meta(adata, feature_df, var_name = "gene_name", normalize = False, count_type = "sum"):
    """
    Group the genes into meta-genes in adata, according to the hard-coded grouping information
    """
    adata_s = adata.copy()
    
    if var_name == "gene_name":
        gene_ids = feature_df["feature_name"].values.squeeze()
        feature_df.index = gene_ids
    elif var_name == "ensembl_id":
        gene_ids = feature_df.index.values.squeeze()
    else:
        raise ValueError("Either gene_name or ensembl_id need to be provided in adata")

    gene_ids = np.intersect1d(adata_s.var.index.values, gene_ids)
    adata_s = adata_s[:, gene_ids]
    print(f"overlapping genes: {len(gene_ids)}")

    counts = adata_s.layers["counts"].toarray()
    if normalize:
        counts = counts/(np.sum(counts, axis = 1, keepdims = True) + 1e-6) *10e4
        counts = np.log1p(counts) 

    meta_labels = feature_df.loc[gene_ids, "labels"].values.squeeze()
    counts_meta = np.zeros((adata_s.shape[0], len(np.unique(feature_df["labels"].values))))
    for label in np.unique(meta_labels):
        if count_type == "sum":
            counts_meta[:, label] = counts[:, meta_labels == label].sum(axis = 1)
        else:
            counts_meta[:, label] = counts[:, meta_labels == label].mean(axis = 1)
        
    counts_meta = sp.csr_matrix(counts_meta)
    adata_meta = anndata.AnnData(X = counts_meta.astype(np.float32), obs = adata_s.obs)

    adata_meta.layers["counts"] = adata_meta.X.copy()
    return adata_meta


# ----------------------------------------------------------------------------------------------------------
#
# Ver. 2. Training data are saved in a joint format,stable version but only for smaller training data
# 
# ----------------------------------------------------------------------------------------------------------

'''

def divide_chunks(counts, nchunks):
    """
    Description:
    -------------
        cut the count matrix into chunks by cells

    Parameters:
    -------------
        counts: the count matrix, can be sparse csr matrix or dense matrix
        nchunks: number of chunks
    
    Return:
    -------------
        counts_chunks: list of counts matrices with length == nchunks
    """
    ncells = counts.shape[0]
    idxs = np.linspace(0, ncells, nchunks + 1)[:nchunks].astype(int)
    counts_chunks = []
    for i, idx in enumerate(idxs):
        if i == len(idxs) - 1:
            counts_chunks.append(counts[idx:, :])
        else:
            idx_next = idxs[i + 1]
            counts_chunks.append(counts[idx:(idx_next), :])
    return counts_chunks

def expr_binning(counts, nbins = 50):
    """
    Description:
    -------------
        bin the count matrix according to the expression values

    Parameters:
    -------------
        counts: the count matrix, sparse csr matrix
        nbins: number of bins
    
    Return:
    -------------
        counts_bin: csr matrix with binned values
    """
    # preprocessing step, run in chunks
    # scMulan uses the maximum expression, alternative: libsize in obs
    max_expr = counts.max(axis = 1).toarray().squeeze()
    bins = np.linspace(np.zeros_like(max_expr), max_expr, nbins, axis = 1)
    # through for loop
    counts_bin = []
    for i in range(counts.shape[0]):
        count_bin = np.digitize(counts[i, :].toarray(), bins[i, :], right=True)
        counts_bin.append(count_bin)
    counts_bin = np.vstack(counts_bin)
    return sp.csr_matrix(counts_bin)

def cell_sentance_const(counts: np.ndarray,
                        drop_zerocount: bool = True,
                        npads: int | None = None
                        ):
    ncells, nfeats = counts.shape

    if npads is None:
        # the padded length should include at leat the cls token, regardless of drop_zerocount or not
        npads = nfeats + 1    
    assert nfeats <= npads - 1

    # assign special token index: cls, mask, and pad token
    # 0~nfeats-1 are gene tokens, cls is nfeats, pad is nfeats+1
    cls_idx = nfeats
    pad_idx = nfeats + 1
    mask_idx = nfeats + 2

    # construct cell sentence
    # init expr as 0
    expr_sent = np.zeros((ncells, npads))
    # init feat as pad_idx
    feat_sent = np.full((ncells, npads), pad_idx)
    # fill in the cls_idx as the first of feat_sent
    feat_sent[:, 0] = cls_idx

    if drop_zerocount:
        for i in range(ncells):
            # use counts_bin if want interger
            expr_i = counts[i, :].toarray().squeeze()
            feat_i = np.arange(nfeats)[expr_i != 0]
            # NOTE: for cls and padding, assign expr_sentence value to be 0, fine with continuous embedding, 
            # but be careful for bin nn.embedding 
            expr_sent[i, 1:(1 + len(feat_i))] = expr_i[expr_i != 0]
            # 1 + len(feat_i) <= 257, NOTE: slicing will include the last one (index 256) if len(feat_i) + 1 > 256
            feat_sent[i, 1:(1 + len(feat_i))] = feat_i
    
    else:
        # the same across all cells, as all genes are included
        feat_sent[:, 1:(nfeats + 1)] = np.arange(nfeats)[None, :]
        expr_sent[:, 1:(nfeats + 1)] = counts.toarray()


    return sp.csr_matrix(expr_sent), sp.csr_matrix(feat_sent)

def tokenize_expr_para(counts_norm, dropzeros = True, nbins = None, npads = None, njobs = 16, nchunks = None):
    from multiprocessing import Pool
    if nchunks is None:
        nchunks = njobs
    counts_chunks = divide_chunks(counts_norm, nchunks)
    del counts_norm

    if nbins is not None:
        args = []
        for chunk_i in range(len(counts_chunks)):
            args.append((counts_chunks[chunk_i], nbins))

        # binning can be done during the training
        with Pool(processes = njobs) as pool:
            counts_bin = pool.starmap(expr_binning, args)
        counts_chunks = counts_bin
    
    args = []
    for chunk_i in range(len(counts_chunks)):
        args.append((counts_chunks[chunk_i], dropzeros, npads))
    
    with Pool(processes = njobs) as pool:
        res = pool.starmap(cell_sentance_const, args)
    
    expr_sent = sp.vstack([x[0] for x in res])
    feat_sent = sp.vstack([x[1] for x in res])

    return expr_sent, feat_sent

class sc_dataset(data.Dataset):
    def __init__(self, expr_path, gene_path, ncells, npads, labels = None, batches = None):
        """
        expr_path: stores the path to the expr data on disk
        gene_path: stores the path to the gene name of cells on disk 
        meta_path: stores the path to the meta data of cells on disk
        """
        super(sc_dataset, self).__init__()
        self.ncells = ncells
        self.npads = npads

        self.expr = np.memmap(expr_path, dtype = "float32", mode = "r", shape = (self.ncells, self.npads))
        self.gene_name = np.memmap(gene_path, dtype = "int32", mode = "r", shape = (self.ncells, self.npads))

        if labels is not None:
            assert len(labels) == ncells
        if batches is not None:
            assert len(batches) == ncells
        self.labels = labels
        self.batches = batches

    def __len__(self):
        return self.ncells
    
    def __getitem__(self, idx):
        expr = torch.tensor(self.expr[idx])
        gene = torch.tensor(self.gene_name[idx])

        if self.labels is not None:
            label = self.labels[idx]
        else:
            label = None

        if self.batches is not None:
            batch = self.batches[idx]
        else:
            batch = None
        
        return expr, gene, label, batch

class sc_dataset_chunk(data.Dataset):
    """
    Difference compared to sc_dataset, load dataset by chunks, reduce overheads
    """
    def __init__(self, expr_path, gene_path, ncells, npads, labels = None, batches = None, batch_size = 128):
        """
        expr_path: stores the path to the expr data on disk
        gene_path: stores the path to the gene name of cells on disk 
        meta_path: stores the path to the meta data of cells on disk
        """
        super(sc_dataset_chunk, self).__init__()
        self.ncells = ncells
        self.npads = npads

        self.expr = np.memmap(expr_path, dtype = "float32", mode = "r", shape = (self.ncells, self.npads))
        self.gene_name = np.memmap(gene_path, dtype = "int32", mode = "r", shape = (self.ncells, self.npads))

        if labels is not None:
            assert len(labels) == ncells
            self.labels = labels
        else:
            self.labels = None
            
        if batches is not None:
            assert len(batches) == ncells
            self.batches = batches
        else:
            self.batches = None

        self.batch_size = batch_size


    def __len__(self):
        # Return the number of batches
        return (self.ncells + self.batch_size - 1) // self.batch_size
    
    def __getitem__(self, idx):

        start_idx = idx * self.batch_size
        end_idx = min((idx + 1) * self.batch_size, self.ncells)
        expr = torch.tensor(self.expr[start_idx:end_idx])
        gene = torch.tensor(self.gene_name[start_idx:end_idx])

        sample = {"expr": expr, "gene": gene}
        if self.labels is not None:
            label = self.labels[start_idx:end_idx]
            sample["label"] = torch.tensor(label, dtype = torch.int32)

        if self.batches is not None:
            batch = self.batches[start_idx:end_idx]
            sample["batch"] = torch.tensor(batch, dtype = torch.int32)

        return sample
    

class sc_dataset_anndata(data.Dataset):
    """
    construct scdataset from anndata
    """
    def __init__(self, adata_meta, dropzeros = True, npads = None, labels = None, batches = None, batch_size = 128, njobs = 1):
        """
        expr_path: stores the path to the expr data on disk
        gene_path: stores the path to the gene name of cells on disk 
        meta_path: stores the path to the meta data of cells on disk
        """
        super(sc_dataset_anndata, self).__init__()

        self.ncells = adata_meta.shape[0]
        if npads is None:
            self.npads = adata_meta.shape[1] + 1 # number of meta-genes plus cls
        else:
            self.npads = npads
        # transform the anndata into sentences
        expr, gene_name = tokenize_expr_para(adata_meta.X, njobs = njobs, nchunks = njobs, npads = self.npads, dropzeros = dropzeros)
        self.expr = expr.toarray().astype("float32")
        self.gene_name = gene_name.toarray().astype("int32")

        if labels is not None:
            assert len(labels) == self.ncells
        if batches is not None:
            assert len(batches) == self.ncells
        self.labels = labels
        self.batches = batches

        self.batch_size = batch_size


    def __len__(self):
        # Return the number of batches
        return (self.ncells + self.batch_size - 1) // self.batch_size
    
    def __getitem__(self, idx):

        start_idx = idx * self.batch_size
        end_idx = min((idx + 1) * self.batch_size, self.ncells)
        expr = torch.tensor(self.expr[start_idx:end_idx])
        gene = torch.tensor(self.gene_name[start_idx:end_idx])

        sample = {"expr": expr, "gene": gene}
        if self.labels is not None:
            label = self.labels[start_idx:end_idx]
            sample["label"] = label

        if self.batches is not None:
            batch = self.batches[start_idx:end_idx]
            sample["batch"] = batch

        return sample

        
def preprocess_anndata(adata, feature_df, var_name = "gene_name", count_type = "mean"):
    """
    Group the genes into meta-genes in adata, according to the hard-coded grouping information
    """
    adata_s = adata.copy()
    
    if var_name == "gene_name":
        gene_ids = feature_df["feature_name"].values.squeeze()
        feature_df.index = gene_ids
    elif var_name == "ensembl_id":
        gene_ids = feature_df.index.values.squeeze()
    else:
        raise ValueError("Either gene_name or ensembl_id need to be provided in adata")

    gene_ids = np.intersect1d(adata_s.var.index.values, gene_ids)
    adata_s = adata_s[:, gene_ids]
    print(f"overlapping genes: {len(gene_ids)}")

    counts = adata_s.layers["counts"].toarray()
    if count_type == "binary":
        counts = (counts > 0).astype(int)

    meta_labels = feature_df.loc[gene_ids, "labels"].values.squeeze()
    counts_meta = np.zeros((adata_s.shape[0], len(np.unique(feature_df["labels"].values))))
    for label in np.unique(meta_labels):
        if count_type == "sum":
            counts_meta[:, label] = counts[:, meta_labels == label].sum(axis = 1)
        elif count_type == "mean":
            counts_meta[:, label] = counts[:, meta_labels == label].mean(axis = 1)
        elif count_type == "binary":
            counts_meta[:, label] = counts[:, meta_labels == label].mean(axis = 1)
        else:
            raise ValueError("count type can only be `sum', `mean', and `binary'.")
        
    counts_meta = sp.csr_matrix(counts_meta)
    adata_meta = anndata.AnnData(X = counts_meta, obs = adata_s.obs)

    adata_meta.layers["counts"] = adata_meta.X.copy().astype(int)

    # preprocess the meta-gene count
    sc.pp.normalize_total(adata_meta, target_sum = 10e4, key_added = "libsize")
    sc.pp.log1p(adata_meta)
    return adata_meta
    

'''