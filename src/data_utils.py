"""
Dataloaders

"""
import warnings
warnings.filterwarnings("ignore")
import torch
import numpy as np
import torch.utils.data as data
import scipy.sparse as sp
import gc
import random
from math import ceil, floor
import anndata
import scanpy as sc

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
    def __init__(self, adata_meta, npads = None, labels = None, batches = None, batch_size = 128, njobs = 1):
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
        expr, gene_name = tokenize_expr_para(adata_meta.X, njobs = njobs, nchunks = njobs, npads = self.npads)
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



def preprocess_anndata(adata, feature_df, var_name = "gene_name", use_bin = False):
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
    if use_bin:
        counts = (counts > 0).astype(int)

    meta_labels = feature_df.loc[gene_ids, "labels"].values.squeeze()
    counts_meta = np.zeros((adata_s.shape[0], len(np.unique(feature_df["labels"].values))))
    for label in np.unique(meta_labels):
        counts_meta[:, label] = counts[:, meta_labels == label].sum(axis = 1)
    counts_meta = sp.csr_matrix(counts_meta)
    
    adata_meta = anndata.AnnData(X = counts_meta, obs = adata_s.obs)

    adata_meta.layers["counts"] = adata_meta.X.copy().astype(int)

    # preprocess the meta-gene count
    sc.pp.normalize_total(adata_meta, target_sum = 10e4, key_added = "libsize")
    sc.pp.log1p(adata_meta)
    return adata_meta
    