import torch
import numpy as np
import pandas as pd

def construct_batch_feats(adata, use_mito = False):
    if use_mito:
        mito_genes = ['MT-ND6', 'MT-CO2', 'MT-CYB', 'MT-ND2', 'MT-ND5', 'MT-CO1', 'MT-ND3', 'MT-ND4', 'MT-ND1', 'MT-ATP6', 'MT-CO3', 'MT-ND4L', 'MT-ATP8']
    else:
        mito_genes = []
    hk_genes = ['AP2M1', 'BSG', 'CD59', 'CSNK2B', 'EDF1', 'EEF2', 'GABARAP', 'HNRNPA2B1', 'HSP90AB1', 'MLF2', 'MRFAP1', 'PCBP1', 'PFDN5', 'PSAP', 'RAB11B', 'RAB1B', 'RAB7A', 'RHOA', 'UBC']
    batch_info = ["assay", "suspension_type", "libsize", "nnz", "raw_mean_nnz"]

    assert "batch_id" in adata.obs.columns
    assert "assay" in adata.obs.columns
    assert "suspension_type" in adata.obs.columns
    # calculate the remaining stats
    X = adata.layers["counts"].toarray()
    # calculate the normalized counts
    adata.layers["counts_norm"] = np.log1p(X/(np.sum(X, axis = 1, keepdims = True) + 1e-4) * 10e4)

    adata.obs["nnz"]  = (X > 0).sum(axis = 1)/X.shape[1]
    adata.obs["libsize"] = X.sum(axis = 1)
    adata.obs["raw_mean_nnz"] = np.array([np.mean(X[i, X[i, :] > 0]) for i in range(X.shape[0])])
    del X

    uniq_batches = np.unique(adata.obs["batch_id"].values)
    batch_feats = pd.DataFrame(data = 0, index = uniq_batches, columns = batch_info + mito_genes + hk_genes)
    for batch in uniq_batches:
        adata_batch = adata[adata.obs["batch_id"] == batch, :]
        assay = np.unique(adata_batch.obs["assay"].values)
        try:
            assert len(assay) == 1
        except:
            raise ValueError(f"there should be only one assay type in batch {batch}")
        batch_feats.loc[batch, "assay"] = assay[0]

        suspension_type = np.unique(adata_batch.obs["suspension_type"].values)
        try:
            assert len(suspension_type) == 1
        except:
            raise ValueError(f"there should be only one suspension type in batch {batch}")      
        batch_feats.loc[batch, "suspension_type"] = suspension_type[0]

        # log-normalization to control the scale
        batch_feats.loc[batch, "libsize"] = np.log1p(adata_batch.obs["libsize"].values.mean())
        batch_feats.loc[batch, "nnz"] = adata_batch.obs["nnz"].values.mean()
        batch_feats.loc[batch, "raw_mean_nnz"] = np.log1p(adata_batch.obs["raw_mean_nnz"].values.mean())

        # calculate for each gene
        for gene in mito_genes + hk_genes:
            if gene in adata_batch.var.index:
                batch_feats.loc[batch, gene] = adata_batch[:, gene].layers["counts_norm"].mean()
        
    return batch_feats


def tokenize_batch_feats(batch_feats, use_mito = False, expr_binsize = 1):
    """
    Description:
    -------------
        Transform the batch_feats table into the digitized table values
    """
    if use_mito:
        mito_genes = ['MT-ND6', 'MT-CO2', 'MT-CYB', 'MT-ND2', 'MT-ND5', 'MT-CO1', 'MT-ND3', 'MT-ND4', 'MT-ND1', 'MT-ATP6', 'MT-CO3', 'MT-ND4L', 'MT-ATP8']
    else:
        mito_genes = []
    hk_genes = ['AP2M1', 'BSG', 'CD59', 'CSNK2B', 'EDF1', 'EEF2', 'GABARAP', 'HNRNPA2B1', 'HSP90AB1', 'MLF2', 'MRFAP1', 'PCBP1', 'PFDN5', 'PSAP', 'RAB11B', 'RAB1B', 'RAB7A', 'RHOA', 'UBC']
    batch_info = ["assay", "suspension_type", "libsize", "nnz", "raw_mean_nnz"]

    # non-value features: assay, suspension_type, codebook hard-coded
    # are there hierarchical structure within the class? EFO save the ontology of assay, should we include the other class? 
    assay_codebook = {"10x 3' v3": 0, "Smart-seq2": 1, "10x 3' v2": 2, "10x 5' v1": 3, "10x 5' transcription profiling": 4, "Seq-Well": 5, 
                      "10x 3' v1": 6, "10x 5' v2": 7, "Seq-Well S3": 8, "Drop-seq": 9, "microwell-seq": 10, "Smart-seq v4": 11,
                      "ScaleBio single cell RNA sequencing": 12, "10x 3' transcription profiling": 13, "TruDrop": 14, "MARS-seq": 15,
                      "CEL-seq2": 16, "SPLiT-seq": 17, "BD Rhapsody Whole Transcriptome Analysis": 18, "BD Rhapsody Targeted mRNA": 19,
                      "sci-RNA-seq": 20} # , "other": -1
    suspension_codebook = {"cell": 0, "nucleus": 1}


    batch_feats = batch_feats[batch_info + mito_genes + hk_genes]
    # now tokenize the batch_feats
    batch_feats_digitize = pd.DataFrame(data = 0, index = batch_feats.index.values, columns = batch_info + mito_genes + hk_genes)

    for batch_id in batch_feats.index.values:
        assay = batch_feats.loc[batch_id, "assay"]
        suspension_type = batch_feats.loc[batch_id, "suspension_type"]

        # instead of making sure that the assay is within the category, maybe create an empty category
        if assay not in assay_codebook.keys():
            print(f"{assay} is not in the training dict")
            batch_feats_digitize.loc[batch_id, "assay"] = -1
        else:
            batch_feats_digitize.loc[batch_id, "assay"] = assay_codebook[assay]

        assert suspension_type in suspension_codebook.keys()
        batch_feats_digitize.loc[batch_id, "suspension_type"] = suspension_codebook[suspension_type]
    

    # value features:
    libsize_bucket = np.arange(0, 15.0, 1)
    batch_feats_digitize["libsize"] = np.digitize(batch_feats["libsize"], libsize_bucket, right = True)
    nnz_bucket = np.arange(0, 1, 0.1)       
    batch_feats_digitize["nnz"] = np.digitize(batch_feats["nnz"], nnz_bucket, right = True)
    expr_bucket = torch.arange(0, 10.0, expr_binsize)
    batch_feats_digitize["raw_mean_nnz"] = np.digitize(batch_feats["raw_mean_nnz"], expr_bucket, right = True)
    for gene in mito_genes + hk_genes:
        batch_feats_digitize[gene] = np.digitize(batch_feats[gene], expr_bucket, right = True)

    num_buckets = [len(assay_codebook), len(suspension_codebook), len(libsize_bucket)+1, len(nnz_bucket)+1, len(expr_bucket)+1] + [len(expr_bucket)+1] * len(mito_genes) + [len(expr_bucket)+1] * len(hk_genes)

    return batch_feats_digitize, num_buckets
    


