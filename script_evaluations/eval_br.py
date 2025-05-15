
# In[]
# --------------------------------------------------------------------------------------------------------------
#
# Evaluate different version of vanilla model
#
# --------------------------------------------------------------------------------------------------------------
import seaborn as sns
import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
import sys, os
import scanpy as sc
sys.path.append("/net/csefiles/xzhanglab/zzhang834/LLM_KD/src")

import eval
import anndata

sns.set_theme()

# In[]
# --------------------------------------------------------------------------------------------------------------
#
# NOTE: Evaluation batch effect removal
#
# --------------------------------------------------------------------------------------------------------------
PROJECT_DIR = "/net/csefiles/xzhanglab/zzhang834/LLM_KD/"

# baseline model
# model_name = "scGPT"
# model_name = "scMulan"
# model_name = "UCE"
# model_name = "scFoundation"
# model_name = "geneformer"

# old model
# model_name = "cp_contrcb1_6_512_256_encbg_level2_1"

# new batch encoder model
# 1. batch encoder
# model_name = f"cp_6_512_256_concat_full_1"
# 2. batch encoder + contrastive
# model_name = f"cp_contrcb1_6_512_256_concat_full_1"
# 3. batch encoder + restart
# model_name = f"cp_6_512_256_concat_rawrestart_1"
# 4. batch encoder + contrastive + restart
# model_name = f"cp_contrcb1_mlm10_dyn_6_512_256_concat_rawrestart_1"
# model_dir = PROJECT_DIR + f"checkpoint/model_6_256_concat_full/{model_name}.pth"

# new vanilla model
# 1. vanilla
# model_name = f"cp_6_512_256_1"
# 2. vanilla + contrastive
# model_name = f"cp_contrcb1_mlm2_dyn_6_512_256_1"
# TODO: 3. vanilla + restart
model_name = f"cp_6_512_256_rawrestart_1"
# TODO: 4. vanilla + contrastive + restart
# model_name = f"cp_contrcb1_mlm10_dyn_6_512_256_rawrestart_1"
model_dir = PROJECT_DIR + f"checkpoint/model_6_256_nobatch/{model_name}.pth"


# some model latent dimension are too high for benchmark
reduce_dim = True

res_dir = PROJECT_DIR + f"results/zs_br/{model_name}/"
if not os.path.exists(res_dir):
    os.makedirs(res_dir)

if model_name in ["scGPT", "scMulan", "UCE", "scFoundation", "geneformer"]:
    # same set of embedding
    test_embed_dir = PROJECT_DIR + f"results/zs_annot/{model_name}/test_embed/"
else:
    test_embed_dir = PROJECT_DIR + f"results/checkpoint/{model_name}/"

for data_case in ["immune_all", "pancreas", "lung_atlas", "covid19"]:
# for data_case in ["covid19"]:
    adata_embed = anndata.read_h5ad(test_embed_dir + f"adata_embed_{data_case}.h5ad")

    if reduce_dim:
        sc.pp.pca(adata_embed, n_comps = 100)
        adata_embed.obsm["latent"] = adata_embed.obsm["X_pca"]
        print(f"reduce feature dimesion to: {adata_embed.obsm["latent"].shape[1]}")
    else:
        adata_embed.obsm["latent"] = adata_embed.X

    if "batch_id" not in adata_embed.obs:
        adata_embed.obs["batch_id"], batch_code = pd.factorize(adata_embed.obs["batch"])

    if data_case != "lung_atlas":
        scores = eval.eval_batch_correction(adata_embed, embed_key = "latent", batch_key = "batch_id", label_key = "label")
        
    else:
        meta_embed_ref = anndata.read_h5ad(PROJECT_DIR + f"results/checkpoint/stable_4_512_level2/adata_embed_{data_case}.h5ad").obs
        adata_embed.obs["patientGroup"] = meta_embed_ref["patientGroup"].values

        adata_embed_ctrl = adata_embed[adata_embed.obs["patientGroup"] == "Ctrl"]
        adata_embed_parenchyma = adata_embed[adata_embed.obs["patientGroup"] == "Parenchyma"]
        adata_embed_nan = adata_embed[adata_embed.obs["patientGroup"] == "nan"]
        scores = eval.eval_batch_correction(adata_embed, embed_key = "latent", batch_key = "batch_id", label_key = "label")
        scores["case"] = "total"
        scores_ctrl = eval.eval_batch_correction(adata_embed_ctrl, embed_key = "latent", batch_key = "batch_id", label_key = "label")
        scores_ctrl["case"] = "ctrl"
        scores_parenchyma = eval.eval_batch_correction(adata_embed_parenchyma, embed_key = "latent", batch_key = "batch_id", label_key = "label")
        scores_parenchyma["case"] = "parenchyma"
        scores_nan = eval.eval_batch_correction(adata_embed_nan, embed_key = "latent", batch_key = "batch_id", label_key = "label")
        scores_nan["case"] = "nan"
        scores = pd.concat([scores, scores_ctrl, scores_parenchyma, scores_nan], axis = 0)
            
    scores.to_csv(res_dir + f"scores_br_{data_case}_pca.csv")


# %%
