
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
import sys
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
# vanilla mlm model
# model_name = "cp_vanilla_4_512_meta_1"
# batch encoding model
# model_name = "cp_vanilla_4_512_meta_enc_trans_level0_1"
# model_name = "cp_vanilla_4_512_meta_enc_trans_wmask_level0_1"
# model_name = "cp_vanilla_4_512_meta_enc_trans_level2_1"
# model_name = "cp_vanilla_4_512_meta_enc_trans_wmask_level2_1"
# model_name = "cp_vanilla_4_512_meta_enc_level2_1"
# model_name = "cp_vanilla_4_512_meta_enc_wmask_level2_1"
# res_dir = PROJECT_DIR + f"results/checkpoint/{model_name}/"

# # finetune model
# model_name = "cp_contrcb1_4_512_meta_nobatch_1"
# model_name = "cp_contrcbproj1_4_512_meta_nobatch_1"
# model_name = "cp_contrcbproj21_4_512_meta_nobatch_1"
# model_name = "cp_contrcbproj1_4_512_meta_enc_trans_wmask_level0_1"

# model_name = "cp_contr1_4_512_meta_enc_trans_wmask_level2_1"
# model_name = "cp_contrcb1_4_512_meta_enc_trans_wmask_level2_1"
# model_name = "cp_contrcbproj1_4_512_meta_enc_trans_wmask_level2_1"
# model_name = "cp_contrcb1_4_512_meta_enc_wmask_level2_1"
# model_name = "cp_contrcbproj1_4_512_meta_enc_wmask_level2_1"

# res_dir = PROJECT_DIR + f"results/checkpoint_finetune/{model_name}/"

# # compress model
# # single-layer
# model_name = "cp_nofrz_4_512_lognorm_1"
# model_name = "cp_frztrans_4_512_lognorm_1"
# model_name = "cp_frzall_4_512_lognorm_1"
# # double layers
# model_name = "cp_frztrans_4_512_lognorm2_1"
# model_name = "cp_frzall_4_512_lognorm2_1"
# model_name = "cp_nofrz_4_512_lognorm2_1"
# model_name = "cp_hybrid_4_512_lognorm2_1"
# model_name = "cp_frztrans_4_512_raw2_1"
# model_name = "cp_nofrz_4_512_raw2_1"
# model_name = "cp_hybrid_4_512_raw2_1"
# res_dir = PROJECT_DIR + f"results/checkpoint_compress/{model_name}/"

# # compress model-finetune
# model_name = "cp_contrcbproj1_4_512_1"
model_name = "cp_contrcb1_4_512_raw2_enc_wmask_level2_1"
res_dir = PROJECT_DIR + f"results/checkpoint_compress_finetune/{model_name}/"

for use_rep in ["latent", "contr"]:
    for data_case in ["immune_all", "pancreas", "lung_atlas"]:
        adata_embed = anndata.read_h5ad(res_dir + f"adata_embed_{data_case}.h5ad")
        if f"X_umap_{use_rep}" in adata_embed.obsm.keys(): 
            if data_case != "lung_atlas":
                scores = eval.eval_batch_correction(adata_embed, embed_key = use_rep, batch_key = "batch_id", label_key = "label")
                
            else:
                adata_embed_ctrl = adata_embed[adata_embed.obs["patientGroup"] == "Ctrl"]
                adata_embed_parenchyma = adata_embed[adata_embed.obs["patientGroup"] == "Parenchyma"]
                adata_embed_nan = adata_embed[adata_embed.obs["patientGroup"] == "nan"]
                scores = eval.eval_batch_correction(adata_embed, embed_key = use_rep, batch_key = "batch_id", label_key = "label")
                scores["case"] = "total"
                scores_ctrl = eval.eval_batch_correction(adata_embed_ctrl, embed_key = use_rep, batch_key = "batch_id", label_key = "label")
                scores_ctrl["case"] = "ctrl"
                scores_parenchyma = eval.eval_batch_correction(adata_embed_parenchyma, embed_key = use_rep, batch_key = "batch_id", label_key = "label")
                scores_parenchyma["case"] = "parenchyma"
                scores_nan = eval.eval_batch_correction(adata_embed_nan, embed_key = use_rep, batch_key = "batch_id", label_key = "label")
                scores_nan["case"] = "nan"
                scores = pd.concat([scores, scores_ctrl, scores_parenchyma, scores_nan], axis = 0)
                
            scores.to_csv(res_dir + f"scores_br_{data_case}_{use_rep}.csv")

# In[]
# --------------------------------------------------------------------------------------------------------------
#
# NOTE: Compare batch-encoding methods
#
# --------------------------------------------------------------------------------------------------------------
# choice 1: batch-encode: no-encoder, encoder, encoder-mask
# choice 2: batch annots: level0, level2
model1 = "cp_vanilla_4_512_meta_enc_trans_level0_1"
model2 = "cp_vanilla_4_512_meta_enc_trans_wmask_level0_1"
model3 = "cp_vanilla_4_512_meta_enc_trans_level2_1"
model4 = "cp_vanilla_4_512_meta_enc_trans_wmask_level2_1"
model5 = "cp_vanilla_4_512_meta_1"

PROJECT_DIR = "/net/csefiles/xzhanglab/zzhang834/LLM_KD/"
res_dir = PROJECT_DIR + f"results/checkpoint/"

for data_case in ["immune_all", "pancreas", "lung_atlas"]:
    # MLM task
    val_loss_dict = {}
    val_loss_dict["level0"] = pd.read_csv(res_dir + model1 + f"/{data_case}_valloss.csv", index_col = 0)
    val_loss_dict["level0_wmask"] = pd.read_csv(res_dir + model2 + f"/{data_case}_valloss.csv", index_col = 0)
    val_loss_dict["level2"] = pd.read_csv(res_dir + model3 + f"/{data_case}_valloss.csv", index_col = 0)
    val_loss_dict["level2_wmask"] = pd.read_csv(res_dir + model4 + f"/{data_case}_valloss.csv", index_col = 0)
    val_loss_dict["nobatch"] = pd.read_csv(res_dir + model5 + f"/{data_case}_valloss.csv", index_col = 0)

    val_loss_dict["level0"]["batch_encode"] = "Encoder"
    val_loss_dict["level0"]["batch_annos"] = "Level0"
    val_loss_dict["level0"]["insert_trans"] = True
    val_loss_dict["level0_wmask"]["batch_encode"] = "Encoder-Mask"
    val_loss_dict["level0_wmask"]["batch_annos"] = "Level0"
    val_loss_dict["level0_wmask"]["insert_trans"] = True
    val_loss_dict["level2"]["batch_encode"] = "Encoder"
    val_loss_dict["level2"]["batch_annos"] = "Level2"
    val_loss_dict["level2"]["insert_trans"] = True
    val_loss_dict["level2_wmask"]["batch_encode"] = "Encoder-Mask"
    val_loss_dict["level2_wmask"]["batch_annos"] = "Level2"
    val_loss_dict["level2_wmask"]["insert_trans"] = True
    val_loss_dict["nobatch"]["batch_encode"] = "No encoder"
    val_loss_dict["nobatch"]["batch_annos"] = "No batch"
    val_loss_dict["nobatch"]["insert_trans"] = False

    val_loss = pd.concat([x for x in val_loss_dict.values()], axis = 0)

    fig = plt.figure(figsize = (8, 10))
    ax = fig.subplots(ncols = 1, nrows = 4)

    for i, mask_prob in enumerate([0.1, 0.2, 0.3, 0.4]):
        sns.barplot(data = val_loss[val_loss["mask prob"] == mask_prob], x = "batch_encode", y = "val mlm", hue = "batch_annos", ax = ax[i])
        ax[i].set_xlabel(None)
        ax[i].set_ylabel(f"Mask Prob {mask_prob}", fontsize = 17)
        for container in ax[i].containers:
            ax[i].bar_label(container, fmt = "%.3f")
        leg = ax[i].legend(loc='upper left', prop={'size': 10}, frameon = False, bbox_to_anchor=(1.04, 1), title = "Batch Label")
    fig.suptitle(data_case, fontsize = 20)
    plt.tight_layout()
    fig.savefig(res_dir + f"compare_batchencoding_{data_case}", bbox_inches = "tight")

    # embedding learning method
    br_scores_dict = {}
    # only have the latent version, not fine-tuned model
    br_scores_dict["level0"] = pd.read_csv(res_dir + model1 + f"/scores_br_{data_case}_latent.csv", index_col = 0)
    br_scores_dict["level0_wmask"] = pd.read_csv(res_dir + model2 + f"/scores_br_{data_case}_latent.csv", index_col = 0)
    br_scores_dict["level2"] = pd.read_csv(res_dir + model3 + f"/scores_br_{data_case}_latent.csv", index_col = 0)
    br_scores_dict["level2_wmask"] = pd.read_csv(res_dir + model4 + f"/scores_br_{data_case}_latent.csv", index_col = 0)
    br_scores_dict["nobatch"] = pd.read_csv(res_dir + model5 + f"/scores_br_{data_case}_latent.csv", index_col = 0)

    br_scores_dict["level0"]["batch_encode"] = "Encoder"
    br_scores_dict["level0"]["batch_annos"] = "Level0"
    br_scores_dict["level0_wmask"]["batch_encode"] = "Encoder-Mask"
    br_scores_dict["level0_wmask"]["batch_annos"] = "Level0"
    br_scores_dict["level2"]["batch_encode"] = "Encoder"
    br_scores_dict["level2"]["batch_annos"] = "Level2"
    br_scores_dict["level2_wmask"]["batch_encode"] = "Encoder-Mask"
    br_scores_dict["level2_wmask"]["batch_annos"] = "Level2"
    br_scores_dict["nobatch"]["batch_encode"] = "No encoder"
    br_scores_dict["nobatch"]["batch_annos"] = "No batch"    

    br_scores = pd.concat([x for x in br_scores_dict.values()], axis = 0)
    if data_case == "lung_atlas":
        # currently the total version is still not working well
        br_scores = br_scores[br_scores["case"] != "total"]

    fig = plt.figure(figsize = (8, 12))
    ax = fig.subplots(ncols = 1, nrows = 3)

    for i, score in enumerate(["ari", "asw", "asw (batch)"]):
        sns.barplot(data = br_scores, x = "batch_encode", y = score, hue = "batch_annos", ax = ax[i])
        ax[i].set_xlabel(None)
        ax[i].set_ylabel(score.upper(), fontsize = 15)
        for container in ax[i].containers:
            ax[i].bar_label(container, fmt = "%.3f", color = "blue")
        leg = ax[i].legend(loc='upper left', prop={'size': 10}, frameon = False, bbox_to_anchor=(1.04, 1), title = "batch annos")
    fig.suptitle(data_case, fontsize = 20)
    plt.tight_layout()
    fig.savefig(res_dir + f"compare_br_batchencoding_{data_case}", bbox_inches = "tight")


# In[]
# choice 1: batch-encode: no-encoder, encoder, encoder-mask
# choice 2: insert transformer: True, False
model1 = "cp_vanilla_4_512_meta_enc_trans_level2_1"
model2 = "cp_vanilla_4_512_meta_enc_trans_wmask_level2_1"
model3 = "cp_vanilla_4_512_meta_enc_level2_1"
model4 = "cp_vanilla_4_512_meta_enc_wmask_level2_1"
model5 = "cp_vanilla_4_512_meta_1"

PROJECT_DIR = "/net/csefiles/xzhanglab/zzhang834/LLM_KD/"
res_dir = PROJECT_DIR + f"results/checkpoint/"
for data_case in ["immune_all", "pancreas", "lung_atlas"]:
    val_loss_dict = {}
    val_loss_dict["level2"] = pd.read_csv(res_dir + model1 + f"/{data_case}_valloss.csv", index_col = 0)
    val_loss_dict["level2_wmask"] = pd.read_csv(res_dir + model2 + f"/{data_case}_valloss.csv", index_col = 0)
    val_loss_dict["level2_notrans"] = pd.read_csv(res_dir + model3 + f"/{data_case}_valloss.csv", index_col = 0)
    val_loss_dict["level2_notrans_wmask"] = pd.read_csv(res_dir + model4 + f"/{data_case}_valloss.csv", index_col = 0)
    val_loss_dict["nobatch"] = pd.read_csv(res_dir + model5 + f"/{data_case}_valloss.csv", index_col = 0)

    val_loss_dict["level2"]["batch_encode"] = "Encoder"
    val_loss_dict["level2"]["batch_annos"] = "Level2"
    val_loss_dict["level2"]["insert_trans"] = True
    val_loss_dict["level2_wmask"]["batch_encode"] = "Encoder-Mask"
    val_loss_dict["level2_wmask"]["batch_annos"] = "Level2"
    val_loss_dict["level2_wmask"]["insert_trans"] = True
    val_loss_dict["level2_notrans"]["batch_encode"] = "Encoder"
    val_loss_dict["level2_notrans"]["batch_annos"] = "Level2"
    val_loss_dict["level2_notrans"]["insert_trans"] = False
    val_loss_dict["level2_notrans_wmask"]["batch_encode"] = "Encoder-Mask"
    val_loss_dict["level2_notrans_wmask"]["batch_annos"] = "Level2"
    val_loss_dict["level2_notrans_wmask"]["insert_trans"] = False
    val_loss_dict["nobatch"]["batch_encode"] = "No encoder"
    val_loss_dict["nobatch"]["batch_annos"] = "No annos"
    val_loss_dict["nobatch"]["insert_trans"] = False

    val_loss = pd.concat([x for x in val_loss_dict.values()], axis = 0)

    fig = plt.figure(figsize = (8, 10))
    ax = fig.subplots(ncols = 1, nrows = 4)

    for i, mask_prob in enumerate([0.1, 0.2, 0.3, 0.4]):
        sns.barplot(data = val_loss[val_loss["mask prob"] == mask_prob], x = "batch_encode", y = "val mlm", hue = "insert_trans", ax = ax[i])
        ax[i].set_xlabel(None)
        ax[i].set_ylabel(f"Mask Prob {mask_prob}", fontsize = 17)
        for container in ax[i].containers:
            ax[i].bar_label(container, fmt = "%.3f")
        leg = ax[i].legend(loc='upper left', prop={'size': 10}, frameon = False, bbox_to_anchor=(1.04, 1), title = "Insert Transformer")
    fig.suptitle(data_case, fontsize = 20)
    plt.tight_layout()
    fig.savefig(res_dir + f"compare_batchencoding2_{data_case}", bbox_inches = "tight")

    # embedding learning method
    br_scores_dict = {}
    # only have the latent version, not fine-tuned model
    br_scores_dict["level2"] = pd.read_csv(res_dir + model1 + f"/scores_br_{data_case}_latent.csv", index_col = 0)
    br_scores_dict["level2_wmask"] = pd.read_csv(res_dir + model2 + f"/scores_br_{data_case}_latent.csv", index_col = 0)
    br_scores_dict["level2_notrans"] = pd.read_csv(res_dir + model3 + f"/scores_br_{data_case}_latent.csv", index_col = 0)
    br_scores_dict["level2_notrans_wmask"] = pd.read_csv(res_dir + model4 + f"/scores_br_{data_case}_latent.csv", index_col = 0)
    br_scores_dict["nobatch"] = pd.read_csv(res_dir + model5 + f"/scores_br_{data_case}_latent.csv", index_col = 0)

    br_scores_dict["level2"]["batch_encode"] = "Encoder"
    br_scores_dict["level2"]["batch_annos"] = "Level2"
    br_scores_dict["level2"]["insert_trans"] = True
    br_scores_dict["level2_wmask"]["batch_encode"] = "Encoder-Mask"
    br_scores_dict["level2_wmask"]["batch_annos"] = "Level2"
    br_scores_dict["level2_wmask"]["insert_trans"] = True
    br_scores_dict["level2_notrans"]["batch_encode"] = "Encoder"
    br_scores_dict["level2_notrans"]["batch_annos"] = "Level2"
    br_scores_dict["level2_notrans"]["insert_trans"] = False
    br_scores_dict["level2_notrans_wmask"]["batch_encode"] = "Encoder-Mask"
    br_scores_dict["level2_notrans_wmask"]["batch_annos"] = "Level2"
    br_scores_dict["level2_notrans_wmask"]["insert_trans"] = False
    br_scores_dict["nobatch"]["batch_encode"] = "No encoder"
    br_scores_dict["nobatch"]["batch_annos"] = "No annos"
    br_scores_dict["nobatch"]["insert_trans"] = False

    br_scores = pd.concat([x for x in br_scores_dict.values()], axis = 0)
    if data_case == "lung_atlas":
        # currently the total version is still not working well
        br_scores = br_scores[br_scores["case"] != "total"]

    fig = plt.figure(figsize = (8, 12))
    ax = fig.subplots(ncols = 1, nrows = 3)

    for i, score in enumerate(["ari", "asw", "asw (batch)"]):
        sns.barplot(data = br_scores, x = "batch_encode", y = score, hue = "insert_trans", ax = ax[i])
        ax[i].set_xlabel(None)
        ax[i].set_ylabel(score.upper(), fontsize = 15)
        for container in ax[i].containers:
            ax[i].bar_label(container, fmt = "%.3f", color = "blue")
        leg = ax[i].legend(loc='upper left', prop={'size': 10}, frameon = False, bbox_to_anchor=(1.04, 1), title = "Insert Transformer")
    fig.suptitle(data_case, fontsize = 20)
    plt.tight_layout()
    fig.savefig(res_dir + f"compare_br_batchencoding2_{data_case}", bbox_inches = "tight")



# In[]
# --------------------------------------------------------------------------------------------------------------
#
# NOTE: Compare compression
#
# --------------------------------------------------------------------------------------------------------------
# three training settings: 1. freeze all, 2. freeze transformer, 3. no freeze
# two decompression: 1. 1 layer, 2. 2 layers
# reconstruction: 1. log-norm, 2. raw

# Compare MLM, NOTE: comparison only be within the log-norm case, cannot compare across
model1 = "cp_frzall_4_512_lognorm_1"
model2 = "cp_frzall_4_512_lognorm2_1"
model3 = "cp_frztrans_4_512_lognorm_1"
model4 = "cp_frztrans_4_512_lognorm2_1"
model5 = "cp_nofrz_4_512_lognorm_1"
model6 = "cp_nofrz_4_512_lognorm2_1"
model7 = "cp_frztrans_4_512_raw2_1"
model8 = "cp_nofrz_4_512_raw2_1"
model9 = "cp_hybrid_4_512_lognorm2_1"
model10 = "cp_hybrid_4_512_raw2_1"

PROJECT_DIR = "/net/csefiles/xzhanglab/zzhang834/LLM_KD/"
res_dir = PROJECT_DIR + f"results/checkpoint_compress/"
for data_case in ["immune_all", "pancreas", "lung_atlas"]:
    val_loss_dict = {}
    val_loss_dict["frzall_lognorm1"] = pd.read_csv(res_dir + model1 + f"/{data_case}_valloss.csv", index_col = 0)
    val_loss_dict["frzall_lognorm2"] = pd.read_csv(res_dir + model2 + f"/{data_case}_valloss.csv", index_col = 0)
    val_loss_dict["frztrans_lognorm1"] = pd.read_csv(res_dir + model3 + f"/{data_case}_valloss.csv", index_col = 0)
    val_loss_dict["frztrans_lognorm2"] = pd.read_csv(res_dir + model4 + f"/{data_case}_valloss.csv", index_col = 0)
    val_loss_dict["nofrz_lognorm1"] = pd.read_csv(res_dir + model5 + f"/{data_case}_valloss.csv", index_col = 0)
    val_loss_dict["nofrz_lognorm2"] = pd.read_csv(res_dir + model6 + f"/{data_case}_valloss.csv", index_col = 0)
    val_loss_dict["frztrans_raw2"] = pd.read_csv(res_dir + model7 + f"/{data_case}_valloss.csv", index_col = 0)
    val_loss_dict["nofrz_raw2"] = pd.read_csv(res_dir + model8 + f"/{data_case}_valloss.csv", index_col = 0)
    val_loss_dict["hybrid_lognorm2"] = pd.read_csv(res_dir + model9 + f"/{data_case}_valloss.csv", index_col = 0)
    val_loss_dict["hybrid_raw2"] = pd.read_csv(res_dir + model10 + f"/{data_case}_valloss.csv", index_col = 0)


    val_loss_dict["frzall_lognorm1"]["train"] = "frz-all"
    val_loss_dict["frzall_lognorm1"]["recon"] = "Lognorm-1layer"
    val_loss_dict["frzall_lognorm2"]["train"] = "frz-all"
    val_loss_dict["frzall_lognorm2"]["recon"] = "Lognorm-2layers"
    val_loss_dict["frztrans_lognorm1"]["train"] = "frz-trans"
    val_loss_dict["frztrans_lognorm1"]["recon"] = "Lognorm-1layer"
    val_loss_dict["frztrans_lognorm2"]["train"] = "frz-trans"
    val_loss_dict["frztrans_lognorm2"]["recon"] = "Lognorm-2layers"
    val_loss_dict["nofrz_lognorm1"]["train"] = "no-frz"
    val_loss_dict["nofrz_lognorm1"]["recon"] = "Lognorm-1layer"
    val_loss_dict["nofrz_lognorm2"]["train"] = "no-frz"
    val_loss_dict["nofrz_lognorm2"]["recon"] = "Lognorm-2layers"
    val_loss_dict["frztrans_raw2"]["train"] = "frz-trans"
    val_loss_dict["frztrans_raw2"]["recon"] = "Raw-2layers"
    val_loss_dict["nofrz_raw2"]["train"] = "no-frz"
    val_loss_dict["nofrz_raw2"]["recon"] = "Raw-2layers"
    val_loss_dict["hybrid_lognorm2"]["train"] = "hybrid"
    val_loss_dict["hybrid_lognorm2"]["recon"] = "Lognorm-2layers"
    val_loss_dict["hybrid_raw2"]["train"] = "hybrid"
    val_loss_dict["hybrid_raw2"]["recon"] = "Raw-2layers"

    val_loss = pd.concat([x for x in val_loss_dict.values()], axis = 0)
    val_loss1 = val_loss[val_loss["recon"] != "Raw-2layers"]
    val_loss2 = val_loss[val_loss["recon"] == "Raw-2layers"]

    fig = plt.figure(figsize = (10, 10))
    ax = fig.subplots(ncols = 1, nrows = 4)

    for i, mask_prob in enumerate([0.1, 0.2, 0.3, 0.4]):
        sns.barplot(data = val_loss1[val_loss1["mask prob"] == mask_prob], x = "train", y = "val mlm", hue = "recon", ax = ax[i])
        ax[i].set_xlabel(None)
        ax[i].set_ylabel(f"Mask Prob {mask_prob}", fontsize = 17)
        for container in ax[i].containers:
            ax[i].bar_label(container, fmt = "%.1f", color = "blue")
        leg = ax[i].legend(loc='upper left', prop={'size': 10}, frameon = False, bbox_to_anchor=(1.04, 1), title = "Recon")
    fig.suptitle(data_case, fontsize = 20)
    plt.tight_layout()
    fig.savefig(res_dir + f"compare_compress_{data_case}", bbox_inches = "tight")

    fig = plt.figure(figsize = (10, 10))
    ax = fig.subplots(ncols = 1, nrows = 4)

    for i, mask_prob in enumerate([0.1, 0.2, 0.3, 0.4]):
        sns.barplot(data = val_loss2[val_loss2["mask prob"] == mask_prob], x = "train", y = "val mlm", hue = "recon", ax = ax[i])
        ax[i].set_xlabel(None)
        ax[i].set_ylabel(f"Mask Prob {mask_prob}", fontsize = 17)
        for container in ax[i].containers:
            ax[i].bar_label(container, fmt = "%.1f", color = "blue")
        leg = ax[i].legend(loc='upper left', prop={'size': 10}, frameon = False, bbox_to_anchor=(1.04, 1), title = "Recon")
    fig.suptitle(data_case, fontsize = 20)
    plt.tight_layout()
    fig.savefig(res_dir + f"compare_compress_nb_{data_case}", bbox_inches = "tight")

    # embedding learning method
    br_scores_dict = {}
    # only have the latent version, not fine-tuned model
    br_scores_dict["frzall_lognorm1"] = pd.read_csv(res_dir + model1 + f"/scores_br_{data_case}_latent.csv", index_col = 0)
    br_scores_dict["frzall_lognorm2"] = pd.read_csv(res_dir + model2 + f"/scores_br_{data_case}_latent.csv", index_col = 0)
    br_scores_dict["frztrans_lognorm1"] = pd.read_csv(res_dir + model3 + f"/scores_br_{data_case}_latent.csv", index_col = 0)
    br_scores_dict["frztrans_lognorm2"] = pd.read_csv(res_dir + model4 + f"/scores_br_{data_case}_latent.csv", index_col = 0)
    br_scores_dict["nofrz_lognorm1"] = pd.read_csv(res_dir + model5 + f"/scores_br_{data_case}_latent.csv", index_col = 0)
    br_scores_dict["nofrz_lognorm2"] = pd.read_csv(res_dir + model6 + f"/scores_br_{data_case}_latent.csv", index_col = 0)
    br_scores_dict["frztrans_raw2"] = pd.read_csv(res_dir + model7 + f"/scores_br_{data_case}_latent.csv", index_col = 0)
    br_scores_dict["nofrz_raw2"] = pd.read_csv(res_dir + model8 + f"/scores_br_{data_case}_latent.csv", index_col = 0)
    br_scores_dict["hybrid_lognorm2"] = pd.read_csv(res_dir + model9 + f"/scores_br_{data_case}_latent.csv", index_col = 0)
    br_scores_dict["hybrid_raw2"] = pd.read_csv(res_dir + model10 + f"/scores_br_{data_case}_latent.csv", index_col = 0)


    br_scores_dict["frzall_lognorm1"]["train"] = "frz-all"
    br_scores_dict["frzall_lognorm1"]["recon"] = "Lognorm-1layer"
    br_scores_dict["frzall_lognorm2"]["train"] = "frz-all"
    br_scores_dict["frzall_lognorm2"]["recon"] = "Lognorm-2layers"
    br_scores_dict["frztrans_lognorm1"]["train"] = "frz-trans"
    br_scores_dict["frztrans_lognorm1"]["recon"] = "Lognorm-1layer"
    br_scores_dict["frztrans_lognorm2"]["train"] = "frz-trans"
    br_scores_dict["frztrans_lognorm2"]["recon"] = "Lognorm-2layers"
    br_scores_dict["nofrz_lognorm1"]["train"] = "no-frz"
    br_scores_dict["nofrz_lognorm1"]["recon"] = "Lognorm-1layer"
    br_scores_dict["nofrz_lognorm2"]["train"] = "no-frz"
    br_scores_dict["nofrz_lognorm2"]["recon"] = "Lognorm-2layers"
    br_scores_dict["frztrans_raw2"]["train"] = "frz-trans"
    br_scores_dict["frztrans_raw2"]["recon"] = "Raw-2layers"
    br_scores_dict["nofrz_raw2"]["train"] = "no-frz"
    br_scores_dict["nofrz_raw2"]["recon"] = "Raw-2layers"
    br_scores_dict["hybrid_lognorm2"]["train"] = "hybrid"
    br_scores_dict["hybrid_lognorm2"]["recon"] = "Lognorm-2layers"
    br_scores_dict["hybrid_raw2"]["train"] = "hybrid"
    br_scores_dict["hybrid_raw2"]["recon"] = "Raw-2layers"

    br_scores = pd.concat([x for x in br_scores_dict.values()], axis = 0)
    if data_case == "lung_atlas":
        # currently the total version is still not working well
        br_scores = br_scores[br_scores["case"] != "total"]

    fig = plt.figure(figsize = (12, 12))
    ax = fig.subplots(ncols = 1, nrows = 3)

    for i, score in enumerate(["ari", "asw", "asw (batch)"]):
        sns.barplot(data = br_scores, x = "train", y = score, hue = "recon", ax = ax[i])
        ax[i].set_xlabel(None)
        ax[i].set_ylabel(score.upper(), fontsize = 15)
        for container in ax[i].containers:
            ax[i].bar_label(container, fmt = "%.3f", color = "blue")
        leg = ax[i].legend(loc='upper left', prop={'size': 10}, frameon = False, bbox_to_anchor=(1.04, 1), title = "Recon")
    fig.suptitle(data_case, fontsize = 20)
    plt.tight_layout()
    fig.savefig(res_dir + f"compare_br_compress_{data_case}", bbox_inches = "tight")


# In[]
# --------------------------------------------------------------------------------------------------------------
#
# NOTE: Compare contrastive-finetuning
#
# --------------------------------------------------------------------------------------------------------------
# Compare contrcb fine-tunning
# NOTE: mask_prob should not be too large for fine-tunning
model1 = "cp_vanilla_4_512_meta_1"

model2 = "cp_contrcb1_4_512_meta_nobatch_1"
model3 = "cp_contrcbproj1_4_512_meta_nobatch_1"
model4 = "cp_contrcbproj21_4_512_meta_nobatch_1"

PROJECT_DIR = "/net/csefiles/xzhanglab/zzhang834/LLM_KD/"
res_dir_vanilla = PROJECT_DIR + f"results/checkpoint/"
res_dir_finetune = PROJECT_DIR + f"results/checkpoint_finetune/"

for data_case in ["immune_all", "pancreas", "lung_atlas"]:
    # MLM task
    val_loss_dict = {}
    val_loss_dict["vanilla"] = pd.read_csv(res_dir_vanilla + model1 + f"/{data_case}_valloss.csv", index_col = 0)
    val_loss_dict["contrcb"] = pd.read_csv(res_dir_finetune + model2 + f"/{data_case}_valloss.csv", index_col = 0)
    val_loss_dict["contrcb-proj"] = pd.read_csv(res_dir_finetune + model3 + f"/{data_case}_valloss.csv", index_col = 0)
    val_loss_dict["contrcb-proj2"] = pd.read_csv(res_dir_finetune + model4 + f"/{data_case}_valloss.csv", index_col = 0)

    val_loss_dict["vanilla"]["model"] = "Orig"
    val_loss_dict["contrcb"]["model"] = "Contr-cb"
    val_loss_dict["contrcb-proj"]["model"] = "Contr-cb-proj"
    val_loss_dict["contrcb-proj2"]["model"] = "Contr-cb-proj2"

    val_loss = pd.concat([x for x in val_loss_dict.values()], axis = 0)

    fig = plt.figure(figsize = (6, 10))
    ax = fig.subplots(ncols = 1, nrows = 4)

    for i, mask_prob in enumerate([0.1, 0.2, 0.3, 0.4]):
        sns.barplot(data = val_loss[val_loss["mask prob"] == mask_prob], x = "model", y = "val mlm", ax = ax[i])
        ax[i].set_xlabel(None)
        ax[i].set_ylabel(f"Mask Prob {mask_prob}", fontsize = 17)
        for container in ax[i].containers:
            ax[i].bar_label(container, fmt = "%.3f")
        # leg = ax[i].legend(loc='upper left', prop={'size': 10}, frameon = False, bbox_to_anchor=(1.04, 1), title = "Batch Label")
    fig.suptitle(data_case, fontsize = 20)
    plt.tight_layout()
    fig.savefig(res_dir_finetune + f"compare_contrastive_{data_case}", bbox_inches = "tight")

    # embedding learning method
    br_scores = []
    for use_rep in ["contr", "latent"]:
        br_scores_dict = {}
        # only latent for vanilla
        if use_rep == "latent":
            br_scores_dict["vanilla"] = pd.read_csv(res_dir_vanilla + model1 + f"/scores_br_{data_case}_latent.csv", index_col = 0)
            br_scores_dict["vanilla"]["model"] = "Orig"
            br_scores_dict["vanilla"]["rep"] = "latent"
            br_scores_dict["contrcb"] = pd.read_csv(res_dir_finetune + model2 + f"/scores_br_{data_case}_latent.csv", index_col = 0)
            br_scores_dict["contrcb"]["model"] = "Contr-cb"
            br_scores_dict["contrcb"]["rep"] = "latent"

        br_scores_dict["contrcb-proj"] = pd.read_csv(res_dir_finetune + model3 + f"/scores_br_{data_case}_{use_rep}.csv", index_col = 0)
        br_scores_dict["contrcb-proj2"] = pd.read_csv(res_dir_finetune + model4 + f"/scores_br_{data_case}_{use_rep}.csv", index_col = 0)        
        br_scores_dict["contrcb-proj"]["model"] = "Contr-cb-proj"
        br_scores_dict["contrcb-proj2"]["model"] = "Contr-cb-proj2"
        
        br_scores_dict["contrcb-proj"]["rep"] = use_rep
        br_scores_dict["contrcb-proj2"]["rep"] = use_rep
        br_scores.append(pd.concat([x for x in br_scores_dict.values()], axis = 0))
    br_scores = pd.concat(br_scores, axis = 0)

    if data_case == "lung_atlas":
        # currently the total version is still not working well
        br_scores = br_scores[br_scores["case"] != "total"]

    fig = plt.figure(figsize = (8, 12))
    ax = fig.subplots(ncols = 1, nrows = 3)

    for i, score in enumerate(["ari", "asw", "asw (batch)"]):
        sns.barplot(data = br_scores, x = "model", y = score, hue = "rep", ax = ax[i])
        ax[i].set_xlabel(None)
        ax[i].set_ylabel(score.upper(), fontsize = 15)
        for container in ax[i].containers:
            ax[i].bar_label(container, fmt = "%.3f", color = "blue")
        leg = ax[i].legend(loc='upper left', prop={'size': 10}, frameon = False, bbox_to_anchor=(1.04, 1), title = "Rep")
    fig.suptitle(data_case, fontsize = 20)
    plt.tight_layout()
    fig.savefig(res_dir_finetune + f"compare_br_contrastive_{data_case}", bbox_inches = "tight")


# In[]
# read score of scGPT
br_scores_scgpt = pd.read_csv(PROJECT_DIR + "results/scGPT/scores_scib.csv", index_col = 0)
br_scores_scgpt["data_case"] = ["immune_all", "pancreas", "lung_atlas"]
br_scores_scgpt["model"] = "scGPT"
br_scores_scgpt["rep"] = "latent"
br_scores_scgpt["case"] = "total"

br_scores_scvi = pd.read_csv(PROJECT_DIR + "results/scvi/scores_scib.csv", index_col = 0)
br_scores_scvi["data_case"] = ["immune_all", "pancreas", "lung_atlas"]
br_scores_scvi["model"] = "scVI"
br_scores_scvi["rep"] = "latent"
br_scores_scvi["case"] = "total"

for data_case in ["immune_all", "pancreas", "lung_atlas"]:
    # embedding learning method
    br_scores = []
    for use_rep in ["contr", "latent"]:
        br_scores_dict = {}
        # only latent for vanilla
        if use_rep == "latent":
            pass
            # br_scores_dict["vanilla"] = pd.read_csv(res_dir_vanilla + model1 + f"/scores_br_{data_case}_latent.csv", index_col = 0)
            # br_scores_dict["vanilla"]["model"] = "Orig"
            # br_scores_dict["vanilla"]["rep"] = "latent"
            # br_scores_dict["contrcb"] = pd.read_csv(res_dir_finetune + model2 + f"/scores_br_{data_case}_latent.csv", index_col = 0)
            # br_scores_dict["contrcb"]["model"] = "Contr-cb"
            # br_scores_dict["contrcb"]["rep"] = "latent"
        else:
            br_scores_dict["contrcb-proj"] = pd.read_csv(res_dir_finetune + model3 + f"/scores_br_{data_case}_{use_rep}.csv", index_col = 0)
            br_scores_dict["contrcb-proj2"] = pd.read_csv(res_dir_finetune + model4 + f"/scores_br_{data_case}_{use_rep}.csv", index_col = 0)        
            br_scores_dict["contrcb-proj"]["model"] = "Contr-cb-proj"
            br_scores_dict["contrcb-proj2"]["model"] = "Contr-cb-proj2"
            
            br_scores_dict["contrcb-proj"]["rep"] = use_rep
            br_scores_dict["contrcb-proj2"]["rep"] = use_rep
            br_scores.append(pd.concat([x for x in br_scores_dict.values()], axis = 0))
    br_scores = pd.concat(br_scores, axis = 0)

    br_scores = pd.concat([br_scores,
                           br_scores_scvi.loc[br_scores_scvi["data_case"] == data_case, br_scores.columns],
                           br_scores_scgpt.loc[br_scores_scgpt["data_case"] == data_case, br_scores.columns]], axis = 0)


    if data_case == "lung_atlas":
        # currently the total version is still not working well
        br_scores = br_scores[br_scores["case"] == "total"]

    fig = plt.figure(figsize = (6, 12))
    ax = fig.subplots(ncols = 1, nrows = 3)

    for i, score in enumerate(["ari", "asw", "asw (batch)"]):
        sns.barplot(data = br_scores, x = "model", y = score, ax = ax[i], width = 0.3)
        ax[i].set_xlabel(None)
        ax[i].set_ylabel(score.upper(), fontsize = 15)
        for container in ax[i].containers:
            ax[i].bar_label(container, fmt = "%.3f", color = "blue")
        # leg = ax[i].legend(loc='upper left', prop={'size': 10}, frameon = False, bbox_to_anchor=(1.04, 1), title = "Rep")
        ax[i].set_ylim([0.2, 1])
    fig.suptitle(data_case, fontsize = 20)
    plt.tight_layout()
    fig.savefig(res_dir_finetune + f"compare_br_contrastive_full_{data_case}", bbox_inches = "tight")



# In[]
# Compare fine-tunning on wmask model
model1 = "cp_vanilla_4_512_meta_enc_trans_wmask_level0_1"
model2 = "cp_vanilla_4_512_meta_enc_trans_wmask_level2_1"
model3 = "cp_contrcbproj1_4_512_meta_enc_trans_wmask_level0_1"
model4 = "cp_contrcbproj1_4_512_meta_enc_trans_wmask_level2_1"
model5 = "cp_contr1_4_512_meta_enc_trans_wmask_level2_1"
model6 = "cp_contrcb1_4_512_meta_enc_trans_wmask_level2_1"
model7 = "cp_contrcb1_4_512_meta_enc_wmask_level2_1"
model8 = "cp_contrcbproj1_4_512_meta_enc_wmask_level2_1"

model9 = "cp_contrcbproj1_4_512_meta_nobatch_1"

PROJECT_DIR = "/net/csefiles/xzhanglab/zzhang834/LLM_KD/"
res_dir_vanilla = PROJECT_DIR + f"results/checkpoint/"
res_dir_finetune = PROJECT_DIR + f"results/checkpoint_finetune/"

for data_case in ["immune_all", "pancreas", "lung_atlas"]:
    # MLM task
    val_loss_dict = {}
    val_loss_dict["vanilla-level0"] = pd.read_csv(res_dir_vanilla + model1 + f"/{data_case}_valloss.csv", index_col = 0)
    val_loss_dict["vanilla-level2"] = pd.read_csv(res_dir_vanilla + model2 + f"/{data_case}_valloss.csv", index_col = 0)
    val_loss_dict["contrcb-proj-level0"] = pd.read_csv(res_dir_finetune + model3 + f"/{data_case}_valloss.csv", index_col = 0)
    val_loss_dict["contrcb-proj-level2"] = pd.read_csv(res_dir_finetune + model4 + f"/{data_case}_valloss.csv", index_col = 0)

    val_loss_dict["contr-level2"] = pd.read_csv(res_dir_finetune + model5 + f"/{data_case}_valloss.csv", index_col = 0)
    val_loss_dict["contrcb-level2"] = pd.read_csv(res_dir_finetune + model6 + f"/{data_case}_valloss.csv", index_col = 0)
    val_loss_dict["contrcb-notrans-level2"] = pd.read_csv(res_dir_finetune + model7 + f"/{data_case}_valloss.csv", index_col = 0)
    val_loss_dict["contrcb-proj-notrans-level2"] = pd.read_csv(res_dir_finetune + model8 + f"/{data_case}_valloss.csv", index_col = 0)
    
    val_loss_dict["contrcb-proj-nobatch"] = pd.read_csv(res_dir_finetune + model9 + f"/{data_case}_valloss.csv", index_col = 0)

    val_loss_dict["vanilla-level0"]["model"] = "vanilla"
    val_loss_dict["vanilla-level0"]["batch"] = "level0"
    val_loss_dict["vanilla-level2"]["model"] = "vanilla"
    val_loss_dict["vanilla-level2"]["batch"] = "level2"
    
    val_loss_dict["contrcb-proj-level0"]["model"] = "contrcb-proj"
    val_loss_dict["contrcb-proj-level0"]["batch"] = "level0"
    val_loss_dict["contrcb-proj-level2"]["model"] = "contrcb-proj"
    val_loss_dict["contrcb-proj-level2"]["batch"] = "level2"

    val_loss_dict["contr-level2"]["model"] = "contr"
    val_loss_dict["contr-level2"]["batch"] = "level2"
    val_loss_dict["contrcb-level2"]["model"] = "contrcb"
    val_loss_dict["contrcb-level2"]["batch"] = "level2"
    
    val_loss_dict["contrcb-notrans-level2"]["model"] = "contrcb-notrans"
    val_loss_dict["contrcb-notrans-level2"]["batch"] = "level2"
    val_loss_dict["contrcb-proj-notrans-level2"]["model"] = "contrcb-proj-notrans"
    val_loss_dict["contrcb-proj-notrans-level2"]["batch"] = "level2"

    val_loss_dict["contrcb-proj-nobatch"]["model"] = "contrcb-proj"
    val_loss_dict["contrcb-proj-nobatch"]["batch"] = "none"
    
    val_loss = pd.concat([x for x in val_loss_dict.values()], axis = 0)

    fig = plt.figure(figsize = (13, 10))
    ax = fig.subplots(ncols = 1, nrows = 4)

    for i, mask_prob in enumerate([0.1, 0.2, 0.3, 0.4]):
        sns.barplot(data = val_loss[val_loss["mask prob"] == mask_prob], x = "model", y = "val mlm", hue = "batch", ax = ax[i])
        ax[i].set_xlabel(None)
        ax[i].set_ylabel(f"Mask Prob {mask_prob}", fontsize = 17)
        for container in ax[i].containers:
            ax[i].bar_label(container, fmt = "%.3f")
        leg = ax[i].legend(loc='upper left', prop={'size': 10}, frameon = False, bbox_to_anchor=(1.04, 1), title = "Batch Label")
    fig.suptitle(data_case, fontsize = 20)
    plt.tight_layout()
    fig.savefig(res_dir_finetune + f"compare_batchenc_{data_case}", bbox_inches = "tight")


    # embedding learning method
    br_scores = []
    for use_rep in ["contr", "latent"]:
        br_scores_dict = {}
        # only latent for vanilla
        if use_rep == "latent":
            br_scores_dict["vanilla-level0"] = pd.read_csv(res_dir_vanilla + model1 + f"/scores_br_{data_case}_latent.csv", index_col = 0)
            br_scores_dict["vanilla-level0"]["model"] = "vanilla"
            br_scores_dict["vanilla-level0"]["batch"] = "level0"
            br_scores_dict["vanilla-level2"] = pd.read_csv(res_dir_vanilla + model2 + f"/scores_br_{data_case}_latent.csv", index_col = 0)
            br_scores_dict["vanilla-level2"]["model"] = "vanilla"
            br_scores_dict["vanilla-level2"]["batch"] = "level2"

            br_scores_dict["contr-level2"] = pd.read_csv(res_dir_finetune + model5 + f"/scores_br_{data_case}_latent.csv", index_col = 0)
            br_scores_dict["contr-level2"]["model"] = "contr"
            br_scores_dict["contr-level2"]["batch"] = "level2"
            br_scores_dict["contrcb-level2"] = pd.read_csv(res_dir_finetune + model6 + f"/scores_br_{data_case}_latent.csv", index_col = 0)
            br_scores_dict["contrcb-level2"]["model"] = "contrcb"
            br_scores_dict["contrcb-level2"]["batch"] = "level2"
            
            br_scores_dict["contrcb-notrans-level2"] = pd.read_csv(res_dir_finetune + model7 + f"/scores_br_{data_case}_latent.csv", index_col = 0)
            br_scores_dict["contrcb-notrans-level2"]["model"] = "contrcb-notrans"
            br_scores_dict["contrcb-notrans-level2"]["batch"] = "level2"

        else:
            br_scores_dict["contrcb-proj-level0"] = pd.read_csv(res_dir_finetune + model3 + f"/scores_br_{data_case}_{use_rep}.csv", index_col = 0)
            br_scores_dict["contrcb-proj-level2"] = pd.read_csv(res_dir_finetune + model4 + f"/scores_br_{data_case}_{use_rep}.csv", index_col = 0)        
            br_scores_dict["contrcb-proj-notrans-level2"] = pd.read_csv(res_dir_finetune + model8 + f"/scores_br_{data_case}_{use_rep}.csv", index_col = 0)
            br_scores_dict["contrcb-proj-nobatch"] = pd.read_csv(res_dir_finetune + model9 + f"/scores_br_{data_case}_{use_rep}.csv", index_col = 0)        
            br_scores_dict["contrcb-proj-level0"]["model"] = "contrcb-proj"
            br_scores_dict["contrcb-proj-level0"]["batch"] = "level0"
            br_scores_dict["contrcb-proj-level2"]["model"] = "contrcb-proj"
            br_scores_dict["contrcb-proj-level2"]["batch"] = "level2"
            br_scores_dict["contrcb-proj-notrans-level2"]["model"] = "contrcb-proj-notrans"
            br_scores_dict["contrcb-proj-notrans-level2"]["batch"] = "level2"
            br_scores_dict["contrcb-proj-nobatch"]["model"] = "contrcb-proj"
            br_scores_dict["contrcb-proj-nobatch"]["batch"] = "none"
            
        br_scores.append(pd.concat([x for x in br_scores_dict.values()], axis = 0))
    br_scores = pd.concat(br_scores, axis = 0)

    if data_case == "lung_atlas":
        # currently the total version is still not working well
        br_scores = br_scores[br_scores["case"] != "total"]

    fig = plt.figure(figsize = (13, 12))
    ax = fig.subplots(ncols = 1, nrows = 3)

    for i, score in enumerate(["ari", "asw", "asw (batch)"]):
        sns.barplot(data = br_scores, x = "model", y = score, ax = ax[i], hue = "batch")
        ax[i].set_xlabel(None)
        ax[i].set_ylabel(score.upper(), fontsize = 15)
        for container in ax[i].containers:
            ax[i].bar_label(container, fmt = "%.3f", color = "blue")
        leg = ax[i].legend(loc='upper left', prop={'size': 10}, frameon = False, bbox_to_anchor=(1.04, 1), title = "Rep")
    fig.suptitle(data_case, fontsize = 20)
    plt.tight_layout()
    fig.savefig(res_dir_finetune + f"compare_br_batchenc_{data_case}", bbox_inches = "tight")


# In[]
# Compare final model
model1 = "cp_contrcbproj1_4_512_1"
model2 = "cp_contrcb1_4_512_raw2_enc_wmask_level2_1"

PROJECT_DIR = "/net/csefiles/xzhanglab/zzhang834/LLM_KD/"
res_dir = PROJECT_DIR + f"results/checkpoint_compress_finetune/"
for data_case in ["immune_all", "pancreas", "lung_atlas"]:
    # MLM task
    val_loss_dict = {}
    val_loss_dict["nobatch"] = pd.read_csv(res_dir + model1 + f"/{data_case}_valloss.csv", index_col = 0)
    val_loss_dict["nobatch"]["model"] = "no-batch"
    val_loss_dict["batch"] = pd.read_csv(res_dir + model2 + f"/{data_case}_valloss.csv", index_col = 0)
    val_loss_dict["batch"]["model"] = "batch"

    val_loss = pd.concat([x for x in val_loss_dict.values()], axis = 0)

    fig = plt.figure(figsize = (6, 10))
    ax = fig.subplots(ncols = 1, nrows = 4)

    for i, mask_prob in enumerate([0.1, 0.2, 0.3, 0.4]):
        sns.barplot(data = val_loss[val_loss["mask prob"] == mask_prob], x = "model", y = "val mlm", ax = ax[i])
        ax[i].set_xlabel(None)
        ax[i].set_ylabel(f"Mask Prob {mask_prob}", fontsize = 17)
        for container in ax[i].containers:
            ax[i].bar_label(container, fmt = "%.3f")
    fig.suptitle(data_case, fontsize = 20)
    plt.tight_layout()
    fig.savefig(res_dir + f"compare_{data_case}", bbox_inches = "tight")

    br_scores_dict = {}
    br_scores_dict["nobatch"] = pd.read_csv(res_dir + model1 + f"/scores_br_{data_case}_contr.csv", index_col = 0)
    br_scores_dict["nobatch"]["model"] = "nobatch"
    br_scores_dict["batch"] = pd.read_csv(res_dir + model2 + f"/scores_br_{data_case}_latent.csv", index_col = 0)
    br_scores_dict["batch"]["model"] = "batch"

    br_scores = pd.concat([x for x in br_scores_dict.values()], axis = 0)

    if data_case == "lung_atlas":
        # currently the total version is still not working well
        br_scores = br_scores[br_scores["case"] != "total"]

    fig = plt.figure(figsize = (6, 10))
    ax = fig.subplots(ncols = 1, nrows = 3)

    for i, score in enumerate(["ari", "asw", "asw (batch)"]):
        sns.barplot(data = br_scores, x = "model", y = score, ax = ax[i])
        ax[i].set_xlabel(None)
        ax[i].set_ylabel(score.upper(), fontsize = 15)
        for container in ax[i].containers:
            ax[i].bar_label(container, fmt = "%.3f", color = "blue")
    fig.suptitle(data_case, fontsize = 20)
    plt.tight_layout()
    fig.savefig(res_dir + f"compare_br_{data_case}", bbox_inches = "tight")

# In[]
# --------------------------------------------------------------------------------------------------------------
#
# NOTE: Old comparison. batch-encoding/insert-transformer/expr-encoding
# Summary:
# If we consistently evaluate the model performance using the mask prediction accuracy (should be a comprehensive metric compared to embedding alignment)
# We evaluate the model performance on stand alone dataset: immune_all, lung_atlas
# 1. using fourier encoding is better than using mlp for continuous expr encoding
# 2. use the onehot batch encoding is the worst (Possible Reason: the batch encoding is all-zero on test)
# 3. use the batch encoding in better than one-hot, which shows that the batch statistics is possibly proving some information
# 4. however, the best performing model is the one without using any batch encoding information (Possible Reason: the batch statistics is not generalizable, more regularization might also be needed).
# 5. Potential regularization: discriminator, metric learning, etc
#
# --------------------------------------------------------------------------------------------------------------
# # choice 1: insert batch into transformer or not
# # choice 2: use onehot encoding or use batch encoding
# model1 = "cp_vanilla_4_512_meta_enc_trans_1"
# model2 = "cp_vanilla_4_512_meta_enc_1"
# model3 = "cp_vanilla_4_512_meta_onehot_trans_1"
# model4 = "cp_vanilla_4_512_meta_onehot_1"
# model5 = "cp_vanilla_4_512_meta_1"

# PROJECT_DIR = "/net/csefiles/xzhanglab/zzhang834/LLM_KD/"
# res_dir = PROJECT_DIR + f"results/checkpoint/"

# # data_case = "immune_all"
# data_case = "pancreas"
# data_case = "lung_atlas"
# val_loss_dict = {}
# val_loss_dict["enc_trans"] = pd.read_csv(res_dir + model1 + f"/{data_case}_valloss.csv", index_col = 0)
# val_loss_dict["enc"] = pd.read_csv(res_dir + model2 + f"/{data_case}_valloss.csv", index_col = 0)
# val_loss_dict["onehot_trans"] = pd.read_csv(res_dir + model3 + f"/{data_case}_valloss.csv", index_col = 0)
# val_loss_dict["onehot"] = pd.read_csv(res_dir + model4 + f"/{data_case}_valloss.csv", index_col = 0)
# val_loss_dict["nobatch"] = pd.read_csv(res_dir + model5 + f"/{data_case}_valloss.csv", index_col = 0)

# val_loss_dict["enc_trans"]["batch_encode"] = "encoder"
# val_loss_dict["enc_trans"]["insert_trans"] = True
# val_loss_dict["enc"]["batch_encode"] = "encoder"
# val_loss_dict["enc"]["insert_trans"] = False
# val_loss_dict["onehot_trans"]["batch_encode"] = "onehot"
# val_loss_dict["onehot_trans"]["insert_trans"] = True
# val_loss_dict["onehot"]["batch_encode"] = "onehot"
# val_loss_dict["onehot"]["insert_trans"] = False
# val_loss_dict["nobatch"]["batch_encode"] = "no batch"
# val_loss_dict["nobatch"]["insert_trans"] = False

# val_loss = pd.concat([x for x in val_loss_dict.values()], axis = 0)

# fig = plt.figure(figsize = (8, 10))
# ax = fig.subplots(ncols = 1, nrows = 4)

# for i, mask_prob in enumerate([0.1, 0.2, 0.3, 0.4]):
#     sns.barplot(data = val_loss[val_loss["mask prob"] == mask_prob], x = "batch_encode", y = "val mlm", hue = "insert_trans", ax = ax[i])
#     ax[i].set_xlabel(None)
#     for container in ax[i].containers:
#         ax[i].bar_label(container, fmt = "%.4f")
#         leg = ax[i].legend(loc='upper left', prop={'size': 15}, frameon = False, bbox_to_anchor=(1.04, 1), title = "insert transformer")
# plt.tight_layout()
# fig.suptitle(data_case, fontsize = 20)
# fig.savefig(res_dir + f"compare_batchencoding_{data_case}", bbox_inches = "tight")


# In[]
# # choice 3: expr encoding, fourier encoding v.s. learnable encoding
# model1 = "cp_vanilla_4_512_meta_enc_trans_1"
# model2 = "cp_vanilla_4_512_meta_enc_trans_nofourier_1"
# model3 = "cp_vanilla_4_512_meta_onehot_trans_1"
# model2 = "cp_vanilla_4_512_meta_onehot_trans_nofourier_1"

# # data_case = "immune_all"
# data_case = "lung_atlas"
# val_loss_dict = {}
# val_loss_dict["enc"] = pd.read_csv(res_dir + model1 + f"/{data_case}_valloss.csv", index_col = 0)
# val_loss_dict["enc_nofourier"] = pd.read_csv(res_dir + model2 + f"/{data_case}_valloss.csv", index_col = 0)
# val_loss_dict["onehot"] = pd.read_csv(res_dir + model3 + f"/{data_case}_valloss.csv", index_col = 0)
# val_loss_dict["onehot_nofourier"] = pd.read_csv(res_dir + model4 + f"/{data_case}_valloss.csv", index_col = 0)

# val_loss_dict["enc"]["batch_encode"] = "encoder"
# val_loss_dict["enc"]["expr_embed"] = "fourier"
# val_loss_dict["enc_nofourier"]["batch_encode"] = "encoder"
# val_loss_dict["enc_nofourier"]["expr_embed"] = "mlp"
# val_loss_dict["onehot"]["batch_encode"] = "onehot"
# val_loss_dict["onehot"]["expr_embed"] = "fourier"
# val_loss_dict["onehot_nofourier"]["batch_encode"] = "onehot"
# val_loss_dict["onehot_nofourier"]["expr_embed"] = "mlp"

# val_loss = pd.concat([x for x in val_loss_dict.values()], axis = 0)

# fig = plt.figure(figsize = (8, 10))
# ax = fig.subplots(ncols = 1, nrows = 4)

# for i, mask_prob in enumerate([0.1, 0.2, 0.3, 0.4]):
#     sns.barplot(data = val_loss[val_loss["mask prob"] == mask_prob], x = "batch_encode", y = "val mlm", hue = "expr_embed", ax = ax[i])
#     ax[i].set_xlabel(None)
#     for container in ax[i].containers:
#         ax[i].bar_label(container, fmt = "%.4f")
#         leg = ax[i].legend(loc='upper left', prop={'size': 15}, frameon = False, bbox_to_anchor=(1.04, 1), title = "expr embedding")

# fig.suptitle(data_case, fontsize = 20)
# fig.savefig(res_dir + f"compare_exprcoding_{data_case}", bbox_inches = "tight")


# In[]
# Compare discriminator fine-tunning
# NOTE: mask_prob should not be too large for fine-tunning
model1 = "cp_vanilla_4_512_meta_enc_trans_1"
model2 = "cp_disc1_4_512_meta_maskprob0.15_1"
PROJECT_DIR = "/net/csefiles/xzhanglab/zzhang834/LLM_KD/"
res_dir1 = PROJECT_DIR + f"results/checkpoint/"
res_dir2 = PROJECT_DIR + f"results/checkpoint_finetune/"

data_case = "immune_all"
# data_case = "lung_atlas"
val_loss_dict = {}
val_loss_dict["orig"] = pd.read_csv(res_dir1 + model1 + f"/{data_case}_valloss.csv", index_col = 0)
val_loss_dict["disc"] = pd.read_csv(res_dir2 + model2 + f"/{data_case}_valloss.csv", index_col = 0)
val_loss_dict["orig"]["model"] = "orig"
val_loss_dict["disc"]["model"] = "disc-maskprob0.15"

val_loss = pd.concat([x for x in val_loss_dict.values()], axis = 0)

fig = plt.figure(figsize = (5, 15))
ax = fig.subplots(ncols = 1, nrows = 4)

for i, mask_prob in enumerate([0.1, 0.2, 0.3, 0.4]):
    sns.barplot(data = val_loss[val_loss["mask prob"] == mask_prob], x = "model", y = "val mlm", ax = ax[i])
    ax[i].set_xlabel(None)
    for container in ax[i].containers:
        ax[i].bar_label(container, fmt = "%.4f")
        # leg = ax[i].legend(loc='upper left', prop={'size': 15}, frameon = False, bbox_to_anchor=(1.04, 1), title = "model")

fig.suptitle(data_case, fontsize = 20)
fig.savefig(res_dir2 + f"compare_disc_{data_case}", bbox_inches = "tight")



