import torch
import numpy as np
import torch.nn as nn
from contrastive import MultiPosConLoss, MultiPosConLossMultiGPUs


# for cell_embed
import scipy.sparse as sparse
import pandas as pd
import anndata
import tqdm


def infer_databatch(model, data_sample, multigpus: bool = True):
        """
        Description:
        ------------
            Forward pass and loss calculation of the foundation model on input data_sample

        Parameters:
        ------------
            model: the transformer foundation model
            data_sample: the input data sample to the model

        Return:
        ------------
            loss, loss_mlm, loss_sup, loss_kd 

        """
        if multigpus:
             model_acc = model.module
        else:
             model_acc = model
        expr_sample = data_sample["expr"].squeeze(0).to(model_acc.device, non_blocking = True)
        gene_sample = data_sample["gene"].squeeze(0).to(model_acc.device, non_blocking = True)
        if "label" in data_sample.keys():
            label_sample = data_sample["label"].squeeze(0).to(model_acc.device, non_blocking = True)
        else:
            label_sample = None

        if "batch" in data_sample.keys():
            batch_sample = data_sample["batch"].squeeze(0).to(model_acc.device, non_blocking = True)
        else:
            batch_sample = None
        
        _, cell_embed, mask = model(gene_sent = gene_sample, expr_sent = expr_sample)

        # calculate the loss
        # 1. MSE loss between the predict expression and ground truth expression
        # NOTE: the expr_sent is log-normalized in advance
        mse_mlm = nn.MSELoss()

        # NOTE: to be removed in the future
        if model_acc.model_config.sample_mlm:
            # sample from the masked pool
            mask_gene_sample = []
            mask_expr_sample = []
            # loop through every single cell within the batch
            for i in range(cell_embed.shape[0]):
                # 1. target gene can be the gene with 0 expression (not included in the gene sentence)
                masked_genes = torch.tensor([x for x in model_acc.gene_idx if x not in gene_sample[i]]).to(model_acc.device)
                masked_exprs = torch.zeros_like(masked_genes)
                # 2. target gene can be the gene with mask
                masked_genes = torch.hstack([masked_genes, gene_sample[i, mask[i]]])
                masked_exprs = torch.hstack([masked_exprs, expr_sample[i, mask[i]]])
                # sample
                idx = np.random.choice(len(masked_genes), 1)
                mask_gene_sample.append(gene_sample[i, idx])
                mask_expr_sample.append(expr_sample[i, idx])

            mask_gene_sample = torch.cat(mask_gene_sample)
            mask_expr_sample = torch.cat(mask_expr_sample)

            # prediction of the target genes with shape (batch_size, 1)
            expr_pred = model_acc.predict_expr(cell_embed = cell_embed, gene_cond = mask_gene_sample, batch_cond = batch_sample)
            loss_mlm = mse_mlm(expr_pred, mask_expr_sample.unsqueeze(1))
        
        else:
            # predict the gene expression (batch_size, n_mgenes)
            expr_pred = model_acc.predict_expr(cell_embed = cell_embed, gene_cond = None, batch_cond = batch_sample)

            recon_expr_sample = torch.vstack([expr_pred[x,y] for x,y in enumerate(gene_sample)]) 
            loss_mlm = ((recon_expr_sample - expr_sample) * mask).pow(2).sum(1).mean()
            if model_acc.model_config.mlm_include_zero:
                # 0-expression gene
                loss_mlm_zeroexpr = [expr_pred[x, model_acc.gene_idx[~torch.isin(model_acc.gene_idx, y)]].pow(2).sum() for x,y in enumerate(gene_sample)]
                loss_mlm += sum(loss_mlm_zeroexpr)/len(loss_mlm_zeroexpr)

        # 2. Classification loss between the predict label and the ground truth label
        if label_sample is not None:
            if model_acc.model_config.sup_type == "classifier":
                # cell type label
                label_pred = model_acc.classifier(cell_embed)
                ce = nn.CrossEntropyLoss()
                loss_sup = ce(label_pred, label_sample)
            
            elif model_acc.model_config.sup_type == "contrastive":
                if multigpus:
                    contr = MultiPosConLossMultiGPUs(temperature = 0.07)
                else:
                    contr = MultiPosConLoss(temperature = 0.07)
                loss_sup = contr(features = cell_embed, labels = label_sample)

            else:
                raise ValueError("`sub_type' can only be classifier or contrastive")
        
        else:
            loss_sup = torch.tensor([0.0], device = model_acc.device)

        # 3. KD loss from teacher model
        loss_kd = torch.tensor([0.0], device = model_acc.device)
        loss = loss_mlm + model_acc.model_config.lamb_sup * loss_sup + model_acc.model_config.lamb_kd * loss_kd

        return loss, loss_mlm, loss_sup, loss_kd





def cell_embed(model, dataloader, mask_prob: float = 0.0, multi_gpus = True):
    """
    Description:
    ------------
        Obtain the model cell embedding for data in dataloader
    
    Parameters:
    ------------
        model: the transformer model
        dataloader: the dataloader for the input data
        mask_prob: the masking probability of data in the forward pass, default is 0

    """
    # NOTE: no token masking for cell embedding extraction
    if multi_gpus:
        model_acc = model.module
    else:
        model_acc = model

    model_acc.model_config.mask_prob = mask_prob

    cell_embeds = []
    labels = []
    batchs = []
    with torch.no_grad():
        for data_sample in tqdm.tqdm(dataloader, desc=f"Calc embed"):
            expr_sample = data_sample["expr"].squeeze(0).to(model_acc.device, non_blocking = True)
            gene_sample = data_sample["gene"].squeeze(0).to(model_acc.device, non_blocking = True)
            
            # Forward pass
            _, cell_embed, mask = model(gene_sent = gene_sample, expr_sent = expr_sample)
            assert mask.sum() == 0

            cell_embeds.append(sparse.csr_matrix(cell_embed.detach().cpu().numpy()))


            if "label" in data_sample.keys():
                labels.append(data_sample["label"].squeeze(0).detach().cpu().numpy())
            else:
                labels.append(np.array([np.nan] * cell_embed.shape[0]))

            if "batch" in data_sample.keys():
                batchs.append(data_sample["batch"].squeeze(0).detach().cpu().numpy())
            else:
                batchs.append(np.array([np.nan] * cell_embed.shape[0]))

    cell_embeds = sparse.vstack(cell_embeds)
    labels = np.concatenate(labels, axis = 0)
    batchs = np.concatenate(batchs, axis = 0)
    meta = pd.DataFrame.from_dict({"labels": labels, "batchs": batchs})
    adata = anndata.AnnData(X = cell_embeds, obs = meta.astype("category"))

    return adata