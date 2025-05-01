import torch
import numpy as np
import torch.nn as nn

# for cell_embed
import scipy.sparse as sparse
import pandas as pd
import anndata
import tqdm

from torch.utils import data
from torch.nn.utils import clip_grad_norm_
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler

# for bf16 training
from torch.amp import autocast, GradScaler

import data_utils as data_utils
import base_model
import graph_pool
import contrastive

CAST_DTYPE = torch.bfloat16

def save_checkpoint(epoch, step, model, optimizer, scheduler, loss, path, multi_gpus = True):
    if multi_gpus:
        model_acc = model.module
    else:
        model_acc = model
    
    checkpoint = {
        'epoch': epoch,
        'step': step,
        'model_state_dict': model_acc.state_dict(),
        'model_config': model_acc.model_config.__dict__, # save the model config for repeated training too
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'loss': loss,
        }
    torch.save(checkpoint, path)
    print(f"Checkpoint saved at epoch {epoch}.")



def infer_databatch(model, data_sample, multigpus: bool = True):
    """
    Description:
    ------------
        Forward pass and loss calculation of the foundation model on input data_sample

    Parameters:
    ------------
        model: the transformer foundation model
        data_sample: the input data sample to the model
        multigpus: boolean value indicating the use of multi-gpus or not

    Return:
    ------------
        loss, loss_mlm, loss_sup, loss_kd 

    """
    if multigpus:
            model_acc = model.module
    else:
            model_acc = model

    # NOTE: data_sample["counts_norm"] of shape (nchunks, chunksize, nfeats), need to reshape to (batchsize, nfeats)
    # batchsize = nchunks * chunksize, if nchunks == 1, the batch is the same as chunk, not an ideal permutation
    expr_sample = data_sample["counts_norm"].reshape(-1, data_sample["counts_norm"].shape[-1]).to(model_acc.device, non_blocking = True)
    batch_sample_id = data_sample["batch"].reshape(-1).to(model_acc.device, non_blocking = True) if "batch" in data_sample.keys() else None
    batch_sample_cat = data_sample["batch_cat"].reshape(-1, data_sample["batch_cat"].shape[-1]).to(model_acc.device, non_blocking = True) if "batch_cat" in data_sample.keys() else None
    batch_sample_cont = data_sample["batch_cont"].reshape(-1, data_sample["batch_cont"].shape[-1]).to(model_acc.device, non_blocking = True) if "batch_cont" in data_sample.keys() else None

    # NOTE: the model is trained to overfit to the input batches, do not generalize well
    # TODO: add masks/drop-outs on batch_factors_cat, mask probability is dynamic, 0.025-0.1 (1/4 of actual mask)
    if model_acc.model_config.mask_batchfactor:
        mask_bf_prob = torch.full(batch_sample_cat.shape, 0.5 * model_acc.model_config.mask_prob).to(batch_sample_cat.device)
        mask_bf = torch.bernoulli(mask_bf_prob).bool()
        batch_sample_cat = batch_sample_cat.masked_fill(mask_bf, 0)

    all_embed, cell_embed, mask_gene = model(counts_norm = expr_sample, batch_factors_cont = batch_sample_cont, batch_factors_cat = batch_sample_cat, batch_ids = batch_sample_id)
    expr_pred, expr_pred_meta = model_acc.predict_expr(cell_embed = cell_embed, batch_factors_cont = batch_sample_cont, batch_factors_cat = batch_sample_cat, batch_ids = batch_sample_id)

    if model_acc.model_config.recon_meta:
        *_, expr_sample_meta = model_acc.gene_compression(gene_embed = model_acc.token_embed, expr = expr_sample, log_norm = True)
        loss_mlm = ((expr_pred_meta - expr_sample_meta) * mask_gene).pow(2).sum(1).mean()
    else:
        # NOTE: the mask here need to be on gene level
        # use log-norm
        norm = base_model.LogNormalization()
        # used softmax output, so norm1 has to be True
        loss_mlm = ((expr_pred - norm(expr_sample, norm_1 = False)) * mask_gene).pow(2).sum(1).mean()

    # --------------------------------------------------------------------------------------------------------------------
    # NOTE: try discriminator here, has to be bincode classifier
    if model_acc.model_config.use_discriminator:
        # load discriminator related info: label_id, and batch_id
        label_sample_id = data_sample["label"].reshape(-1).to(model_acc.device, non_blocking = True)
        # remove unknown samples
        known_samples = (label_sample_id != model_acc.label_unknown)
        label_sample_id = label_sample_id[known_samples]
        batch_sample_id = batch_sample_id[known_samples]
        cell_embed = cell_embed[known_samples]
        # for each cell type (label), bincode shows in which batch it exists
        label_batch = model_acc.label_dict["label_batch"][label_sample_id, :].bool().to(model_acc.device)
        # [NOTE opt] use of label_batch should improve the performance
        # label_batch = None
        batch_pred = model_acc.calc_discriminator(cell_embed, label_batch)

        ce = nn.CrossEntropyLoss()
        loss_batch = ce(batch_pred, batch_sample_id.long())
    else:
        loss_batch  = torch.tensor([0.0], device = model_acc.device)
    
    # --------------------------------------------------------------------------------------------------------------------
    # add metric learning
    # 2. Classification loss between the predict label and the ground truth label
    if model_acc.model_config.sup_type is not None:
        label_sample_id = data_sample["label"].reshape(-1).to(model_acc.device, non_blocking = True)
        unknown_samples = (label_sample_id == model_acc.label_unknown)

        # remove unknown for better calculation
        cell_embed = cell_embed[~unknown_samples]
        label_sample_id = label_sample_id[~unknown_samples]
        if model_acc.model_config.sup_type == "contrastive":
            batch_label_contr = None
        else:
            batch_label_contr = batch_sample_id[~unknown_samples]

        if model_acc.model_config.sup_type == "contrcb-proj":
            cell_embed = nn.functional.normalize(model_acc.project_contrastive(cell_embed))


        contr = contrastive.SupContrLoss(temperature = 0.07)
        # To be incorporated in the future for larger batch in multi-gpus case
        # contr = contrastive.SupContrLossMultiGPUs(label_asso_mtx = model_acc.contrastive_label_mtx, temperature = 0.07, unknown_label = model_acc.label_unknown)
        contr_label_mtx_sample = model_acc.contrastive_label_mtx[label_sample_id.unsqueeze(1), label_sample_id.unsqueeze(0)]
        loss_sup = contr(features = cell_embed, label_mtx = contr_label_mtx_sample, batch_ids = batch_label_contr)

        # fine-tune for classification task
        if model_acc.model_config.use_classifier:
            label_sample_bincode = model_acc.label_dict["label_bincode"][label_sample_id]
            # NOTE: the cell_embed is linear transformed if use contrcb-proj
            label_pred = model_acc.classifier(cell_embed)
            loss_sup += contrastive.bce_weighted(pred = label_pred, target = label_sample_bincode, weight = model_acc.classifier_weight)
            
    else:
        loss_sup = torch.tensor([0.0], device = model_acc.device)
    # --------------------------------------------------------------------------------------------------------------------


    # mincut loss
    # S = model_acc.gene_compression.get_score()
    # l_mincut, l_ortho = graph_pool.mincut_loss(A = model_acc.gene_compression.A, S = S, add_orthogonality = True)
    l_mincut, l_ortho = model_acc.gene_compression.mincut_loss(add_ortho = True)
    
    loss = loss_mlm + model_acc.model_config.lamb_disc * loss_batch + model_acc.model_config.lamb_mincut * (l_mincut + 0.01 * l_ortho) + model_acc.model_config.lamb_sup * loss_sup

    return loss, {"mlm": loss_mlm.item(), "disc": loss_batch.item(), "mincut": l_mincut.item(), "ortho": l_ortho.item(), "metric": loss_sup.item()}


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
    if multi_gpus:
        model_acc = model.module
    else:
        model_acc = model
    
    # evaluation model
    model_acc.eval()
    # update the mask_prob for evaluation
    if mask_prob is not None:
        model_acc.model_config.mask_prob = mask_prob

    if model_acc.model_config.use_fastatten:
        # because flashattention only accept 16bit model
        enable_casting = True
    else:
        enable_casting = False

    cell_embeds = []
    cell_embeds_contr = []
    labels = []
    batchs = []
    with torch.no_grad():
        for data_sample in tqdm.tqdm(dataloader, desc=f"Calc embed"):

            with autocast(device_type="cuda", dtype = CAST_DTYPE, enabled = enable_casting):
                expr_sample = data_sample["counts_norm"].reshape(-1, data_sample["counts_norm"].shape[-1]).to(model_acc.device, non_blocking = True)
                batch_sample_cat = data_sample["batch_cat"].reshape(-1, data_sample["batch_cat"].shape[-1]).to(model_acc.device, non_blocking = True) if "batch_cat" in data_sample.keys() else None
                batch_sample_cont = data_sample["batch_cont"].reshape(-1, data_sample["batch_cont"].shape[-1]).to(model_acc.device, non_blocking = True) if "batch_cont" in data_sample.keys() else None

                all_embed, cell_embed, mask_gene = model(counts_norm = expr_sample, batch_factors_cont = batch_sample_cont, batch_factors_cat = batch_sample_cat, batch_ids = None)                    
                cell_embeds.append(sparse.csr_matrix(cell_embed.to(torch.float32).detach().cpu().numpy()))  
                # use contrcb-projector
                if model_acc.model_config.sup_type == "contrcb-proj":
                    cell_embed_contr = nn.functional.normalize(model_acc.project_contrastive(cell_embed))
                    cell_embeds_contr.append(sparse.csr_matrix(cell_embed_contr.to(torch.float32).detach().cpu().numpy()))

            if "label" in data_sample.keys():
                labels.append(data_sample["label"].reshape(-1).detach().cpu().numpy())
            else:
                labels.append(np.array([np.nan] * cell_embed.shape[0]))

            if "batch" in data_sample.keys():
                batchs.append(data_sample["batch"].reshape(-1).detach().cpu().numpy())
            else:
                batchs.append(np.array([np.nan] * cell_embed.shape[0]))

    cell_embeds = sparse.vstack(cell_embeds)
    labels = np.concatenate(labels, axis = 0)
    batchs = np.concatenate(batchs, axis = 0)
    meta = pd.DataFrame.from_dict({"labels": labels, "batchs": batchs})
    adata = anndata.AnnData(X = cell_embeds, obs = meta.astype("category"))

    if model_acc.model_config.sup_type == "contrcb-proj":
        adata.obsm["contr"] = sparse.vstack(cell_embeds_contr)

    return adata




def train_multigpus(model, global_rank, dataset_dict, writer, initial_epoch, initial_step, log_step):

    """
    Description:
    ------------
        The training function of foundation model

    Parameters:
    ------------
        model: transformer model
        train_loader: the training data loader
        val_loader: the validation data loader
        optimizer: the optimizer of the model
        scheduler: the scheduler of the model
        writer: the tensorboard writer
        TODO: ADD
    """
    print(f"GPU {global_rank} - Loading dataset...")
    # Need to normalize the data, the min_chunksize = 64, so batchsize 512 = 8 samples * 64
    min_chunksize = 64
    label_colname = dataset_dict["label_colname"]
    batch_colname = dataset_dict["batch_colname"]
    val_dataset = data_utils.sc_partition(data_path = dataset_dict["DIR"], batch_feats = dataset_dict["batch_dict"], min_chunksize = min_chunksize, normalize = model.module.model_config.lognorm_data)
    val_dataset.load_partition(idx = dataset_dict["num_partitions"] - 1, label_colname = label_colname, batch_colname = batch_colname, data_prefix = dataset_dict["data_prefix"], meta_prefix = dataset_dict["meta_prefix"]) # use last chunk
    val_loader = data.DataLoader(val_dataset, batch_size = model.module.model_config.batch_size//min_chunksize, shuffle = False, pin_memory = True,
                                 sampler = DistributedSampler(val_dataset, shuffle = False), num_workers = 8, prefetch_factor = 8)
    train_dataset = data_utils.sc_partition(data_path = dataset_dict["DIR"], batch_feats = dataset_dict["batch_dict"], min_chunksize = min_chunksize, normalize = model.module.model_config.lognorm_data)
    print(f"GPU {global_rank} - Done.")

    scaler = GradScaler()
    # NOTE: training loop
    checkpoint_counter = 0
    if model.module.model_config.use_fastatten:
        torch.backends.cuda.enable_flash_sdp(True)
        torch.backends.cuda.enable_mem_efficient_sdp(True)
        torch.backends.cuda.enable_math_sdp(True)
        
    for epoch in range(initial_epoch, model.module.model_config.n_epoch):
        step = 0
        running_loss, running_loss_mlm, running_loss_disc, running_loss_metric, running_loss_mincut, running_loss_ortho = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0

        # shuffle the partition for each epoch
        for partition_idx in np.random.permutation(dataset_dict["num_partitions"] - 1):
            print(f"GPU {global_rank} - Start training Epoch {epoch:02d}, Partition {partition_idx:02d}...")
            torch.cuda.empty_cache()
            # load training dataset for partition_idx
            train_dataset.load_partition(idx = partition_idx, label_colname = label_colname, batch_colname = batch_colname, data_prefix = dataset_dict["data_prefix"], meta_prefix = dataset_dict["meta_prefix"])
            # shuffle in distributed sampler
            train_loader = data.DataLoader(train_dataset, batch_size = model.module.model_config.batch_size//min_chunksize, shuffle = False, pin_memory = True,
                                        sampler = DistributedSampler(train_dataset, shuffle = True), num_workers = 8, prefetch_factor = 8)

            batch_iterator = tqdm.tqdm(train_loader, desc=f"Processing Epoch {epoch:02d}, Partition {partition_idx:02d} on rank {global_rank}", disable = global_rank != 0)

            for data_sample in batch_iterator:
                model.module.train()
                model.module.optimizer.zero_grad()
            
                if step < initial_step:
                    step += 1
                    continue

                with autocast(device_type="cuda", dtype = CAST_DTYPE):
                    loss, loss_item = infer_databatch(model, data_sample, multigpus = True)

                scaler.scale(loss).backward()

                #Unscale the optimizer to clip gradients
                scaler.unscale_(model.module.optimizer)
                # clip gradient
                max_grad_norm = 1.0 
                clip_grad_norm_(model.module.parameters(), max_grad_norm)


                scaler.step(model.module.optimizer)
                scaler.update()
                model.module.scheduler.step()

                # NOTE: log the results
                running_loss += loss.item()
                running_loss_mlm += loss_item["mlm"]
                running_loss_disc += loss_item["disc"]
                running_loss_metric += loss_item["metric"]
                running_loss_mincut += loss_item["mincut"]
                running_loss_ortho += loss_item["ortho"]
            
                if step % log_step == log_step - 1:
                    # calculate for each gpus
                    running_loss /= log_step
                    running_loss_mlm /= log_step
                    running_loss_disc /= log_step
                    running_loss_metric /= log_step
                    running_loss_mincut /= log_step
                    running_loss_ortho /= log_step
                    if writer is not None:
                        cum_step = epoch * len(train_loader) * dataset_dict["num_partitions"] + step + 1
                        # only write/print the running loss for one gpu with writer
                        writer.add_scalar("Train Loss (TOTAL)", running_loss, cum_step)
                        writer.add_scalar("Train Loss (MLM)", running_loss_mlm, cum_step)
                        writer.add_scalar("Train Loss (DISC)", running_loss_disc, cum_step)
                        writer.add_scalar("Train Loss (METRIC)", running_loss_metric, cum_step)
                        writer.add_scalar("Train Loss (MINCUT)", running_loss_mincut, cum_step)
                        writer.add_scalar("Train Loss (ORTHO)", running_loss_ortho, cum_step)
                        writer.add_scalar("Learning rate", model.module.scheduler.get_last_lr()[0], cum_step)
                        print(f"Epoch: {epoch}, Step: {step + 1}/{len(train_loader) * dataset_dict["num_partitions"]}, Learning rate: {model.module.scheduler.get_last_lr()[0]:.2e}, Mask prob: {model.module.model_config.mask_prob:.4f}, \
                              Train Loss (TOTAL): {running_loss:.4f}, Train Loss (MLM):{running_loss_mlm:.4f}, Train Loss (DISC): {running_loss_disc:.4f}, Train Loss (METRIC): {running_loss_metric:.4f}, Train Loss (MINCUT): {running_loss_mincut:.4f}, Train Loss (ORTHO): {running_loss_ortho:.4f}")

                    running_loss, running_loss_mlm, running_loss_disc, running_loss_metric, running_loss_mincut, running_loss_ortho = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
                    checkpoint_counter += 1

                    # model evaluation and checkpoint saving
                    if (checkpoint_counter == 10):
                        model.module.eval()
                        with torch.no_grad():
                            val_loss, val_loss_mlm, val_loss_disc, val_loss_metric, val_loss_mincut, val_loss_ortho = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
                            for data_sample in val_loader:
                                with autocast(device_type="cuda", dtype = CAST_DTYPE):

                                    loss, loss_item = infer_databatch(model, data_sample, multigpus = True)                            
                                    val_loss += loss.item()
                                    val_loss_mlm += loss_item["mlm"]
                                    val_loss_disc += loss_item["disc"]
                                    val_loss_metric += loss_item["metric"]
                                    val_loss_mincut += loss_item["mincut"]
                                    val_loss_ortho += loss_item["ortho"]

                            # log the values
                            val_loss /= len(val_loader)
                            val_loss_mlm /= len(val_loader)
                            val_loss_disc /= len(val_loader)
                            val_loss_metric /= len(val_loader)
                            val_loss_mincut /= len(val_loader)
                            val_loss_ortho /= len(val_loader)
                            
                            if writer is not None:
                                cum_step = epoch * len(train_loader) * dataset_dict["num_partitions"] + step + 1
                                writer.add_scalar("Val Loss (TOTAL)", val_loss, cum_step)
                                writer.add_scalar("Val Loss (MLM)", val_loss_mlm, cum_step)
                                writer.add_scalar("Val Loss (DISC)", val_loss_disc, cum_step)
                                writer.add_scalar("Val Loss (METRIC)", val_loss_metric, cum_step)
                                writer.add_scalar("Val Loss (MINCUT)", val_loss_mincut, cum_step)
                                writer.add_scalar("Val Loss (ORTHO)", val_loss_ortho, cum_step)
                                writer.add_scalar("Mask prob", model.module.model_config.mask_prob, cum_step)
                                print(f"Epoch: {epoch}, Step: {step + 1}/{len(train_loader) * dataset_dict["num_partitions"]}, Val Loss (TOTAL): {val_loss:.4f}, Val Loss (MLM): {val_loss_mlm:.4f}, Val Loss (DISC): {val_loss_disc:.4f}, \
                                      Val Loss (METRIC): {val_loss_metric:.4f}, Val Loss (MINCUT): {val_loss_mincut:.4f}, Val Loss (ORTHO): {val_loss_ortho:.4f}")

                                # save only for the writer gpu
                                save_checkpoint(epoch = epoch, step = step, model = model, optimizer = model.module.optimizer, scheduler = model.module.scheduler, loss = running_loss,
                                                path = f"{model.module.model_config.checkpoint_path}{model.module.model_config.checkpoint_prefix}_{epoch}_{step + 1}.pth")
                        
                        checkpoint_counter = 0                

                    # update the mask_prob 
                    if model.module.model_config.dynamic_maskprob:
                        model.module.model_config.mask_prob += model.module.model_config.maskprob_step
                    # update the reverse gradient weight 
                    if model.module.model_config.use_discriminator:
                        model.module.model_config.lamb_reverse += model.module.model_config.lamb_reverse_step

                    # sync all gpus after eval
                    dist.barrier()
                step += 1   
        # end of the epoch, reset the init step as 0
        initial_step = 0

    # save the final model, also only for the writer gpu
    if writer is not None:
        save_checkpoint(epoch = model.module.model_config.n_epoch, step = 0, model = model, optimizer = model.module.optimizer, scheduler = model.module.scheduler, loss = running_loss,
                        path = f"{model.module.model_config.checkpoint_path}{model.module.model_config.checkpoint_prefix}_{model.module.model_config.n_epoch}.pth")
        




def train_singlegpu(model, dataset_dict, writer, initial_epoch, initial_step, log_step):

    """
    Description:
    ------------
        The training function of foundation model

    Parameters:
    ------------
        model: transformer model
        train_loader: the training data loader
        val_loader: the validation data loader
        optimizer: the optimizer of the model
        scheduler: the scheduler of the model
        writer: the tensorboard writer
        TODO: ADD
    """
    print(f"GPU - Loading dataset...")
    # Need to normalize the data, the min_chunksize = 64, so batchsize 512 = 8 samples * 64
    min_chunksize = 64
    label_colname = dataset_dict["label_colname"]
    batch_colname = dataset_dict["batch_colname"]
    val_dataset = data_utils.sc_partition(data_path = dataset_dict["DIR"], batch_feats = dataset_dict["batch_dict"], min_chunksize = min_chunksize, normalize = model.model_config.lognorm_data)
    val_dataset.load_partition(idx = dataset_dict["num_partitions"] - 1, label_colname = label_colname, batch_colname = batch_colname, data_prefix = dataset_dict["data_prefix"], meta_prefix = dataset_dict["meta_prefix"]) # use last chunk
    val_loader = data.DataLoader(val_dataset, batch_size = model.model_config.batch_size//min_chunksize, shuffle = False, pin_memory = True, num_workers = 8, prefetch_factor = 8)
    train_dataset = data_utils.sc_partition(data_path = dataset_dict["DIR"], batch_feats = dataset_dict["batch_dict"], min_chunksize = min_chunksize, normalize = model.model_config.lognorm_data)
    print(f"GPU - Done.")

    scaler = GradScaler()
    # NOTE: training loop
    checkpoint_counter = 0
    if model.model_config.use_fastatten:
        torch.backends.cuda.enable_flash_sdp(True)
        torch.backends.cuda.enable_mem_efficient_sdp(True)
        torch.backends.cuda.enable_math_sdp(True)
        
    for epoch in range(initial_epoch, model.model_config.n_epoch):
        step = 0
        running_loss, running_loss_mlm, running_loss_disc, running_loss_metric, running_loss_mincut, running_loss_ortho = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0

        # shuffle the partition for each epoch
        for partition_idx in np.random.permutation(dataset_dict["num_partitions"] - 1):
            print(f"GPU - Start training Epoch {epoch:02d}, Partition {partition_idx:02d}...")
            torch.cuda.empty_cache()
            # load training dataset for partition_idx
            train_dataset.load_partition(idx = partition_idx, label_colname = label_colname, batch_colname = batch_colname, data_prefix = dataset_dict["data_prefix"], meta_prefix = dataset_dict["meta_prefix"])
            # shuffle in distributed sampler
            train_loader = data.DataLoader(train_dataset, batch_size = model.model_config.batch_size//min_chunksize, shuffle = True, pin_memory = True, num_workers = 8, prefetch_factor = 8)

            batch_iterator = tqdm.tqdm(train_loader, desc=f"Processing Epoch {epoch:02d}, Partition {partition_idx:02d}")

            for data_sample in batch_iterator:
                model.train()
                model.optimizer.zero_grad()
            
                if step < initial_step:
                    step += 1
                    continue

                with autocast(device_type="cuda", dtype = CAST_DTYPE):
                    loss, loss_item = infer_databatch(model, data_sample, multigpus = False)

                scaler.scale(loss).backward()

                #Unscale the optimizer to clip gradients
                scaler.unscale_(model.optimizer)
                # clip gradient
                max_grad_norm = 1.0 
                clip_grad_norm_(model.parameters(), max_grad_norm)


                scaler.step(model.optimizer)
                scaler.update()
                model.scheduler.step()

                # NOTE: log the results
                running_loss += loss.item()
                running_loss_mlm += loss_item["mlm"]
                running_loss_disc += loss_item["disc"]
                running_loss_metric += loss_item["metric"]
                running_loss_mincut += loss_item["mincut"]
                running_loss_ortho += loss_item["ortho"]

                if step % log_step == log_step - 1:
                    # calculate for each gpus
                    running_loss /= log_step
                    running_loss_mlm /= log_step
                    running_loss_disc /= log_step
                    running_loss_metric /= log_step
                    running_loss_mincut /= log_step
                    running_loss_ortho /= log_step

                    cum_step = epoch * len(train_loader) * dataset_dict["num_partitions"] + step + 1
                    # only write/print the running loss for one gpu with writer
                    writer.add_scalar("Train Loss (TOTAL)", running_loss, cum_step)
                    writer.add_scalar("Train Loss (MLM)", running_loss_mlm, cum_step)
                    writer.add_scalar("Train Loss (DISC)", running_loss_disc, cum_step)
                    writer.add_scalar("Train Loss (METRIC)", running_loss_metric, cum_step)
                    writer.add_scalar("Train Loss (MINCUT)", running_loss_mincut, cum_step)
                    writer.add_scalar("Train Loss (ORTHO)", running_loss_ortho, cum_step)
                    writer.add_scalar("Learning rate", model.scheduler.get_last_lr()[0], cum_step)

                    print(f"Epoch: {epoch}, Step: {step + 1}/{len(train_loader) * dataset_dict["num_partitions"]}, Learning rate: {model.scheduler.get_last_lr()[0]:.2e}, Mask prob: {model.model_config.mask_prob:.4f}, \
                            Train Loss (TOTAL): {running_loss:.4f}, Train Loss (MLM):{running_loss_mlm:.4f}, Train Loss (DISC): {running_loss_disc:.4f}, Train Loss (METRIC): {running_loss_metric:.4f}, Train Loss (MINCUT): {running_loss_mincut:.4f}, Train Loss (ORTHO): {running_loss_ortho:.4f}")
                    
                    running_loss, running_loss_mlm, running_loss_disc, running_loss_metric, running_loss_mincut, running_loss_ortho = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
                    checkpoint_counter += 1

                    # model evaluation and checkpoint saving
                    if (checkpoint_counter == 10):
                        model.eval()
                        with torch.no_grad():
                            val_loss, val_loss_mlm, val_loss_disc, val_loss_metric, val_loss_mincut, val_loss_ortho = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
                            for data_sample in val_loader:
                                with autocast(device_type="cuda", dtype = CAST_DTYPE):

                                    loss, loss_item = infer_databatch(model, data_sample, multigpus = False)                            
                                    val_loss += loss.item()
                                    val_loss_mlm += loss_item["mlm"]
                                    val_loss_disc += loss_item["disc"]
                                    val_loss_metric += loss_item["metric"]
                                    val_loss_mincut += loss_item["mincut"]
                                    val_loss_ortho += loss_item["ortho"]

                            # log the values
                            val_loss /= len(val_loader)
                            val_loss_mlm /= len(val_loader)
                            val_loss_disc /= len(val_loader)
                            val_loss_metric /= len(val_loader)
                            val_loss_mincut /= len(val_loader)
                            val_loss_ortho /= len(val_loader)

                            cum_step = epoch * len(train_loader) * dataset_dict["num_partitions"] + step + 1
                            writer.add_scalar("Val Loss (TOTAL)", val_loss, cum_step)
                            writer.add_scalar("Val Loss (MLM)", val_loss_mlm, cum_step)
                            writer.add_scalar("Val Loss (DISC)", val_loss_disc, cum_step)
                            writer.add_scalar("Val Loss (METRIC)", val_loss_metric, cum_step)
                            writer.add_scalar("Val Loss (MINCUT)", val_loss_mincut, cum_step)
                            writer.add_scalar("Val Loss (ORTHO)", val_loss_ortho, cum_step)
                            writer.add_scalar("Mask prob", model.model_config.mask_prob, cum_step)

                            print(f"Epoch: {epoch}, Step: {step + 1}/{len(train_loader) * dataset_dict["num_partitions"]}, Val Loss (TOTAL): {val_loss:.4f}, Val Loss (MLM): {val_loss_mlm:.4f}, Val Loss (DISC): {val_loss_disc:.4f}, \
                                    Val Loss (METRIC): {val_loss_metric:.4f}, Val Loss (MINCUT): {val_loss_mincut:.4f}, Val Loss (ORTHO): {val_loss_ortho:.4f}")

                            save_checkpoint(epoch = epoch, step = step, model = model, optimizer = model.optimizer, scheduler = model.scheduler, loss = running_loss,
                                            path = f"{model.model_config.checkpoint_path}{model.model_config.checkpoint_prefix}_{epoch}_{step + 1}.pth", multi_gpus = False)
                                                
                        checkpoint_counter = 0                

                    # update the mask_prob 
                    if model.model_config.dynamic_maskprob:
                        model.model_config.mask_prob += model.model_config.maskprob_step
                    # update the reverse gradient weight 
                    if model.model_config.use_discriminator:
                        model.model_config.lamb_reverse += model.model_config.lamb_reverse_step

                    # sync all gpus after eval
                step += 1   
        initial_step = 0

    save_checkpoint(epoch = model.model_config.n_epoch, step = 0, model = model, optimizer = model.optimizer, scheduler = model.scheduler, loss = running_loss,
                    path = f"{model.model_config.checkpoint_path}{model.model_config.checkpoint_prefix}_{model.model_config.n_epoch}.pth", multi_gpus = False)
