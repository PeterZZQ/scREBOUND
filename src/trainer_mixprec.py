import torch
import numpy as np
import torch.nn as nn
from src.contrastive import MultiPosConLoss, MultiPosConLossMultiGPUs, bce_weighted
from src.transformer_model import gradient_reversal


# for cell_embed
import scipy.sparse as sparse
import pandas as pd
import anndata
import tqdm

from torch.nn.utils import clip_grad_norm_
import torch.distributed as dist

# for bf16 training
from torch.amp import autocast, GradScaler

CAST_DTYPE = torch.bfloat16


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
        # scale the expression feature to be between 0~1
        expr_sample = data_sample["expr"].squeeze(0).to(model_acc.device, non_blocking = True)
        gene_sample = data_sample["gene"].squeeze(0).to(model_acc.device, non_blocking = True)
        
        with torch.no_grad():
            # binning the data
            if model_acc.model_config.count_mode == "digitize":
                expr_sample /= torch.max(expr_sample, dim = 1, keepdim = True).values
                bins_array = torch.arange(0, 1, 1/model_acc.n_bins).to(model_acc.device, non_blocking = True)
                expr_sample = torch.bucketize(expr_sample, bins_array)
            else:
                # normalize the expression value range between 0 and 1
                expr_sample /= torch.max(expr_sample, dim = 1, keepdim = True).values


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
        # predict the gene expression (batch_size, n_mgenes)
        expr_pred = model_acc.predict_expr(cell_embed = cell_embed, batch_cond = batch_sample)

        if model_acc.model_config.with_padding:
            # cost too much time, save in advance        
            recon_expr_sample = torch.vstack([expr_pred[x,y.long()] for x,y in enumerate(gene_sample)]) 
            loss_mlm = ((recon_expr_sample - expr_sample) * mask).pow(2).sum(1).mean()
            # if model_acc.model_config.mlm_include_zero:
            #     # 0-expression gene
            #     loss_mlm_zeroexpr = [expr_pred[x, model_acc.gene_idx[~torch.isin(model_acc.gene_idx, y)]].pow(2).sum() for x,y in enumerate(gene_sample)]
            #     loss_mlm += sum(loss_mlm_zeroexpr)/len(loss_mlm_zeroexpr)
        else:
            # NOTE: the expr_sample is of the same ordering across cells, 
            # directly use the first expr_sample.shape[1] tokens to save time
            recon_expr_sample = expr_pred[:, :expr_sample.shape[1]]
            loss_mlm = ((recon_expr_sample - expr_sample) * mask).pow(2).sum(1).mean()


        # 2. Classification loss between the predict label and the ground truth label
        if (label_sample is not None) and (model_acc.model_config.sup_type is not None):
            # NOTE: label_sample is (n_batch, n_class), find unknown labels
            mask_samples = (label_sample == model_acc.label_mask).all(dim=1)
            
            if model_acc.model_config.sup_type == "classifier":
                label_pred = model_acc.classifier(cell_embed)
                # remove masked sample for loss calculation
                loss_sup = bce_weighted(pred = label_pred[~mask_samples], target = label_sample[~mask_samples], weight = model_acc.classifier_weight)
            
            elif (model_acc.model_config.sup_type == "contrastive") or (model_acc.model_config.sup_type == "contrastive-cb"):
                # mask_samples cannot be used to downsample input as this will cause the tensor size mismatch for multi-gpus combining
                # make the corresponding label_sample to be all 0 so that unknown will always be neutral samples
                # make sure mask sample always is ancestor (always remove) and remove mask_sample in contrastive too
                label_sample[mask_samples] = 0
                if multigpus:
                    # TODO: bug within multigpus version, gradient explode easily
                    contr = MultiPosConLossMultiGPUs(temperature = 0.07, use_bincode_label = True)
                    # contr = MultiPosConLoss(temperature = 0.07, use_bincode_label = True)
                else:
                    contr = MultiPosConLoss(temperature = 0.07, use_bincode_label = True)

                if model_acc.model_config.sup_type == "contrastive":
                    # classify the cells into 3 categories instead of two:
                    # positive, negative, neutral
                    # supervised contrastive loss generate a similarity matrix for every cell
                    # the ground truth is the cell x cell pos/neg matrix, incorporate mutual too
                    loss_sup = contr(features = cell_embed, labels = label_sample, mask_samples = mask_samples)
                
                else:
                    loss_sup = contr(features = cell_embed, labels = label_sample, batchs = batch_sample)
                
            else:
                raise ValueError("sup_type not right")

        
        else:
            loss_sup = torch.tensor([0.0], device = model_acc.device)

        
        # NOTE: try discriminator here, has to be bincode classifier
        if model_acc.model_config.use_discriminator:
            # remove unknown samples
            mask_samples = (label_sample == model_acc.label_mask).all(dim=1)
            # reverse the gradient of the model
            reversed_embed = gradient_reversal(cell_embed[~mask_samples], model_acc.model_config.lamb_reverse)
            # pass the reversed gradient into discriminator
            batch_pred = model_acc.discriminator(reversed_embed, label_sample[~mask_samples])
            ce = nn.CrossEntropyLoss()
            # ce expect int64 labels
            loss_batch = ce(batch_pred, batch_sample.long())
        else:
            loss_batch  = torch.tensor([0.0], device = model_acc.device)


        # 3. KD loss from teacher model
        # loss_kd = torch.tensor([0.0], device = model_acc.device)
        loss = loss_mlm + model_acc.model_config.lamb_sup * loss_sup + model_acc.model_config.lamb_disc * loss_batch #model_acc.model_config.lamb_kd * loss_kd

        return loss, loss_mlm, loss_sup, loss_batch



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

    if model_acc.model_config.use_fastatten:
        # because flashattention only accept 16bit model
        enable_casting = True
    else:
        enable_casting = False

    cell_embeds = []
    labels = []
    batchs = []
    with torch.no_grad():
        for data_sample in tqdm.tqdm(dataloader, desc=f"Calc embed"):

            with autocast(device_type="cuda", dtype = CAST_DTYPE, enabled = enable_casting):
                expr_sample = data_sample["expr"].squeeze(0).to(model_acc.device, non_blocking = True)
                gene_sample = data_sample["gene"].squeeze(0).to(model_acc.device, non_blocking = True)

                # binning the data
                with torch.no_grad():
                    # binning the data
                    if model_acc.model_config.count_mode == "digitize":
                        expr_sample /= torch.max(expr_sample, dim = 1, keepdim = True).values
                        bins_array = torch.arange(0, 1, 1/model_acc.n_bins).to(model_acc.device, non_blocking = True)
                        expr_sample = torch.bucketize(expr_sample, bins_array)
                    else:
                        # normalize the expression value range between 0 and 1
                        expr_sample /= torch.max(expr_sample, dim = 1, keepdim = True).values

                # Forward pass
                _, cell_embed, mask = model_acc(gene_sent = gene_sample, expr_sent = expr_sample)
            
            cell_embeds.append(sparse.csr_matrix(cell_embed.to(torch.float32).detach().cpu().numpy()))  

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



def save_checkpoint(epoch, step, model, optimizer, scheduler, loss, path):
    checkpoint = {
        'epoch': epoch,
        'step': step,
        'model_state_dict': model.module.state_dict(),
        'model_config': model.module.model_config, # save the model config for repeated training too
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'loss': loss
    }
    torch.save(checkpoint, path)
    print(f"Checkpoint saved at epoch {epoch}.")



def train_multigpus(model, global_rank, train_loader, val_loader, optimizer, scheduler, writer, initial_epoch, initial_step, log_step):
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
    scaler = GradScaler()

    # NOTE: dynamic masking probability, small to large
    if model.module.model_config.dynamic_maskprob:
        mask_prob_init = 0.2
        mask_prob_end = 0.5
        mask_prob_step = (mask_prob_end - mask_prob_init) / len(train_loader) / (model.module.model_config.n_epoch - initial_epoch) * log_step
        model.module.model_config.mask_prob = mask_prob_init
    
    if model.module.model_config.use_discriminator:
        lamb_reverse_init = 1.0 # the lamb_reverse is static
        lamb_reverse_end = 1.0
        lamb_reverse_step = (lamb_reverse_end - lamb_reverse_init) / len(train_loader) / (model.module.model_config.n_epoch - initial_epoch) * log_step
        model.module.model_config.lamb_reverse = lamb_reverse_init
    else:
        model.module.model_config.lamb_reverse = 0.0

    # NOTE: training loop
    checkpoint_counter = 0
    if model.module.model_config.use_fastatten:
        torch.backends.cuda.enable_flash_sdp(True)
        torch.backends.cuda.enable_mem_efficient_sdp(True)
        torch.backends.cuda.enable_math_sdp(True)
        
    for epoch in range(initial_epoch, model.module.model_config.n_epoch):
        torch.cuda.empty_cache()
        # Disable tqdm on all nodes except the rank 0 GPU on each server
        batch_iterator = tqdm.tqdm(train_loader, desc=f"Processing Epoch {epoch:02d} on rank {global_rank}", disable = global_rank != 0)

        # NOTE: Training
        running_loss = 0.0
        running_loss_mlm = 0.0
        running_loss_sup = 0.0
        running_loss_kd = 0.0      

        for step, data_sample in enumerate(batch_iterator):
            model.module.train()
            optimizer.zero_grad()
            
            if step < initial_step:
                continue

            with autocast(device_type="cuda", dtype = CAST_DTYPE):
                # NOTE: need to process the label into bincode
                if model.module.model_config.sup_type is not None:
                    data_sample["label"] = model.module.label_bincode[data_sample["label"],:]

                loss, loss_mlm, loss_sup, loss_kd = infer_databatch(model, data_sample, multigpus = True)

            scaler.scale(loss).backward()

            #Unscale the optimizer to clip gradients
            scaler.unscale_(optimizer)
            # clip gradient
            max_grad_norm = 1.0 
            clip_grad_norm_(model.module.parameters(), max_grad_norm)

            # # NOTE: check gradient underflow
            # for name, param in model.module.named_parameters():
            #     if param.grad is not None:
            #         if (param.grad.abs() < 1e-7).all():  
            #             print(f"Possible underflow in {name}")
            #             assert False

            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            # NOTE: log the results
            running_loss += loss.item()
            running_loss_mlm += loss_mlm.item()
            running_loss_sup += loss_sup.item()
            running_loss_kd += loss_kd.item()
            
            if step % log_step == log_step - 1:
                interval = (step % log_step)
                running_loss /= interval
                running_loss_mlm /= interval
                running_loss_sup /= interval
                running_loss_kd /= interval
                # NOTE: writer is not None only when global_rank == 0, make sure only one thread write the result
                if writer is not None:
                    writer.add_scalar("Train Loss (TOTAL)", running_loss, epoch * len(train_loader) + step + 1)
                    writer.add_scalar("Train Loss (MLM)", running_loss_mlm, epoch * len(train_loader) + step + 1)
                    writer.add_scalar("Train Loss (CLASS)", running_loss_sup, epoch * len(train_loader) + step + 1)
                    writer.add_scalar("Train Loss (DISC)", running_loss_kd, epoch * len(train_loader) + step + 1)
                    writer.add_scalar("Learning rate", scheduler.get_last_lr()[0], epoch * len(train_loader) + step + 1)

                    print(f"Epoch: {epoch}, Step: {step + 1}/{len(train_loader)}, Learning rate: {scheduler.get_last_lr()[0]:.2e}, Train Loss (TOTAL): {running_loss:.4f}, Train Loss (MLM): {running_loss_mlm:.4f}, Train Loss (CLASS): {running_loss_sup:.4f}, Train Loss (DISC): {running_loss_kd:.4f}")
                
                running_loss = 0.0
                running_loss_mlm = 0.0
                running_loss_sup = 0.0
                running_loss_kd = 0.0

                checkpoint_counter += 1

                # update the mask_prob 
                if model.module.model_config.dynamic_maskprob:
                    model.module.model_config.mask_prob += mask_prob_step
                    # constraint to not exceed end
                    model.module.model_config.mask_prob = min(model.module.model_config.mask_prob, mask_prob_end)
                
                # update the reverse gradient weight 
                if model.module.model_config.use_discriminator:
                    model.module.model_config.lamb_reverse += lamb_reverse_step
                    # contraint to not exceed end
                    model.module.model_config.lamb_reverse = min(model.module.model_config.lamb_reverse, lamb_reverse_end)


                # model evaluation and checkpoint saving
                # only the first for evaluation
                # if (global_rank == 0) & (checkpoint_counter == 10):
                # all gpus for evalution                
                if (checkpoint_counter == 10):
                    model.module.eval()
                    with torch.no_grad():
                        val_loss = 0.0
                        val_loss_mlm = 0.0
                        val_loss_sup = 0.0
                        val_loss_kd = 0.0
                        for data_sample in val_loader:
                            with autocast(device_type="cuda", dtype = CAST_DTYPE):
                                # NOTE: need to process the label into bincode
                                if model.module.model_config.sup_type is not None:
                                    data_sample["label"] = model.module.label_bincode[data_sample["label"],:]

                                loss, loss_mlm, loss_sup, loss_kd = infer_databatch(model, data_sample, multigpus = True)                            
                                val_loss += loss.item()
                                val_loss_mlm += loss_mlm.item()
                                val_loss_sup += loss_sup.item()
                                val_loss_kd += loss_kd.item()

                        # log the values
                        val_loss /= len(val_loader)
                        val_loss_mlm /= len(val_loader)
                        val_loss_sup /= len(val_loader)
                        val_loss_kd /= len(val_loader)
                        if writer is not None:
                            writer.add_scalar("Val Loss (TOTAL)", val_loss, epoch * len(train_loader) + step + 1)
                            writer.add_scalar("Val Loss (MLM)", val_loss_mlm, epoch * len(train_loader) + step + 1)
                            writer.add_scalar("Val Loss (CLASS)", val_loss_sup, epoch * len(train_loader) + step + 1)
                            writer.add_scalar("Val Loss (DISC)", val_loss_kd, epoch * len(train_loader) + step + 1)
                            writer.add_scalar("Mask prob", model.module.model_config.mask_prob, epoch * len(train_loader) + step + 1)
                            writer.add_scalar("Disc lamb", model.module.model_config.lamb_reverse, epoch * len(train_loader) + step + 1)
                            print(f"Epoch: {epoch}, Step: {step + 1}/{len(train_loader)}, Val Loss (TOTAL): {val_loss:.4f}, Val Loss (MLM): {val_loss_mlm:.4f}, Val Loss (CLASS): {val_loss_sup:.4f}, Val Loss (DISC): {val_loss_kd:.4f}")

                            # save only for the writer gpu
                            save_checkpoint(epoch = epoch, step = step, model = model, optimizer = optimizer, scheduler = scheduler, loss = running_loss,
                                            path = f"{model.module.model_config.checkpoint_path}{model.module.model_config.checkpoint_prefix}_{epoch}_{step}.pth")
                    
                    checkpoint_counter = 0                
                # sync all gpus after eval
                dist.barrier()
               
            initial_step = 0

    # save the final model, also only for the writer gpu
    if writer is not None:
        save_checkpoint(epoch = model.module.model_config.n_epoch, step = 0, model = model, optimizer = optimizer, scheduler = scheduler, loss = running_loss,
                        path = f"{model.module.model_config.checkpoint_path}{model.module.model_config.checkpoint_prefix}_{model.module.model_config.n_epoch}.pth")
        
