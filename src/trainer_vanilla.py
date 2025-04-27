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

import data_utils

CAST_DTYPE = torch.bfloat16

def save_checkpoint(epoch, step, model, optimizer, scheduler, loss, path):
    checkpoint = {
        'epoch': epoch,
        'step': step,
        'model_state_dict': model.module.state_dict(),
        'model_config': model.module.model_config, # save the model config for repeated training too
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'loss': loss,
        'compression_mask': model.module.compression_mask
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
        batch_sample = data_sample["batch"].reshape(-1).to(model_acc.device, non_blocking = True)
        
        all_embed, cell_embed, mask = model(counts_norm = expr_sample)
        expr_pred, expr_pred_meta = model_acc.predict_expr(cell_embed = cell_embed, batch_cond = batch_sample)

        loss_mlm = ((expr_pred_meta - model_acc.gene_compression(expr_sample)) * mask[:, 1:]).pow(2).sum(1).mean()
        return loss_mlm


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

    # update the mask_prob for evaluation
    if mask_prob is not None:
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
                expr_sample = data_sample["counts_norm"].reshape(-1, data_sample["counts_norm"].shape[-1]).to(model_acc.device, non_blocking = True)
                all_embed, cell_embed, mask = model(counts_norm = expr_sample)
            
            cell_embeds.append(sparse.csr_matrix(cell_embed.to(torch.float32).detach().cpu().numpy())) 
            if "batch" in data_sample.keys(): 
                batchs.append(data_sample["batch"].reshape(-1).detach().cpu().numpy())
            else:
                batchs.append(np.array([np.nan] * cell_embed.shape[0]))

            if "label" in data_sample.keys():
                labels.append(data_sample["label"].reshape(-1).detach().cpu().numpy())
            else:
                labels.append(np.array([np.nan] * cell_embed.shape[0]))

    cell_embeds = sparse.vstack(cell_embeds)
    labels = np.concatenate(labels, axis = 0)
    batchs = np.concatenate(batchs, axis = 0)
    meta = pd.DataFrame.from_dict({"labels": labels, "batchs": batchs})
    adata = anndata.AnnData(X = cell_embeds, obs = meta.astype("category"))

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
    val_dataset = data_utils.sc_partition(data_path = dataset_dict["DIR"], min_chunksize = min_chunksize, normalize = model.module.model_config.lognorm_data)
    val_dataset.load_partition(idx = dataset_dict["num_partitions"] - 1, label_colname = "label_id", batch_colname = "batch_id", data_prefix = dataset_dict["data_prefix"], meta_prefix = dataset_dict["meta_prefix"]) # use last chunk
    val_loader = data.DataLoader(val_dataset, batch_size = model.module.model_config.batch_size//min_chunksize, shuffle = False, pin_memory = True,
                                 sampler = DistributedSampler(val_dataset, shuffle = False), num_workers = 8, prefetch_factor = 8)
    train_dataset = data_utils.sc_partition(data_path = dataset_dict["DIR"], min_chunksize = min_chunksize, normalize = model.module.model_config.lognorm_data)
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
        running_loss = 0.0
        # shuffle the partition for each epoch
        for partition_idx in np.random.permutation(dataset_dict["num_partitions"] - 1):
            print(f"GPU {global_rank} - Start training Epoch {epoch:02d}, Partition {partition_idx:02d}...")
            torch.cuda.empty_cache()
            # load training dataset for partition_idx
            train_dataset.load_partition(idx = partition_idx, label_colname = "label_id", batch_colname = "batch_id", data_prefix = dataset_dict["data_prefix"], meta_prefix = dataset_dict["meta_prefix"])
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
                    loss = infer_databatch(model, data_sample, multigpus = True)

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
            
                if step % log_step == log_step - 1:
                    # calculate for each gpus
                    running_loss /= log_step
                    if writer is not None:
                        cum_step = epoch * len(train_loader) * dataset_dict["num_partitions"] + step + 1
                        # only write/print the running loss for one gpu with writer
                        writer.add_scalar("Train Loss (TOTAL)", running_loss, cum_step)
                        writer.add_scalar("Learning rate", model.module.scheduler.get_last_lr()[0], cum_step)

                        print(f"Epoch: {epoch}, Step: {step + 1}/{len(train_loader) * dataset_dict["num_partitions"]}, Learning rate: {model.module.scheduler.get_last_lr()[0]:.2e}, Mask prob: {model.module.model_config.mask_prob:.4f}, Train Loss (TOTAL): {running_loss:.4f}")
                    
                    running_loss = 0.0
                    checkpoint_counter += 1

                    # model evaluation and checkpoint saving
                    if (checkpoint_counter == 10):
                        model.module.eval()
                        with torch.no_grad():
                            val_loss = 0.0
                            for data_sample in val_loader:
                                with autocast(device_type="cuda", dtype = CAST_DTYPE):

                                    loss = infer_databatch(model, data_sample, multigpus = True)                            
                                    val_loss += loss.item()

                            # log the values
                            val_loss /= len(val_loader)
                            if writer is not None:
                                cum_step = epoch * len(train_loader) * dataset_dict["num_partitions"] + step + 1
                                writer.add_scalar("Val Loss (TOTAL)", val_loss, cum_step)
                                writer.add_scalar("Mask prob", model.module.model_config.mask_prob, cum_step)
                                print(f"Epoch: {epoch}, Step: {step + 1}/{len(train_loader) * dataset_dict["num_partitions"]}, Val Loss (TOTAL): {val_loss:.4f}")

                                # save only for the writer gpu
                                save_checkpoint(epoch = epoch, step = step, model = model, optimizer = model.module.optimizer, scheduler = model.module.scheduler, loss = running_loss,
                                                path = f"{model.module.model_config.checkpoint_path}{model.module.model_config.checkpoint_prefix}_{epoch}_{step + 1}.pth")
                        
                        checkpoint_counter = 0                

                    # update the mask_prob 
                    if model.module.model_config.dynamic_maskprob:
                        model.module.model_config.mask_prob += model.module.model_config.maskprob_step

                    # sync all gpus after eval
                    dist.barrier()
                step += 1   
        initial_step = 0

    # save the final model, also only for the writer gpu
    if writer is not None:
        save_checkpoint(epoch = model.module.model_config.n_epoch, step = 0, model = model, optimizer = model.module.optimizer, scheduler = model.module.scheduler, loss = running_loss,
                        path = f"{model.module.model_config.checkpoint_path}{model.module.model_config.checkpoint_prefix}_{model.module.model_config.n_epoch}.pth")
        




def train_singlegpu(model, dataset_dict, initial_epoch, initial_step, log_step):

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
    val_dataset = data_utils.sc_partition(data_path = dataset_dict["DIR"], min_chunksize = min_chunksize, normalize = False)
    val_dataset.load_partition(idx = dataset_dict["num_partitions"] - 1, label_colname = "label_id", batch_colname = "batch_id", data_prefix = dataset_dict["data_prefix"], meta_prefix = dataset_dict["meta_prefix"]) # use last chunk
    val_loader = data.DataLoader(val_dataset, batch_size = model.model_config.batch_size//min_chunksize, shuffle = False, pin_memory = True, num_workers = 8, prefetch_factor = 8)
    train_dataset = data_utils.sc_partition(data_path = dataset_dict["DIR"], min_chunksize = min_chunksize, normalize = False)
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
        running_loss = 0.0
        # shuffle the partition for each epoch
        for partition_idx in np.random.permutation(dataset_dict["num_partitions"] - 1):
            print(f"GPU - Start training Epoch {epoch:02d}, Partition {partition_idx:02d}...")
            torch.cuda.empty_cache()
            # load training dataset for partition_idx
            train_dataset.load_partition(idx = partition_idx, label_colname = "label_id", batch_colname = "batch_id", data_prefix = dataset_dict["data_prefix"], meta_prefix = dataset_dict["meta_prefix"])
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
                    loss = infer_databatch(model, data_sample, multigpus = False)

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
            
                if step % log_step == log_step - 1:
                    # calculate for each gpus
                    running_loss /= log_step

                    print(f"Epoch: {epoch}, Step: {step + 1}/{len(train_loader) * dataset_dict["num_partitions"]}, Learning rate: {model.scheduler.get_last_lr()[0]:.2e}, Mask prob: {model.model_config.mask_prob:.4f}, Train Loss (TOTAL): {running_loss:.4f}")
                    
                    running_loss = 0.0
                    checkpoint_counter += 1

                    # model evaluation and checkpoint saving
                    if (checkpoint_counter == 10):
                        model.eval()
                        with torch.no_grad():
                            val_loss = 0.0
                            for data_sample in val_loader:
                                with autocast(device_type="cuda", dtype = CAST_DTYPE):

                                    loss = infer_databatch(model, data_sample, multigpus = False)                            
                                    val_loss += loss.item()

                            # log the values
                            val_loss /= len(val_loader)
                            print(f"Epoch: {epoch}, Step: {step + 1}/{len(train_loader) * dataset_dict["num_partitions"]}, Val Loss (TOTAL): {val_loss:.4f}")
                        
                        checkpoint_counter = 0                

                    # update the mask_prob 
                    if model.model_config.dynamic_maskprob:
                        model.model_config.mask_prob += model.model_config.maskprob_step

                step += 1   
        initial_step = 0



'''
def train_singlegpu(model, train_loader, val_loader, optimizer, scheduler, writer, initial_epoch, initial_step, log_step):
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
    if model.model_config.dynamic_maskprob:
        mask_prob_init = 0.2
        mask_prob_end = 0.5
        mask_prob_step = (mask_prob_end - mask_prob_init) / len(train_loader) / (model.model_config.n_epoch - initial_epoch) * log_step
        model.model_config.mask_prob = mask_prob_init
    

    # NOTE: training loop
    checkpoint_counter = 0
    if model.model_config.use_fastatten:
        torch.backends.cuda.enable_flash_sdp(True)
        torch.backends.cuda.enable_mem_efficient_sdp(True)
        torch.backends.cuda.enable_math_sdp(True)
        
    for epoch in range(initial_epoch, model.model_config.n_epoch):
        torch.cuda.empty_cache()
        # Disable tqdm on all nodes except the rank 0 GPU on each server
        batch_iterator = tqdm.tqdm(train_loader, desc=f"Processing Epoch {epoch:02d}")

        # NOTE: Training
        running_loss = 0.0

        for step, data_sample in enumerate(batch_iterator):
            model.train()
            optimizer.zero_grad()
            
            if step < initial_step:
                continue

            with autocast(device_type="cuda", dtype = CAST_DTYPE):
                # NOTE: need to process the label into bincode
                if model.model_config.sup_type is not None:
                    data_sample["label"] = model.label_bincode[data_sample["label"],:]

                loss = infer_databatch(model, data_sample, multigpus = False)

            scaler.scale(loss).backward()

            #Unscale the optimizer to clip gradients
            scaler.unscale_(optimizer)
            # clip gradient
            max_grad_norm = 1.0 
            clip_grad_norm_(model.parameters(), max_grad_norm)


            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            # NOTE: log the results
            running_loss += loss.item()
            
            if step % log_step == log_step - 1:
                # calculate for each gpus
                running_loss /= log_step
                print(f"Epoch: {epoch}, Step: {step + 1}/{len(train_loader)}, Learning rate: {scheduler.get_last_lr()[0]:.2e}, Train Loss (TOTAL): {running_loss:.4f}")
                
                running_loss = 0.0
                checkpoint_counter += 1
                
                # model evaluation and checkpoint saving
                if (checkpoint_counter == 10):
                    model.eval()
                    with torch.no_grad():
                        val_loss = 0.0
                        for data_sample in val_loader:
                            with autocast(device_type="cuda", dtype = CAST_DTYPE):
                                # NOTE: need to process the label into bincode
                                if model.model_config.sup_type is not None:
                                    data_sample["label"] = model.module.label_bincode[data_sample["label"],:]

                                loss = infer_databatch(model, data_sample, multigpus = False)                            
                                val_loss += loss.item()

                        # log the values
                        val_loss /= len(val_loader)
                        print(f"Epoch: {epoch}, Step: {step + 1}/{len(train_loader)}, Val Loss (TOTAL): {val_loss:.4f}")

                    checkpoint_counter = 0                
                
                # update the mask_prob 
                if model.model_config.dynamic_maskprob:
                    model.model_config.mask_prob += mask_prob_step
                    # constraint to not exceed end
                    model.model_config.mask_prob = min(model.model_config.mask_prob, mask_prob_end)
               
            initial_step = 0

'''
