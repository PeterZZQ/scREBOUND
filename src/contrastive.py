import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.functional import sigmoid, softmax

def compute_cross_entropy(p, q):
    q = nn.functional.log_softmax(q, dim=-1)
    loss = torch.sum(p * q, dim=-1)
    return - loss.mean()

def stablize_logits(logits):
    logits_max, _ = torch.max(logits, dim=-1, keepdim=True)
    logits = logits - logits_max.detach()
    return logits

@torch.no_grad()
def concat_all_gather(tensor):
    """
    Performs all_gather operation on the provided tensors.
    *** Warning ***: torch.distributed.all_gather has no gradient.
    NOTE: requires that the tensor to be the same size across gpus, which is not true with the filtered samples
    """
    tensors_gather = [torch.ones_like(tensor)
                      for _ in range(dist.get_world_size())]
    dist.all_gather(tensors_gather, tensor, async_op=False)

    # output = torch.cat(tensors_gather, dim=0)
    output = tensors_gather
    return output


def exact_equal(A, B, chunk_size = 16):
    # result = torch.all(torch.eq(A[:, None, :], B[None, :, :]), dim=2) 
    # return result

    results = []
    for start in range(0, A.shape[0], chunk_size):
        end = start + chunk_size
        partial = torch.all(A[start:end, None, :] == B[None, :, :], dim=2)
        results.append(partial)
    result = torch.cat(results, dim=0)
    return result

def full_equal(A, B):
    return (A @ B.T) != 0


# Memory consumption too large
# def find_ancestor(A, B):
#     """
#     Find the ancester cells
#     Using bincode, each cell's ancester cell should have at least some common 1s with the cell and no other 1s
#     including itself
#     return a cell by cell binary matrix, where 1 denote ancestor, 0 denote non-ancestor
#     """
#     # result = torch.sum(torch.logical_or(A[:, None, :], B[None, :, :]), dim = 2)
#     result = torch.sum(A[:, None, :] | A[None, :, :], dim=2)
#     # result = torch.sum((A[:, None, :] + B[None, :, :]) > 0, dim = 2)
#     return result <= torch.sum(A, dim = 1, keepdim = True)

def find_ancestor(A, B, chunk_size = 16):
    results = []
    for start in range(0, A.shape[0], chunk_size):
        end = start + chunk_size
        # partial = torch.sum(A[start:end, None, :].bool() | B[None, :, :].bool(), dim=2)
        partial = torch.sum(torch.logical_or(A[start:end, None, :], B[None, :, :]), dim=2)
        results.append(partial)

    result = torch.cat(results, dim=0)
    return result <= torch.sum(A, dim = 1, keepdim = True)

def find_descend(A, B):
    """ 
    Find the descendent cells
    using bincode, the descendent cells should have 1 for all 1s in the cell bincode
    including itself
    return a cell by cell binary matrix, where 1 denote descendent, 0 denote non-descendent
    """
    result = A @ B.T 
    return result == torch.sum(A, dim = 1, keepdim = True)


class SupContrLoss(nn.Module):
    """
    Supervised contrastive loss
    """
    def __init__(self, temperature: float = 0.1):
        super(SupContrLoss, self).__init__()
        self.temperature = temperature
    
    def forward(self, features, label_mtx, batch_ids = None):
        """
        features: feature embedding, of shape (ncells, nfeats)
        label_mtx: binary identification matrix, of shape (ncells, ncells), pos: 1, neutral: 0, neg: -1
        """
        device = features.device
        
        # update the label_mtx is batch is not None
        if batch_ids is not None:
            # NOTE: if the batch label is provided, the contrastive loss is applied only across batches to better remove batch effect
            # positive sample only includes the samples of the same cell type across batchs, but should the samples of the same cell type within the same batch be negative samples?
            batch_ids = batch_ids.contiguous().view(-1, 1)
            # (ncells, ncells) batch identification matrix, sample cell type in same batch: 1, remaining batch: 0
            extra_neutral = torch.eq(batch_ids, batch_ids.T).float().to(device) * (label_mtx == 1).float()
            # remove self-similarity
            assert torch.all(torch.diag(extra_neutral) == 1)
            # these samples are neutral
            label_mtx[extra_neutral.bool()] = 0
        else:
            # remove self-similarity
            extra_neutral = torch.eye(label_mtx.shape[0]).to(label_mtx.device)
            label_mtx[extra_neutral] = 0

        # -------------------------------------------
        # Contrastive loss with ground truth matrix provided, label_mtx
        # compute logits
        logits = torch.matmul(features, features.T) / self.temperature
        # optional: minus the largest logit to stablize logits
        logits = stablize_logits(logits)

        neutral_label = (label_mtx == 0)
        logits.masked_fill_(neutral_label, -1e7)
        # compute ground-truth distribution
        # for each sample, sum mask between the sample and all remaining samples, and clamp by 1 min to prevent 0 sum
        # more neighboring, more even the p is after normalization
        pos_label = (label_mtx == 1).float()
        p = pos_label / pos_label.sum(1, keepdim=True).clamp(min=1.0)

        # cross entropy loss, select and calculate only on non-neutral mask 
        loss = compute_cross_entropy(p, logits)

        # -------------------------------------------

        return loss
    


class SupContrLossMultiGPUs(nn.Module):
    """
    Supervised contrastive loss
    """
    def __init__(self, label_asso_mtx: torch.Tensor, temperature: float = 0.1, unknown_label: int | None = None):
        super(SupContrLoss, self).__init__()
        self.label_asso_mtx = label_asso_mtx
        self.unknown_label = unknown_label
        self.temperature = temperature
    
    def forward(self, features, label_ids, batch_ids = None):
        """
        features: feature embedding, of shape (ncells, nfeats)
        label_mtx: binary identification matrix, of shape (ncells, ncells), pos: 1, neutral: 0, neg: -1
        """
        device = features.device

        local_batch_size = features.size(0)
        # get all features across gpus
        all_features = torch.cat(concat_all_gather(features), dim = 0)
        # get all labels across gpus
        all_label_ids = torch.cat(concat_all_gather(label_ids), dim = 0)

        # label by all_labels association matrix
        label_asso = self.label_asso_mtx[label_ids.unsqueeze(1), all_label_ids.unsqueeze(0)]

        # remove self-similarity, 0 for self-similarity
        self_sim_mask = torch.scatter(torch.ones_like(label_asso), 1,
                                      torch.arange(label_asso.shape[0]).view(-1, 1).to(device) +
                                      local_batch_size * dist.get_rank(),
                                      0).bool().to(device)

        # update the label_mtx is batch is not None
        if batch_ids is not None:
            # NOTE: if the batch label is provided, the contrastive loss is applied only across batches to better remove batch effect
            # positive sample only includes the samples of the same cell type across batchs, but should the samples of the same cell type within the same batch be negative samples?
            batch_ids = batch_ids.contiguous().view(-1, 1)
            all_batch_ids = torch.cat(concat_all_gather(batch_ids), dim = 0)

            # (ncells, ncells) batch identification matrix, sample cell type in same batch: 1, remaining batch: 0
            extra_neutral = torch.eq(batch_ids, all_batch_ids.T).float().to(device) * (label_asso == 1).float()
            # remove self-similarity
            assert torch.all(extra_neutral[~self_sim_mask] == 1)
            # these samples are neutral
            label_asso[extra_neutral.bool()] = 0
        else:
            # remove self-similarity
            label_asso[~self_sim_mask] = 0


        # -------------------------------------------
        # Contrastive loss with ground truth matrix provided, label_mtx
        # compute logits
        logits = torch.matmul(features, all_features.T) / self.temperature

        # drop the unknown after calculating all the metrics
        if self.unknown_label is not None:
            keep_idx = (label_ids != self.unknown_label)
            all_keep_idx = (all_label_ids != self.unknown_label)
            
            label_asso = label_asso[keep_idx][:, all_keep_idx]
            logits = logits[keep_idx][:, all_keep_idx]

        # optional: minus the largest logit to stablize logits
        logits = stablize_logits(logits)
        neutral_label = (label_asso == 0)
        logits.masked_fill_(neutral_label, -1e7)
        # compute ground-truth distribution
        # for each sample, sum mask between the sample and all remaining samples, and clamp by 1 min to prevent 0 sum
        # more neighboring, more even the p is after normalization
        pos_label = (label_asso == 1).float()
        p = pos_label / pos_label.sum(1, keepdim=True).clamp(min=1.0)

        # cross entropy loss, select and calculate only on non-neutral mask 
        loss = compute_cross_entropy(p, logits)
        # -------------------------------------------

        return loss



class _MultiPosConLoss(nn.Module):
    """
    Multi-Positive Contrastive Loss: https://arxiv.org/pdf/2306.00984.pdf
    """

    def __init__(self,
                 temperature: float = 0.1,
                 use_bincode_label: bool = False):
        super(_MultiPosConLoss, self).__init__()
        self.temperature = temperature
        self.use_bincode_label = use_bincode_label

    def set_temperature(self, temp=0.1):
        self.temperature = temp

    def forward(self, features, labels, batchs = None, mask_samples = None):
        """
        NOTE: mask_samples should include all unknowns, not necessary for single-gpu, just drop them
        """
        device = features.device
        # normalize the features, cell embed already normalized
        # features = nn.functional.normalize(features, dim=-1, p=2)

        # ground truth
        if self.use_bincode_label:
            # for every label bincode, only when the neighboring label bincode have the same 1 position (and more) have inner product == label.sum
            # these neighbors are either same type (exactly same) or descendent (same and more), these are positive 
            mask = find_descend(labels, labels).float().to(device)
            neutral_mask = (find_ancestor(labels, labels).float() - exact_equal(labels, labels).float()).to(device)
            # TODO: save memory, precalculate the label by label ancestral descendent mask, then sample it 
        else:
            # NOTE: the integer label, exactly the same
            labels = labels.contiguous().view(-1, 1)
            mask = torch.eq(labels, labels.T).float().to(device) 
            full_mask = None
            neutral_mask = None     

        if batchs is not None:
            # NOTE: if the batch label is provided, the contrastive loss is applied only across batches to better remove batch effect
            # positive sample only includes the samples of the same cell type across batchs, but should the samples of the same cell type within the same batch be negative samples?
            batchs = batchs.contiguous().view(-1, 1)
            mask_batch = torch.eq(batchs, batchs.T).float().to(device) * mask
            mask_batch.fill_diagonal_(0)
            # neutral mask also include the cells of the same cell type and from the same batch
            neutral_mask += mask_batch
            # also need to update the mask
            mask = mask * (1 - mask_batch)

        # compute logits
        logits = torch.matmul(features, features.T) / self.temperature
        # TODO: Adjust the temperature to increase the cross-batch alignment penalty

        # optional: minus the largest logit to stablize logits
        logits = stablize_logits(logits)

        # NOTE: remove gradient calculation 
        # 1. remove self-similarity
        mask.fill_diagonal_(0)  
        logits.fill_diagonal_(-1e9)
        # 2. neutral mask
        if neutral_mask is not None:
            logits = logits - neutral_mask * 1e9
        # compute ground-truth distribution
        # for each sample, sum mask between the sample and all remaining samples, and clamp by 1 min to prevent 0 sum
        # more neighboring, more even the p is after normalization
        p = mask / mask.sum(1, keepdim=True).clamp(min=1.0)

        # cross entropy loss, select and calculate only on non-neutral mask 
        loss = compute_cross_entropy(p, logits)

        return loss



class _MultiPosConLossMultiGPUs(nn.Module):
    """
    Multi-Positive Contrastive Loss: https://arxiv.org/pdf/2306.00984.pdf
    """

    def __init__(self,
                 temperature: float = 0.1,
                 use_bincode_label: bool = False):
        super(_MultiPosConLossMultiGPUs, self).__init__()
        self.temperature = temperature
        self.use_bincode_label = use_bincode_label

    def set_temperature(self, temp=0.1):
        self.temperature = temp

    def forward(self, features, labels, batchs = None, mask_samples = None):

        device = features.device
        feats = features
        # feats = nn.functional.normalize(features, dim=-1, p=2)
        local_batch_size = feats.size(0)

        # get all masked samples
        # all_mask_samples = torch.cat(concat_all_gather(mask_samples), dim = 0)
        # concatenate all features across devices (for multi-gpus training)
        all_feats = torch.cat(concat_all_gather(feats), dim = 0)
        # concatenate all labels across devices (for multi-gpus training)
        all_labels = concat_all_gather(labels) 

        # compute the mask based on labels
        # labels by all_labels mask, 1 for the same label, 0 for different label

        if self.use_bincode_label:
            # NOTE: bincode label, use the binary vector
            # mask = exact_equal(labels, all_labels).float()
            mask = []
            neutral_mask = []
            # save gpu memory
            for labels_i in all_labels:
                mask.append(find_descend(labels, labels_i).float())
                neutral_mask.append((find_ancestor(labels, labels_i).float() - exact_equal(labels, labels_i).float()))
            mask = torch.cat(mask, dim = 1)
            neutral_mask = torch.cat(neutral_mask, dim = 1)
        else:
            # NOTE: the integer label, exactly the same
            mask = torch.eq(labels.view(-1, 1), all_labels.contiguous().view(1, -1)).float()
            full_mask = None
            neutral_mask = None

        # remove self-similarity, 0 for self-similarity
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(mask.shape[0]).view(-1, 1).to(device) +
            local_batch_size * dist.get_rank(),
            0
        )

        # NOTE: remove gradient calculation of certain logits 
        # set self-similarity to be -1e9, so that softmax produce 0
        logits = torch.matmul(feats, all_feats.T) / self.temperature
        # optional: minus the largest logit to stablize logits
        logits = stablize_logits(logits)

        # remove self-similarity
        mask = mask * logits_mask

        if batchs is not None:
            all_batchs = concat_all_gather(batchs)
            # NOTE: if the batch label is provided, the contrastive loss is applied only across batches to better remove batch effect
            batchs = batchs.contiguous().view(-1, 1)
            mask_batch = torch.eq(batchs.view(-1,1), all_batchs.contiguous().view(1, -1)).float().to(device) * mask
            # neutral mask also include the cells of the same cell type and from the same batch
            neutral_mask += mask_batch
            # also need to update the mask
            mask = mask * (1 - mask_batch)

        logits = logits - (1 - logits_mask) * 1e9
        # 2. neutral mask
        if neutral_mask is not None:
            logits = logits - neutral_mask * 1e9
        
        # Now remove the unknown
        logits = logits[~mask_samples, :]#[:, ~all_mask_samples]
        mask = mask[~mask_samples, :]#[:, ~all_mask_samples]
        # compute ground-truth distribution
        p = mask / mask.sum(1, keepdim=True).clamp(min=1.0)

        # TODO: need to drop neutral for cross-entropy calculation in each cell
        loss = compute_cross_entropy(p, logits)
        return loss




def bce_weighted(pred, target, weight = None, eps = 1e-12):
    """
    Compute the binary cross-entropy loss from scratch.

    Args:
        pred (torch.Tensor): Predicted probabilities (values between 0 and 1) of shape (N, *).
        target (torch.Tensor): Ground truth binary labels (0 or 1) with the same shape as pred.
        eps (float): A small value to avoid log(0).

    Returns:
        torch.Tensor: The mean binary cross-entropy loss.
    """
    # Clamp predictions to avoid log(0)

    pred = torch.clamp(sigmoid(pred), eps, 1 - eps)
    
    # Compute the element-wise binary cross-entropy loss
    if weight is None:
        loss = - (target * torch.log(pred) + (1 - target) * torch.log(1 - pred))
    else:
        weight = softmax(weight, dim = 0)
        loss = - (target * torch.log(pred) * weight[None, :] + (1 - target) * torch.log(1 - pred) * weight[None, :])
        
    # Return the mean loss over all elements
    return loss.sum(dim = 1).mean()


'''
class _MultiPosConLossMultiGPUs(nn.Module):
    """
    Multi-Positive Contrastive Loss: https://arxiv.org/pdf/2306.00984.pdf
    """

    def __init__(self, temperature=0.1):
        super(_MultiPosConLossMultiGPUs, self).__init__()
        self.temperature = temperature
        self.logits_mask = None
        self.mask = None
        self.last_local_batch_size = None

    def set_temperature(self, temp=0.1):
        self.temperature = temp

    def forward(self, features, labels, batchs = None):

        device = features.device
        feats = nn.functional.normalize(features, dim=-1, p=2)
        local_batch_size = feats.size(0)

        # concatenate all features across devices (for multi-gpus training)
        all_feats = concat_all_gather(feats)
        # concatenate all labels across devices (for multi-gpus training)
        all_labels = concat_all_gather(labels)  

        # compute the mask based on labels
        if local_batch_size != self.last_local_batch_size:
            # labels by all_labels mask, 1 for the same label, 0 for different label
            mask = torch.eq(labels.view(-1, 1),
                            all_labels.contiguous().view(1, -1)).float().to(device)
            self.logits_mask = torch.scatter(
                torch.ones_like(mask),
                1,
                torch.arange(mask.shape[0]).view(-1, 1).to(device) +
                local_batch_size * dist.get_rank(),
                0
            )

            self.last_local_batch_size = local_batch_size
            self.mask = mask * self.logits_mask

        mask = self.mask

        # compute logits
        logits = torch.matmul(feats, all_feats.T) / self.temperature
        logits = logits - (1 - self.logits_mask) * 1e9

        # optional: minus the largest logit to stablize logits
        logits = stablize_logits(logits)

        # compute ground-truth distribution
        p = mask / mask.sum(1, keepdim=True).clamp(min=1.0)
        loss = compute_cross_entropy(p, logits)

        return loss
'''