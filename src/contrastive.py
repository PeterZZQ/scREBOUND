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

    output = torch.cat(tensors_gather, dim=0)
    return output


def exact_equal(A, B):
    result = torch.all(torch.eq(A[:, None, :], B[None, :, :]), dim=2) 
    return result

def full_equal(A, B):
    return (A @ B.T) != 0

class MultiPosConLoss(nn.Module):
    """
    Multi-Positive Contrastive Loss: https://arxiv.org/pdf/2306.00984.pdf
    """

    def __init__(self,
                 temperature: float = 0.1,
                 use_bincode_label: bool = False):
        super(MultiPosConLoss, self).__init__()
        self.temperature = temperature
        self.use_bincode_label = use_bincode_label

    def set_temperature(self, temp=0.1):
        self.temperature = temp

    def forward(self, features, labels, batchs = None):
        device = features.device
        # normalize the features
        features = nn.functional.normalize(features, dim=-1, p=2)

        # ground truth
        if self.use_bincode_label:
             # NOTE: bincode label, use the binary vector
            mask = exact_equal(labels, labels).float().to(device)
            # full mask, bin_code label have common ancestor, larger number of positives
            full_mask = full_equal(labels, labels).float().to(device)
            # neutral mask: full mask except for the exact mask
            neutral_mask = full_mask - mask
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



class MultiPosConLossMultiGPUs(nn.Module):
    """
    Multi-Positive Contrastive Loss: https://arxiv.org/pdf/2306.00984.pdf
    """

    def __init__(self,
                 temperature: float = 0.1,
                 use_bincode_label: bool = False):
        super(MultiPosConLossMultiGPUs, self).__init__()
        self.temperature = temperature
        self.use_bincode_label = use_bincode_label

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
        # labels by all_labels mask, 1 for the same label, 0 for different label

        if self.use_bincode_label:
            # NOTE: bincode label, use the binary vector
            mask = exact_equal(labels, all_labels).float().to(device)
            # full mask, bin_code label have common ancestor, larger number of positives
            full_mask = full_equal(labels, all_labels).float().to(device)
            # neutral mask: full mask except for the exact mask
            neutral_mask = full_mask - mask
        else:
            # NOTE: the integer label, exactly the same
            mask = torch.eq(labels.view(-1, 1), all_labels.contiguous().view(1, -1)).float().to(device)
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