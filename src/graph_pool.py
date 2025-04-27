import torch 
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import base_model

# ----------------------------------------------------------------------------------------
#
# Learnable gene grouping
#
# ----------------------------------------------------------------------------------------
def knn_graph(features, k, include_self: bool = True):
    """
    features: Tensor of shape [N, D] (N nodes, D features)
    k: Number of nearest neighbors
    Returns: Adjacency matrix [N, N] (binary, symmetric)
    """
    print("construct knn graph from embedding.")
    N = features.size(0)

    # Compute pairwise squared Euclidean distance matrix: [N, N]
    dist = torch.cdist(features, features, p=2)  # L2 distance

    # Mask self-distances
    dist.fill_diagonal_(float('inf'))

    # Find k smallest distances per row (i.e. nearest neighbors)
    knn_indices = dist.topk(k, dim=1, largest=False).indices  # [N, k]

    # Create adjacency matrix
    A = torch.zeros(N, N, device=features.device)
    A.scatter_(1, knn_indices, 1.0)  # Directed edges

    # Make the graph undirected (optional)
    A = torch.maximum(A, A.T)
    A = A + torch.eye(A.size(0)).to(A.device)

    return A


class GCNLayer(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super(GCNLayer, self).__init__()
        self.linear = nn.Linear(in_features, out_features, bias=bias)

    def forward(self, X, A):
        """
        X: Node features, shape [N, F]
        A: Adjacency matrix, shape [N, N] (dense, unnormalized)
        """
        # Add self-loops
        I = torch.eye(A.size(0), device=A.device)
        A_hat = A + I

        # Compute degree matrix
        D_hat = torch.diag(torch.sum(A_hat, dim=1))

        # Normalize adjacency
        D_inv_sqrt = torch.pow(D_hat, -0.5)
        D_inv_sqrt[torch.isinf(D_inv_sqrt)] = 0.0
        A_norm = D_inv_sqrt @ A_hat @ D_inv_sqrt

        # GCN propagation
        out = A_norm @ X
        out = self.linear(out)
        return out
    
class DiffPool(nn.Module):
    def __init__(self, in_features, n_meta, n_embed):
        super(DiffPool, self).__init__()
        self.embed = GCNLayer(in_features = in_features, out_features = n_embed)
        self.assign_mat = GCNLayer(in_features = in_features, out_features = n_meta)

    def forward(self, X, A):
        # node/gene embed, of shape (n_genes, n_embed)
        z_l = self.embed(X, A)
        # node/gene assignment of shape (n_genes, n_meta)
        s_l = F.softmax(self.assign_mat(X, A), dim=-1)
        
        # Pooling
        # meta embed, of shape (n_meta, n_embed) 
        X_next = s_l.T @ z_l
        # meta graph, of shape (n_meta, n_meta)
        A_next = s_l.T @ A @ s_l

        return X_next, A_next, s_l



class MincutPool(nn.Module):
    def __init__(self, in_features, n_meta):
        super(MincutPool, self).__init__()
        self.assign_mat = nn.Sequential(nn.Linear(in_features = in_features, out_features = 512),
                                        nn.ReLU(),
                                        nn.Linear(in_features = 512, out_features = n_meta),
                                        nn.Softmax(dim = 1)
                                        )
    
    def forward(self, X, A):
        s_l = self.assign_mat(X)
        X_pool = s_l.T @ X
        A_pool = s_l.T @ A @ s_l

        # calculate the loss
        # normalize A
        D = torch.diag(torch.sum(A_pool, dim=1))
        D_inv_sqrt = torch.pow(D, -0.5)
        D_inv_sqrt[torch.isinf(D_inv_sqrt)] = 0.0
        A_norm = D_inv_sqrt @ A_pool @ D_inv_sqrt
        denom = s_l.T @ D @ s_l

        # min-cut loss
        L_c = -torch.trace(A_norm)/(torch.trace(denom) + 1e-7)
        inner_prod = s_l.T @ s_l
        inner_prod /= inner_prod.pow(2).sum().sqrt()
        # ortho-loss
        L_o = (inner_prod - torch.eye(inner_prod.shape[0])).pow(2).sum().sqrt()

        return X_pool, A_pool, s_l, L_c, L_o




def mincut_loss(A, S, add_orthogonality=True, eps=1e-9):
    """
    A: [N, N] Adjacency matrix (un-normalized)
    S: [N, k] Soft pooling assignment matrix (softmax over k)
    """
    N, k = S.shape
    D = torch.diag(A.sum(dim=1))  # Degree matrix

    # MinCut numerator and denominator
    cut = torch.trace(S.T @ A @ S)
    assoc = torch.trace(S.T @ D @ S)
    mincut_loss = -cut / (assoc + eps)

    if add_orthogonality:
        # Orthogonality regularization
        # SS = S.T @ S
        # SS_norm = SS / (torch.norm(S, p='fro')**2 + eps)

        S_norm = S/S.pow(2).sum(0, keepdim = True).sqrt()
        SS_norm = S_norm.T @ S_norm
        I = torch.eye(k, device=A.device)
        orth_loss = torch.norm(SS_norm - I, p='fro')

    else:
        orth_loss = torch.tensor([0]).to(mincut_loss.device)

    return mincut_loss, orth_loss
    

class GenePoolVanilla(nn.Module):
    def __init__(self, A, n_meta = 256, s_init = None, temp = 0.5):
        super(GenePoolVanilla, self).__init__()
        # Naive gene pool, no random walk, directly parameterize the pooling score
        self.A = A
        if s_init is None:
            self.S = nn.Parameter(torch.rand((self.A.shape[0], n_meta)))
        else:
            self.S = nn.Parameter(s_init.clone())
        self.softmax = nn.Softmax(dim = 1)
        self.temp = temp
        # self.normalization = base_model.MinMaxNormalization(eps = 1e-6)

    def get_score(self):
        return self.softmax(self.S/self.temp)
    
    def forward(self, expr, gene_embed, log_norm: bool = True):
        S = self.get_score()
        expr_pool = expr @ S
        gene_pool = S.T @ gene_embed

        if log_norm:
            expr_pool = expr_pool/(torch.sum(expr_pool, dim = 1, keepdim = True) + 1e-4) * 10e4
            expr_pool = torch.log1p(expr_pool)
        
        return S, gene_pool, expr_pool # self.normalization(expr_pool)
    

    def mincut_loss(self, add_ortho: bool = False):
        return mincut_loss(self.A, self.get_score(), add_orthogonality = add_ortho)


class GenePoolRW(nn.Module):
    def __init__(self, gene_embed, k_neighs = 10, n_hops = 4, n_meta = 256):
        super(GenePoolRW, self).__init__()
        self.gene_embed = gene_embed
        # graph construction, mnn symmetric graph
        A = knn_graph(self.gene_embed, k = k_neighs)
        # calculate the random walk
        self.R = A/(A.sum(dim = 1, keepdim = True) + 1e-6)
        
        # 4-step neighbors
        self.Rs = []
        self.weight = nn.Parameter(torch.randn(n_hops + 1))
        for hop in range(n_hops + 1):
            if hop == 0:
                self.Rs.append(torch.eye(self.R.shape[0]).to(self.R.device))
            else:
                self.Rs.append(torch.linalg.matrix_power(self.R, hop))

        self.assign_mat = nn.Sequential(nn.Linear(in_features = self.gene_embed.shape[0], out_features = n_meta),
                                        nn.Softmax(dim = 1))

    def forward(self, expr, log_norm: bool = True):

        weight_rw = F.softmax(self.weight)
        # 0 step, identity matrix
        R_comb = 0
        for weight, R in zip(weight_rw, self.Rs):
            R_comb += weight * R
        R_norm = R_comb/(R_comb.sum(dim = 1, keepdim = True) + 1e-6)

        s_l = self.assign_mat(R_norm)

        expr_pool = expr @ s_l
        gene_pool = s_l.T @ self.gene_embed

        if log_norm:
            expr_pool = expr_pool/(torch.sum(expr_pool, dim = 1, keepdim = True) + 1e-4) * 10e4
            expr_pool = torch.log1p(expr_pool)

        return R_comb, s_l, expr_pool, gene_pool 