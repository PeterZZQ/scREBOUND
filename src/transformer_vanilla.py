# Directly use the transformer provided by torch
import torch
import torch.nn as nn
import math
from dataclasses import dataclass
import base_model
from data_utils import set_seed

import flash_transformer_layer as flash_model

@dataclass
class ModelConfig:

    batch_size: int # Batch size
    n_epoch: int # Number of epochs to train
    lr: float # Learning rate
    d_embed: int
    n_head: int
    d_hidden: int
    n_layer: int
    d_output: int
    dropout: float
    mask_prob: float
    dynamic_maskprob: bool
    
    # model stats
    use_fastatten: bool
    precision: torch.dtype

    # paths
    checkpoint_path: str
    checkpoint_prefix: str
    pretrain_path: str | None

    lognorm_data: bool

def get_default_config() -> ModelConfig:

    return ModelConfig(
        batch_size=1024,
        n_epoch=3,
        lr=5e-4,
        d_embed=512, # dimension of each head == d_embed/n_head 512 = 8*64, 768 = 12*64
        n_head=8,
        d_hidden=2048,
        n_layer=4,
        d_output=256,
        dropout=0.05,
        mask_prob=0.15,
        dynamic_maskprob=False,
        use_fastatten=False,
        precision=torch.float32,
        pretrain_path=None,
        checkpoint_path="checkpoint/",
        checkpoint_prefix="checkpoint",
        lognorm_data=True
    )


class TransformerModel(nn.Module):

    def __init__(self, compression_mask: torch.Tensor | None, token_embed: torch.Tensor, model_config: ModelConfig, n_label: int,
                 n_batch: int, device: torch.device, seed: int = 0):
        """
        token_dim: the token dimensions, 5120 in UCE, following ESM2
        n_embed: the transformed token dimensions after the embedding, 1280 in UCE
        n_head: number of the attention head, 20 in UCE
        d_hid: the hidden layer dimensions, 5120 in UCE 
        n_layers: the number of layers, 4 layers in UCE github, in paper 33 layers
        output_dim: the output dimensions
        dropout: dropout rate
        """
        super().__init__()

        set_seed(seed)
        
        self.model_type = 'Transformer'
        self.model_config = model_config
        self.device = device

        # conditions
        self.n_batch = n_batch
        self.n_label = n_label
        # NOTE: token embed includes gene + cls special tokens
        self.n_tokens = token_embed.shape[0]

        # assign the index of the special token, hardcoded
        self.cls_idx = self.n_tokens - 1
        self.gene_idx = torch.arange(self.n_tokens - 1).to(self.device)

        self.mask_embed = nn.Parameter(torch.randn((1, self.model_config.d_embed)))

        # store the token embedding, of the shape (ntokens, n_token_dims)
        self.token_embed = nn.Embedding.from_pretrained(token_embed)
        # fix the token values, NOTE: UCE didn't fix the token_embed weight
        self.token_embed.weight.requires_grad = False

        # compression and decompression network
        if compression_mask is not None:
            self.compression_mask = compression_mask.to(self.device)
            self.gene_compression = CompressionModel(compression_mask = self.compression_mask)
            self.gene_decompression = DecompressionModel(compression_mask = self.compression_mask)
            # NOTE: sanity check, fix the gradient
            for param in self.gene_compression.parameters():
                param.requires_grad = False
            for param in self.gene_decompression.parameters():
                param.requires_grad = False

        else:
            # no transformation
            self.compression_mask = torch.eye(self.n_tokens - 1).to(self.device)
            # still need to min-max the input for compression
            self.gene_compression = MinMaxNormalization(eps = 1e-6)
            self.gene_decompression = nn.Identity()


        # gene name encoder, input: dimension of gene name embedding, token_dim
        # output: the n_embed, adopted from UCE
        self.gene_encoder = nn.Sequential(nn.Linear(token_embed.shape[1], self.model_config.d_embed),
                                          nn.GELU(),
                                          nn.LayerNorm(self.model_config.d_embed))


        # fourier position embedding
        self.expr_encoder = nn.Sequential(nn.Linear(32, self.model_config.d_embed),
                                            nn.GELU(),
                                            nn.LayerNorm(self.model_config.d_embed))


        
        if model_config.use_fastatten:
            # NOTE: need to use mixed precision
            self.transformer_encoder = flash_model.TransformerBlocks(d_model = self.model_config.d_embed,
                                                                    n_head = self.model_config.n_head,
                                                                    num_layers = self.model_config.n_layer,
                                                                    dim_feedforward = self.model_config.d_hidden,
                                                                    dropout = self.model_config.dropout,
                                                                    activation = "gelu")

        else:            
            self.transformer_encoder = base_model.TransformerBlocks(d_model = self.model_config.d_embed,
                                                                    n_head = self.model_config.n_head,
                                                                    num_layers = self.model_config.n_layer,
                                                                    dim_feedforward = self.model_config.d_hidden,
                                                                    dropout = self.model_config.dropout,
                                                                    activation = "gelu")

        # transform the output of the transformer m_embed -> output_dim
        self.decoder = nn.Sequential(base_model.full_block(self.model_config.d_embed, 256, self.model_config.dropout),
                                     base_model.full_block(256, self.model_config.d_output, self.model_config.dropout),
                                     base_model.full_block(self.model_config.d_output, self.model_config.d_output, self.model_config.dropout),
                                     nn.Linear(self.model_config.d_output, self.model_config.d_output))


        self.expr_predictor = base_model.decoder(
            n_input = self.model_config.d_output,
            n_output = self.n_tokens - 1, # number of genes, remove the cls
            n_cat_list = [self.n_batch],
            n_layers = 4,
            n_hidden = 512,
            dropout_rate = self.model_config.dropout,
            inject_covariates = True, # NOTE: True for deep injection
            use_batch_norm = False,
            use_layer_norm = True,
            dtype = self.model_config.precision
        )
        
        self.to(self.device)


    def forward(self, counts_norm: torch.Tensor, mask: torch.Tensor | None = None):
        """
        Parameters:
        --------------
            gene_sent: the gene name sentence, of the shape (n_batchs, n_tokens)
            expr_sent: the gene expression sentence, of the shape (n_batchs, n_tokens)
        """
        # only do min-max normalization for fourier value encoding
        counts_norm = self.gene_compression(counts_norm)
        # construct the gene name sentence (match position embedding)
        gene_sent = torch.tensor([[self.cls_idx] + [x for x in range(counts_norm.shape[1])]]).repeat(counts_norm.shape[0], 1).to(counts_norm.device)
        # construct the gene expression sentence (match the word embedding)
        expr_sent = torch.hstack([torch.zeros((counts_norm.shape[0], 1)).to(counts_norm.device), counts_norm])

        if mask is None:
            # add mask to the input data
            mask_prob = torch.full(gene_sent.shape, self.model_config.mask_prob).to(self.device)
            # do not add mask on padding and cls positions
            mask_prob[:, 0] = 0
            # sample mask using bernoulli
            mask = torch.bernoulli(mask_prob).bool()

        gene_embed = self.token_embed(gene_sent.long())
        gene_embed = self.gene_encoder(gene_embed) * math.sqrt(self.model_config.d_embed)
        expr_embed = base_model.fourier_positional_encoding(expr_sent.unsqueeze(2), embedding_dim = 32)
        expr_embed = self.expr_encoder(expr_embed) * math.sqrt(self.model_config.d_embed)
        
        # updated mask
        mask_embed = self.mask_embed.view(1, 1, -1)
        expr_embed = torch.where(mask.unsqueeze(-1), mask_embed, expr_embed)
        embed = self.transformer_encoder((gene_embed + expr_embed).permute(1, 0, 2), src_key_padding_mask = None)

        embed = self.decoder(embed) # batch x seq_len x 128
        cell_embed = embed[0, :, :] # select only the CLS token.
        cell_embed = nn.functional.normalize(cell_embed, dim=1) # Normalize.
        return embed, cell_embed, mask


    def predict_expr(self, cell_embed, batch_cond):
        expr_pred_meta = self.expr_predictor(cell_embed, batch_cond.reshape(-1,1))
        expr_pred = self.gene_decompression(expr_pred_meta)
        return expr_pred, expr_pred_meta


class MinMaxNormalization(nn.Module):
    def __init__(self, eps: float = 1e-6):
        super(MinMaxNormalization, self).__init__()
        self.eps = eps  # A small value to avoid division by zero

    def forward(self, x):
        # Compute the min and max along the specified dimensions
        x_max = x.max(dim=1, keepdim=True)[0]
        # x_min is always 0
        
        # Normalize to [0, 1]
        x_normalized = x / (x_max + self.eps)
        return x_normalized

class CompressionModel(nn.Module):
    def __init__(self, compression_mask):
        super(CompressionModel, self).__init__()
        self.compression_mask = compression_mask

        self.compression_proj = nn.Parameter(torch.ones(self.compression_mask.shape))
        # NOTE: dim = 1, for each original gene, the sum of weight pointing out of the gene is 1
        # dim = 0, for each meta-gene, the sum of weight pointing to the gene is 1
        # self.softmax_act = nn.Softmax(dim = 1)
        self.softmax_act = nn.Softmax(dim = 0)
        # 1 for keep, 0 for remove
        # normalize output between 0 and 1
        self.normalization = MinMaxNormalization(eps = 1e-6)

    def forward(self, counts):
        # make the masked position value extremely small
        proj_mtx = self.compression_proj.masked_fill(self.compression_mask == 0, -1e7)
        proj_mtx = self.softmax_act(proj_mtx)
        counts_meta = counts @ proj_mtx
        # the model does not work without normalization
        # log-transform need as value is skewed
        counts_meta = counts_meta/(torch.sum(counts_meta, dim = 1, keepdim = True) + 1e-4) * 10e4
        counts_meta = torch.log1p(counts_meta)
        return self.normalization(counts_meta)

        # # NOTE: sanity check, worked
        # counts = counts @ self.compression_mask
        # counts = counts/(torch.sum(counts, dim = 1, keepdim = True) + 1e-4) * 10e4
        # counts = torch.log1p(counts)
        # return self.normalization(counts)


class DecompressionModel(nn.Module):
    def __init__(self, compression_mask):
        super(DecompressionModel, self).__init__()
        self.decompression_proj = nn.Parameter(torch.ones(compression_mask.T.shape))
        # TODO: should dim be 0 or 1??
        # self.softmax_act = nn.Softmax(dim = 0)
        self.softmax_act = nn.Softmax(dim = 1)
        self.compression_mask = compression_mask

    def forward(self, counts_meta):
        proj_mtx = self.decompression_proj.masked_fill(self.compression_mask.T == 0, -1e7)
        proj_mtx = self.softmax_act(proj_mtx)
        return counts_meta @ proj_mtx

from torch.autograd import Function
# gradient reversal, necessary for discriminator training
class GradientReversalFunction(Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.neg() * ctx.alpha, None

def gradient_reversal(x, alpha=1.0):
    return GradientReversalFunction.apply(x, alpha)

