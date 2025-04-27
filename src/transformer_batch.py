# Directly use the transformer provided by torch
import torch
import torch.nn as nn
import math
import numpy as np

from dataclasses import dataclass
import base_model
from data_utils import set_seed
import graph_pool

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
    recon_meta: bool
    use_fourier: bool

    # batch
    batch_enc: str
    insert_transformer: bool

    # discriminator
    use_discriminator: bool
    lamb_disc: float
    lamb_mincut: float

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
        recon_meta=True,
        use_fourier=True,
        batch_enc="encoder",
        insert_transformer=True,
        use_discriminator=False,
        lamb_disc=1,
        lamb_mincut=1,
        use_fastatten=False,
        precision=torch.float32,
        pretrain_path=None,
        checkpoint_path="checkpoint/",
        checkpoint_prefix="checkpoint",
        lognorm_data=True
    )


class ConditionalPredictor(nn.Module):
    def __init__(self, input_dim: int, condition_dim: int, output_dim: int, n_layers: int, dropout: float, deep_inj: bool):
        super().__init__()

        latent_dim = 512
        self.deep_inj = deep_inj
        self.layers = nn.ModuleList()
        self.layers.append(base_model.full_block(input_dim + condition_dim, latent_dim, p_drop = dropout))

        for layer in range(1, n_layers - 1):
            self.layers.append(base_model.full_block(latent_dim + condition_dim if self.deep_inj else latent_dim, latent_dim, p_drop = dropout))

        self.layers.append(nn.Linear(latent_dim + condition_dim if self.deep_inj else latent_dim, output_dim))

        # predict non-negative values
        # self.activation = nn.Sigmoid()
        self.activation = nn.Softplus()

    def forward(self, x, cond):
        for idx, layer in enumerate(self.layers):
            if (not self.deep_inj) and (idx > 0):
                x = layer(x)
            else:
                if cond is not None:
                    x = layer(torch.cat([x, cond], dim = 1))
                else:
                    x = layer(x)

        x = self.activation(x)
        return x


class TransformerModel(nn.Module):

    def __init__(self, token_dict: dict, batch_dict: dict, label_dict: dict, model_config: ModelConfig, device: torch.device, seed: int = 0):
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

        if self.model_config.batch_enc == "encoder":
            # cannot insert into the transformer if use one-hot encoding
            self.batch_encoder = base_model.encoder_batchfactor(n_cont_feat= batch_dict["n_cont_feats"],
                                                                n_cat_list = batch_dict["n_cat_list"],
                                                                n_embed = self.model_config.d_embed)

            self.batch_proj = nn.Sequential(nn.Linear(self.model_config.d_embed, self.model_config.d_embed),
                                            nn.GELU(),
                                            nn.LayerNorm(self.model_config.d_embed))
        
        elif self.model_config.batch_enc == "onehot":
            # transform to one-hot encoding, learn cell embedding for each of these one-hot encoding
            self.batch_encoder = nn.Embedding(num_embeddings = batch_dict["cats"].shape[0], embedding_dim = self.model_config.d_embed)
            self.batch_proj = nn.Identity()


        # genepool
        # ------------------ genepool ------------------
        n_meta = 256
        token_embed = token_dict["gene_embed"].to(self.model_config.precision)
        cls_embed = token_dict["cls_embed"].to(self.model_config.precision)
        
        self.token_embed = token_embed.to(self.device)
        self.cls_embed = cls_embed.to(self.device)
        self.gene_compression = graph_pool.GenePoolVanilla(A = token_dict["token_graph"].to(device), n_meta = n_meta, s_init = None, temp = token_dict["temp"]).to(device)
        self.gene_compression.load_state_dict(token_dict["gpoolvanilla"])
        for param in self.gene_compression.parameters():
            param.requires_grad = False
        self.gene_decompression = nn.Identity()

        if self.model_config.recon_meta:
            self.mask_embed = nn.Parameter(torch.randn((1, self.model_config.d_embed)))
            self.n_genes = n_meta
        else:
            self.n_genes = token_embed.shape[0] 
        # ----------------------------------------------

        # gene name encoder, input: dimension of gene name embedding, token_dim
        # output: the n_embed, adopted from UCE
        self.gene_encoder = nn.Sequential(nn.Linear(token_embed.shape[1], self.model_config.d_embed),
                                            nn.GELU(),
                                            nn.LayerNorm(self.model_config.d_embed))
                                                      




        
        if self.model_config.use_fourier:
            # fourier position embedding
            self.expr_encoder = nn.Sequential(nn.Linear(32, self.model_config.d_embed),
                                                nn.GELU(),
                                                nn.LayerNorm(self.model_config.d_embed))
        else:
            # the continuous value range between 0 to 1, which make the training unstable, use minmax to stablize
            self.expr_norm = base_model.MinMaxNormalization()
            # self.expr_norm = nn.LayerNorm(n_meta)
            self.expr_encoder = nn.Sequential(nn.Linear(1, self.model_config.d_embed),
                                                nn.GELU(),
                                            #   nn.LayerNorm(self.model_config.d_embed),
                                                nn.Linear(self.model_config.d_embed, self.model_config.d_embed),
                                                nn.GELU(),
                                            #   nn.LayerNorm(self.model_config.d_embed),
                                                nn.Linear(self.model_config.d_embed, self.model_config.d_embed),
                                                nn.GELU(),
                                                nn.LayerNorm(self.model_config.d_embed))

            # self.expr_encoder = nn.Sequential(base_model.full_block(in_features = 1, out_features = self.model_config.d_embed, p_drop = 0.0),
            #                                   base_model.full_block(in_features = self.model_config.d_embed, out_features = self.model_config.d_embed, p_drop = 0.0),
            #                                   nn.Linear(self.model_config.d_embed, self.model_config.d_embed),
            #                                   nn.GELU(),
            #                                   nn.LayerNorm(self.model_config.d_embed))


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


        if self.model_config.batch_enc == "encoder":
            self.expr_predictor = ConditionalPredictor(input_dim = self.model_config.d_output, condition_dim = self.model_config.d_embed,
                                                    output_dim = self.n_genes, n_layers = 4, dropout = self.model_config.dropout, deep_inj = True)
        elif self.model_config.batch_enc == "onehot":
            # one-hot + a linear embedding
            self.expr_predictor = ConditionalPredictor(input_dim = self.model_config.d_output, condition_dim = self.model_config.d_embed,
                                                    output_dim = self.n_genes, n_layers = 4, dropout = self.model_config.dropout, deep_inj = True)
        else:
            self.expr_predictor = ConditionalPredictor(input_dim = self.model_config.d_output, condition_dim = 0,
                                                    output_dim = self.n_genes, n_layers = 4, dropout = self.model_config.dropout, deep_inj = True)

        if model_config.use_discriminator:
            self.label_dict = label_dict
            self.label_dict["label_batch"] = torch.tensor(self.label_dict["label_batch"]).bool().to(self.device)
            self.label_unknown = np.where(self.label_dict["label_code"] == "unknown")[0][0]
            self.batch_dict = batch_dict
            n_batch = self.batch_dict["cats"].shape[0]

            self.discriminator = nn.Sequential(base_model.full_block(self.model_config.d_output, 512, self.model_config.dropout),
                                               base_model.full_block(512, 512, self.model_config.dropout),
                                               base_model.full_block(512, 512, self.model_config.dropout),
                                               nn.Linear(512, n_batch))
            


        else:
            self.discriminator = None
        
        self.to(self.device)


    def freeze_fm_gradient(self, freeze_trans: bool, freeze_predictor: bool, freeze_batchenc: bool, freeze_compression: bool, freeze_discriminator: bool):
        self.freeze_compression = freeze_compression
        self.freeze_batchenc = freeze_batchenc
        self.freeze_trans = freeze_trans
        self.freeze_predictor = freeze_predictor
        self.freeze_discriminator = freeze_discriminator

        if self.freeze_compression:
            for param in self.gene_compression.parameters():
                param.requires_grad = False
            for param in self.gene_decompression.parameters():
                param.requires_grad = False

        if self.freeze_batchenc:
            for param in self.batch_encoder.parameters():
                param.requires_grad = False
            for param in self.batch_proj.parameters():
                param.requires_grad = False
        
        if self.freeze_trans:
            for param in self.transformer_encoder.parameters():
                param.requires_grad = False
            for param in self.decoder.parameters():
                param.requires_grad = False
            for param in self.expr_encoder.parameters():
                param.requires_grad = False
            for param in self.gene_encoder.parameters():
                param.requires_grad = False
            if self.model_config.recon_meta:
                self.mask_embed.requires_grad = False
        
        if self.freeze_predictor:
            for param in self.expr_predictor.parameters():
                param.requires_grad = False
        
        if (self.freeze_discriminator) & (self.model_config.use_discriminator):
            for param in self.discriminator.parameters():
                param.requires_grad = False
            


        

    def forward(self, counts_norm: torch.Tensor, batch_factors_cont: torch.Tensor, batch_factors_cat: torch.Tensor, batch_ids: torch.Tensor | None):
        """
        Parameters:
        --------------
            expr_sent: the gene expression sentence, of the shape (n_batchs, n_tokens)
        """
        
        if not self.model_config.recon_meta:
            # mask added on gene
            mask_prob = torch.full(counts_norm.shape, self.model_config.mask_prob).to(self.device)
            mask_gene = torch.bernoulli(mask_prob).bool()             
            counts_norm = counts_norm.masked_fill(mask_gene, 0)

        # reconstruct meta-gene expr, mask is added on meta-gene
        S, token_embed_meta, counts_norm_meta = self.gene_compression(gene_embed = self.token_embed, expr = counts_norm, log_norm = True)
        gene_embed = torch.vstack([self.cls_embed, token_embed_meta])
        assert gene_embed.shape[0] == 257
        gene_embed = self.gene_encoder(gene_embed).unsqueeze(0).repeat(counts_norm.shape[0], 1, 1)     

        # construct the gene expression sentence (match the word embedding)
        expr_sent = torch.hstack([torch.zeros((counts_norm_meta.shape[0], 1)).to(counts_norm_meta.device), counts_norm_meta])
        # TODO: still need to replace fourier with some other encoding (3 layer with gelu activation)

        if self.model_config.use_fourier:
            expr_embed = base_model.fourier_positional_encoding(expr_sent.unsqueeze(2), embedding_dim = 32)
        else:
            # training unstable, use minmax to stabilize
            expr_embed = self.expr_norm(expr_sent).unsqueeze(2)
        expr_embed = self.expr_encoder(expr_embed)

        if (self.model_config.insert_transformer) and (self.model_config.batch_enc is not None):
            if self.model_config.batch_enc == "onehot":
                if batch_ids is not None:
                    batch_embed = self.batch_encoder(batch_ids)
                else:
                    batch_embed = torch.zeros((gene_embed.shape[0], gene_embed.shape[2]), device = gene_embed.device)

            else:    
                batch_embed, _ = self.batch_encoder(batch_factors_cont = batch_factors_cont, batch_factors_cat = batch_factors_cat)
            batch_embed = self.batch_proj(batch_embed)

            # insert condition, make (258) tokens, sentence composition [cls, expr1, expr2, ..., exprm, cond]
            gene_sent_embed = torch.cat([gene_embed, batch_embed.unsqueeze(1)], dim = 1) * math.sqrt(self.model_config.d_embed)
            expr_sent_embed = torch.cat([expr_embed, torch.zeros((expr_embed.shape[0], 1, expr_embed.shape[2])).to(expr_embed.device)], dim = 1) * math.sqrt(self.model_config.d_embed)
        
        else:
            # insert condition, make (258) tokens, sentence composition [cls, expr1, expr2, ..., exprm, cond]
            gene_sent_embed = gene_embed * math.sqrt(self.model_config.d_embed)
            expr_sent_embed = expr_embed * math.sqrt(self.model_config.d_embed)
        
        # updated mask
        if self.model_config.recon_meta:
            # add mask to the input data
            mask_prob = torch.full(gene_sent_embed.shape[:2], self.model_config.mask_prob).to(self.device)
            # do not add mask on cls and cond positions
            mask_prob[:, 0] = 0
            if self.model_config.insert_transformer:
                # masking the last token (batch token) if insert_transformer is True (batch token not exist)
                mask_prob[:, -1] = 0

            # sample mask using bernoulli
            mask = torch.bernoulli(mask_prob).bool()
            mask_embed = self.mask_embed.view(1, 1, -1)
            expr_sent_embed = torch.where(mask.unsqueeze(-1), mask_embed, expr_sent_embed)

            if self.model_config.insert_transformer:
                mask_gene = mask[:, 1:-1]
            else:
                mask_gene = mask[:, 1:]

        embed = self.transformer_encoder((gene_sent_embed + expr_sent_embed).permute(1, 0, 2), src_key_padding_mask = None)
        embed = self.decoder(embed)
        cell_embed = embed[0, :, :]
        cell_embed = nn.functional.normalize(cell_embed, dim=1) # Normalize.
        return embed, cell_embed, mask_gene


    def predict_expr(self, cell_embed: torch.Tensor, batch_factors_cont: torch.Tensor, batch_factors_cat: torch.Tensor, batch_ids: torch.Tensor):

        # incorporate the batch condition
        if self.model_config.batch_enc == "onehot":
            if batch_ids is not None:
                batch_embed = self.batch_encoder(batch_ids)
            else:
                batch_embed = torch.zeros((cell_embed.shape[0], self.model_config.d_embed), device = cell_embed.device)
            batch_embed = self.batch_proj(batch_embed)

        elif self.model_config.batch_enc == "encoder":
            batch_embed, _ = self.batch_encoder(batch_factors_cont = batch_factors_cont, batch_factors_cat = batch_factors_cat)
            batch_embed = self.batch_proj(batch_embed)
        
        else:
            batch_embed = None
        
        expr_pred_meta = self.expr_predictor(x = cell_embed, cond = batch_embed)
        expr_pred = self.gene_decompression(expr_pred_meta)
        return expr_pred, expr_pred_meta



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