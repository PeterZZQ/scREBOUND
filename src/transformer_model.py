# Directly use the transformer provided by torch
import torch
import torch.nn as nn
import math
from dataclasses import dataclass
import base_model

from flash_transformer_layer import FlashTransformerEncoderLayer

@dataclass
class ModelConfig:

    batch_size: int # Batch size
    n_epoch: int # Number of epochs to train
    lr: float # Learning rate
    n_warmup_stp_lr: int # the number of steps for warmup of learning rate
    d_embed: int
    n_head: int
    d_hidden: int
    n_layer: int
    d_output: int
    dropout: float
    mask_prob: float
    dynamic_maskprob: bool
    lamb_kd: float
    lamb_sup: float
    sup_type: str | None

    # prediction manner
    mlm_include_zero: bool
    # whether use continuous or digitized count in transformer
    count_mode: str

    # remove batch effect
    deep_injection: bool
    use_discriminator: bool
    lamb_disc: float

    # model stats
    use_fastatten: bool
    precision: torch.dtype

    # paths
    checkpoint_path: str
    checkpoint_prefix: str
    pretrain_path: str | None = None


def get_default_config() -> ModelConfig:

    return ModelConfig(
        batch_size=1024,
        n_epoch=3,
        lr=5e-4,
        n_warmup_stp_lr=1000,
        d_embed=512, # dimension of each head == d_embed/n_head 512 = 8*64, 768 = 12*64
        n_head=8,
        d_hidden=2048,
        n_layer=4,
        d_output=256,
        dropout=0.05,
        mask_prob=0.15,
        dynamic_maskprob=False,
        lamb_kd=0.0,
        lamb_sup=1.0,
        mlm_include_zero=False,
        count_mode="continuous",
        deep_injection=False,
        use_discriminator=False,
        lamb_disc=1.0,
        use_fastatten=False,
        precision=torch.float32,
        checkpoint_path="checkpoint/",
        checkpoint_prefix="checkpoint",
        sup_type="classifier",
    )


def full_block(in_features, out_features, p_drop=0.1):
    return nn.Sequential(
        nn.Linear(in_features, out_features, bias=True),
        nn.LayerNorm(out_features),
        nn.GELU(),
        nn.Dropout(p=p_drop),
    )

class TransformerModel(nn.Module):

    def __init__(self, token_embed: torch.Tensor, model_config: ModelConfig, n_label: int,
                 n_batch: int, device: torch.device, teacher_params: dict = None):
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
        self.model_type = 'Transformer'
        self.model_config = model_config
        self.device = device

        # conditions
        self.n_batch = n_batch
        self.n_label = n_label

        self.n_tokens, self.n_token_dims = token_embed.shape

        # assign the index of the special token, hardcoded
        self.cls_idx = self.n_tokens - 3
        self.pad_idx = self.n_tokens - 2
        self.mask_idx = self.n_tokens - 1
        self.gene_idx = torch.arange(self.n_tokens - 3).to(self.device)

        # store the token embedding, of the shape (ntokens, n_token_dims)
        self.token_embed = nn.Embedding.from_pretrained(token_embed)
        # fix the token values, NOTE: UCE didn't fix the token_embed weight
        self.token_embed.weight.requires_grad = False

        # gene name encoder, input: dimension of gene name embedding, token_dim
        # output: the n_embed, adopted from UCE
        self.gene_encoder = nn.Sequential(nn.Linear(token_embed.shape[1], self.model_config.d_embed),
                                          nn.GELU(),
                                          nn.LayerNorm(self.model_config.d_embed))

        if self.model_config.count_mode == "digitize":
            # NOTE: 2 version to digitized expr: 
            # 1. nn.Embedding for discrete value
            self.n_bins = 10
            # +1 for 0 
            self.expr_encoder = nn.Embedding(self.n_bins + 1, self.model_config.d_embed)
            # 2. TODO: position embedding for continuous value
        else:
            # for continuous value
            self.expr_encoder = nn.Sequential(nn.Linear(1, self.model_config.d_embed),
                                            nn.GELU(),
                                            nn.LayerNorm(self.model_config.d_embed))

        
        # Construct the transformer layers, use torch transform directly, or use transformer in base_model.py
        # n_hidden = 2048, n_embed = 512
        if model_config.use_fastatten:
            # NOTE: need to use mixed precision
            encoder_layers = FlashTransformerEncoderLayer(d_model = self.model_config.d_embed, 
                                                        nhead = self.model_config.n_head,
                                                        dim_feedforward = self.model_config.d_hidden,
                                                        dropout = self.model_config.dropout)

            self.transformer_encoder = nn.TransformerEncoder(encoder_layers, model_config.n_layer)

        else:            
            self.transformer_encoder = base_model.TransformerBlocks(d_model = self.model_config.d_embed,
                                                                    n_head = self.model_config.n_head,
                                                                    num_layers = self.model_config.n_layer,
                                                                    dim_feedforward = self.model_config.d_hidden,
                                                                    dropout = self.model_config.dropout,
                                                                    activation = "gelu")

        # transform the output of the transformer m_embed -> output_dim
        self.decoder = nn.Sequential(full_block(self.model_config.d_embed, 256, self.model_config.dropout),
                                     full_block(256, self.model_config.d_output, self.model_config.dropout),
                                     full_block(self.model_config.d_output, self.model_config.d_output, self.model_config.dropout),
                                     nn.Linear(self.model_config.d_output, self.model_config.d_output))

        # deep injection, same as the MLP above with the choice of deep injection
        if self.model_config.deep_injection:
            self.expr_predictor = base_model.decoder(
                n_input = self.model_config.d_output,
                n_output = self.n_tokens - 1,
                n_cat_list = [self.n_batch],
                n_layers = 4,
                n_hidden = 512,
                dropout_rate = self.model_config.dropout,
                inject_covariates = self.model_config.deep_injection, # NOTE: True for deep injection
                use_batch_norm = False,
                use_layer_norm = True,
                dtype = self.model_config.precision
            )
        else:
            self.expr_predictor = nn.Sequential(
                full_block(self.model_config.d_output + self.n_batch, 2048, self.model_config.dropout), # add the batch condition
                full_block(2048, 512, self.model_config.dropout),
                full_block(512, 128, self.model_config.dropout),
                nn.Linear(128, self.n_tokens - 1) # remove the masked token (not in the original gene sentance)
            )


        if model_config.sup_type == "classifier":
            self.classifier = nn.Sequential(
                full_block(self.model_config.d_output, 2048, self.model_config.dropout), # no batch condition
                nn.Linear(2048, self.n_label)            
            )
            # weights for self.n_label classifiers
            # self.classifier_weight = nn.Parameter(torch.randn(self.n_label))
            # BASELINE: weight for self.n_label classifiers, the same for all classes
            self.classifier_weight = 1/self.n_label * torch.ones(self.n_label).to(self.device)

        else:
            self.classifier = None

        if model_config.use_discriminator:
            self.discriminator = base_model.decoder(
                    n_input = self.model_config.d_output,
                    n_output = self.n_batch,
                    n_cat_list = [self.n_label],
                    n_layers = 2,
                    n_hidden = 512,
                    dropout_rate = self.model_config.dropout,
                    inject_covariates = self.model_config.deep_injection, # NOTE: True for deep injection
                    use_batch_norm = False,
                    use_layer_norm = True            
            )
        else:
            self.discriminator = None


        if teacher_params is not None:
            self.teacher_encoder = nn.Linear(sum(teacher_params["teacher_dims"]), self.model_config.d_output)
        
        self.to(self.device)


    def forward(self, gene_sent: torch.Tensor, expr_sent: torch.Tensor, mask: torch.Tensor | None = None):
        """
        Parameters:
        --------------
            gene_sent: the gene name sentence, of the shape (n_batchs, n_tokens)
            expr_sent: the gene expression sentence, of the shape (n_batchs, n_tokens)
        """
        # padding mask: position of the padding place
        padding_mask = (gene_sent == self.pad_idx)

        if mask is None:
            # add mask to the input data
            mask_prob = torch.full(gene_sent.shape, self.model_config.mask_prob).to(self.device)
            # do not add mask on padding and cls positions
            mask_prob[:, 0] = 0
            mask_prob = mask_prob * (~ padding_mask)
            # sample mask using bernoulli
            mask = torch.bernoulli(mask_prob).bool()

        # only place where mask_idx is used
        gene_sent_wmask = gene_sent.long()
        gene_sent_wmask[mask] = self.mask_idx
        expr_sent_wmask = expr_sent.clone()
        expr_sent_wmask[mask] = 0 # NOTE: no other genes have 0 expression in the sentence, except for the padding

        # with autocast(device_type="cuda", enabled = False):
        # obtain the gene name embedding given the gene sentence, (n_batchs, n_tokens, n_token_dims)
        gene_embed = self.token_embed(gene_sent_wmask)
        # updated gene_embed of the shape (batch_size, n_tokens, n_embed), TODO: math.sqrt(self.n_embed) was included in UCE, why??
        gene_embed = self.gene_encoder(gene_embed) * math.sqrt(self.model_config.d_embed)

        if self.model_config.count_mode == "digitize":
            expr_embed = self.expr_encoder(expr_sent_wmask.long()).to(self.model_config.precision) * math.sqrt(self.model_config.d_embed)
        else:
            expr_embed = self.expr_encoder(expr_sent_wmask.unsqueeze(2)) * math.sqrt(self.model_config.d_embed)

        # with autocast(device_type="cuda", enabled = False):
        # src_key_padding_mask: the padding position is true, remainings are false
        # input should be of the shape (n_tokens, n_batches, n_token_dims)
        embed = self.transformer_encoder((gene_embed + expr_embed).permute(1, 0, 2), src_key_padding_mask = padding_mask)

        # with autocast(device_type="cuda", enabled = False):
        embed = self.decoder(embed) # batch x seq_len x 128
        # embedding = torch.mul(gene_output, mask.t().unsqueeze(2)).sum(0) # average over non zero genes
        # In the new format, the cls token, which is at the 0 index mark, is the output.
        cell_embed = embed[0, :, :] # select only the CLS token.
        cell_embed = nn.functional.normalize(cell_embed, dim=1) # Normalize.
        return embed, cell_embed, mask


    def predict_expr(self, cell_embed, batch_cond):
        expr_pred = self.expr_predictor(cell_embed, batch_cond.reshape(-1,1))
        return expr_pred



    def KD_loss(self, student_embed, teacher_embeds):
        """
        # NOTE: currently only consider one-layer
        student_embeds: of the shape (nlayers, ntokens, nbatches, ndims_stud), without mask
        teacher_embeds: list of teachers, each of the shape (nlayers, ntokens, nbatches, ndims_teacher)
        """
        teacher_embed = torch.concat(teacher_embeds, dim = -1)
        teacher_embed = self.teacher_encoder(teacher_embed)
        # mse between teacher embed and student embed
        loss_kd = nn.MSELoss()
        return loss_kd(teacher_embed, student_embed)




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

