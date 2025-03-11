from typing import Iterable
import torch
from torch import nn as nn
import torch.nn.functional as F
import collections
from torch.amp import autocast
    
def identity(x):
    return x

def one_hot(index, n_cat, dtype = torch.bfloat16) -> torch.Tensor:
    """One hot a tensor of categories."""
    onehot = torch.zeros(index.size(0), n_cat, device=index.device)
    onehot.scatter_(1, index.type(torch.long), 1)
    return onehot.type(dtype)


class FCLayers(nn.Module):
    """
    A helper class to build fully-connected layers for a neural network.

    Parameters
    ----------
    n_in
        The dimensionality of the input
    n_out
        The dimensionality of the output
    n_cat_list
        A list containing, for each category of interest,
        the number of categories. Each category will be
        included using a one-hot encoding.
    n_layers
        The number of fully-connected hidden layers
    n_hidden
        The number of nodes per hidden layer
    dropout_rate
        Dropout rate to apply to each of the hidden layers
    use_batch_norm
        Whether to have `BatchNorm` layers or not
    use_layer_norm
        Whether to have `LayerNorm` layers or not
    use_activation
        Whether to have layer activation or not
    bias
        Whether to learn bias in linear layers or not
    inject_covariates
        Whether to inject covariates in each layer, or just the first (default).
    activation_fn
        Which activation function to use
    """

    def __init__(
        self,
        n_in: int,
        n_out: int,
        n_cat_list: Iterable[int] = None,
        n_layers: int = 1,
        n_hidden: int = 128,
        dropout_rate: float = 0.1,
        use_batch_norm: bool = False,
        use_layer_norm: bool = False,
        use_activation: bool = False,
        bias: bool = True,
        inject_covariates: bool = False,
        activation_fn: nn.Module = nn.GELU,
        dtype: torch.dtype = torch.float32
    ):
        super().__init__()
        self.inject_covariates = inject_covariates
        layers_dim = [n_in] + (n_layers - 1) * [n_hidden] + [n_out]

        if n_cat_list is not None:
            # n_cat = 1 will be ignored
            self.n_cat_list = [n_cat if n_cat > 1 else 0 for n_cat in n_cat_list]
        else:
            self.n_cat_list = []
        
        # self.n_cat_list = [n_batches]
        # cat_dim = n_batches
        cat_dim = sum(self.n_cat_list)
        self.fc_layers = nn.Sequential(
            collections.OrderedDict(
                [
                    (
                        "Layer {}".format(i),
                        nn.Sequential(
                            nn.Linear(
                                # inject cat_dim into the first layer if deeply_inject_covariates is False
                                # to all dimensions if the deeply_inject_covariates is True
                                n_in + cat_dim * self.inject_into_layer(i),
                                n_out,
                                bias=bias,
                            ),
                            # non-default params come from defaults in original Tensorflow implementation
                            nn.BatchNorm1d(n_out, momentum=0.01, eps=0.001)
                            if use_batch_norm
                            else None,
                            nn.LayerNorm(n_out, elementwise_affine=False)
                            if use_layer_norm
                            else None,
                            activation_fn() if use_activation else None,
                            nn.Dropout(p=dropout_rate) if dropout_rate > 0 else None,
                        ),
                    )
                    for i, (n_in, n_out) in enumerate(
                        zip(layers_dim[:-1], layers_dim[1:])
                    )
                ]
            )
        )

        self.dtype = dtype

    def inject_into_layer(self, layer_num) -> bool:
        """Helper to determine if covariates should be injected."""
        user_cond = layer_num == 0 or (layer_num > 0 and self.inject_covariates)
        return user_cond

    def set_online_update_hooks(self, hook_first_layer=True):
        self.hooks = []

        def _hook_fn_weight(grad):
            categorical_dims = sum(self.n_cat_list)
            new_grad = torch.zeros_like(grad)
            if categorical_dims > 0:
                new_grad[:, -categorical_dims:] = grad[:, -categorical_dims:]
            return new_grad

        def _hook_fn_zero_out(grad):
            return grad * 0

        for i, layers in enumerate(self.fc_layers):
            for layer in layers:
                if i == 0 and not hook_first_layer:
                    continue
                if isinstance(layer, nn.Linear):
                    if self.inject_into_layer(i):
                        w = layer.weight.register_hook(_hook_fn_weight)
                    else:
                        w = layer.weight.register_hook(_hook_fn_zero_out)
                    self.hooks.append(w)
                    b = layer.bias.register_hook(_hook_fn_zero_out)
                    self.hooks.append(b)

    def forward(self, x: torch.Tensor, *cat_list: int):
        """
        Forward computation on ``x``.

        Parameters
        ----------
        x
            tensor of values with shape ``(n_in,)``
        cat_list
            list of category membership(s) for this sample
        x: torch.Tensor

        Returns
        -------
        py:class:`torch.Tensor`
            tensor of shape ``(n_out,)``

        """
        one_hot_cat_list = []  # for generality in this list many indices useless.

        if len(self.n_cat_list) > len(cat_list):
            raise ValueError(
                "nb. categorical args provided doesn't match init. params."
            )
        # self.n_cat_list = [n_batches], cat is a int indicating the current batch
        for n_cat, cat in zip(self.n_cat_list, cat_list):
            if n_cat and cat is None:
                raise ValueError("cat not provided while n_cat != 0 in init. params.")
            if n_cat > 1:  # n_cat = 1 will be ignored - no additional information
                if cat.size(1) != n_cat:
                    one_hot_cat = one_hot(cat, n_cat, dtype = self.dtype)
                else:
                    one_hot_cat = cat  # cat has already been one_hot encoded
                one_hot_cat_list += [one_hot_cat]
        for i, layers in enumerate(self.fc_layers):
            for layer in layers:
                if layer is not None:
                    if isinstance(layer, nn.BatchNorm1d):
                        if x.dim() == 3:
                            x = torch.cat(
                                [(layer(slice_x)).unsqueeze(0) for slice_x in x], dim=0
                            )
                        else:
                            x = layer(x)
                    else:
                        if isinstance(layer, nn.Linear) and self.inject_into_layer(i):
                            if x.dim() == 3:
                                one_hot_cat_list_layer = [
                                    o.unsqueeze(0).expand(
                                        (x.size(0), o.size(0), o.size(1))
                                    )
                                    for o in one_hot_cat_list
                                ]
                            else:
                                one_hot_cat_list_layer = one_hot_cat_list
                            x = torch.cat((x, *one_hot_cat_list_layer), dim=-1)
                        x = layer(x)
        return x   
    

class decoder(nn.Module):
    def __init__(
        self,
        n_input: int,
        n_output: int,
        n_cat_list: Iterable[int] = None,
        n_layers: int = 1,
        n_hidden: int = 128,
        dropout_rate: float = 0.1,
        inject_covariates: bool = False,
        use_batch_norm: bool = False,
        use_layer_norm: bool = True,
        dtype: torch.dtype = torch.float32
    ):
        super().__init__()
        self.fc = FCLayers(
            n_in=n_input,
            n_out=n_hidden,
            n_cat_list=n_cat_list,
            n_layers=n_layers - 1,
            n_hidden=n_hidden,
            dropout_rate=dropout_rate,
            inject_covariates=inject_covariates,
            use_batch_norm=use_batch_norm,
            use_layer_norm=use_layer_norm,
            use_activation=True,
            bias=True,
            activation_fn=nn.GELU,
            dtype = dtype
        )
        self.linear_out = nn.Linear(n_hidden, n_output)
        # NOTE: the expression value is non-negative, but doesn't normalize by 1
        # use softplus to constrain the non-negativity
        # self.output_act = nn.Softmax(dim=-1)
        # self.output_act = nn.Softplus()
    
    def forward(self, x: torch.Tensor, *cat_list: int):
        x = self.fc(x, *cat_list)
        # y = self.output_act(self.linear_out(x))
        y = self.linear_out(x)
        return y


class TransformerLayer(nn.Module):
    def __init__(self, d_model: int, nhead: int, dim_feedforward: int, dropout: float, activation: str = "relu"):
        super(TransformerLayer, self).__init__()
        
        # Multi-head self attention
        self.self_attn = nn.MultiheadAttention(embed_dim = d_model, num_heads = nhead, dropout = dropout)
        
        # Feed-forward network
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        
        # Layer Normalization
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
        # Dropout layers
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        
        # Activation function
        if activation == "relu":
            self.activation = F.relu
        elif activation == "gelu":
            self.activation = F.gelu
        else:
            raise ValueError("activation can only be `relu' or `gelu'")


    def forward(self, x, src_key_padding_mask = None):
        # Apply mixed precision to the entire TransformerEncoderLayer
        attn_output = self.self_attn(x, x, x, key_padding_mask=src_key_padding_mask)[0]
        
        # Force layer normalization and softmax to run in FP32 for stability
        with autocast(device_type = "cuda", enabled=False):
            x = self.norm1(x + self.dropout1(attn_output))
        
        # Continue the feedforward part in mixed precision
        feedforward_output = self.linear2(self.dropout(self.activation(self.linear1(x))))
        
        # Again, force normalization to run in FP32
        with autocast(device_type = "cuda", enabled=False):
            x = self.norm2(x + self.dropout2(feedforward_output))
        
        return x
    

class TransformerBlocks(nn.Module):
    def __init__(self, d_model: int, n_head: int, num_layers: int, dim_feedforward: int, dropout: float, activation: str = "gelu"):
        super(TransformerBlocks, self).__init__()
        self.layers = nn.ModuleList([TransformerLayer(d_model, n_head, dim_feedforward, dropout, activation) for _ in range(num_layers)])
    
    def forward(self, src, src_key_padding_mask = None):
        for layer in self.layers:
            src = layer(src, src_key_padding_mask)
        return src 


def fourier_positional_encoding(x: torch.Tensor, embedding_dim: int):
    """
    Computes the Fourier-based positional embedding for a continuous value x in [0, 1].
    
    Args:
        x (float): A continuous value in the range [0, 1].
        embedding_dim (int): The dimensionality of the positional embedding.
        
    Returns:
        numpy.ndarray: The positional embedding vector.
    """
    assert torch.max(x) <= 1, "x must be between 0 and 1"
    assert torch.min(x) >= 0
    # Half of the embedding dimension will be used for sin, and half for cos
    half_dim = embedding_dim // 2
    
    # Define the frequencies as powers of 2
    frequencies = 2 ** torch.arange(half_dim).to(x.device)
    
    # Compute the sine and cosine components
    sin_components = torch.sin(frequencies * x)
    cos_components = torch.cos(frequencies * x)
    
    # Concatenate the sin and cos components
    positional_embedding = torch.concat([sin_components, cos_components], dim = -1)
    
    return positional_embedding

# class CustomTransformerEncoderLayer(nn.Module):
#     def __init__(self, d_model: int, nhead: int, dim_feedforward: int, dropout: float, dtype: torch.dtype = torch.float32):
#         super(CustomTransformerEncoderLayer, self).__init__()
#         self.layer = nn.TransformerEncoderLayer(d_model=d_model, 
#                                                 nhead=nhead, 
#                                                 dim_feedforward=dim_feedforward, 
#                                                 dropout=dropout)

#         self.dtype = dtype
#         if self.dtype != torch.float32:
#             print(f"cast the model to {self.dtype}")
        

#     def forward(self, x, src_key_padding_mask = None):
#         # Apply mixed precision to the entire TransformerEncoderLayer
#         attn_output = self.layer.self_attn(x, x, x, key_padding_mask=src_key_padding_mask)[0]
        
#         # Force layer normalization and softmax to run in FP32 for stability
#         with autocast(device_type = "cuda", enabled=False):
#             x = self.layer.norm1(attn_output + x)
        
#         # Continue the feedforward part in mixed precision
#         feedforward_output = self.layer.linear2(self.layer.dropout(self.layer.activation(self.layer.linear1(x))))
        
#         # Again, force normalization to run in FP32
#         with autocast(device_type = "cuda", enabled=False):
#             x = self.layer.norm2(feedforward_output + x)
        
#         return x



# class CustomTransformer(nn.Module):
#     def __init__(self, d_model, n_head, num_layers, dim_feedforward=2048, dropout=0.1, dtype=torch.float32):
#         super(CustomTransformer, self).__init__()
#         self.layers = nn.ModuleList([CustomTransformerEncoderLayer(d_model, n_head, dim_feedforward, dropout, dtype) for _ in range(num_layers)])
    
#     def forward(self, src, src_key_padding_mask = None):
#         for layer in self.layers:
#             src = layer(src, src_key_padding_mask)
#         return src