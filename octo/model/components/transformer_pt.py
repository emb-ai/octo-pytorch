# adapted from https://github.com/google-research/vision_transformer/blob/main/vit_jax/models_vit.py
from typing import Callable, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F 

from octo.model.components.base import TokenGroup
from octo.utils.typing import Dtype, PRNGKey, Shape, Union
from octo.model.components.jax_pt import FromJaxModel, LinearPt, LayerNormPt

class MAPHeadPt(nn.Module, FromJaxModel):
    def __init__(self, *args, **kwargs):
        raise NotImplementedError
    
    def load_jax_weights(self, params):
        raise NotImplementedError


class AddPositionEmbsPt(nn.Module, FromJaxModel):
    """Adds learned positional embeddings to the inputs.

    Attributes:
      posemb_init: positional embedding initializer.
    """

    def __init__(self, hid_dim=512, max_len=5000):
        super().__init__()
        self.max_len = max_len
        self.hid_dim = hid_dim
        pos_embed = torch.randn(max_len, self.hid_dim) * 0.02
        self.register_parameter('pos_embed', pos_embed) # trainable

    def load_jax_weights(self, jax_params):
        with torch.no_grad():
            self.pos_embed = self._get_param(jax_params['pos_embedding'][0])
        self.max_len, self.hid_dim = self.pos_embed.shape
    
    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Applies the AddPositionEmbs module.

        Args:
          inputs: Inputs to the layer.

        Returns:
          Output tensor with shape `(bs, timesteps, in_dim)`.
        """
        # inputs.shape is (batch_size, seq_len, emb_dim).
        assert inputs.dim() == 3, (
            "Number of dimensions should be 3," f" but it is: {inputs.dim()}"
        )
        
        b, l, d = inputs.shape
        assert d == self.hid_dim, (
            "Dimensionality of input tensor doesn't match positional embedding dimensionality"
        )
        assert l <= self.max_len, (
            f"Sequence length is larger than max_len={self.max_len}"
        )
        
        pe = self.pos_embed[:l, :].expand(b, l, d)
        return inputs + pe
      
      
      
class MlpBlockPt(nn.Module, FromJaxModel):
    """Transformer MLP / feed-forward block."""

    def __init__(
        self,
        inp_dim: int,
        mlp_dim: int,
        out_dim: Optional[int] = None,
        dropout_rate: float = 0.1,
    ):
        super().__init__()
        self.inp_dim = inp_dim
        self.mlp_dim = mlp_dim
        self.out_dim = out_dim
        self.dropout_rate = dropout_rate

        # These will be initialized in the forward pass when we know the input shape
        self.dense1 = LinearPt(inp_dim, mlp_dim)
        self.dense2 = LinearPt(mlp_dim, inp_dim) if out_dim is None else LinearPt(mlp_dim, out_dim)
        self.dropout1 = nn.Dropout(p=dropout_rate)
        self.dropout2 = nn.Dropout(p=dropout_rate)
        
    def load_jax_weights(self, params):
        self.dense1.load_jax_weights(params['Dense_0'])
        self.dense2.load_jax_weights(params['Dense_1'])


    def forward(self, inputs: torch.Tensor, deterministic: bool = False) -> torch.Tensor:
        """Applies Transformer MlpBlock module."""
        assert inputs.shape[-1] == self.inp_dim, "Input tensor dimension does not match MLP block input dimension"

        x = self.dense1(inputs)
        x = F.gelu(x)
        x = self.dropout1(x) if not deterministic else x
        output = self.dense2(x)
        output = self.dropout2(output) if not deterministic else output
        return output
    
class MultiheadAttentionPt(nn.MultiheadAttention, FromJaxModel):
    def load_jax_weights(self, params):
        q_weight = self._get_param(params['query']['kernel'])
        features, num_heads, head_dim = q_weight.shape
        embed_dim = num_heads * head_dim
        
        q_weight = q_weight.permute((1, 2, 0)).reshape(embed_dim, embed_dim)
        # q_weight = torch.tensor(params['key']['kernel'].transpose(1, 0, 2).reshape(num_heads * head_dim, -1)).float()
        k_weight = self._get_param(params['key']['kernel']).permute((1, 2, 0)).reshape(embed_dim, embed_dim)
        # k_weight = torch.tensor(params['value']['kernel'].transpose(1, 0, 2).reshape(num_heads * head_dim, -1)).float()
        v_weight = self._get_param(params['value']['kernel']).permute((1, 2, 0)).reshape(embed_dim, embed_dim)
        # v_weight = torch.tensor(params['value']['kernel'].transpose(1, 0, 2).reshape(num_heads * head_dim, -1)).float()
        q_bias = self._get_param(params['query']['bias']).reshape(embed_dim)
        k_bias = self._get_param(params['key']['bias']).reshape(embed_dim)
        v_bias = self._get_param(params['value']['bias']).reshape(embed_dim)
                
        if not self._qkv_same_embed_dim:
            with torch.no_grad():
                self.q_proj_weight = nn.Parameter(q_weight)
                self.k_proj_weight = nn.Parameter(k_weight)
                self.v_proj_weight = nn.Parameter(v_weight)
        else:
            in_proj = torch.cat((q_weight, k_weight, v_weight), dim=0)
            with torch.no_grad():
                self.in_proj_weight = nn.Parameter(in_proj)
                
        bias = torch.cat((q_bias, k_bias, v_bias), dim=0)
        with torch.no_grad():
            self.in_proj_bias = nn.Parameter(bias)
        
        with torch.no_grad():
            self.out_proj.weight = nn.Parameter(self._get_param(params['out']['kernel']).permute((1, 2, 0)).reshape(embed_dim, embed_dim))
            self.out_proj.bias = self._get_param(params['out']['bias'])

class Encoder1DBlockPt(nn.Module, FromJaxModel):
    """Transformer encoder layer.

    Attributes:
      mlp_dim: dimension of the mlp on top of attention block.
      num_heads: Number of heads in MultiHeadAttention
      dropout_rate: dropout rate.
      attention_dropout_rate: dropout for attention heads.
    """

    def __init__(
        self,
        input_dim,
        mlp_dim: int,
        num_heads: int,
        dropout_rate: float = 0.1,
        attention_dropout_rate: float = 0.1
    ):
        super().__init__()
        self.input_dim = input_dim
        self.mlp_dim = mlp_dim
        self.num_heads = num_heads
        self.dropout_rate = dropout_rate
        self.attention_dropout_rate = attention_dropout_rate

        self.layer_norm1 = LayerNormPt(input_dim)
        self.self_attention = MultiheadAttentionPt(
            embed_dim=input_dim,
            num_heads=num_heads,
            dropout=attention_dropout_rate,
            batch_first=True
        )
        self.dropout1 = nn.Dropout(dropout_rate)

        self.layer_norm2 = LayerNormPt(input_dim)
        self.mlp_block = MlpBlockPt(
            inp_dim=input_dim,
            mlp_dim=mlp_dim,
            dropout_rate=dropout_rate
        )
        
    def load_jax_weights(self, params):
        self.layer_norm1.load_jax_weights(params['LayerNorm_0'])
        self.layer_norm2.load_jax_weights(params['LayerNorm_1'])
        self.self_attention.load_jax_weights(params['MultiHeadDotProductAttention_0'])
        self.mlp_block.load_jax_weights(params['MlpBlock_0'])

    def forward(self, inputs: torch.Tensor, attention_mask: torch.Tensor, deterministic: bool = False):
        """Applies Encoder1DBlock module.

        Args:
          inputs: Inputs to the layer.
          attention_mask: Mask for attention mechanism.
          deterministic: Dropout will not be applied when set to true.

        Returns:
          output after transformer encoder block.
        """
        assert inputs.dim() == 3, f"Expected (batch, seq, hidden) got {inputs.shape}"

        # Attention block.
        x = self.layer_norm1(inputs)
        attention_mask = attention_mask.to(torch.bool)
        # x, _ = self.self_attention(x, x, x, key_padding_mask=~attention_mask, need_weights=False)
        x, _ = self.self_attention(x, x, x, key_padding_mask=None, need_weights=False)
        x = self.dropout1(x) if not deterministic else x
        x = x + inputs

        # MLP block.
        y = self.layer_norm2(x)
        y = self.mlp_block(y, deterministic=deterministic)

        return x + y
    

