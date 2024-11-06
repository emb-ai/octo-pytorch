# adapted from https://github.com/google-research/vision_transformer/blob/main/vit_jax/models_vit.py
from typing import Callable, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F 

from octo.model.components.base import TokenGroupPt
from octo.utils.typing import Dtype, PRNGKey, Shape, Union, Tuple

from octo.model.components.jax_pt import FromJaxModel, LinearPt, LayerNormPt

class MAPHeadPt(nn.Module, FromJaxModel):
    """Multihead Attention Pooling.
    PyTorch version of the MAP head from Big Vision.
    """
    def __init__(
        self,
        input_dim: int,
        mlp_dim: Optional[int] = None,
        num_heads: int = 8,
        num_readouts: int = 1
    ):
        super().__init__()
        self.input_dim = input_dim
        self.mlp_dim = mlp_dim if mlp_dim is not None else 4 * input_dim
        self.num_heads = num_heads
        self.num_readouts = num_readouts

        # Initialize probe parameter
        self.probe = nn.Parameter(torch.empty(1, num_readouts, input_dim))
        nn.init.xavier_uniform_(self.probe)

        # Initialize layers
        self.attention = MultiheadAttentionPt(
            embed_dim=input_dim,
            num_heads=num_heads,
            batch_first=True
        )
        self.layer_norm = LayerNormPt(input_dim)
        self.mlp_block = MlpBlockPt(
            input_dim,
            mlp_dim=self.mlp_dim,
            dropout_rate=0.0  # No dropout as per original
        )

    def forward(
        self, 
        x: Union[torch.Tensor, TokenGroupPt], 
        train: bool = True
    ) -> torch.Tensor:
        if isinstance(x, TokenGroupPt):
            x, mask = x.tokens, x.mask
        else:
            mask = None

        # Handle input reshaping
        *batch_dims, l, d = x.shape
        x = x.reshape(-1, l, d)
        batch_size = x.shape[0]

        # Expand probe to batch size
        probe = self.probe.expand(batch_size, -1, -1)

        # Prepare attention mask if provided
        if mask is not None:
            mask = mask.reshape(-1, l)
            # Convert mask to PyTorch attention mask format
            # PyTorch uses True to mask out values
            attention_mask = ~mask

        # Apply attention
        # PyTorch's MultiheadAttention expects different signature than JAX
        out, _ = self.attention(
            query=probe,
            key=x,
            value=x,
            key_padding_mask=attention_mask if mask is not None else None
        )

        # Apply layer norm and residual connection with MLP
        y = self.layer_norm(out)
        out = out + self.mlp_block(y, deterministic=not train)

        # Reshape output back to original batch dimensions
        out = out.reshape(*batch_dims, self.num_readouts, d)
        return out

    @property
    def output_dim(self) -> int:
        return self.input_dim


class AddPositionEmbsPt(nn.Module, FromJaxModel):
    """Adds learned positional embeddings to the inputs.

    Attributes:
      TODO
    """

    def __init__(self, emb_shape: Tuple, history_dim = None):
        super().__init__()
        self.history_dim = history_dim
        pos_embed = torch.randn(*emb_shape) * 0.02
        self.pos_embed = nn.Parameter(pos_embed) # trainable

    def load_jax_weights(self, jax_params):
        pos_embed = self._get_param(jax_params[0])
        self.assign_new_value('pos_embed', pos_embed)
        
    
    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Applies the AddPositionEmbs module.

        Args:
          inputs: Inputs to the layer.

        Returns:
          Output tensor with shape `(bs, timesteps, in_dim)`.
        """
        # inputs.shape is (batch_size, seq_len, emb_dim).
        bs = inputs.shape[0]
        # assert inputs[0].shape ==  self.pos_embed.shape 
        
        pe = self.pos_embed
        if self.history_dim is not None:
            history_len = inputs[0].shape[self.history_dim]
            pe = pe[(slice(None),) * self.history_dim + (slice(0, history_len),)]
        else:
            assert pe.shape == inputs[0].shape
        
        pe = pe.unsqueeze(0).expand(bs, *pe.shape)
        return inputs + pe
        
        
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
        
        k_weight = self._get_param(params['key']['kernel']).permute((1, 2, 0)).reshape(embed_dim, embed_dim)
       
        v_weight = self._get_param(params['value']['kernel']).permute((1, 2, 0)).reshape(embed_dim, embed_dim)
        
        q_bias = self._get_param(params['query']['bias']).reshape(embed_dim)
        k_bias = self._get_param(params['key']['bias']).reshape(embed_dim)
        v_bias = self._get_param(params['value']['bias']).reshape(embed_dim)
                
        if not self._qkv_same_embed_dim:
            self.assign_new_value('q_proj_weight', q_weight)
            self.assign_new_value('k_proj_weight', k_weight)
            self.assign_new_value('v_proj_weight', v_weight)
        else:
            in_proj = torch.cat((q_weight, k_weight, v_weight), dim=0)
            self.assign_new_value('in_proj_weight', in_proj)
                
        bias = torch.cat((q_bias, k_bias, v_bias), dim=0)
        self.assign_new_value('in_proj_bias', bias)
        
        with torch.no_grad():
            out_proj_weight = self._get_param(params['out']['kernel']).permute((2, 0, 1)).reshape(embed_dim, embed_dim)
            out_proj_bias = self._get_param(params['out']['bias'])
            self.assign_new_value('weight', out_proj_weight, self.out_proj)
            self.assign_new_value('bias', out_proj_bias, self.out_proj)

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
        deterministic: bool = True,
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
            dropout=0. if deterministic else attention_dropout_rate,
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
        if attention_mask is not None:
            attention_mask = attention_mask.to(torch.bool)
            # attention_mask = ~attention_mask
        x, _ = self.self_attention(x, x, x, attn_mask=attention_mask, need_weights=False)
        x = self.dropout1(x) if not deterministic else x
        x = x + inputs

        # MLP block.
        y = self.layer_norm2(x)
        y = self.mlp_block(y, deterministic=deterministic)

        return x + y
    

class TransformerPt(nn.Module, FromJaxModel):
    """Transformer Model Encoder for sequence to sequence translation.

    Attributes:
      num_layers: number of layers
      mlp_dim: dimension of the mlp on top of attention block
      num_attention_heads: Number of heads in nn.MultiheadAttention
      dropout_rate: dropout rate.
      attention_dropout_rate: dropout rate in self attention.
    """

    def __init__(
        self,
        token_embedding_size: int,
        num_layers: int,
        mlp_dim: int,
        num_attention_heads: int,
        dropout_rate: float = 0.1,
        attention_dropout_rate: float = 0.1,
        add_position_embedding: bool = False
    ):
        super().__init__()
        self.token_embedding_size = token_embedding_size
        self.num_layers = num_layers
        self.mlp_dim = mlp_dim
        self.num_attention_heads = num_attention_heads
        self.dropout_rate = dropout_rate
        self.attention_dropout_rate = attention_dropout_rate
        self.add_position_embedding = add_position_embedding

        if self.add_position_embedding:
            self.position_embedding = AddPositionEmbsPt(
                hid_dim=self.token_embedding_size
            )
        self.dropout = nn.Dropout(dropout_rate)

        self.encoder_blocks = nn.ModuleList([
            Encoder1DBlockPt(
                input_dim=self.token_embedding_size,
                mlp_dim=self.mlp_dim,
                num_heads=self.num_attention_heads,
                dropout_rate=self.dropout_rate,
                attention_dropout_rate=self.attention_dropout_rate,
            ) for _ in range(self.num_layers)
        ])
        self.layer_norm = LayerNormPt(self.token_embedding_size)

    def forward(self, x: torch.Tensor, attention_mask: torch.Tensor, train: bool = True) -> torch.Tensor:
        """Applies Transformer model on the inputs.

        Args:
          x: Inputs to the layer.
          attention_mask: Mask for attention mechanism.
          train: Set to `True` when training.

        Returns:
          output of a transformer encoder.
        """
        assert x.dim() == 3  # (batch, len, emb)

        if self.add_position_embedding:
            x = self.position_embedding(x)
            x = self.dropout(x) if train else x

        # Input Encoder
        for encoder_block in self.encoder_blocks:
            x = encoder_block(x, attention_mask, deterministic=not train)
        encoded = self.layer_norm(x)

        return encoded

    def load_jax_weights(self, jax_params):
        if self.add_position_embedding:
            self.position_embedding.load_jax_weights(jax_params['position_embedding'])
            
        for i in range(len(self.encoder_blocks)):
            self.encoder_blocks[i].load_jax_weights(jax_params[f'encoderblock_{i}'])
        
        self.layer_norm.load_jax_weights(jax_params['encoder_norm'])
        

def common_transformer_sizes_pt(transformer_size: str) -> (int, dict):
    """
    Args:
        transformer_size (str): The size of the transformer. One of "dummy", "vanilla", "vit_s", "vit_b", "vit_l", "vit_h"

    Returns:
            token_embedding_size (int): The size of the token embeddings
            transformer_kwargs (dict): The kwargs to pass to the transformer

    """
    assert transformer_size in [
        "dummy",
        "vanilla",
        "vit_t",
        "vit_s",
        "vit_b",
        "vit_l",
        "vit_h",
    ]
    default_params = {
        "attention_dropout_rate": 0.0,
        "add_position_embedding": False,
    }

    TRANSFORMER_SIZES = {
        "dummy": dict(
            token_embedding_size=256,
            num_layers=1,
            mlp_dim=256,
            num_attention_heads=2,
            dropout_rate=0.1,
        ),
        "vanilla": dict(
            token_embedding_size=256,
            num_layers=4,
            mlp_dim=1024,
            num_attention_heads=8,
            dropout_rate=0.1,
        ),
        "vit_t": dict(
            token_embedding_size=192,
            num_layers=12,
            mlp_dim=768,
            num_attention_heads=3,
            dropout_rate=0.0,
        ),
        "vit_s": dict(
            token_embedding_size=384,
            num_layers=12,
            mlp_dim=1536,
            num_attention_heads=6,
            dropout_rate=0.0,
        ),
        "vit_b": dict(
            token_embedding_size=768,
            num_layers=12,
            mlp_dim=3072,
            num_attention_heads=12,
            dropout_rate=0.0,
        ),
        "vit_l": dict(
            token_embedding_size=1024,
            num_layers=24,
            mlp_dim=4096,
            num_attention_heads=16,
            dropout_rate=0.1,
        ),
        "vit_h": dict(
            token_embedding_size=1280,
            num_layers=32,
            mlp_dim=5120,
            num_attention_heads=16,
            dropout_rate=0.1,
        ),
    }

    TOKEN_DIMS = {
        "dummy": 256,
        "vanilla": 256,
        "vit_t": 192,
        "vit_s": 384,
        "vit_b": 768,
        "vit_l": 1024,
        "vit_h": 1280,
    }

    return TOKEN_DIMS[transformer_size], {
        **default_params,
        **TRANSFORMER_SIZES[transformer_size],
    }
