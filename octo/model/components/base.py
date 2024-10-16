import flax
import jax
import jax.numpy as jnp

from octo.utils.typing import Sequence

import torch
from dataclasses import dataclass 

@flax.struct.dataclass
class TokenGroup:
    """A group of tokens that have semantic meaning together (e.g. the tokens for a single observation)

    Attributes:
        tokens: jax.Array of shape (..., n_tokens, token_dim)
        mask: jax.Array of shape (..., n_tokens) indicating which tokens are valid (1) vs padding (0)
    """

    tokens: jax.typing.ArrayLike
    mask: jax.typing.ArrayLike

    @classmethod
    def create(
        cls, tokens: jax.typing.ArrayLike, mask: jax.typing.ArrayLike = None, **kwargs
    ):
        if mask is None:
            mask = jnp.ones(tokens.shape[:-1])
        assert mask.ndim == tokens.ndim - 1
        return cls(tokens, mask, **kwargs)

    @classmethod
    def concatenate(cls, group_list: Sequence["TokenGroup"], axis=-2):
        data = jnp.concatenate([t.tokens for t in group_list], axis=axis)
        mask = jnp.concatenate([t.mask for t in group_list], axis=axis + 1)
        return cls(data, mask)



@dataclass
class TokenGroupPt:
    """A group of tokens that have semantic meaning together (e.g. the tokens for a single observation)

    Attributes:
        tokens: torch.tensor of shape (..., n_tokens, token_dim)
        mask: torch.tensor of shape (..., n_tokens) indicating which tokens are valid (1) vs padding (0)
    """

    tokens: torch.tensor
    mask: torch.tensor

    @classmethod
    def create(
        cls, tokens: torch.tensor, mask: torch.tensor = None, **kwargs
    ):
        if mask is None:
            mask = torch.ones(tokens.shape[:-1], device=tokens.device)
        assert mask.ndim == tokens.ndim - 1
        return cls(tokens, mask, **kwargs)

    @classmethod
    def concatenate(cls, group_list: Sequence["TokenGroupPt"], axis=-2):
        data = torch.cat([t.tokens for t in group_list], dim=axis)
        mask = torch.cat([t.mask for t in group_list], dim=axis + 1)
        return cls(data, mask)

    def to_numpy(self):
        return TokenGroup(
            tokens=self.tokens.detach().cpu().numpy(),
            mask=self.mask.detach().cpu().numpy()
        )