from abc import ABC, abstractmethod
import logging
from typing import Dict, Optional, Tuple

import distrax
from einops import rearrange
import torch
import torch.nn as nn
import torch.nn.functional as F

from octo.model.components.base import TokenGroup, TokenGroupPt
from octo.model.components.diffusion import cosine_beta_schedule, create_diffusion_model
from octo.model.components.tokenizers import BinTokenizer
from octo.model.components.transformer_pt import MAPHeadPt
from octo.model.components.unet import ConditionalUnet1D, unet_squaredcos_cap_v2
from octo.model.components.jax_pt import FromJaxModel
from octo.model.components.jax_pt import LinearPt


def masked_mean(x, mask):
    # mask = jnp.broadcast_to(mask, x.shape)
    mask = mask.float()
    return torch.mean(x * mask) / torch.clip(torch.mean(mask), min=1e-5, max=None)


def continuous_loss(
    pred_value: torch.tensor,
    ground_truth_value: torch.tensor,
    mask: torch.tensor,
    loss_type: str = "mse",
) -> torch.tensor:
    """
    Args:
        pred_value: shape (batch_dims...)
        ground_truth_value: continuous values w/ shape (batch_dims...)
        mask: broadcastable to ground_truth
    """
    if loss_type == "mse":
        loss = torch.square(pred_value - ground_truth_value)
    elif loss_type == "l1":
        loss = torch.abs(pred_value - ground_truth_value)
    else:
        raise ValueError(f"Invalid loss type: {loss_type}")

    loss = masked_mean(loss, mask)

    mse = torch.square(pred_value - ground_truth_value)
    mse = masked_mean(mse, mask)
    return loss, {
        "loss": loss,
        "mse": mse,
    }

class ActionHead(ABC):
    """Action prediction modules that take in the transformer token outputs and predict actions.

    Each action head here does chunked action prediction: i.e. at every timestep, it tries to predict the next
    `action_horizon` actions into the future from that timestep.  Setting `action_horizon=1` corresponds to
    the typical action prediction setup.
    """

    @abstractmethod
    def loss(
        self,
        transformer_outputs: Dict[str, TokenGroup],
        actions: torch.tensor,
        timestep_pad_mask: torch.tensor,
        action_pad_mask: torch.tensor,
        train: bool = True,
    ) -> Tuple[torch.tensor, Dict[str, torch.tensor]]:
        raise NotImplementedError

    @abstractmethod
    def predict_action(
        self,
        transformer_outputs: Dict[str, TokenGroup],
        argmax: bool = False,
        sample_shape: Tuple[int, ...] = (),
        temperature: float = 1.0,
        train: bool = False,
        embodiment_action_dim: Optional[int] = None,
    ) -> torch.tensor:
        """Predict the action for the last timestep in the window. Returns shape
        (*sample_shape, batch_size, action_horizon, action_dim).
        """
        raise NotImplementedError


class ContinuousActionHeadPt(nn.Module, ActionHead, FromJaxModel):
    """Predicts continuous actions (as opposed to discretized).

    Continuous actions are predicted by tanh squashing the model output to [-max_action, max_action], and then
    optimized using a standard regression loss.

    You may create an embedding by either mean-pooling across tokens (use_map=False) or using multi-head
    attention pooling (use_map=True). It is recommended to use MAP when decoding from the observation token
    stream.
    """
    
    def __init__(self, readout_key: str,
        input_dim: int,
        use_map: bool = False,
        action_horizon: int = 1,
        action_dim: int = 7,
        max_action: float = 5.0,
        loss_type: str = "mse"
    ):
        super().__init__()
        self.readout_key = readout_key
        self.use_map = use_map  
        self.action_horizon = action_horizon
        self.action_dim = action_dim
        self.max_action = max_action
        self.loss_type = loss_type
        self.input_dim = input_dim
        
        if self.use_map:
            self.map_head = MAPHeadPt(self.input_dim)
        self.mean_proj = LinearPt(self.input_dim, self.action_horizon * self.action_dim)

    @property
    def pt_to_jax_args_map(self):
        pt_to_jax = {
            'mean_proj': (self.mean_proj.load_jax_weights, 'mean_proj'),
        }
        if self.use_map:
            pt_to_jax['map_head'] = (self.map_head.load_jax_weights, 'map_head')
        return pt_to_jax
    
    def forward(
        self, transformer_outputs: Dict[str, TokenGroupPt], train: bool = True
    ) -> torch.tensor:
        """
        Returns:
            mean: Predicted actions w/ shape (batch_size, window_size, action_horizon, action_dim)
        """
        token_group = transformer_outputs[self.readout_key]
        assert token_group.tokens.ndim == 4, (
            f"Expected token_group.tokens to have shape (batch_size, window_size, num_tokens, embedding_size), "
            f"but got shape {token_group.tokens.shape}"
        )
        if self.use_map:  # Multi-head attention pooling
            embeddings = self.map_head(token_group, train=train)[:, :, 0]
        else:  # mean pooling
            embeddings = token_group.tokens.mean(axis=-2)
        # Now, embeddings is (batch_size, window_size, embedding_size)

        mean = self.mean_proj(embeddings)
        mean = rearrange(
            mean, "b w (h a) -> b w h a", h=self.action_horizon, a=self.action_dim
        )
        mean = F.tanh(mean / self.max_action) * self.max_action
        return mean

    def loss(
        self,
        transformer_outputs: Dict[str, TokenGroup],
        actions: torch.tensor,
        timestep_pad_mask: torch.tensor,
        action_pad_mask: torch.tensor,
        train: bool = True,
    ) -> Tuple[torch.tensor, Dict[str, torch.tensor]]:
        """Computes the loss for the action regression objective.

        Args:
            transformer_ouputs: must contain self.readout_key with shape (batch_size, window_size, num_tokens,
                embedding_size)
            actions: shape (batch_size, window_size, action_horizon, action_dim)
            timestep_pad_mask: boolean array (batch, window_size) which is True if the timestep is not a padding timestep
            action_pad_mask: boolean array (same shape as actions) which is True if the action dimension is not a padding dimension

        Returns:
            loss: float
            metrics: dict
        """
        # # (batch, window_size, action_horizon, action_dim)
        mean = self(transformer_outputs, train=train)

        # # combine the timestep pad mask with the action pad mask
        mask = timestep_pad_mask[:, :, None, None] & action_pad_mask

        loss, metrics = continuous_loss(mean, actions, mask, loss_type=self.loss_type)
        # # Sum over action dimension instead of averaging
        loss = loss * self.action_dim
        metrics["loss"] = metrics["loss"] * self.action_dim
        metrics["mse"] = metrics["mse"].detach() * self.action_dim
        return loss, metrics

    def predict_action(
        self,
        transformer_outputs: Dict[str, TokenGroup],
        train: bool = True,
        *args,
        sample_shape: tuple = (),
        **kwargs,
    ) -> torch.tensor:
        """Convenience methods for predicting actions for the final timestep in the window."""
        # only get the last timestep in the window
        pass
        # mean = self(transformer_outputs, train=train)[:, -1]
        # return jnp.broadcast_to(mean, sample_shape + mean.shape)




class MSEActionHeadPt(ContinuousActionHeadPt):
    def __init__(self, 
                 input_dim: int,
                readout_key: str,
                action_horizon: int = 1,
                action_dim: int = 7,
    ):
        super().__init__(
            input_dim = input_dim,
            readout_key = readout_key, 
            use_map = True,
            action_horizon = action_horizon,
            action_dim = action_dim,
            max_action = 5.0,
            loss_type = "mse"
        )

class L1ActionHeadPt(ContinuousActionHeadPt):
    def __init__(self, 
                 input_dim: int,
                readout_key: str,
                use_map = True,
                action_horizon: int = 1,
                action_dim: int = 7,
                max_action = 5.0,
    ):
        super().__init__(
            input_dim = input_dim,
            readout_key = readout_key, 
            use_map = use_map,
            action_horizon = action_horizon,
            action_dim = action_dim,
            max_action = max_action,
            loss_type = "l1"
        )
        
        
class DiffusionActionHeadPt(nn.Module, FromJaxModel):
    """
    To be implemented ... 
    """
    def __init__(
        self,
        input_dim: int,
        readout_key: str,
        use_map: bool = False,
        action_horizon: int = 1,
        action_dim: int = 7,
        max_action: float = 5.0,
        loss_type: str = "mse",
        time_dim: int = 32,
        num_blocks: int = 3,
        dropout_rate: float = 0.0,
        hidden_dim: int = 256,
        use_layer_norm: bool = True,
        diffusion_steps: int = 20,
        n_diffusion_samples: int = 1
    ):
        super().__init__()
        self.input_dim = input_dim
        self.readout_key = readout_key
        self.use_map = use_map
        self.action_horizon = action_horizon
        self.action_dim = action_dim
        self.max_action = max_action
        self.loss_type = loss_type
        self.time_dim = time_dim
        self.num_blocks = num_blocks
        self.dropout_rate = dropout_rate
        self.hidden_dim = hidden_dim
        self.use_layer_norm = use_layer_norm
        self.diffusion_steps = diffusion_steps
        self.n_diffusion_samples = n_diffusion_samples

        # Initialize MAP head if needed
        if self.use_map:
            self.map_head = MAPHeadPt(
                input_dim=input_dim,
                num_heads=8,  # You might want to make this configurable
                num_readouts=1
            )

        # Create the diffusion model (score network)
        # self.diffusion_model = create_diffusion_model(
        #     action_dim=self.action_dim * self.action_horizon,
        #     embedding_dim=embedding_dim,
        #     time_dim=self.time_dim,
        #     num_blocks=self.num_blocks,
        #     dropout_rate=self.dropout_rate,
        #     hidden_dim=self.hidden_dim,
        #     use_layer_norm=self.use_layer_norm,
        # )

        # Create beta schedule
        # betas = torch.from_numpy(cosine_beta_schedule(self.diffusion_steps)).float()
        # self.register_buffer('betas', betas)
        # self.register_buffer('alphas', 1 - betas)
        # self.register_buffer('alpha_hats', torch.cumprod(1 - betas, dim=0))

    @property
    def pt_to_jax_args_map(self):
        # {
        # pt_module_name: (load_func, jax_param_key),
        # ...
        # }
        return{} 
    
    def forward(
        self,
        transformer_outputs: Dict[str, TokenGroupPt],
        time: Optional[torch.Tensor] = None,
        noisy_actions: Optional[torch.Tensor] = None,
        train: bool = True,
    ) -> torch.Tensor:
        raise NotImplementedError
        """Performs a single forward pass through the diffusion model."""
        token_group = transformer_outputs[self.readout_key]
        assert token_group.tokens.dim() == 4, (
            f"Expected token_group.tokens to have shape (batch_size, window_size, num_tokens, embedding_size), "
            f"but got shape {tuple(token_group.tokens.shape)}"
        )

        if self.use_map:  # Multi-head attention pooling
            embeddings = self.map_head(token_group, train=train)[:, :, 0]
        else:  # mean pooling
            embeddings = token_group.tokens.mean(dim=-2)
        # Now, embeddings is (batch_size, window_size, embedding_size)

        # Handle initialization case
        if time is None or noisy_actions is None:
            device = embeddings.device
            time = torch.zeros(*embeddings.shape[:2], 1, device=device)
            noisy_actions = torch.zeros(
                *embeddings.shape[:2],
                self.action_dim * self.action_horizon,
                device=device
            )

        pred_eps = self.diffusion_model(embeddings, noisy_actions, time, train=train)
        return pred_eps

    @property
    def device(self) -> torch.device:
        """Get the device of the model."""
        raise NotImplementedError
        return next(self.parameters()).device
