from abc import ABC, abstractmethod
import logging
from typing import Dict, Optional, Tuple


from einops import rearrange
import torch
import torch.nn as nn
import torch.nn.functional as F

from octo.model.components.base import TokenGroup
from octo.model.components.base_pt import TokenGroupPt
from octo.model.components.diffusion_pt import (
    cosine_beta_schedule,
    create_diffusion_model,
)
from octo.model.components.tokenizers import BinTokenizer
from octo.model.components.transformer_pt import MAPHeadPt
from octo.model.components.unet_pt import ConditionalUnet1D, unet_squaredcos_cap_v2
from octo.model.components.jax_pt import FromJaxModel, LinearPt


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

    def __init__(
        self,
        readout_key: str,
        input_dim: int,
        use_map: bool = False,
        action_horizon: int = 1,
        action_dim: int = 7,
        max_action: float = 5.0,
        loss_type: str = "mse",
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
            "mean_proj": (self.mean_proj.load_jax_weights, "mean_proj"),
        }
        if self.use_map:
            pt_to_jax["map_head"] = (self.map_head.load_jax_weights, "map_head")
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

        mean = self(transformer_outputs, train=train)[:, -1]
        return mean
        # return jnp.broadcast_to(mean, sample_shape + mean.shape)


class MSEActionHeadPt(ContinuousActionHeadPt):
    def __init__(
        self,
        input_dim: int,
        readout_key: str,
        action_horizon: int = 1,
        action_dim: int = 7,
    ):
        super().__init__(
            input_dim=input_dim,
            readout_key=readout_key,
            use_map=True,
            action_horizon=action_horizon,
            action_dim=action_dim,
            max_action=5.0,
            loss_type="mse",
        )


class L1ActionHeadPt(ContinuousActionHeadPt):
    def __init__(
        self,
        input_dim: int,
        readout_key: str,
        use_map=True,
        action_horizon: int = 1,
        action_dim: int = 7,
        max_action=5.0,
    ):
        super().__init__(
            input_dim=input_dim,
            readout_key=readout_key,
            use_map=use_map,
            action_horizon=action_horizon,
            action_dim=action_dim,
            max_action=max_action,
            loss_type="l1",
        )


class DiffusionActionHeadPt(nn.Module, FromJaxModel):
    def __init__(
        self,
        readout_key: str,
        input_dim: int = 384,
        use_map: bool = False,
        time_input_dim: int = 1,
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
        n_diffusion_samples: int = 1,
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
        self.time_input_dim = time_input_dim
        
        # Initialize MAP head if needed
        if self.use_map:
            self.map_head = MAPHeadPt(
                input_dim=self.input_dim,
            )

        self.diffusion_model = create_diffusion_model(
            input_dim=self.input_dim,
            time_input_dim=self.time_input_dim,
            out_dim=self.action_dim * self.action_horizon,
            time_dim=self.time_dim,
            num_blocks=self.num_blocks,
            dropout_rate=self.dropout_rate,
            hidden_dim=self.hidden_dim,
            use_layer_norm=self.use_layer_norm,
        )

        betas = cosine_beta_schedule(self.diffusion_steps)
        alphas = 1 - betas
        alpha_hats = torch.cumprod(alphas, dim=0)
        self.register_buffer('betas', betas, persistent=False)
        self.register_buffer('alphas', alphas, persistent=False)
        self.register_buffer('alpha_hats', alpha_hats, persistent=False)
        

    @property
    def pt_to_jax_args_map(self):
        # {
        # pt_module_name: (load_func, jax_param_key),
        # ...
        # }
        return {
            "diffusion_model": (self.diffusion_model.load_jax_weights, "diffusion_model")
        }

    def forward(
        self,
        transformer_outputs: Dict[str, TokenGroupPt],
        time: Optional[torch.Tensor] = None,
        noisy_actions: Optional[torch.Tensor] = None,
        train: bool = True,
    ) -> torch.Tensor:
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
            time = torch.zeros(*embeddings.shape[:2], 1, device=device, dtype=torch.float32)
            noisy_actions = torch.zeros(
                *embeddings.shape[:2],
                self.action_dim * self.action_horizon,
                device=device,
            )

        pred_eps = self.diffusion_model(embeddings, noisy_actions, time)
        return pred_eps

    def loss(
        self,
        transformer_outputs: Dict[str, TokenGroupPt],
        actions: torch.tensor,
        timestep_pad_mask: torch.tensor,
        action_pad_mask: torch.tensor,
        train: bool = True,
    ) -> Tuple[torch.tensor, Dict[str, torch.tensor]]:

        batch_size, window_size = timestep_pad_mask.shape

        # fold action_dim and action_horizon into one dimension
        actions_flat = rearrange(actions, "b w h a -> b w (h a)")
        actions_flat = torch.clip(actions_flat, -self.max_action, self.max_action)

        time = torch.randint(
            low=0,
            high=self.diffusion_steps,
            size=(self.n_diffusion_samples, batch_size, window_size, 1),
            device=actions.device,
        )
        noise = torch.randn(
            size=(self.n_diffusion_samples,) + actions_flat.shape, device=actions.device
        )

        scale = torch.sqrt(self.alpha_hats[time])
        std = torch.sqrt(1 - self.alpha_hats[time])
        noisy_actions = scale * actions_flat[None] + std * noise

        pred_eps = self(
            transformer_outputs, train=train, time=time, noisy_actions=noisy_actions
        )

        # combine the timestep pad mask with the action pad mask
        mask = timestep_pad_mask[:, :, None, None] & action_pad_mask
        # flatten the mask to match the flat actions
        mask = rearrange(mask, "b w h a -> b w (h a)")
        # add a dimension to the mask for n_diffusion_samples
        mask = mask[None]

        loss, metrics = continuous_loss(pred_eps, noise, mask, loss_type=self.loss_type)
        # Sum over action dimension instead of averaging
        loss = loss * self.action_dim
        metrics["loss"] = metrics["loss"] * self.action_dim
        metrics["mse"] = metrics["mse"] * self.action_dim
        return loss, metrics

    def predict_action(
        self,
        transformer_outputs: Dict[str, TokenGroupPt],
        train: bool = True,
        embodiment_action_dim: Optional[int] = None,
        sample_shape: tuple = (),
        generator: torch.Generator = None,
        **kwargs
    ) -> torch.tensor:
        device=transformer_outputs[self.readout_key].tokens.device
        batch_size, window_size = transformer_outputs[self.readout_key].tokens.shape[:2]
        action_mask = torch.ones(
            *sample_shape,
            batch_size,
            window_size,
            self.action_horizon,
            self.action_dim,
            dtype=bool,
            device=device,
        )
        if embodiment_action_dim is not None:
            action_mask[..., embodiment_action_dim:] = False
        flat_action_mask = rearrange(action_mask, "... p a -> ... (p a)")

        current_x = torch.randn(
            (
                *sample_shape,
                batch_size,
                window_size,
                self.action_horizon * self.action_dim,
            ),
            generator=generator,
            device=device
        )
        for time in torch.arange(self.diffusion_steps - 1, -1, -1):
            input_time = torch.broadcast_to(time, (*current_x.shape[:-1], 1)).to(device)
            eps_pred = self(
                transformer_outputs,
                time=input_time,
                noisy_actions=current_x,
                train=train,
            )

            alpha_1 = 1 / torch.sqrt(self.alphas[time])
            alpha_2 = (1 - self.alphas[time]) / torch.sqrt(
                1 - self.alpha_hats[time]
            )
            current_x = alpha_1 * (current_x - alpha_2 * eps_pred)
            
            z = torch.empty_like(current_x).normal_(generator=generator)

            current_x = current_x + (time > 0) * (torch.sqrt(self.betas[time]) * z)
            current_x = torch.clip(current_x, -self.max_action, self.max_action)
            current_x = torch.where(
                flat_action_mask, current_x, torch.sqrt(1 - self.alpha_hats[time]) * z
            )

        actions = rearrange(
            current_x, "... (h a) -> ... h a", h=self.action_horizon, a=self.action_dim
        )
        return actions[..., -1, :, :]


class UNetDDPMActionHead(nn.Module, FromJaxModel):
    """Predicts actions using a diffusion process and a U-Net architecture (unlike MLP above)

    Only a single pass through the transformer is done to obtain an action embedding at each timestep. The
    actions are then predicted using a diffusion process conditioned on this embedding. The diffusion model
    architecture is an 1D unet based on the implementation from Chi et al: https://arxiv.org/abs/2303.04137

    You may create an embedding by either mean-pooling across tokens (use_map=False) or using multi-head
    attention pooling (use_map=True). It is recommended to use MAP when decoding from the observation token
    stream.
    """

    def __init__(
        self,
        readout_key: str,
        input_dim: int,
        action_dim: int,
        action_horizon: int,
        use_map: bool = (False,),
        flatten_tokens: bool = (False,),
        timesteps: int = 100,
        max_action: float = 1.0,
        clip_sample: Optional[float] = None,
        variance_type: str = "fixed_large",
        *args,
        **kwargs,
    ):
        super().__init__()
        self.readout_key = readout_key
        self.input_dim = input_dim
        self.action_dim = action_dim
        self.action_horizon = action_horizon
        self.use_map = use_map
        self.flatten_tokens = flatten_tokens
        self.timesteps = timesteps
        self.max_action = max_action
        self.clip_sample = clip_sample
        self.variance_type = variance_type

        self.action_proj = nn.Linear(self.input_dim, self.action_dim)
        betas = unet_squaredcos_cap_v2(self.timesteps).float()
        self.alphas = 1.0 - betas  # So betas = 1 - alphas
        self.alphas_cumprod = torch.cumprod(self.alphas, axis=0)

        self.model = ConditionalUnet1D(
            down_features=(256, 512, 1024),
            mid_layers=2,
            time_features=128,
            kernel_size=5,
        )
        if self.use_map:
            self.map_head = MAPHeadPt(
                input_dim=self.input_dim,
            )

    def forward(
        self,
        transformer_outputs: Dict[str, TokenGroup],
        time: Optional[torch.Tensor] = None,
        noisy_actions: Optional[torch.Tensor] = None,
        train: bool = True,
    ) -> torch.Tensor:
        """Performs a single forward pass through the diffusion model."""
        token_group = transformer_outputs[self.readout_key]
        assert token_group.tokens.ndim == 4, (
            f"Expected token_group.tokens to have shape (batch_size, window_size, num_tokens, embedding_size), "
            f"but got shape {token_group.tokens.shape}"
        )

        if self.use_map:  # Multi-head attention pooling
            assert not self.flatten_tokens, "Cannot use MAP token and flattening!"
            embeddings = self.map_head(token_group, train=train)[:, :, 0]
        elif self.flatten_tokens:  # concatenate tokens in final dim
            embeddings = token_group.tokens.reshape((*token_group.tokens.shape[:2], -1))
        else:  # mean pooling
            embeddings = token_group.tokens.mean(axis=-2)
        # Now, embeddings is (batch_size, window_size, embedding_size)

        pred_eps = self.model(embeddings, action=noisy_actions, time=time, train=train)
        pred_eps = self.action_proj(pred_eps)
        return pred_eps

    def loss(
        self,
        transformer_outputs: Dict[str, TokenGroup],
        actions: torch.Tensor,
        action_pad_mask: torch.Tensor,
        timestep_pad_mask: torch.Tensor,
        train: bool = True,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Computes the loss for the diffusion objective.

        Args:
            transformer_ouputs: must contain self.readout_key with shape (batch_size, window_size, num_tokens,
                embedding_size)
            actions: shape (batch_size, >= window_size + action_horizon - 1, action_dim)
            action_pad_mask: boolean array (same shape as actions) which is True if the action dimension is not a padding dimension
            timestep_pad_mask: boolean array (batch, window_size) which is True if the timestep is not a padding timestep

        Returns:
            loss: float
            metrics: dict
        """
        batch_size, window_size = timestep_pad_mask.shape[:2]

        actions = torch.clip(actions, -self.max_action, self.max_action)

        # piggy-back on the dropout rng chain for diffusion rng
        time = torch.randint(
            0,
            self.timesteps,
            (batch_size, window_size, 1),
            device=actions.device,
        )
        noise = torch.randn(actions.shape, device=actions.device)

        # Add noise to the action according to the schedule
        sqrt_alpha_prod = torch.sqrt(self.alphas_cumprod[time[:, None]])  # (B, 1, 1)
        sqrt_one_minus_alpha_prod = torch.sqrt(
            1 - self.alphas_cumprod[time[:, None]]
        )  # (B, 1, 1)
        noisy_actions = sqrt_alpha_prod * actions + sqrt_one_minus_alpha_prod * noise

        pred_eps = self(
            transformer_outputs, train=train, time=time, noisy_actions=noisy_actions
        )

        # combine the timestep-level pad mask with the action-dimension-level pad mask
        mask = (
            torch.broadcast_to(action_pad_mask[:, None, None, :], actions.shape)
            * timestep_pad_mask
        )

        loss, metrics = continuous_loss(pred_eps, noise, mask, loss_type="mse")
        # Sum over action dimension instead of averaging
        loss = loss * self.action_dim
        metrics["loss"] = metrics["loss"] * self.action_dim
        metrics["mse"] = metrics["mse"] * self.action_dim
        return loss, metrics

    def predict_action(
        self,
        transformer_outputs: Dict[str, TokenGroup],
        train: bool = True,
        embodiment_action_dim: Optional[int] = None,
    ) -> torch.Tensor:
        """
        Code inspired by diffusers:
        https://github.com/huggingface/diffusers/blob/main/src/diffusers/schedulers/scheduling_ddpm_flax.py
        """
        batch_size, window_size = transformer_outputs[self.readout_key].tokens.shape[:2]

        action_mask = torch.ones(
            (
                batch_size,
                window_size,
                self.action_horizon,
                self.action_dim,
            ),
            dtype=bool,
            device=transformer_outputs[self.readout_key].tokens.device,
        )

        if embodiment_action_dim is not None:
            action_mask = action_mask.at[..., embodiment_action_dim:].set(False)
        else:
            logging.warning(
                "embodiment_action_dim is highly recommended for diffusion action head"
                " if any action dimensions were masked during training"
            )

        sample = torch.randn(
            (
                batch_size,
                window_size,
                self.action_horizon,
                self.action_dim,
            ),
            device=transformer_outputs[self.readout_key].tokens.device,
        )

        for i in range(self.timesteps):
            
            time = self.timesteps - 1 - i
            # Note that here time is (B, 1, 1) where as in loss in is (B, 1)
            time = torch.broadcast_to(time, (sample.shape[0], 1, 1))
            alpha = self.alphas[time]
            alpha_prod_t = self.alphas_cumprod[time]
            alpha_prod_t_prev = torch.where(
                time > 0,
                self.alphas_cumprod[time - 1],
                torch.array(1.0, dtype=float),
            )

            # Run the model. Reduce time to (B, 1) for the model.
            eps = self(
                transformer_outputs,
                time=time,
                noisy_actions=sample,
                train=train,
            )

            # Predict x_0, clip if desired.
            orig = (sample - torch.sqrt(1 - alpha_prod_t) * eps) / torch.sqrt(alpha_prod_t)
            if self.clip_sample is not None:
                orig = torch.clip(orig, -self.clip_sample, self.clip_sample)

            # Compute x_{t-1} using x_0
            orig_coeff = torch.sqrt(alpha_prod_t_prev) * (1 - alpha) / (1 - alpha_prod_t)
            current_coeff = (
                torch.sqrt(alpha) * (1 - alpha_prod_t_prev) / (1 - alpha_prod_t)
            )

            prev = orig_coeff * orig + current_coeff * sample

            # Add noise according to the schedule
            variance = (1 - alpha_prod_t_prev) / (1 - alpha_prod_t) * (1 - alpha)
            if self.variance_type == "fixed_large":
                variance = 1 - alpha
            elif self.variance_type == "fixed_small":
                variance = torch.clip(variance, a_min=1e-20)
            else:
                raise ValueError("Invalid schedule provided")

            variance = torch.where(
                time > 0, variance, torch.zeros(eps.shape, dtype=float, device=sample.device)
            )
            z = torch.randn(shape=sample.shape, dtype=float, device=sample.device)
            prev = prev + torch.sqrt(variance) * z

            # set non-eval actions to the noise that would have been seen during training
            sample = torch.where(action_mask, prev, torch.sqrt(1 - alpha_prod_t) * z)

        noisy_action = sample


        return noisy_action
