from typing import Tuple

import torch.nn as nn
import torch

default_init = nn.init.xavier_uniform


def mish(x):
    return x * torch.tanh(torch.nn.functional.softplus(x))


def unet_squaredcos_cap_v2(timesteps, s=0.008):
    t = torch.linspace(0, timesteps, timesteps + 1) / timesteps
    alphas_cumprod = torch.cos((t + s) / (1 + s) * torch.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0, 0.999)


class SinusoidalPosEmb(nn.Module):
    def __init__(self, features: int):
        super().__init__()
        self.features = features

    def forward(self, x):
        half_features = self.features // 2
        emb = torch.log(10000) / (half_features - 1)
        emb = torch.exp(torch.arange(half_features) * -emb)
        emb = x * emb
        emb = torch.concatenate((torch.sin(emb), torch.cos(emb)), dim=-1)
        return emb


class Downsample1d(nn.Module):
    def __init__(self, features: int):
        super().__init__()
        self.conv = nn.Conv1d(features, kernel_size=3, stride=2)

    def forward(self, x):
        return self.conv(x)


class Upsample1d(nn.Module):
    def __init__(self, features: int):
        super().__init__()
        self.conv = nn.ConvTranspose1d(features, kernel_size=4, stride=2)

    def forward(self, x):
        return self.conv(x)


class Conv1dBlock(nn.Module):
    """
    Conv1d --> GroupNorm --> Mish
    """

    def __init__(self, features: int, kernel_size: int, n_groups: int):
        super().__init__()
        self.conv = nn.Conv1d(
            features, kernel_size=kernel_size, stride=1, padding=kernel_size // 2
        )
        self.norm = nn.GroupNorm(n_groups, features)
        self.act = mish

    def forward(self, x):
        x = self.conv(x)
        x = self.norm(x)
        x = self.act(x)
        return x


class ConditionalResidualBlock1D(nn.Module):
    def __init__(
        self, features: int, kernel_size: int, n_groups: int, residual_proj: bool
    ):
        super().__init__()
        self.features = features
        self.kernel_size = kernel_size
        self.n_groups = n_groups
        self.residual_proj = residual_proj

        self.conv1 = Conv1dBlock(features, kernel_size, n_groups)
        self.conv2 = Conv1dBlock(features, kernel_size, n_groups)

        if residual_proj:
            self.residual_proj = nn.Conv1d(features, kernel_size=1, stride=1, padding=0)

        self.cond_linear = nn.Linear(features, features * 2)

    def forward(self, x, cond):
        residual = x
        x = self.conv1(x)

        cond = self.cond_linear(mish(cond))
        scale, bias = torch.split(cond, 2, axis=-1)
        # Scale, bias are (B, D) and x is shape (B, T, D)
        # We need to broadcast over time, so choose axis = -2
        x = x * scale.unsqueeze(-2) + bias.unsqueeze(-2)
        x = self.conv2(x)

        if self.residual_proj:
            residual = self.residual_proj(residual)

        return x + residual


class ConditionalUnet1D(nn.Module):
    def __init__(
        self,
        down_features: Tuple[int],
        mid_layers: int,
        kernel_size: int,
        n_groups: int,
        time_features: int,
    ):
        super().__init__()
        self.down_features = down_features
        self.mid_layers = mid_layers
        self.kernel_size = kernel_size
        self.n_groups = n_groups
        self.time_features = time_features
        self.pos_emb = SinusoidalPosEmb(time_features)
        self.linear1 = nn.Linear(time_features, 4 * time_features)
        self.linear2 = nn.Linear(time_features, time_features)

        self.downs = nn.ModuleList()
        for i, features in enumerate(down_features):
            self.downs.append(
                ConditionalResidualBlock1D(
                    features,
                    kernel_size,
                    n_groups,
                    residual_proj=True,
                )
            )
            self.downs.append(
                ConditionalResidualBlock1D(
                    features,
                    kernel_size,
                    n_groups,
                    residual_proj=False,
                )
            )
            if i != len(down_features) - 1:
                self.downs.append(Downsample1d(features))

        self.mids = nn.ModuleList()
        for _ in range(mid_layers):
            self.mids.append(
                ConditionalResidualBlock1D(
                    down_features[-1],
                    kernel_size,
                    n_groups,
                )
            )

        self.ups = nn.ModuleList()
        for i, features in enumerate(reversed(down_features[:-1])):
            self.ups.append(
                ConditionalResidualBlock1D(
                    features,
                    kernel_size,
                    n_groups,
                    residual_proj=True,
                )
            )
            self.ups.append(
                ConditionalResidualBlock1D(
                    features,
                    kernel_size,
                    n_groups,
                    residual_proj=False,
                )
            )
            self.ups.append(Upsample1d(features))

        self.final_conv = Conv1dBlock(
            down_features[0], kernel_size, n_groups
        )

    def forward(self, obs, action, time, train: bool = False):
        # Embed the timestep
        time = self.pos_emb(time)
        time = self.linear1(time)
        time = mish(time)
        time = self.linear2(time)
        # Define conditioning as time and observation
        cond = torch.cat((obs, time), dim=-1)

        # Project Down
        hidden_reps = []
        for i, features in enumerate(self.down_features):
            # We always project to the dimension on the first residual connection.
            action = self.downs[i](action, cond)
            action = self.downs[i + 1](action, cond)
            if i != 0:
                hidden_reps.append(action)
            if i != len(self.down_features) - 1:
                # If we aren't the last step, downsample
                action = self.downs[i + 2](action)

        # Mid Layers
        for i in range(self.mid_layers):
            action = self.mids[i](action, cond)

        # Project Up
        for features, hidden_rep in reversed(
            list(zip(self.down_features[:-1], hidden_reps, strict=False))
        ):
            action = torch.cat((action, hidden_rep), dim=-1)  # concat on feature dim
            # Always project since we are adding in the hidden rep
            action = self.ups[i](action, cond)
            action = self.ups[i + 1](action, cond)
            # Upsample
            action = self.ups[i + 2](action)

        # Should be the same as the input shape
        action = self.final_conv(action)
        return action
