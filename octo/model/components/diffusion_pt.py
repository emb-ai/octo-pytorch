# copied from: https://raw.githubusercontent.com/rail-berkeley/bridge_data_v2/main/jaxrl_m/networks/diffusion_nets.py
import logging

import torch.nn as nn
import torch

from octo.model.components.jax_pt import FromJaxModel, LinearPt, LayerNormPt, ParamNode

default_init = nn.init.xavier_uniform


def cosine_beta_schedule(timesteps, s=0.008):
    """
    cosine schedule
    as proposed in https://openreview.net/forum?id=-NEXDKk8gZ
    """
    steps = timesteps + 1
    t = torch.linspace(0, timesteps, steps) / timesteps
    alphas_cumprod = torch.cos((t + s) / (1 + s) * torch.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0, 0.999)


class ScoreActorPt(nn.Module, FromJaxModel):
    def __init__(self, time_preprocess, cond_encoder, reverse_network):
        super().__init__()
        self.time_preprocess = time_preprocess
        self.cond_encoder = cond_encoder
        self.reverse_network = reverse_network

    def forward(self, obs_enc, actions, time):
        """
        Args:
            obs_enc: (bd..., obs_dim) where bd... is broadcastable to batch_dims
            actions: (batch_dims..., action_dim)
            time: (batch_dims..., 1)
        """
        t_ff = self.time_preprocess(time)
        cond_enc = self.cond_encoder(t_ff)
        if obs_enc.shape[:-1] != cond_enc.shape[:-1]:
            new_shape = cond_enc.shape[:-1] + (obs_enc.shape[-1],)
            logging.debug(
                "Broadcasting obs_enc from %s to %s", obs_enc.shape, new_shape
            )
            obs_enc = torch.broadcast_to(obs_enc, new_shape)

        reverse_input = torch.concatenate([cond_enc, obs_enc, actions], dim=-1)
        eps_pred = self.reverse_network(reverse_input)
        return eps_pred

    def pt_to_jax_args_map(self):
        # {
        # pt_module_name: (load_func, jax_param_key),
        # ...
        # }
        return {
            "time_preprocess": ParamNode(submodule=self.time_preprocess, jax_param_names='time_preprocess'),
            "cond_encoder": ParamNode(submodule=self.cond_encoder, jax_param_names='cond_encoder'),
            "reverse_network": ParamNode(submodule=self.reverse_network, jax_param_names='reverse_network')
        }

class FourierFeaturesPt(nn.Module, FromJaxModel):
    def __init__(self, input_size, output_size, learnable=True):
        super().__init__()
        self.output_size = output_size
        self.input_size = input_size
        self.learnable = learnable
        self.w = nn.Parameter(
            torch.randn(self.output_size // 2, self.input_size, dtype=torch.float32) * 0.2
        ) # <- nn.initializers.normal(0.2),

    def forward(self, x):
        if self.learnable:
            f = 2 * torch.pi * x @ self.w.T
        else:
            half_dim = self.output_size // 2
            f = torch.log(10000) / (half_dim - 1)
            f = torch.exp(torch.arange(half_dim) * -f)
            f = x * f
        return torch.concatenate([torch.cos(f), torch.sin(f)], dim=-1)
    
    def pt_to_jax_args_map(self):
        # {
        # pt_module_name: (load_func, jax_param_key),
        # ...
        # }
        return {
            "w": ParamNode(load_func=self._set_terminal_param, jax_param_names='kernel')
        }


class MLPPt(nn.Module, FromJaxModel):
    def __init__(self, input_dim, hidden_dims, activation=nn.SiLU(), activate_final=False, use_layer_norm=False, dropout_rate=None):
        super().__init__()
        self.use_layer_norm = use_layer_norm
        self.activate_final = activate_final
        self.dropout_rate = dropout_rate
        
        layers = []
        hidden_dims = [input_dim] + list(hidden_dims)
        
        for i in range(len(hidden_dims) - 1):
            layers.append(LinearPt(hidden_dims[i], hidden_dims[i + 1])) # <- , kernel_init=default_init()
            if i + 1 < len(hidden_dims) - 1 or activate_final:
                if dropout_rate is not None and dropout_rate > 0:
                    layers.append(nn.Dropout(rate=dropout_rate))
                if use_layer_norm:
                    layers.append(LayerNormPt(hidden_dims[i + 1]), eps=1e-6)
                layers.append(activation)
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)
    
    def pt_to_jax_args_map(self):
        # {
        # pt_module_name: (load_func, jax_param_key),
        # ...
        # }
        pt_to_jax_dict = {}
        i = 0
        num_layer = 0
        while i < len(self.layers):
            pt_to_jax_dict[f"layers.{i}"] = ParamNode(submodule=self.layers[i], jax_param_names=f"Dense_{num_layer}")
            i += 1
            if num_layer + 1 < len(self.layers) - 1 or self.activate_final:
                if self.dropout_rate is not None and self.dropout_rate > 0:
                    i += 1
                if self.use_layer_norm:
                    pt_to_jax_dict[f"layers.{i}"] = ParamNode(submodule=self.layers[i], jax_param_names=f"LayerNorm_{num_layer}")
                    i += 1
            i += 1
            num_layer += 1
                
        return pt_to_jax_dict


class MLPResNetBlockPt(nn.Module, FromJaxModel):
    def __init__(self, input_dim, features, act, dropout_rate=None, use_layer_norm=False):
        super().__init__()
        self.input_dim = input_dim
        self.features = features
        self.act = act
        self.dropout_rate = dropout_rate
        self.use_layer_norm = use_layer_norm
        self.dropout = nn.Dropout(p=dropout_rate)
        self.layer_norm = LayerNormPt(input_dim, eps=1e-6) if use_layer_norm else nn.Identity()
        self.linear1 = LinearPt(features, features * 4)
        self.linear2 = LinearPt(features * 4, features)
        if input_dim != features:
            self.residual = LinearPt(features, input_dim)
        else:
            self.residual = nn.Identity()

    def forward(self, x):
        residual = x
        if self.dropout_rate is not None and self.dropout_rate > 0:
            x = self.dropout(x)
        if self.use_layer_norm:
            x = self.layer_norm(x)
        x = self.linear1(x)
        x = self.act(x)
        x = self.linear2(x)
        residual = self.residual(residual)
        return residual + x

    def pt_to_jax_args_map(self):
        # {
        # pt_module_name: (load_func, jax_param_key),
        # ...
        # }
        pt_to_jax_dict = {
            "linear1": ParamNode(submodule=self.linear1, jax_param_names="Dense_0"),
            "linear2": ParamNode(submodule=self.linear2, jax_param_names="Dense_1"),
        }
        if self.use_layer_norm:
            pt_to_jax_dict["layer_norm"] = ParamNode(submodule=self.layer_norm, jax_param_names="LayerNorm_0")
        
        if self.input_dim != self.features:
            pt_to_jax_dict["residual"] = ParamNode(submodule=self.residual, jax_param_names="Dense_2")
        
        return pt_to_jax_dict

class MLPResNetPt(nn.Module, FromJaxModel):
    def __init__(self, input_dim, num_blocks, out_dim, dropout_rate=None, use_layer_norm=False, hidden_dim=256, activation=nn.SiLU()):
        super().__init__()
        self.input_dim = input_dim
        self.num_blocks = num_blocks
        self.out_dim = out_dim
        self.dropout_rate = dropout_rate
        self.use_layer_norm = use_layer_norm
        self.hidden_dim = hidden_dim
        self.activation = activation
        self.linear1 = LinearPt(input_dim, hidden_dim) # <- default_init()
        self.blocks = nn.Sequential(
            MLPResNetBlockPt(hidden_dim, hidden_dim, activation, dropout_rate, use_layer_norm)
        )

        for _ in range(num_blocks - 1):
            self.blocks.append(MLPResNetBlockPt(hidden_dim, hidden_dim, activation, dropout_rate, use_layer_norm))

        self.linear2 = LinearPt(hidden_dim, out_dim) # <- default_init()



    def forward(self, x):
        x = self.linear1(x)
        x = self.blocks(x)
        x = self.activation(x)
        x = self.linear2(x) # <- default_init()
        return x

    def pt_to_jax_args_map(self):
        # {
        # pt_module_name: (load_func, jax_param_key),
        # ...
        # }
        pt_to_jax_dict = {
            "linear1": ParamNode(submodule=self.linear1, jax_param_names="Dense_0"),
            "linear2": ParamNode(submodule=self.linear2, jax_param_names="Dense_1"),
        }
        for i in range(len(self.blocks)):
            pt_to_jax_dict[f"blocks.{i}"] = ParamNode(submodule=self.blocks[i], jax_param_names=f"MLPResNetBlock_{i}")
        
        return pt_to_jax_dict

def create_diffusion_model(
    input_dim: int,
    time_input_dim: int,
    out_dim: int,
    time_dim: int,
    num_blocks: int,
    dropout_rate: float,
    hidden_dim: int,
    use_layer_norm: bool,
):
    return ScoreActorPt(
        FourierFeaturesPt(time_input_dim, time_dim, learnable=True),
        MLPPt(time_dim, (2 * time_dim, time_dim)),
        MLPResNetPt(
            input_dim + out_dim + time_dim,
            num_blocks,
            out_dim,
            dropout_rate=dropout_rate,
            hidden_dim=hidden_dim,
            use_layer_norm=use_layer_norm,
        ),
    )
