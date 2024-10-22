from abc import ABC, abstractmethod
from dataclasses import dataclass
import logging 
from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F

import jax
import flax
import flax.linen as flax_nn

import numpy as np

from octo.utils.spec import ModuleSpec


class FromJaxModel(ABC):

    @abstractmethod
    def load_jax_weights(self, jax_params=None):
        pass
    
    def _get_param(self, jax_conv_params):
        weight = torch.from_numpy(jax_conv_params.copy()).float()
        return nn.Parameter(weight)

    def _get_conv_params(self, jax_conv_params):
        weight = torch.from_numpy(jax_conv_params['kernel'].transpose((3, 2, 0, 1)).copy()).float()
        bias = torch.from_numpy(jax_conv_params['bias'].copy()).float()
        return nn.Parameter(weight), nn.Parameter(bias)
    
    def _get_linear_params(self, jax_conv_params):
        weight = torch.from_numpy(jax_conv_params['kernel'].transpose((1, 0)).copy()).float()
        bias = torch.from_numpy(jax_conv_params['bias'].copy()).float()
        return nn.Parameter(weight), nn.Parameter(bias)
    
    
    # @classmethod
    # @abstractmethod
    # def from_jax(cls, jax_config, checkpointer, init_args):
    #     jax_model = ModuleSpec.instantiate(jax_config)()
    #     params_shape = jax.eval_shape(
    #             partial(jax_model.init, train=False), jax.random.PRNGKey(0), *init_args
    #         )["params"]
    #     step = checkpointer.latest_step()
    #     params = checkpointer.restore(step, params_shape)
    #     instance = cls(jax_module=jax_model, jax_params=params)
    #     instance.load_jax_weights(params)
    #     return instance
        
    
    def test_forward(self, inputs_pt, inputs_jax, rtol=0.001, atol=0.001):
        raise NotImplementedError
        if self.jax_module is None:
            logging.warning("Original jax module unspecified. Can't test inference")
            return False
        with torch.no_grad():
            output_pt = self.forward(*inputs_pt)
        output_jax = self.jax_module.apply(
            self.jax_params,
            *inputs_jax,
        )
        is_tokens_equal = np.isclose(output_pt.to_numpy().tokens, output_jax.tokens, rtol=rtol, atol=atol)
        is_masks_equal = np.all(output_pt.to_numpy().mask == output_jax.mask)
        
        return is_tokens_equal & is_masks_equal
        
class LinearPt(nn.Linear, FromJaxModel):
    """Linear Layer"""
    def load_jax_weights(self, jax_params=None):
        with torch.no_grad():
            self.weight, self.bias = self._get_linear_params(jax_params)

class ConvPt(nn.Conv2d, FromJaxModel):
    """Ordinary convolution"""
    def load_jax_weights(self, jax_params=None):
        with torch.no_grad():
            self.weight, self.bias = self._get_conv_params(jax_params)

class LayerNormPt(nn.LayerNorm, FromJaxModel):
    def load_jax_weights(self, jax_params=None):
        weight = torch.from_numpy(jax_params['scale'].copy()).float()
        bias = torch.from_numpy(jax_params['bias'].copy()).float()
        with torch.no_grad():
            self.weight, self.bias = nn.Parameter(weight), nn.Parameter(bias)

class GroupNormPt(nn.GroupNorm, FromJaxModel):
    def load_jax_weights(self, jax_params=None):
        weight = torch.from_numpy(jax_params['scale'].copy()).float()
        bias = torch.from_numpy(jax_params['bias'].copy()).float()
        with torch.no_grad():
            self.weight, self.bias = nn.Parameter(weight), nn.Parameter(bias)
        

class StdConvPt(nn.Conv2d, FromJaxModel):
    """Convolution with weight standardization."""
    
    def load_jax_weights(self, jax_params=None):
        with torch.no_grad():
            self.weight, self.bias = self._get_conv_params(jax_params)

    def forward(self, x):
        w = self.weight
        v, m = torch.var_mean(w, dim=[1, 2, 3], keepdim=True, unbiased=False)
        w = (w - m) / torch.sqrt(v + 1e-10)
        return F.conv2d(x, w, self.bias, self.stride, self.padding, self.dilation, self.groups)

class LayerNormPt(nn.LayerNorm, FromJaxModel):
    def load_jax_weights(self, jax_params=None):
        weight = torch.from_numpy(jax_params['scale'].copy()).float()
        bias = torch.from_numpy(jax_params['bias'].copy()).float()
        with torch.no_grad():
            self.weight, self.bias = nn.Parameter(weight), nn.Parameter(bias)
