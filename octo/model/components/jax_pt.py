from abc import ABC, abstractmethod
from dataclasses import dataclass
import logging 
from functools import partial

import torch
import torch.nn as nn

import jax
import flax
import flax.linen as flax_nn

import numpy as np

from octo.utils.spec import ModuleSpec

# @dataclass
class FromJaxModel(ABC):
    # jax_module: flax_nn.Module = None
    # jax_params: dict = None
    
    # @abstractmethod
    # def forward(self,):
    #     pass
    
    @abstractmethod
    def load_jax_weights(self, jax_params=None):
        pass
    
    def _get_conv_params(self, jax_conv_params):
        weight = torch.from_numpy(jax_conv_params['kernel'].transpose((3, 2, 0, 1)).copy()).float()
        bias = torch.from_numpy(jax_conv_params['bias'].copy()).float()
        return nn.Parameter(weight), nn.Parameter(bias)
        
    def _conv_load_handler(self, jax_conv_params):
        with torch.no_grad():
            self.weight, self.bias = self._get_conv_params(jax_conv_params)
    
    def _get_linear_params(self, jax_conv_params):
        weight = torch.from_numpy(jax_conv_params['kernel'].transpose((1, 0)).copy()).float()
        bias = torch.from_numpy(jax_conv_params['bias'].copy()).float()
        return nn.Parameter(weight), nn.Parameter(bias)
       
    def _linear_load_handler(self, jax_linear_params):
        with torch.no_grad():
            self.weight, self.bias = self._get_linear_params(jax_linear_params)
    
    def _get_groupnorm_params(self, jax_groupnorm_params):
        weight = torch.from_numpy(jax_groupnorm_params['scale'].copy()).float()
        bias = torch.from_numpy(jax_groupnorm_params['bias'].copy()).float()
        return nn.Parameter(weight), nn.Parameter(bias)
    
    def _groupnorm_load_handler(self, jax_groupnorm_params):
        with torch.no_grad():
            self.weight, self.bias = self._get_groupnorm_params(jax_groupnorm_params)
    
    
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
        
        

