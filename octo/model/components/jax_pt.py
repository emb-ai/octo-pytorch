from abc import ABC, abstractmethod
from dataclasses import dataclass
import logging 
from functools import partial
from octo.utils.typing import Tuple, Any, Union
from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F

import jax
import flax
import flax.linen as flax_nn

import numpy as np

from octo.utils.spec import ModuleSpec

DEFAULT_PT_JAX_DICT =  {
    'weight': 'kernel',
    'bias': 'bias'
}

class FromJaxModel(ABC):

    def check_is_main_key_in_params(self, jax_params, key_jax, key_pt=None):
        is_ok = True
        if not self._check_key(jax_params, key_jax):
            is_ok = False
            uninitialized_params = list(self.pt_to_jax_args_map.keys())
            if key_pt is not None:
                uninitialized_params = self._add_key(uninitialized_params, key_pt, separator='.')
            unused_jax_params = []
            return is_ok, uninitialized_params, unused_jax_params
        return is_ok, [], []
    
    def load_jax_weights(self, jax_params, key_jax=None, key_pt=None) -> Tuple[List, List]:
        """
        Load JAX weights from jax_params[key] dict. 

        Args:
            jax_params (dict): dictionary with JAX weights 
            key (str, optional): key for dictiona. Defaults to None.

        Returns:
            Tuple[List, List]: list of uninitialized parameters from `self.pt_to_jax_map_dict`,
                and list of unused parameters from `jax_params`
        """
        jax_submodule_params = jax_params
        if key_jax is not None:
            is_ok, uninitialized_params, unused_jax_params = self.check_is_main_key_in_params(jax_params, key_jax, key_pt)
            if not is_ok:
                return uninitialized_params, unused_jax_params
        
            jax_submodule_params = jax_params[key_jax]
        
        pt_submodule_params = list(self._pt_to_jax_args_map().keys())
        
        uninitialized_params = []
        unused_jax_params = list(jax_submodule_params.keys())
        
        submodules_uninitialized_params = []
        submodules_unused_jax_params = []
        
        for pt_submodule_param_name in pt_submodule_params:
            submodule_load_func, jax_submodule_param_key \
                = self._pt_to_jax_args_map()[pt_submodule_param_name]
            
            if not self._check_key(jax_submodule_params, jax_submodule_param_key):
                uninitialized_params.append(pt_submodule_param_name)
                continue
            unused_jax_params.remove(jax_submodule_param_key)
                
            cur_subm_uninit_params, cur_subm_unused = submodule_load_func(
                jax_submodule_params,
                jax_submodule_param_key,
                key_pt = pt_submodule_param_name
            )
            submodules_uninitialized_params += cur_subm_uninit_params
            submodules_unused_jax_params += cur_subm_unused
        
        uninitialized_params += submodules_uninitialized_params
        unused_jax_params += submodules_unused_jax_params
        
        if key_pt is not None:
            uninitialized_params = self._add_key(uninitialized_params, key_pt, separator='.')
        if key_jax is not None:
            unused_jax_params = self._add_key(unused_jax_params, key_jax, separator='/')
        return uninitialized_params, unused_jax_params
        
    
    def _check_key(self, param_dict, key):
        return key in param_dict
    
    
    
    def _add_key(self, list, key, separator='.'):
        return [f'{key}{separator}{l}' for l in list]
    
    def _pt_to_jax_args_map(self):
        __pt_to_jax_dict = getattr(self, '__pt_to_jax_dict', None)
        if not __pt_to_jax_dict is None:
            return __pt_to_jax_dict
        
        pt_to_jax_args = self.pt_to_jax_args_map
        
        # check correctness
        jax_keys = [val[1] for _, val in pt_to_jax_args.items()]
        if len(jax_keys) > len(set(jax_keys)):
            raise ValueError("JAX keys in pt_to_jax_args_map are not unique. \
That means you are trying to load weights from the same JAX parameter to many PyTorch parameters.")
        
        setattr(self, '__pt_to_jax_dict', pt_to_jax_args)
        
        return pt_to_jax_args
    
    @property
    @abstractmethod
    def pt_to_jax_args_map(self):
        return {
            # pt_module_name: (load_func, jax_param_key),
            # ...
        }
        
    @property
    def num_of_params_to_init(self):
        return len(self._pt_to_jax_args_map().keys())
    
    def assign_new_value(self, name: str, parameter: nn.Parameter, module: nn.Module = None, strict_shapes=True):
        old_values = getattr(self, name, None) if module is None else getattr(module, name, None)
        assert old_values is not None, f"No such parameter: {name}"
        
        
        if old_values.shape != parameter.shape:
            if strict_shapes:
                raise AssertionError(f"New value of '{name}' in {self.__class__.__name__} has shape {parameter.shape}, but {old_values.shape} expected.\n \
Set strict_shapes=False to enable automatic shape change.")
            else:
                logging.warning(f'Shape of parameter {name} in {self.__class__.__name__} changed its size: {old_values.shape} -> {parameter.shape}.\n \
Specify strict_shapes=True to disable automatic shape change.')
        
        if module:
            module.register_parameter(name, nn.Parameter(parameter))
        else:
            self.register_parameter(name, nn.Parameter(parameter))        
        
    def _get_param(self, jax_params, keys: Union[str, Tuple[str]]):
        jax_params = jax_params.copy()
        
        if isinstance(keys, tuple) and len(keys) == 1:
            keys = keys[0]
            
        if isinstance(keys, str):
            if self._check_key(jax_params, keys):
                return nn.Parameter(torch.tensor(np.array(jax_params[keys]).copy())), keys
            return None, keys
        else:
            key = keys[0]
            keys = keys[1:]
            if self._check_key(jax_params, key):
                if len(keys) == 1:
                    keys = keys[0]
                value, ret_key = self._get_param(jax_params[key], keys)
                ret_key = f'{key}/{ret_key}'
                return value, ret_key
            return None, key
    
    def _set_terminal_param(self, jax_params, key_jax, key_pt, transform_function=None, **kwargs):
        if key_jax not in jax_params:
            return [f'{key_pt}'], []
        jax_param = jax_params[key_jax]
        
        if transform_function is not None:
            jax_param = transform_function(jax_param)
            
        weight = torch.from_numpy(np.array(jax_param).copy()).float()
        
        self.assign_new_value(key_pt, nn.Parameter(weight), **kwargs)
        
        return [], []
    
    
    def _set_param(self, jax_params, key_jax, key_pt):
        return self._set_terminal_param(jax_params, key_jax, key_pt, None)
    
    def _set_conv_weight(self, jax_params, key_jax, key_pt):
        def t(jax_conv_weight):
            return jax_conv_weight.transpose((3, 2, 0, 1))
        return self._set_terminal_param(jax_params, key_jax, key_pt, t)
    
    def _set_linear_weight(self, jax_params, key_jax, key_pt):
        def t(jax_conv_weight):
            return jax_conv_weight.transpose((1, 0))
        return self._set_terminal_param(jax_params, key_jax, key_pt, t)
    
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
    @property
    def pt_to_jax_args_map(self):
        return {
            'weight': (self._set_linear_weight, 'kernel'),
            'bias': (self._set_param, 'bias')
        }
    
class ConvPt(nn.Conv2d, FromJaxModel):
    """Ordinary convolution"""
    
    @property
    def pt_to_jax_args_map(self):
        return {
            'weight': (self._set_conv_weight, 'kernel'),
            'bias': (self._set_param, 'bias')
        }


class GroupNormPt(nn.GroupNorm, FromJaxModel):
    # def load_jax_weights(self, jax_params, key_jax, key_pt=None) -> Tuple[List, List]:
    #     return super().load_jax_weights(jax_params, key_jax, key_pt)
    
    @property
    def pt_to_jax_args_map(self):
        return {
            'weight': (self._set_param, 'scale'),
            'bias': (self._set_param, 'bias')
        }
        

class StdConvPt(nn.Conv2d, FromJaxModel):
    """Convolution with weight standardization."""
    
    @property
    def pt_to_jax_args_map(self):
        return {
            'weight': (self._set_conv_weight, 'kernel'),
            'bias': (self._set_param, 'bias')
        }

    def forward(self, x):
        w = self.weight
        v, m = torch.var_mean(w, dim=[1, 2, 3], keepdim=True, unbiased=False)
        w = (w - m) / torch.sqrt(v + 1e-10)
        return F.conv2d(x, w, self.bias, self.stride, self.padding, self.dilation, self.groups)


class LayerNormPt(nn.LayerNorm, FromJaxModel):
    
    @property
    def pt_to_jax_args_map(self):
        return {
            'weight': (self._set_param, 'scale'),
            'bias': (self._set_param, 'bias')
        }
