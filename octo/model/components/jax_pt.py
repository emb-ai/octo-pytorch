from abc import ABC, abstractmethod
from dataclasses import dataclass, asdict
import logging 
from functools import partial
import re
from octo.utils.typing import Tuple, Any, Union
from typing import List, Callable, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from enum import Enum

import jax
import flax
import flax.linen as flax_nn

import numpy as np

from octo.utils.spec import ModuleSpec
from octo.utils.train_utils_pt import _flatten_dict

DEFAULT_PT_JAX_DICT =  {
    'weight': 'kernel',
    'bias': 'bias'
}


class WeightsCopyingRule(Enum):
    # Use shape of JAX parameter
    USE_SOURCE_SHAPE = 'jax_shape' # deprecated
    
    # Use shape of Pt parameter.
    USE_TARGET_SHAPE = 'pt_shape'
    
    # JAX and Pt parameter shapes must match.
    # Otherwise AssertionError is raised
    STRICT_MATCH = 'strict_shapes'
    
    # Do not copy weights
    SKIP = 'skip_param'

def _add_prefix_to_elems(prefix: str, l: list):
    return [prefix + elem for elem in l]

def _add_prefix_to_val(prefix: str, d: dict):
    ret_dict = {}
    for key, val in d.items():
        if isinstance(val, str):
            ret_dict[key] = prefix + val
        elif isinstance(val, list):
            ret_dict[key] = _add_prefix_to_elems(prefix, val)
        else:
            raise ValueError
    return ret_dict

def _add_prefix_to_key(prefix: str, d: dict):
    return {prefix + key: val for key, val in d.items()}
        

class FromJaxModel(ABC):
    def get_params_names_dict(self):
        subtree = {}
        for key, node in self.pt_to_jax_args_map().items():
            jax_name = asdict(node)['jax_param_names']
            if node.is_leaf:
                subtree[key] = jax_name
            else:
                assert isinstance(jax_name, str)
                subtree.update(_add_prefix_to_key(
                    f'{key}.',
                    _add_prefix_to_val(f'{jax_name}/', node.submodule.get_params_names_dict())
                ))
        return subtree
    
    def get_leaf_properties(self, prop_name):
        subtree = {}
        for key, node in self.pt_to_jax_args_map().items():
            if node.is_leaf:
                subtree[key] = asdict(node)[prop_name]
            else:
                subtree[key] = node.submodule.get_leaf_properties(prop_name)
        return subtree
    
    def get_copying_rule_dict(self):
        return _flatten_dict(self.get_leaf_properties('copying_rule'))
    
    def get_load_func_dict(self):
        return _flatten_dict(self.get_leaf_properties('load_func'))
    
    def load_jax_weights(
        self,
        jax_params_dict: dict,
        skip_keys: list = [],
        skip_keys_regex: str = None,
        non_strict_keys: list = [],
        non_strict_keys_regex: str = None
    ):
        if len(skip_keys) > 0 and skip_keys_regex is not None:
            logging.warning("skip_keys and skip_keys_regex are both provided. Next use only skip_keys")
        
        if len(non_strict_keys) > 0 and non_strict_keys_regex is not None:
            logging.warning("non_strict_keys and non_strict_keys_regex are both provided. Next use only non_strict_keys")
        
        state_dict = self.state_dict()
        
        if len(skip_keys) == 0 and skip_keys_regex is not None:
            r = re.compile(skip_keys_regex)
            skip_keys = list(filter(r.match, state_dict))
             
        if len(skip_keys) > 0:
            logging.info(f"Following keys will be SKIPPED during initialization: {skip_keys}")
            
        if len(non_strict_keys) == 0 and non_strict_keys_regex is not None:
            r = re.compile(non_strict_keys_regex)
            non_strict_keys = list(filter(r.match, state_dict))
             
        if len(non_strict_keys) > 0:
            logging.info(f"Following keys will be initialized, even if shapes are differ: {non_strict_keys}")
        
        
        pt_to_jax_dict = self.get_params_names_dict()
        jax_keys_to_skip = [pt_to_jax_dict[key] for key in pt_to_jax_dict if key in skip_keys]
        
        pt_to_jax_dict = dict(filter(lambda x: not x[0] in skip_keys, pt_to_jax_dict.items()))
        
        copying_rules = self.get_copying_rule_dict()
        copying_functions = self.get_load_func_dict()
        
        for key in non_strict_keys:
            copying_rules[key] = WeightsCopyingRule.USE_TARGET_SHAPE
        
        jax_params_dict_flat = _flatten_dict(jax_params_dict, sep='/')
        jax_params_dict_flat = dict(filter(lambda x: not x[0] in jax_keys_to_skip, jax_params_dict_flat.items()))
        
        new_state_dict = {}
        
        for pt_key, jax_keys in pt_to_jax_dict.items():
            if not self.check_keys(jax_keys, jax_params_dict_flat):
                continue
            new_state_dict.update(
                copying_functions[pt_key](
                    pt_key,
                    jax_keys,
                    jax_params_dict_flat,
                    state_dict,
                    copying_rules[pt_key]
                )
            )
            
            if isinstance(jax_keys, list):
                for jk in jax_keys:
                    del jax_params_dict_flat[jk]
            else:
                del jax_params_dict_flat[jax_keys]
            
            # del pt_to_jax_dict[pt_key]
        
        missing_keys, unexpected_keys = self.load_state_dict(new_state_dict, strict=False)
        missing_keys = list(set(missing_keys) - set(skip_keys))
        
    
        skipped_keys = jax_params_dict_flat.keys()
        
        if len(missing_keys) == 0:
            logging.info("All parameters were successfully initialized!")
        else:
            logging.warning(f"Following parameters were not initialized ({len(missing_keys)} total): {missing_keys}")
            
        if len(skipped_keys) > 0:
            logging.warning(f"Following JAX parameters were skipped during initialization ({len(skipped_keys)} total): {skipped_keys}")
        
        if len(unexpected_keys) > 0:
            logging.warning(f"Following Pt parameters were unexpected during loading ({len(unexpected_keys)} total): {unexpected_keys}")
        
        return missing_keys, skipped_keys, unexpected_keys

    @abstractmethod
    def pt_to_jax_args_map(self):
        return {
            # pt_module_name: ParamNode(jax_param_names, load_func, submodule)
            # ...
        } 
        
    def check_keys(self, keys: Union[str, List[str]], d: dict):
        if isinstance(keys, list):
            for k in keys:
                if not k in d:
                    return False
            return True
        elif isinstance(keys, str):
            return keys in d
        else:
            raise ValueError
    
    def _set_terminal_param(
        self, 
        pt_param_name: str, 
        jax_param_names: Union[str, List[str]], 
        jax_params: dict, 
        state_dict_pt: dict, 
        copying_rule: WeightsCopyingRule, 
        transform_function: Callable = None
    ):
        if isinstance(jax_param_names, list):
            cur_param = tuple(jax_params[k] for k in jax_param_names)
        elif isinstance(jax_param_names, str):
            cur_param = jax_params[jax_param_names]
        else:
            raise ValueError
        
        if transform_function is not None:
            cur_param = transform_function(cur_param)
            
        weight = torch.from_numpy(np.array(cur_param).copy()).float()
        
        # old_values = getattr(self, pt_param_name, None)
        old_values = state_dict_pt.get(pt_param_name, None)
        assert old_values is not None, f"No such parameter: {pt_param_name}"
        
        if copying_rule == WeightsCopyingRule.STRICT_MATCH:
            if old_values.shape != weight.shape:
                raise AssertionError(f"New value of '{pt_param_name}' has shape {weight.shape}, but {old_values.shape} expected")
        
        elif copying_rule == WeightsCopyingRule.USE_TARGET_SHAPE:
            if old_values.dim() != weight.dim():
                raise AssertionError(f"Shapes of '{pt_param_name}' and {weight.shape} are incompatible: {old_values.dim()} and {weight.dim()} has different number if dimensions")

            if old_values.shape != weight.shape:
                logging.info(f"Copying weights: {jax_param_names} (shape: {weight.shape}) -> {pt_param_name} (shape: {old_values.shape})")
                weight = self._copy_weights_with_diff_shapes(weight, old_values.clone())
            
        elif copying_rule == WeightsCopyingRule.SKIP:
            weight = old_values
            
        else:
            raise ValueError(f"Copying rule {copying_rule} is not supported")
        
        return {pt_param_name: weight}
    
    def _copy_weights_with_diff_shapes(self, src: torch.tensor, dst: torch.tensor):
        dst_slice = []
        src_slice = []
        for idx in np.array(dst.shape) - np.array(src.shape):
            if idx == 0:
                dst_slice.append(slice(None))
                src_slice.append(slice(None))
            if idx > 0:
                dst_slice.append(slice(0, -idx))
                src_slice.append(slice(None))
            if idx < 0:
                dst_slice.append(slice(None))
                src_slice.append(slice(0, idx))
        dst[dst_slice] = src[src_slice]
        return dst
    
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


@dataclass
class ParamNode:
    jax_param_names: Union[str, List[str]]
    load_func: Callable = None
    submodule: FromJaxModel = None
    copying_rule: WeightsCopyingRule = WeightsCopyingRule.STRICT_MATCH
    
    def __post_init__(self):
        if self.load_func is None and self.submodule is None:
            raise ValueError("ParamNode is either leaf node (load_func is not None and submodule is None) or non-leaf (load_func is None and submodule is not None)")
        if self.load_func is not None and self.submodule is not None:
            raise ValueError("ParamNode is either leaf node (load_func is not None and submodule is None) or non-leaf (load_func is None and submodule is not None)")
    
    @property
    def is_leaf(self):
        return self.submodule is None


def _transpose_matrix(jax_linear_weight):
    return jax_linear_weight.transpose((1, 0))

def _transpose_conv_kernel(jax_conv_weight):
        return jax_conv_weight.transpose((3, 2, 0, 1))

class LinearPt(nn.Linear, FromJaxModel):
    """Linear Layer"""
    
    def pt_to_jax_args_map(self):
        return {
            'weight': ParamNode(
                load_func=partial(
                    self._set_terminal_param, 
                    transform_function=_transpose_matrix
                ), 
                jax_param_names='kernel'
            ),
            'bias': ParamNode(load_func=self._set_terminal_param, jax_param_names='bias')
        }
    
class ConvPt(nn.Conv2d, FromJaxModel):
    """Ordinary convolution"""
    
    def pt_to_jax_args_map(self):
        return {
           'weight': ParamNode(
                load_func=partial(
                    self._set_terminal_param, 
                    transform_function=_transpose_conv_kernel 
                ), 
                jax_param_names='kernel'
            ),
            'bias': ParamNode(load_func=self._set_terminal_param, jax_param_names='bias')
        }


class GroupNormPt(nn.GroupNorm, FromJaxModel):
    
    def pt_to_jax_args_map(self):
        return {
            'weight': ParamNode(load_func=self._set_terminal_param, jax_param_names='scale'),
            'bias': ParamNode(load_func=self._set_terminal_param, jax_param_names='bias')
        }
        

class StdConvPt(nn.Conv2d, FromJaxModel):
    """Convolution with weight standardization."""
    
    def pt_to_jax_args_map(self):
        return {
            'weight': ParamNode(
                load_func=partial(
                    self._set_terminal_param, 
                    transform_function=_transpose_conv_kernel 
                ), 
                jax_param_names='kernel'
            ),
            'bias': ParamNode(load_func=self._set_terminal_param, jax_param_names='bias')
        }

    def forward(self, x):
        w = self.weight
        v, m = torch.var_mean(w, dim=[1, 2, 3], keepdim=True, unbiased=False)
        w = (w - m) / torch.sqrt(v + 1e-10)
        return F.conv2d(x, w, self.bias, self.stride, self.padding, self.dilation, self.groups)


class LayerNormPt(nn.LayerNorm, FromJaxModel):
    
    def pt_to_jax_args_map(self):
        return {
            'weight': ParamNode(load_func=self._set_terminal_param, jax_param_names='scale'),
            'bias': ParamNode(load_func=self._set_terminal_param, jax_param_names='bias')
        }
