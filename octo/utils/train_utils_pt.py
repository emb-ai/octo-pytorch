import math
import re
from typing import List, Any, Callable
import logging
from functools import partial

import torch
import torch.nn as nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR

import numpy as np

def regex_match(regex_keys, x):
    return any([re.match('.' + r_key, x) for r_key in regex_keys])


def regex_filter(regex_keys, xs):
    return list(filter(lambda x: regex_match(regex_keys, x), xs))

def freeze_weights_pt(
    module: nn.Module,
    frozen_keys: List[str]
):
    """
    Freezes all weights in params_or_params_shape whose keys fnmatch the ones in frozen_keys.
    Example usage:
        tx = freeze_weights(tx, model.params, ["octo_transformer.*"])
    """
    logging.info(f"Freezing parameters that include the following keys: {frozen_keys}.")
    
    param_names = [name for name, _ in module.named_parameters()]
    frozen_keys = ['.' + key for key in frozen_keys] # TODO
    selected_params = regex_filter(frozen_keys, param_names)
    
    for p_name, p in module.named_parameters():
        if p_name in selected_params:
            p.requires_grad = False
        
    logging.debug("Frozen params:", selected_params)
    total_params = sum([p.numel() for p in module.parameters()])
    trainable_params = sum([p.numel() for p in module.parameters() if p.requires_grad==True])
    frozen_params = sum([p.numel() for p in module.parameters() if p.requires_grad==False])
    
    logging.info(f"Num trainable params: {trainable_params:,}.")
    logging.info(f"Num frozen params: {total_params - trainable_params:,}.")
    logging.info("To see a detailed list of frozen params, set logging level to DEBUG.")

def _flatten_dict(d, parent_key: str = '', sep: str ='.'):
    items = {}
    for k, v in d.items():
        if sep is None:
            new_key = (*parent_key, k) if parent_key else (k,)
        else:
            new_key = parent_key + sep + k if parent_key else k
        if isinstance(v, dict):
            items.update(_flatten_dict(v, new_key, sep=sep).items())
        else:
            items[new_key] = v
    return items

def tree_leaves(d):
    return list(_flatten_dict(d).values())

def _add_to_dict(d, keys, val):
    if len(keys) == 1:
        key = keys[0]
        d[key] = val
        return d
    key = keys[0]
    if key not in d:
        d[key] = _add_to_dict({}, keys[1:], val)
    else:
        d[key] = _add_to_dict(d[key], keys[1:], val)
    return d

def _unflatten_dict(d):
    new_dict = {}
    for key, val in d.items():
        if len(new_dict) == 0:
            new_dict = _add_to_dict({}, key, val)
        else:
            new_dict = _add_to_dict(new_dict, key, val)
    return new_dict

def _jax_config_to_pt_config(config):
    if isinstance(config, dict):
        config_pt = {}
        if 'module' in config:
            config_pt['args'] = config['args']
            config_pt['kwargs'] = {key: _jax_config_to_pt_config(val) for key, val in config['kwargs'].items()}
            config_pt['module'] = config['module']
            config_pt['name'] = config['name']
            if not config['module'].endswith('_pt'):
                config_pt['module'] = config_pt['module'] + '_pt'
            if not config['name'].endswith('Pt'):
                config_pt['name'] = config_pt['name'] + 'Pt'
        else:
            config_pt = {key: _jax_config_to_pt_config(val) for key, val in config.items()}
        return config_pt
    else:
        return config


def _np2pt(data, device=None, dtype=None):
    if isinstance(data, dict):
        return {key: _np2pt(val, device) for key, val in data.items()}
    elif isinstance(data, np.ndarray):
        if len(data.shape) == 4 and data.dtype == np.uint8:
            data = data.transpose((0, 3, 1, 2)) #NHWC -> NCHW
        elif len(data.shape) == 5 and data.dtype == np.uint8:
            data = data.transpose((0, 1, 4, 2, 3)) #NTHWC -> NTCHW
    t = torch.tensor(data, device=device, dtype=dtype)
    return t

def _to_device(data, device):
    if isinstance(data, dict):
        return {key: _to_device(val, device) for key, val in data.items()}
    elif isinstance(data, torch.Tensor):
        return data.to(device)
    else:
        raise ValueError(f"Unsupported data type: {type(data)}")

def regex_match(regex_keys, x):
    return any([re.match(r_key, x) for r_key in regex_keys])


def regex_filter(regex_keys, xs):
    return list(filter(lambda x: regex_match(regex_keys, x), xs))


def tree_map(fn: Callable, tree: dict, is_leaf: Callable[[Any], bool] | None = None, **kwargs) -> dict:
    tree_flat = _flatten_dict(tree)
    if is_leaf is None:
        is_leaf = lambda x: True
    mapped_tree = {}
    for key, val in tree_flat.items():
        mapped_tree[key] =  fn(val, **kwargs) if is_leaf(val) else val
    return _unflatten_dict(mapped_tree)



## Copy from transformers

def _get_cosine_schedule_with_warmup_lr_lambda(
    current_step: int, *, num_warmup_steps: int, num_training_steps: int, num_cycles: float
):
    if current_step < num_warmup_steps:
        return float(current_step) / float(max(1, num_warmup_steps))
    progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
    return max(0.0, 0.5 * (1.0 + math.cos(math.pi * float(num_cycles) * 2.0 * progress)))

def get_cosine_schedule_with_warmup(
    optimizer: Optimizer, num_warmup_steps: int, num_training_steps: int, num_cycles: float = 0.5, last_epoch: int = -1
):
    """
    Create a schedule with a learning rate that decreases following the values of the cosine function between the
    initial lr set in the optimizer to 0, after a warmup period during which it increases linearly between 0 and the
    initial lr set in the optimizer.

    Args:
        optimizer ([`~torch.optim.Optimizer`]):
            The optimizer for which to schedule the learning rate.
        num_warmup_steps (`int`):
            The number of steps for the warmup phase.
        num_training_steps (`int`):
            The total number of training steps.
        num_cycles (`float`, *optional*, defaults to 0.5):
            The number of waves in the cosine schedule (the defaults is to just decrease from the max value to 0
            following a half-cosine).
        last_epoch (`int`, *optional*, defaults to -1):
            The index of the last epoch when resuming training.

    Return:
        `torch.optim.lr_scheduler.LambdaLR` with the appropriate schedule.
    """

    lr_lambda = partial(
        _get_cosine_schedule_with_warmup_lr_lambda,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps,
        num_cycles=num_cycles,
    )
    return LambdaLR(optimizer, lr_lambda, last_epoch)