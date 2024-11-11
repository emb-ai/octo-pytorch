import re
import torch
import torch.nn as nn
from typing import List
import logging
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
        # new_key = parent_key + sep + k if parent_key else k
        new_key = (*parent_key, k) if parent_key else (k,)
        if isinstance(v, dict):
            items.update(_flatten_dict(v, new_key, sep=sep).items())
        else:
            items[new_key] = v
    return items

def _jax_config_to_pt_config(config):
    if isinstance(config, dict):
        config_pt = {}
        if 'module' in config:
            config_pt['args'] = config['args']
            config_pt['kwargs'] = {key: _jax_config_to_pt_config(val) for key, val in config['kwargs'].items()}
            config_pt['module'] = config['module'] + '_pt'
            config_pt['name'] = config['name'] + 'Pt'
        else:
            config_pt = {key: _jax_config_to_pt_config(val) for key, val in config.items()}
        return config_pt
    else:
        return config


def _np2pt(data, device=None):
    if isinstance(data, dict):
        return {key: _np2pt(val, device) for key, val in data.items()}
    elif isinstance(data, np.ndarray):
        if len(data.shape) == 4 and data.dtype == np.uint8:
            data = data.transpose((0, 3, 1, 2)) #NHWC -> NCHW
        elif len(data.shape) == 5 and data.dtype == np.uint8:
            data = data.transpose((0, 1, 4, 2, 3)) #NTHWC -> NTCHW
        t = torch.tensor(data, device=device)
        return t
    
def regex_match(regex_keys, x):
    return any([re.match(r_key, x) for r_key in regex_keys])


def regex_filter(regex_keys, xs):
    return list(filter(lambda x: regex_match(regex_keys, x), xs))
