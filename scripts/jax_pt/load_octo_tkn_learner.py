import torch
from octo.utils.spec import ModuleSpec

import os
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
import orbax.checkpoint
# from flax.training import orbax_utilss
from functools import partial
from pathlib import Path
from PIL import Image
import requests
import matplotlib.pyplot as plt
import numpy as np
import jax

import json

from octo.model.octo_model import OctoModel
from octo.model.octo_model_pt import OctoModelPt, _np2pt, _jax_config_to_pt_config
from octo.utils.spec import ModuleSpec
from octo.model.octo_module import OctoModule
from octo.model.octo_module_pt import OctoModulePt
from octo.model.components.vit_encoders import StdConv
from octo.model.components.transformer import Encoder1DBlock, Transformer
from octo.model.components.transformer_pt import Encoder1DBlockPt
from octo.data.utils.text_processing import HFTokenizer

bs = 4
np.random.seed(42)
obs_jax = {
    "image_primary": np.random.randint(0, 255, (bs, 2, 256, 256, 3)), 
    # "image_wrist": np.random.randint(0, 255, (bs, 2, 128, 128, 3)),
    "timestep_pad_mask": np.array([[True, True]] * bs),
    "proprio": np.random.randn(bs, 2, 14),
}

obs_pt = {
    "image_primary": torch.tensor(obs_jax['image_primary'].transpose((0, 1, 4, 2, 3)), device='cuda:0'), 
    # "image_wrist": torch.tensor(obs_jax['image_wrist'].transpose((0, 1, 4, 2, 3)), device=device),
    "timestep_pad_mask": torch.tensor(obs_jax['timestep_pad_mask'], device='cuda:0'),
    "proprio": torch.tensor(obs_jax['proprio'], device='cuda:0').float(),
}


model_jax = OctoModel.load_pretrained("/home/jovyan/shares/SR006.nfs2/soshin/repo/octo/02_JAX_BIG_MODEL_1EP_DEBUG")
task_jax = model_jax.create_tasks(texts=["please work aboba"] * bs)

acions_jax = model_jax.sample_actions(
        observations=obs_jax,
        tasks=task_jax,
        timestep_pad_mask = np.array([[True, True]] * bs),
        train = False,
        unnormalization_statistics=None,
        argmax=True,
    )

print(acions_jax)
# exit()

# model_pt = OctoModelPt.load_pretrained_from_jax('/home/jovyan/shares/SR006.nfs2/soshin/repo/octo/02_JAX_BIG_MODEL_1EP_DEBUG')['octo_model']
model_meta = OctoModelPt.load_config_and_meta_from_jax(
    '/home/jovyan/shares/SR006.nfs2/soshin/repo/octo/02_JAX_BIG_MODEL_1EP_DEBUG',
    
)
model_meta['config']['model']['num_tokens_dict'] = {
    'primary': 8,
    # 'wrist': 64,
    'language': 16,
    'proprio': 14,
    'action': 1
}
model_pt = OctoModelPt.from_config(**model_meta, verbose=True)
model_pt.load_weights_from_jax('/home/jovyan/shares/SR006.nfs2/soshin/repo/octo/02_JAX_BIG_MODEL_1EP_DEBUG')
model_pt.to('cuda:0')

task_pt = model_pt.create_tasks(texts=["please work aboba"] * bs, device='cuda:0')


actions_pt = model_pt.sample_actions(
        observations=obs_pt,
        tasks=task_pt,
        timestep_pad_mask = torch.tensor([[True, True]] * bs, device='cuda:0'),
        train = False,
        unnormalization_statistics=None,
        argmax=True,
    )

print(np.allclose(acions_jax, actions_pt.detach().cpu().numpy(), 0.0, 0.001))
# print(np.allclose(transformer_outputs_jax['readout_action'].tokens, transformer_outputs_pt['readout_action'].tokens.detach().cpu().numpy(), 0.0, 0.013))

model_pt.save_pretrained(0, '02_PT_DEBUG')

model_pt_loaded = OctoModelPt.load_pretrained('02_PT_DEBUG', 0)['octo_model']
model_pt_loaded.to('cuda:0')



task_pt = model_pt_loaded.create_tasks(texts=["please work aboba"] * bs, device='cuda:0')


actions_pt_loaded = model_pt_loaded.sample_actions(
        observations=obs_pt,
        tasks=task_pt,
        timestep_pad_mask = torch.tensor([[True, True]] * bs, device='cuda:0'),
        train = False,
        unnormalization_statistics=None,
        argmax=True,
    )
print(np.allclose(actions_pt_loaded.detach().cpu().numpy(), actions_pt.detach().cpu().numpy(), 0.0, 0.001))