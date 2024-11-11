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
from octo.model.octo_model_pt import OctoModelPt, load_np_example_batch, _np2pt
from octo.utils.spec import ModuleSpec
from octo.model.octo_module import OctoModule
from octo.model.octo_module_pt import OctoModulePt
from octo.model.components.vit_encoders import StdConv
from octo.model.components.transformer import Encoder1DBlock, Transformer
from octo.model.components.transformer_pt import Encoder1DBlockPt
from octo.data.utils.text_processing import HFTokenizer

def load_params():
    model = OctoModel.load_pretrained("hf://rail-berkeley/octo-small-1.5")

    IMAGE_URL = "https://rail.eecs.berkeley.edu/datasets/bridge_release/raw/bridge_data_v2/datacol2_toykitchen7/drawer_pnp/01/2023-04-19_09-18-15/raw/traj_group0/traj0/images0/im_12.jpg"
    img = np.array(Image.open(requests.get(IMAGE_URL, stream=True).raw).resize((256, 256)))

    img = img[np.newaxis,np.newaxis,...]
    observation = {"image_primary": img, "timestep_pad_mask": np.array([[True]])}
    task = model.create_tasks(texts=["pick up the fork"])
    init_args = (
        observation,
        task,
        np.array([[True,]])
    )


    checkpoint_path = Path('/home/jovyan/shares/SR006.nfs2/soshin/repo/octo/try_save')
    config_path = checkpoint_path / 'config.json'
    with open(config_path, 'r') as f:
        config = json.load(f)
        
    module = OctoModule.create(**config["model"])
    params_shape = jax.eval_shape(
                partial(module.init, train=False), jax.random.PRNGKey(0), *init_args
            )["params"]
    checkpointer = orbax.checkpoint.CheckpointManager(
                checkpoint_path, orbax.checkpoint.PyTreeCheckpointer()
            )
    step = checkpointer.latest_step()
    params = checkpointer.restore(step, params_shape)
    
    return params, config, model



jax_params, config, octo_model = load_params()

device='cuda:0'

text_processor = HFTokenizer(
    tokenizer_name = 't5-base',
    tokenizer_kwargs = {
        "max_length": 16,
        "padding": "max_length",
        "truncation": True,
        "return_tensors": "np",
    },
    encode_with_model = False,
)

new_config = {
    'heads': None,
    'observation_tokenizers': {
        'primary': {
            'args': [],
            'kwargs': {
                'encoder': {
                    'args': [],
                    'kwargs': {},
                    'module': 'octo.model.components.vit_encoders_pt',
                    'name': 'SmallStem16Pt'
                },
                'obs_stack_keys': ['image_primary'],
                'task_stack_keys': ['image_primary']
            },
            'module': 'octo.model.components.tokenizers_pt',
            'name': 'ImageTokenizerPt'
        },
        # 'wrist': {
        #     'args': [],
        #     'kwargs': {
        #         'encoder': {
        #             'args': [],
        #             'kwargs': {},
        #             'module': 'octo.model.components.vit_encoders_pt',
        #             'name': 'SmallStem16Pt'
        #             },
        #         'obs_stack_keys': ['image_wrist'],
        #         'task_stack_keys': ['image_wrist']
        #     },
        #     'module': 'octo.model.components.tokenizers_pt',
        #     'name': 'ImageTokenizerPt'
        # }
    },
    'task_tokenizers': {
        'language': {
            'args': [],
            'kwargs': {
                'encoder': 't5-base', 
                'finetune_encoder': False
            },
        'module': 'octo.model.components.tokenizers_pt',
        'name': 'LanguageTokenizerPt'
        }
    },
    'readouts': {'action': 1},
    'transformer_kwargs': {
        'add_position_embedding': False,
        'attention_dropout_rate': 0.0,
        'dropout_rate': 0.0,
        'mlp_dim': 1536,
        'num_attention_heads': 6,
        'num_layers': 12
    },
    'token_embedding_size': 384,
    'max_horizon': 10,
    'repeat_task_tokens': True,
    'use_correct_attention': True
}

module = OctoModulePt.create(**new_config)

model = OctoModelPt(module, text_processor, None, None, None)
example_batch = load_np_example_batch('/home/jovyan/shares/SR006.nfs2/soshin/repo/octo/try_save')
model.example_batch = _np2pt(example_batch)

language_instruction = "pick up the fork"
tasks_pt = model.create_tasks(texts=[language_instruction] * 10, device=device)

uninitialized_params, unused_jax_params = model.module.load_jax_weights(jax_params)
print(uninitialized_params, unused_jax_params)
model.module.eval()
model.module.to(device)

obs = {
    "image_primary": torch.randint(0, 256, (10, 2, 3, 256, 256)).to(device),
    "timestep_pad_mask" :torch.tensor([[True, True]]).to(device),
    # "image_wrist": torch.randint(0, 256, (10, 2, 3, 128, 128)), 
}

res = model.module(
    obs,
    tasks_pt,
    timestep_pad_mask=torch.tensor([[True, True]]).to(device),
    train=False,
    verbose=True
)

print(res[0]['readout_action'].tokens)
# print(res[0]['readout_action'].tokens[0, 0])

#==========================

task_jax = octo_model.create_tasks(texts=[language_instruction] * 10)

module = OctoModule.create(**config["model"])

obs_jax = {
    "image_primary": obs['image_primary'].detach().cpu().numpy().transpose((0, 1, 3, 4, 2)),
    # "image_wrist": obs['image_wrist'].detach().cpu().numpy().transpose((0, 1, 3, 4, 2)), 
}

transformer_outputs = module.octo_transformer.apply(
            {"params": jax_params['octo_transformer']},
            obs_jax,
            task_jax,
            np.array([[True, True]]),
            train=False,
            verbose=True
        )

print(transformer_outputs['readout_action'].tokens)
# print(transformer_outputs['readout_action'].tokens[0, 0])

print(np.all(np.isclose(res[0]['readout_action'].tokens.detach().cpu().numpy(), transformer_outputs['readout_action'].tokens, 0.01, 0.01))) 
