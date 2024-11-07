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
from octo.model.octo_model_pt import OctoModelPt
from octo.utils.spec import ModuleSpec
from octo.model.octo_module import OctoModule
from octo.model.octo_module_pt import OctoModulePt
from octo.model.components.vit_encoders import StdConv
from octo.model.components.transformer import Encoder1DBlock, Transformer
from octo.model.components.transformer_pt import Encoder1DBlockPt
from octo.data.utils.text_processing import HFTokenizer


device='cuda:0'
model_pt, _, uninitialized_params, unused_jax_params = OctoModelPt.load_pretrained_from_jax("hf://rail-berkeley/octo-small-1.5")
print(uninitialized_params, unused_jax_params)
model_pt.module.heads = None
model_pt.to(device)
model_pt.module.eval()

language_instruction = "pick up the fork"
tasks_pt = model_pt.create_tasks(texts=[language_instruction] * 10)

obs = {
    "image_primary": torch.randint(0, 256, (10, 2, 3, 256, 256)).to(device),
    "timestep_pad_mask": torch.tensor([[True, True]]).to(device),
    # "image_wrist": torch.randint(0, 256, (10, 2, 3, 128, 128)), 
}


res = model_pt.run_transformer(obs, tasks_pt, torch.tensor([[True, True]]).to(device), False)
print(res)

#=========

model_jax = OctoModel.load_pretrained("hf://rail-berkeley/octo-small-1.5")

IMAGE_URL = "https://rail.eecs.berkeley.edu/datasets/bridge_release/raw/bridge_data_v2/datacol2_toykitchen7/drawer_pnp/01/2023-04-19_09-18-15/raw/traj_group0/traj0/images0/im_12.jpg"
img = np.array(Image.open(requests.get(IMAGE_URL, stream=True).raw).resize((256, 256)))

img = img[np.newaxis,np.newaxis,...]
observation = {"image_primary": img, "timestep_pad_mask": np.array([[True]])}
task = model_jax.create_tasks(texts=["pick up the fork"])
init_args = (
    observation,
    task,
    np.array([[True,]])
)


checkpoint_path = Path('/home/jovyan/.cache/huggingface/hub/models--rail-berkeley--octo-small-1.5/snapshots/dc9aa3019f764726c770814b27e4ab0fc6e32a58')
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


task_jax = model_jax.create_tasks(texts=[language_instruction] * 10)

module = OctoModule.create(**config["model"])

obs_jax = {
    "image_primary": obs['image_primary'].detach().cpu().numpy().transpose((0, 1, 3, 4, 2)),
    # "image_wrist": obs['image_wrist'].detach().cpu().numpy().transpose((0, 1, 3, 4, 2)), 
}

transformer_outputs = module.octo_transformer.apply(
            {"params": params['octo_transformer']},
            obs_jax,
            task_jax,
            np.array([[True, True]]),
            train=False,
            verbose=True
        )

print(transformer_outputs['readout_action'].tokens)
# print(transformer_outputs['readout_action'].tokens[0, 0])

print(np.all(np.isclose(res[0]['readout_action'].tokens.detach().cpu().numpy(), transformer_outputs['readout_action'].tokens, 0.01, 0.01))) 


