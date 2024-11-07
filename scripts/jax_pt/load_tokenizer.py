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
import flax.linen as nn
import json

from octo.data.utils.text_processing import HFTokenizer
from octo.model.octo_model import OctoModel
from octo.model.octo_model_pt import OctoModelPt, load_np_example_batch, _np2pt
from octo.utils.spec import ModuleSpec
from octo.model.octo_module import OctoModule
from octo.model.components.vit_encoders import StdConv
from octo.model.components.vit_encoders_pt import StdConvPt
#==============================
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

obs_jax = {
    "image_primary": np.random.randint(0, 255, (10, 2, 256, 256, 3)), 
    "timestep_pad_mask": np.array([[True, True]])
}
task_jax = model.create_tasks(texts=["pick up the fork"] * 10)
img_tokenizer = ModuleSpec.instantiate(config['model']['observation_tokenizers']['primary'])()
jax_out = img_tokenizer.apply(
            {"params": params['octo_transformer']['observation_tokenizers_primary']},
            obs_jax,
            task_jax,
            False
        )
print(jax_out)
# exit()

#==============================
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
model_pt = OctoModelPt(None, text_processor, None, None, None)
model_pt.example_batch = _np2pt(load_np_example_batch("hf://rail-berkeley/octo-small-1.5"))


new_config = {'args': [],
 'kwargs': {'encoder': {'args': [],
   'kwargs': {},
   'module': 'octo.model.components.vit_encoders_pt',
   'name': 'SmallStem16Pt'},
  'obs_stack_keys': ['image_primary'],
  'task_stack_keys': ['image_primary']},
 'module': 'octo.model.components.tokenizers_pt',
 'name': 'ImageTokenizerPt'}
new_config

tokenizer_pt = ModuleSpec.instantiate(new_config)()
print(tokenizer_pt.state_dict().keys())

uninitialized_params, unused_jax_params = tokenizer_pt.load_jax_weights(params['octo_transformer'], 'observation_tokenizers_primary', 'img_tokenizer')
print(uninitialized_params, unused_jax_params)
tokenizer_pt.eval()

obs_pt = {
    "image_primary": torch.tensor(obs_jax["image_primary"]).permute((0, 1, 4, 2, 3)), 
    "timestep_pad_mask": torch.tensor([[True, True]])
}
task_pt = model_pt.create_tasks(texts=["pick up the fork"] * 10)

with torch.no_grad():
    output_pt = tokenizer_pt(
        obs_pt,
        task_pt,
        False
    )

# print(output_pt.shape)
# print(output_pt)


print(np.all(np.isclose(output_pt.tokens.cpu().numpy(), jax_out.tokens, 0.001, 0.001)))
