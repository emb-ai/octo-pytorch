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
from octo.utils.spec import ModuleSpec
from octo.model.octo_module import OctoModule
from octo.model.components.vit_encoders import StdConv
from octo.model.components.vit_encoders_pt import StdConvPt
import flax.linen as flax_nn
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

lang_tokenizer = ModuleSpec.instantiate(config['model']['task_tokenizers']['language'])()
jax_out = lang_tokenizer.apply(
            {"params": params['octo_transformer']['task_tokenizers_language']},
            None,
            task,
            False
        )



print(jax_out.tokens.shape)
print(jax_out.tokens)
# exit()

#==============================



new_config = {
    'args': [],
   'kwargs': {'encoder': 't5-base', 'finetune_encoder': False},
   'module': 'octo.model.components.tokenizers_pt',
   'name': 'LanguageTokenizerPt'}
new_config

lang_tokenizer = ModuleSpec.instantiate(new_config)()
# print(lang_tokenizer.state_dict().keys())

lang_tokenizer.load_jax_weights(params['octo_transformer']['task_tokenizers_language'])
lang_tokenizer.eval()

# task_pt = {key: torch.from_numpy(val.copy()) for key, val in task.items() if type(val) == np.ndarray}
task_pt = {}
task_pt['pad_mask_dict'] = {key: torch.from_numpy(val.copy()) for key, val in task['pad_mask_dict'].items()}
task_pt['language_instruction'] = {key: torch.from_numpy(val.copy()) for key, val in task['language_instruction'].items()}
with torch.no_grad():
    output_pt = lang_tokenizer(
        None,
        task_pt
    )

print(output_pt.tokens.shape)
print(output_pt.tokens)


print(np.all(np.isclose(output_pt.tokens.cpu().numpy(), jax_out.tokens, 0.001, 0.001)))
print(np.all(np.isclose(output_pt.mask.cpu().numpy(), jax_out.mask, 0.001, 0.001)))