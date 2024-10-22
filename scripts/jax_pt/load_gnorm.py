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


gnorm = flax_nn.GroupNorm()
x = np.random.randn(1, 128, 128, 32)
jax_out = gnorm.apply(
            {"params": params['octo_transformer']['observation_tokenizers_primary']['SmallStem16_0']['GroupNorm_0']},
            x
        )
print(jax_out.shape)
print(jax_out.transpose((0, 3, 1, 2)))
# exit()

#==============================







new_config = {
    'args': [32, 32],
   'kwargs': {},
   'module': 'octo.model.components.vit_encoders_pt',
   'name': 'GroupNormPt'}
new_config

gnorm = ModuleSpec.instantiate(new_config)()
print(gnorm.state_dict().keys())

gnorm.load_jax_weights(params['octo_transformer']['observation_tokenizers_primary']['SmallStem16_0']['GroupNorm_0'])
gnorm.eval()

x_pt = torch.from_numpy(x).permute((0, 3, 1, 2)).float()

with torch.no_grad():
    output_pt = gnorm(
        x_pt
    )

print(output_pt.shape)
print(output_pt)


print(np.all(np.isclose(output_pt.cpu().numpy(), jax_out.transpose((0, 3, 1, 2)), 0.001, 0.001)))
