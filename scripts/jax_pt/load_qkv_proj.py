import torch
import torch.nn.functional as F
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
import flax.linen as nn_flax
import json
import jax.numpy as jnp

from octo.model.octo_model import OctoModel
from octo.utils.spec import ModuleSpec
from octo.model.octo_module import OctoModule
from octo.model.components.vit_encoders import StdConv
from octo.model.components.transformer import Encoder1DBlock, MlpBlock
from octo.model.components.transformer_pt import Encoder1DBlockPt
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


dense = nn_flax.linear.DenseGeneral(
        axis=-1,
        dtype=jnp.float32,
        param_dtype=jnp.float32,
        features=(6, 64),
        use_bias=True,
        )


np.random.seed(42)
inputs_q = np.random.randn(1, 2, 384)
print(inputs_q[0, 0, :10])

params = params['octo_transformer']['BlockTransformer_0']['Transformer_0']['encoderblock_0']['MultiHeadDotProductAttention_0']

jax_out = dense.apply(
            {"params": params['query']},
            inputs_q,
        )
# print(jax_out.shape)
jax_out = jax_out.reshape((1, 2, 384))
# print(jax_out)
# exit()

#==============================

q_pt = torch.from_numpy(inputs_q).float()

# w = params['kernel'].copy().transpose((1, 2, 0)).reshape((384, -1))
w = params['query']['kernel'].copy().transpose((1, 2, 0)).reshape((384, -1))
w = torch.from_numpy(w).float()

b = params['query']['bias'].copy().reshape(384)
b = torch.from_numpy(b).float()

with torch.no_grad():
    output_pt = F.linear(q_pt, w, b)

# print(output_pt.shape)
# print(output_pt)


print(np.all(np.isclose(output_pt.cpu().numpy(), jax_out, 0.001, 0.001)))


#================================

dense = nn_flax.linear.DenseGeneral(
        features=384,
        axis=(-2, -1),
        dtype=jnp.float32,
        param_dtype=jnp.float32,
        use_bias=False,
        )

x = np.random.randn(1, 2, 6, 64)

jax_out = dense.apply(
            {"params": params['out']},
            x,
        )

# print(jax_out.shape)
# print(jax_out)

x_pt = torch.from_numpy(x).float().reshape((1, 2, -1))

# w = params['kernel'].copy().transpose((1, 2, 0)).reshape((384, -1))
w = params['out']['kernel'].copy().transpose((2, 0, 1)).reshape((384, -1))
w = torch.from_numpy(w).float()

b = params['out']['bias'].copy()
b = torch.from_numpy(b).float()

with torch.no_grad():
    output_pt = F.linear(x_pt, w)

# print(output_pt.shape)
# print(output_pt)


print(np.all(np.isclose(output_pt.cpu().numpy(), jax_out, 0.001, 0.001)))