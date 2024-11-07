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


attn = nn_flax.MultiHeadDotProductAttention(
        dtype=jnp.float32,
        kernel_init=nn_flax.initializers.xavier_uniform(),
        broadcast_dropout=False,
        deterministic=True,
        dropout_rate=0.1,
        num_heads=6,
        )
np.random.seed(42)
x = np.random.randn(10, 100, 384)
print(x[0, 0, :10])
att_mask = np.ones((100, 100)) - np.tri(100, 100, k=-1)
jax_out = attn.apply(
            {"params": params['octo_transformer']['BlockTransformer_0']['Transformer_0']['encoderblock_0']['MultiHeadDotProductAttention_0']},
            x,
            x,
        )
print(jax_out.shape)
print(jax_out[0, 0])
# exit()

#==============================



new_config = {
    'args': [],
   'kwargs': {'embed_dim':  384, 'num_heads': 6, 'dropout': 0.1, 'batch_first': True, 'add_bias_kv': False},
   'module': 'octo.model.components.transformer_pt',
   'name': 'MultiheadAttentionPt'}
new_config

attn_pt = ModuleSpec.instantiate(new_config)()
print(attn_pt.state_dict().keys())

uninitialized_params, unused_jax_params = attn_pt.load_jax_weights(params['octo_transformer']['BlockTransformer_0']['Transformer_0']['encoderblock_0'], 'MultiHeadDotProductAttention_0', 'MultiheadAttentionPt')
attn_pt.eval()

print(uninitialized_params, unused_jax_params)

x_pt = torch.from_numpy(x).float()
att_mask_pt = torch.from_numpy(att_mask).float()
att_mask_pt = att_mask_pt.expand(10, -1, -1)

with torch.no_grad():
    output_pt, _ = attn_pt(x_pt, x_pt, x_pt, key_padding_mask=None, need_weights=False)

print(output_pt.shape)
print(output_pt[0, 0])


print(np.all(np.isclose(output_pt.cpu().numpy(), jax_out, 0.001, 0.001)))
