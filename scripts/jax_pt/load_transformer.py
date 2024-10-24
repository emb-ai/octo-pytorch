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
from octo.model.components.transformer import Encoder1DBlock, Transformer
from octo.model.components.transformer_pt import Encoder1DBlockPt
#==============================

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
    
    return params, config

def test_encoder(params, config):
    encoder = Encoder1DBlock(
            mlp_dim=1536,
            num_heads=6
            )
    x = np.random.randn(10, 100, 384)
    att_mask =  np.tri(100, 100, k=1)
    att_mask = np.random.randn(100, 100) > 0.5

    att_mask = np.tile(att_mask, (10, 6, 1, 1)) 
    print(att_mask.shape)

    jax_out = encoder.apply(
                {"params": params['octo_transformer']['BlockTransformer_0']['Transformer_0']['encoderblock_0']},
                x,
                att_mask,
                deterministic=True
            )
    # print(jax_out.shape)
    # print(jax_out[0, 0])
    # exit()

    #==============================







    new_config = {
        'args': [],
    'kwargs': {'input_dim':  384, 'mlp_dim': 1536, 'num_heads':6},
    'module': 'octo.model.components.transformer_pt',
    'name': 'Encoder1DBlockPt'}
    new_config

    encoder_pt = ModuleSpec.instantiate(new_config)()
    print(encoder_pt.state_dict().keys())

    encoder_pt.load_jax_weights(params['octo_transformer']['BlockTransformer_0']['Transformer_0']['encoderblock_0'])
    encoder_pt.eval()

    x_pt = torch.from_numpy(x).float()
    att_mask_pt = torch.from_numpy(att_mask[0, 0]).to(torch.bool)
    att_mask_pt = ~att_mask_pt

    with torch.no_grad():
        output_pt = encoder_pt(
            x_pt,
            att_mask_pt
        )

    # print(output_pt.shape)
    # print(output_pt[0, 0])


    print(np.all(np.isclose(output_pt.cpu().numpy(), jax_out, 0.001, 0.001)))


#=======================

def test_transformer(params, config):
    transformer = Transformer(
        **config['model']['transformer_kwargs']
    )

    x = np.random.randn(10, 100, 384)
    att_mask = np.random.randn(100, 100) > 0.5

    att_mask = np.tile(att_mask, (10, config['model']['transformer_kwargs']['num_attention_heads'], 1, 1)) 
    jax_out = transformer.apply(
                {"params": params['octo_transformer']['BlockTransformer_0']['Transformer_0']},
                x,
                att_mask,
                train=False
            )
    print(jax_out.shape)
    print(jax_out[0, 0])
    
    
    new_config = {
        'args': [],
        'kwargs': {'d_model':  384, 'num_layers': 12, 'mlp_dim': 1536, 'num_attention_heads':6},
        'module': 'octo.model.components.transformer_pt',
        'name': 'TransformerPt'
        }

    transformer_pt = ModuleSpec.instantiate(new_config)()
    transformer_pt.load_jax_weights(params['octo_transformer']['BlockTransformer_0']['Transformer_0'])
    transformer_pt.eval()

    x_pt = torch.from_numpy(x).float()
    att_mask_pt = torch.from_numpy(att_mask[0, 0]).to(torch.bool)
    att_mask_pt = ~att_mask_pt
    
    with torch.no_grad():
        output_pt = transformer_pt(
            x_pt,
            att_mask_pt
        )

    print(output_pt.shape)
    print(output_pt[0, 0])


    print(np.all(np.isclose(output_pt.cpu().numpy(), jax_out, 0.01, 0.01)))
    
test_transformer(*load_params())