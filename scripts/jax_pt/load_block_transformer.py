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
import jax.numpy as jnp
import json

from octo.model.octo_model import OctoModel
from octo.utils.spec import ModuleSpec
from octo.model.octo_module import OctoModule
from octo.model.components.block_transformer import BlockTransformer, PrefixGroup, TimestepGroup, AttentionRule
from octo.model.components.block_transformer_pt import PrefixGroupPt, TimestepGroupPt

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


def load_params_modular(all_prefix_groups, all_timestep_groups):

    init_args = (
        all_prefix_groups,
        all_timestep_groups,
    )


    checkpoint_path = Path('/home/jovyan/.cache/huggingface/hub/models--rail-berkeley--octo-small-1.5/snapshots/dc9aa3019f764726c770814b27e4ab0fc6e32a58')
    config_path = checkpoint_path / 'config.json'
    with open(config_path, 'r') as f:
        config = json.load(f)
        
    module = BlockTransformer(
        transformer_kwargs = config['model']['transformer_kwargs'],
        use_correct_attention=config['model']['use_correct_attention'],
        
    )
    params_shape = jax.eval_shape(
                partial(module.init, train=False, verbose=True), jax.random.PRNGKey(0), *init_args
            )["params"]
    
    checkpointer = orbax.checkpoint.CheckpointManager(
                checkpoint_path, orbax.checkpoint.PyTreeCheckpointer()
            )
    step = checkpointer.latest_step()
    params = checkpointer.restore(step, params_shape)
    
    return module, params, config

def get_inputs():
    
    task_attention_rules = {"task_*": AttentionRule.CAUSAL}
    
    observation_attention_rules = {
            "task_*": AttentionRule.CAUSAL,
            "obs_*": AttentionRule.CAUSAL,
        }
    bs, prefix_n_tk, obs_n_tkn, obs_wrist_n_tkn, horizon, dim = 10, 25, 35, 15, 4, 384
    # bs, prefix_n_tk, obs_n_tkn, obs_wrist_n_tkn, horizon, dim = 1, 2, 1, 2, 2, 2
    
    np.random.seed(0)
    prefix_tokens = np.random.randn(bs, prefix_n_tk, dim)
    prefix_mask = np.random.randn(bs, prefix_n_tk,) > 0.
    all_prefix_groups = [
        PrefixGroup(
                    tokens=prefix_tokens,
                    mask=prefix_mask,
                    name='task_language',
                    attention_rules=task_attention_rules,
                )
    ]
    
    prefix_tokens_pt = torch.from_numpy(prefix_tokens).float()
    prefix_mask_pt =  torch.from_numpy(prefix_mask).float()
    all_prefix_groups_pt = [
        PrefixGroupPt(
                    tokens=prefix_tokens_pt,
                    mask=prefix_mask_pt,
                    name='task_language',
                    attention_rules=task_attention_rules,
                )
    ]
    
    obs_main_tokens = np.random.randn(bs, horizon, obs_n_tkn, dim)
    obs_main_mask = np.random.randn(bs, horizon, obs_n_tkn,) > 0.
    all_timestep_groups = []
    all_timestep_groups.append(
                TimestepGroup(
                    tokens=obs_main_tokens,
                    mask=obs_main_mask,
                    name='obs_main',
                    attention_rules=observation_attention_rules,
                )
            )
    obs_main_tokens_pt = torch.from_numpy(obs_main_tokens).float()
    obs_main_mask_pt = torch.from_numpy(obs_main_mask).float()
    all_timestep_groups_pt = []
    all_timestep_groups_pt.append(
                TimestepGroupPt(
                    tokens=obs_main_tokens_pt,
                    mask=obs_main_mask_pt,
                    name='obs_main',
                    attention_rules=observation_attention_rules,
                )
            )
    
    obs_wrist_tokens = np.random.randn(bs, horizon, obs_wrist_n_tkn, dim)
    obs_wrist_mask = np.random.randn(bs, horizon, obs_wrist_n_tkn,) > 0.
    all_timestep_groups.append(
                TimestepGroup(
                    tokens=obs_wrist_tokens,
                    mask=obs_wrist_mask,
                    name='obs_wrist',
                    attention_rules=observation_attention_rules,
                )
            )
    
    obs_wrist_tokens_pt = torch.from_numpy(obs_wrist_tokens).float()
    obs_wrist_mask_pt = torch.from_numpy(obs_wrist_mask).float()
    all_timestep_groups_pt.append(
                TimestepGroupPt(
                    tokens=obs_wrist_tokens_pt,
                    mask=obs_wrist_mask_pt,
                    name='obs_wrist',
                    attention_rules=observation_attention_rules,
                )
            )
    
    task_tokens = all_prefix_groups[0].tokens[:, jnp.newaxis, :, :]
    ws = all_timestep_groups[0].tokens.shape[1]
    task_tokens = jnp.tile(task_tokens, [1, ws, 1, 1])
    task_pad_mask = all_prefix_groups[0].mask[:, jnp.newaxis, :]
    task_pad_mask = jnp.tile(task_pad_mask, [1, ws, 1])
    group_name = f"obs_{all_prefix_groups[0].name}"
    all_timestep_groups.append(
        TimestepGroup(
            tokens=task_tokens,
            mask=task_pad_mask,
            name=group_name,
            attention_rules=observation_attention_rules,
        )
    )
    
    
    task_tokens_pt = torch.from_numpy(np.array(task_tokens))
    task_pad_mask_pt = torch.from_numpy(np.array(task_pad_mask))
    all_timestep_groups_pt.append(
        TimestepGroupPt(
            tokens=task_tokens_pt,
            mask=task_pad_mask_pt,
            name=group_name,
            attention_rules=observation_attention_rules,
        )
    )



    group_name = f"readout_action"
    readout_attention_rules = {
        "task_*": AttentionRule.CAUSAL,
        "obs_*": AttentionRule.CAUSAL,
        group_name: AttentionRule.CAUSAL,
    }  # Attend to tasks, all previous observations, and *only it's own own readout*
    
    readout_tokens = jnp.zeros(
                (bs, horizon, 1, dim)
            )
    
    readout_mask = jnp.ones((bs, horizon, 1))
    all_timestep_groups.append(
        TimestepGroup(
            tokens=readout_tokens,
            mask=readout_mask,
            name=group_name,
            attention_rules=readout_attention_rules,
        )
    )
    
    
    group_name = f"readout_action"
    readout_attention_rules = {
        "task_*": AttentionRule.CAUSAL,
        "obs_*": AttentionRule.CAUSAL,
        group_name: AttentionRule.CAUSAL,
    }  # Attend to tasks, all previous observations, and *only it's own own readout*
    
    readout_tokens_pt = torch.from_numpy(np.array(readout_tokens)).float()
    
    readout_mask_pt = torch.from_numpy(np.array(readout_mask)).int()
    all_timestep_groups_pt.append(
        TimestepGroupPt(
            tokens=readout_tokens_pt,
            mask=readout_mask_pt,
            name=group_name,
            attention_rules=readout_attention_rules,
        )
    )
    
    return all_prefix_groups, all_timestep_groups, all_prefix_groups_pt, all_timestep_groups_pt
    
    #=============
def test_block_transformer(
        params, 
        config,
        all_prefix_groups,
        all_timestep_groups,
        all_prefix_groups_pt,
        all_timestep_groups_pt,
        device='cuda:0'
    ):
    block_transformer = BlockTransformer(
        transformer_kwargs = config['model']['transformer_kwargs'],
        use_correct_attention=config['model']['use_correct_attention'],
        
    )
    
    prefix_outputs, timestep_outputs = block_transformer.apply(
        {'params': params['octo_transformer']['BlockTransformer_0']},
        all_prefix_groups,
        all_timestep_groups,
        train=False,
        verbose=True,
    )
    
    print(prefix_outputs[0].tokens[0, 0], timestep_outputs[0].tokens[0, 0, 0])
    
    #=============
    
    new_config = {
        'args': [],
        'kwargs': {'transformer_kwargs': {
                'add_position_embedding': False,
                'attention_dropout_rate': 0.0,
                'dropout_rate': 0.0,
                'token_embedding_size': 384,
                'mlp_dim': 1536,
                'num_attention_heads': 6,
                'num_layers': 12},
            'use_correct_attention': config['model']['use_correct_attention']
        },
        'module': 'octo.model.components.block_transformer_pt',
        'name': 'BlockTransformerPt'
        }
    
    transformer_pt = ModuleSpec.instantiate(new_config)()
    transformer_pt.load_jax_weights(params['octo_transformer']['BlockTransformer_0'])
    transformer_pt.to(device)
    transformer_pt.eval()
    
    all_prefix_groups_pt = [group.to(device) for group in all_prefix_groups_pt]
    all_timestep_groups_pt = [group.to(device) for group in all_timestep_groups_pt]
    
    prefix_outputs_pt, timestep_outputs_pt = transformer_pt(
        all_prefix_groups_pt,
        all_timestep_groups_pt,
        False,
        verbose=True
    )
    print(prefix_outputs_pt[0].tokens[0, 0], timestep_outputs_pt[0].tokens[0, 0, 0])
    print(np.all(np.isclose(prefix_outputs_pt[0].tokens.detach().cpu().numpy(), prefix_outputs[0].tokens, 0.001, 0.001)))
    print(np.all(np.isclose(timestep_outputs_pt[0].tokens.detach().cpu().numpy(), timestep_outputs[0].tokens, 0.001, 0.001)))
    
def test_block_transformer_trunkated(
        params, config,
        all_prefix_groups,
        all_timestep_groups,
        all_prefix_groups_pt,
        all_timestep_groups_pt,
    ):
    
    block_transformer = BlockTransformer(
        transformer_kwargs = config['model']['transformer_kwargs'],
        use_correct_attention=config['model']['use_correct_attention'],
        
    )
    
    # input_tokens = block_transformer.assemble_input_tokens(all_prefix_groups, all_timestep_groups)

    # attention_mask = block_transformer.generate_attention_mask(all_prefix_groups, all_timestep_groups)
    
    attention_mask = block_transformer.get_attn_mask(all_prefix_groups, all_timestep_groups)
    
    print(attention_mask)
    # exit()
    # print(prefix_outputs[0].tokens[0, 0], timestep_outputs[0].tokens[0, 0, 0])
    
    #=============
    
    new_config = {
        'args': [],
        'kwargs': {'transformer_kwargs': {
            'add_position_embedding': False,
            'attention_dropout_rate': 0.0,
            'dropout_rate': 0.0,
            'token_embedding_size': 384,
            'mlp_dim': 1536,
            'num_attention_heads': 6,
            'num_layers': 12},
            'use_correct_attention': config['model']['use_correct_attention']
                
        },
        'module': 'octo.model.components.block_transformer_pt',
        'name': 'BlockTransformerPt'
        }

    transformer_pt = ModuleSpec.instantiate(new_config)()
    transformer_pt.load_jax_weights(params['octo_transformer']['BlockTransformer_0'])
    transformer_pt.eval()
    
    # input_tokens_pt = transformer_pt.assemble_input_tokens(all_prefix_groups_pt, all_timestep_groups_pt)
        
    # attention_mask_pt = transformer_pt.generate_attention_mask(all_prefix_groups_pt, all_timestep_groups_pt)
    
    attention_mask_pt = transformer_pt.get_attn_mask(all_prefix_groups_pt, all_timestep_groups_pt)
    
    print(attention_mask_pt.int())
    # print(prefix_outputs_pt[0].tokens[0, 0], timestep_outputs_pt[0].tokens[0, 0, 0])
    # print(np.all(np.isclose(input_tokens_pt.detach().cpu().numpy(), input_tokens, 0.001, 0.001)))
    print(np.all(np.isclose(attention_mask_pt.detach().cpu().numpy(), attention_mask, 0.01, 0.01)))    


if __name__ == '__main__':
    device = 'cuda:0'
    test_block_transformer(*load_params(), *get_inputs(), device=device)
    # test_block_transformer_trunkated(*load_params(), *get_inputs())


