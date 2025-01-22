# Written by Dibya
import logging
from typing import Dict, Optional
from functools import partial

import torch
import torch.nn as nn


from octo.model.components.base_pt import TokenGroupPt
from octo.model.components.transformer_pt import AddPositionEmbsPt
from octo.model.components.block_transformer import AttentionRule
from octo.model.components.block_transformer_pt import (
    BlockTransformerPt,
    PrefixGroupPt,
    TimestepGroupPt,
    
)
from octo.utils.train_utils_pt import _flatten_dict

from octo.utils.spec import ModuleSpec
from octo.utils.typing import Data, Sequence
from octo.model.components.jax_pt import FromJaxModel, LinearPt, ParamNode

class OctoTransformerPt(nn.Module, FromJaxModel):
    """
    This module forms the base of the Octo architecture.

    The core idea is to run a causal transformer on the following sequence,

        [task, observation 0, observation 1, observation 2, ...]

    The task is tokenized using a set of *task tokenizers* (for example, a tokenizer that processes the
    language instruction into tokens, or one that processes the goal images into tokens).

    The observation at each timestep is tokenized using a set of *observation tokenizers*
    (for example, a tokenizer that processes the primary image into tokens, or one that processes
    the wrist image into tokens).

    We introduce additional tokens ("readouts") that "read out" the information in the transformer for
    downstream action or value prediction. For example, we may have an "action" readout that provides
    embeddings that are useful for predicting actions, and a "value" readout with embeddings that are useful
    for predicting values.

    The transformer is a blockwise-causal transformer, where each timestep only attends to the same or
    previous timesteps.  The easiest way to understand how the model works is to run:

    ```
        >>> model(observations, tasks, timestep_pad_mask, verbose=True)
    ```

    Generally, the model runs the transformer on something like the following sequence:

    [
        <task language tokens>,
        <t=0 "image_primary" tokens>, <t=0 "image_wrist" tokens>, <t=0 readout_action tokens>, ...
        <t=1 "image_primary" tokens>, <t=1 "image_wrist" tokens>, <t=1 readout_action tokens>, ...
        <t=2 "image_primary" tokens>, <t=2 "image_wrist" tokens>, <t=2 readout_action tokens>, ...
        ...
    ]

    The observation tokens attend to the task prefix, and to all observation tokens in the same or previous
    timesteps. So, "image_wrist" can attend to "image_primary" and vice versa.

    Readouts provide a mechanism for "reading out" the information in the transformer. They are designed to
    only *read* from the sequence before it, without the ability to influence (i.e. write) the computation for
    any of the non-readout tokens. By design, different readouts (e.g. "action" vs "value") are completely
    independent of each other, meaning they can be run separately without affecting each other.

    Args:
        TODO
    """

    def __init__(self, 
                 observation_tokenizers: Dict[str, nn.Module], 
                 task_tokenizers: Dict[str, nn.Module],
                 readouts: Dict[str, int],
                transformer_kwargs: Dict,
                token_embedding_size: int,
                max_horizon: int,
                repeat_task_tokens: bool,
                use_correct_attention: bool = False,
                max_horizon_dim: int = 10,
                num_tokens_dict: Dict[str, int] = {
                    'primary': 256,
                    'wrist': 64,
                    'language': 16,
                    'action': 1
                },
                 ):
        super().__init__()
        self.task_tokenizers = nn.ModuleDict(task_tokenizers)
        self.observation_tokenizers = nn.ModuleDict(observation_tokenizers)
        self.readouts = readouts
        self.transformer_kwargs = transformer_kwargs
        self.token_embedding_size = token_embedding_size
        self.max_horizon = max_horizon
        self.repeat_task_tokens = repeat_task_tokens
        self.use_correct_attention = use_correct_attention
        self.max_horizon_dim = max_horizon_dim
        self.transformer_kwargs['token_embedding_size'] = self.token_embedding_size
            
        self.block_transformer = BlockTransformerPt(
            transformer_kwargs = self.transformer_kwargs,
            use_correct_attention=self.use_correct_attention,
            )
        
        # Create projection layers for each tokenizer
        self.task_projections = nn.ModuleDict({
            f"task_{name}_projection": LinearPt(tok.output_dim, token_embedding_size)
            for name, tok in self.task_tokenizers.items()
        })
        self.obs_projections = nn.ModuleDict({
            f"obs_{name}_projection": LinearPt(tok.output_dim, token_embedding_size)
            for name, tok in self.observation_tokenizers.items()
        })
        
        self.pos_embeddings_names = []
        for name, tok in self.task_tokenizers.items():
            setattr(self, f"task_{name}_pos_embedding", self._create_positional_embedding((num_tokens_dict[name], self.token_embedding_size)))
            self.pos_embeddings_names.append(f"task_{name}_pos_embedding")
        # self.pos_embeddings = nn.ParameterDict({
        #     f"task_{name}_pos_embedding": self._create_positional_embedding((num_tokens_dict[name], self.token_embedding_size))
        #     for name, tok in self.task_tokenizers.items()
        # })
        
        for name, tok in self.observation_tokenizers.items():
            setattr(self, f"obs_{name}_pos_embedding", self._create_positional_embedding((self.max_horizon_dim, num_tokens_dict[name], self.token_embedding_size)))
            self.pos_embeddings_names.append(f"obs_{name}_pos_embedding")
        # self.pos_embeddings.update(
        #     nn.ParameterDict({
        #         f"obs_{name}_pos_embedding": self._create_positional_embedding((self.max_horizon_dim, num_tokens_dict[name], self.token_embedding_size))
        #         for name, tok in self.observation_tokenizers.items()
        #     })
        # )
        
        for readout_name in self.readouts:
            setattr(self, f"readout_{readout_name}_pos_embedding", self._create_positional_embedding((self.max_horizon_dim, num_tokens_dict[readout_name], self.token_embedding_size)))
            self.pos_embeddings_names.append(f"readout_{readout_name}_pos_embedding")
        # self.pos_embeddings.update(
        #     nn.ParameterDict({
        #         f"readout_{readout_name}_pos_embedding": self._create_positional_embedding((self.max_horizon_dim, num_tokens_dict[readout_name], self.token_embedding_size))
        #         for readout_name in self.readouts
        #     })
        # )
        
    def _create_positional_embedding(self, shape):
        pe = torch.randn(*shape) * 0.02
        return nn.Parameter(pe)
         
    
    def pt_to_jax_args_map(self):
        # {
        # pt_module_name: (load_func, jax_param_key),
        # ...
        # }
        pt_to_jax = {
            "block_transformer": ParamNode(submodule=self.block_transformer, jax_param_names='BlockTransformer_0')
        }
        for name in self.observation_tokenizers:
            # if self.observation_tokenizers[name].num_of_params_to_init > 0:
            pt_to_jax[f'observation_tokenizers.{name}'] = ParamNode(
                submodule=self.observation_tokenizers[name],
                jax_param_names=f'observation_tokenizers_{name}'
            )
        
        for name in self.task_tokenizers:
            pt_to_jax[f'task_tokenizers.{name}'] = ParamNode(
                submodule=self.task_tokenizers[name],
                jax_param_names=f'task_tokenizers_{name}'
            )
        
        for name in self.task_projections:
            pt_to_jax[f'task_projections.' + name] = ParamNode(
                submodule=self.task_projections[name],
                jax_param_names=name
            )
            
        for name in self.obs_projections:
            pt_to_jax[f'obs_projections.' + name] = ParamNode(
                submodule=self.obs_projections[name],
                jax_param_names=name
            )
            
        for name in self.pos_embeddings_names:
            pt_to_jax[name] = ParamNode(
                    load_func=partial(self._set_terminal_param, transform_function=lambda x: x[0]),
                    jax_param_names=name
                )
        return pt_to_jax
    
    def _add_positional_embedding(self, inputs, pe, history_dim=None):
        bs = inputs.shape[0]
        if history_dim is not None:
            history_len = inputs[0].shape[history_dim]
            pe = pe[(slice(None),) * history_dim + (slice(0, history_len),)]
        else:
            assert pe.shape == inputs[0].shape
        pe = pe.unsqueeze(0).expand(bs, *pe.shape)
        assert inputs.shape == pe.shape
        return inputs + pe
            
    def forward(self,
        observations,
        tasks,
        timestep_pad_mask: torch.tensor,
        readouts: Optional[Sequence[str]] = None,
        train: bool = False,
        verbose: bool = False,
        save_attention_mask: bool = False
        ):
        """
        Args:
            TODO

        Returns:
            TODO
        Note: Horizon can be anything <= max_horizon.
        """
        if readouts is None:
            readouts = list(self.readouts.keys())

        #
        # Check that all inputs are valid
        #

        assert set(readouts).issubset(
            set(self.readouts.keys())
        ), "readouts must be specified in the model config"

        batch_size, horizon = next(iter(observations.values())).shape[:2]
        assert horizon <= self.max_horizon, "horizon must be <= max_horizon"
        assert all(x.shape[1] == horizon for x in _flatten_dict(observations).values()), \
            "observations must have the same horizon"

        #
        # Attention rules for the transformer
        #

        # Tasks attend to all other tasks, but not to observations or readouts
        task_attention_rules = {"task_*": AttentionRule.CAUSAL}

        # Observations attend to all tasks and all other observations tokens causally,
        # e.g. at same timestep or before, but do not attend to readouts

        observation_attention_rules = {
            "task_*": AttentionRule.CAUSAL,
            "obs_*": AttentionRule.CAUSAL,
        }

        #
        # Create inputs for the transformer
        #

        all_prefix_groups = []
        all_timestep_groups = []

        #
        # First, add the task tokens
        #

        for name, tok in self.task_tokenizers.items():
            group_name = f"task_{name}"
            # Receive inputs from tokenizer and cast to embedding size
            tokenizer_output: TokenGroupPt = tok(observations, tasks, train=train)
            if tokenizer_output is None:
                logging.warning(f"Skipping task tokenizer: {group_name}")
                continue

            task_tokens = self.task_projections[f"{group_name}_projection"](tokenizer_output.tokens)
            # task_tokens shape is (batch, n_tokens, token_embedding_size)

            # Add positional embedding
            task_tokens = self._add_positional_embedding(
                task_tokens, 
                getattr(self, f'{group_name}_pos_embedding')
            )

            all_prefix_groups.append(
                PrefixGroupPt(
                    tokens=task_tokens,
                    mask=tokenizer_output.mask,
                    name=group_name,
                    attention_rules=task_attention_rules,
                )
            )

        #
        # Next, add the observation tokens
        #

        for name, tok in self.observation_tokenizers.items():
            group_name = f"obs_{name}"
            # Receive inputs from tokenizer and cast to embedding size
            tokenizer_output: TokenGroupPt = tok(observations, tasks, train=train)
            if tokenizer_output is None:
                logging.warning(f"Skipping observation tokenizer: {group_name}")
                continue

            obs_tokens = self.obs_projections[f"{group_name}_projection"](tokenizer_output.tokens)
            # obs_tokens shape is (batch, horizon, n_tokens, token_embedding_size)

            # Add positional embedding
            obs_tokens = self._add_positional_embedding(
                obs_tokens, 
                getattr(self, f'{group_name}_pos_embedding'),
                history_dim=0
            )

            # Update mask to account for which timesteps are padding
            obs_pad_mask = timestep_pad_mask.unsqueeze(-1) & tokenizer_output.mask

            all_timestep_groups.append(
                TimestepGroupPt(
                    tokens=obs_tokens,
                    mask=obs_pad_mask,
                    name=group_name,
                    attention_rules=observation_attention_rules,
                )
            )
        if self.repeat_task_tokens:
            # logging.info(
            #     "repeating task tokens at each timestep to perform cross-modal attention"
            # )
            # get task tokens
            for task in all_prefix_groups:
                # lang (batch, n_tokens, token_embedding_size)
                task_tokens = task.tokens.unsqueeze(1)
                task_tokens = task_tokens.expand(-1, horizon, -1, -1)
                task_pad_mask = task.mask.unsqueeze(1).expand(-1, horizon, -1)
                group_name = f"obs_{task.name}"
                all_timestep_groups.append(
                    TimestepGroupPt(
                        tokens=task_tokens,
                        mask=task_pad_mask,
                        name=group_name,
                        attention_rules=observation_attention_rules,
                    )
                )

        #
        # Finally, add the readout tokens
        #

        device = next(self.parameters()).device
        for readout_name in readouts:
            group_name = f"readout_{readout_name}"
            # Readouts do not correspond to any inputs, just positional embeddings
            n_tokens_for_readout = self.readouts[readout_name]
            readout_tokens = torch.zeros(
                (batch_size, horizon, n_tokens_for_readout, self.token_embedding_size),
                device=device
            )

            # Add positional embedding
            readout_tokens = self._add_positional_embedding(
                readout_tokens, 
                getattr(self, f'{group_name}_pos_embedding'),
                history_dim=0
            )
            
            readout_mask = torch.ones((batch_size, horizon, n_tokens_for_readout), device=device, dtype=torch.bool)
            readout_attention_rules = {
                "task_*": AttentionRule.CAUSAL,
                "obs_*": AttentionRule.CAUSAL,
                group_name: AttentionRule.CAUSAL,
            }  # Attend to tasks, all previous observations, and *only it's own own readout*

            all_timestep_groups.append(
                TimestepGroupPt(
                    tokens=readout_tokens,
                    mask=readout_mask,
                    name=group_name,
                    attention_rules=readout_attention_rules,
                )
            )

        # Run the transformer!
        assert (
            self.transformer_kwargs.get("add_position_embedding", False) is False
        ), "Already added positional embeddings to the tokens"

        prefix_outputs, timestep_outputs = self.block_transformer(
            all_prefix_groups,
            all_timestep_groups,
            train=train,
            verbose=verbose,
            save_attention_mask=save_attention_mask
        )
        outputs = {}
        outputs.update(
            {
                group.name: TokenGroupPt(group.tokens, group.mask)
                for group in prefix_outputs
            }
        )
        outputs.update(
            {
                group.name: TokenGroupPt(group.tokens, group.mask)
                for group in timestep_outputs
            }
        )

        if len(prefix_outputs) > 0:
            outputs["task"] = TokenGroupPt.concatenate(
                [TokenGroupPt(group.tokens, group.mask) for group in prefix_outputs]
            )

        outputs["obs"] = TokenGroupPt.concatenate(
            [
                TokenGroupPt(group.tokens, group.mask)
                for group in timestep_outputs
                if group.name.startswith("obs_")
            ],
            axis=-2,
        )

        return outputs


class OctoModulePt(nn.Module, FromJaxModel):
    def __init__(self, octo_transformer: OctoTransformerPt, heads: Dict[str, nn.Module], initialize_heads: bool = True):
        super().__init__()
        self.octo_transformer = octo_transformer
        self.heads = nn.ModuleDict(heads)
        self.initialize_heads = initialize_heads
     
    def pt_to_jax_args_map(self):
        # {
        # pt_module_name: (load_func, jax_param_key),
        # ...
        # }
        pt_to_jax = {
            "octo_transformer": ParamNode(submodule=self.octo_transformer, jax_param_names='octo_transformer'),
        }
        if self.heads and self.initialize_heads:
            pt_to_jax["heads.action"] = ParamNode(submodule=self.heads['action'], jax_param_names='heads_action')
            
        return pt_to_jax
        
    
    @classmethod
    def create(cls,
        observation_tokenizers: Dict[str, nn.Module],
        task_tokenizers: Dict[str, nn.Module],
        heads: Dict[str, nn.Module],
        readouts: Dict[str, int],
        transformer_kwargs: Dict,
        token_embedding_size: int,
        max_horizon: int,
        repeat_task_tokens: bool = False,
        use_correct_attention: bool = False,
        initialize_heads: bool = True,
        num_tokens_dict: dict = {
            'primary': 256,
            'wrist': 64,
            'language': 16,
            'action': 1
        }
    ) -> "OctoModulePt":
        """
        Canonical way to create an OctoModule from configuration.

        Args:
            TODO
        """
        observation_tokenizer_defs = {
            k: ModuleSpec.instantiate(spec)()
            for k, spec in observation_tokenizers.items()
        }
        task_tokenizer_defs = {
            k: ModuleSpec.instantiate(spec)() for k, spec in task_tokenizers.items()
        }

        head_defs = None # TODO remove
        if heads:
            head_defs = {}
            
            # append number of input dimensions to kwargs
            for k, spec in heads.items():
                head_defs[k] = spec
                if not 'input_dim' in head_defs[k]['kwargs']:
                    head_defs[k]['kwargs']['input_dim'] = token_embedding_size
                    
            head_defs = {
                k: ModuleSpec.instantiate(spec)() for k, spec in head_defs.items()
            }
                    
        model_def = OctoTransformerPt(
            observation_tokenizers=observation_tokenizer_defs,
            task_tokenizers=task_tokenizer_defs,
            readouts=readouts,
            token_embedding_size=token_embedding_size,
            max_horizon=max_horizon,
            repeat_task_tokens=repeat_task_tokens,
            transformer_kwargs=transformer_kwargs,
            use_correct_attention=use_correct_attention,
            num_tokens_dict=num_tokens_dict
        )

        return cls(
            octo_transformer=model_def,
            heads=head_defs,
            initialize_heads=initialize_heads
        )
        
    
    
    def forward(self, 
                observations, 
                tasks, 
                timestep_pad_mask, 
                action_pad_mask=None,
                gt_actions=None,
                train=True, 
                transformer_only=False,
                verbose=False,
                save_attention_mask=False
                ):
        transformer_outputs = self.octo_transformer(
            observations, tasks, timestep_pad_mask, train=train, verbose=verbose, save_attention_mask=save_attention_mask
        )
        head_outputs = {}
        if self.heads and not transformer_only: # TODO: remove if
            for head_name, head in self.heads.items():
                if train:
                    head_outputs[head_name] = head.loss(transformer_outputs, 
                                                        gt_actions,
                                                        timestep_pad_mask,
                                                        action_pad_mask,
                                                        train=train)
                else:
                    head_outputs[head_name] = head(transformer_outputs, train=train)
        return transformer_outputs, head_outputs
    
