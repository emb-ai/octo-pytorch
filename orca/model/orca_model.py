# Written by Dibya
import logging

import flax.linen as nn
import jax
import jax.numpy as jnp

from orca.model.components.base import TokenGroup
from orca.model.components.block_transformer import (
    AttentionRule,
    BlockTransformer,
    PrefixGroup,
    TimestepGroup,
)
from orca.utils.typing import Data, Dict, Optional, Sequence


class OrcaTransformer(nn.Module):
    """
    This module forms the base of the ORCA model.

    The core idea is to run a causal transformer on the following sequence,

        [task, observation 0, observation 1, observation 2, ...]

    but with additional groups of tokens ("readouts") that provide
    a way of "reading out" the information in the transformer.

    For example, we may have an "action" readout that provides embeddings that are
    useful for predicting actions, and a "value" readout with embeddings that are useful for
    predicting values.


    The transformer is a blockwise-causal transformer, where each timestep only attends to the same or previous timesteps.

    When called, the module requests a set of readouts, and performs a forward pass of the transformer on the following sequence:

        [
        <task tokens>,
        <observation ts0 tokens>, <readout1 ts0 tokens>, <readout2 ts0 tokens>, ...
        <observation ts1 tokens>, <readout1 ts1 tokens>, <readout2 ts1 tokens>, ...
        ...
    ]

    The observation tokens attend to the task prefix, and to all observation tokens in the same or previous timesteps.
    Readouts attend to everything observation tokens do, but are not attended to by observation or task tokens. All
    tokens within the same group and same timestep (e.g. "observation ts0 tokens") fully attend to each other.

    By this design, each readout does not influence the computation happening in the task or observation tokens,
    and each readout is **independent of one another**. This allows us to hot-swap in different
    readouts at any time (e.g. we can run with the action readout or the value readout or both at the same time).


    Args:
        observations_tokenizers (Sequence[nn.Module]): List of flax modules for tokenizing the observations.
            The output of each tokenizer is concatenated to form the observation tokens.
        task_tokenizers (Sequence[nn.Module]): List of flax modules for tokenizing the task.
            The output of each tokenizer is concatenated to form the task token prefix.
        readouts (Dict[str, int]): Dictionary of {readout_name: n_tokens_for_readout}
        transformer_kwargs (Dict): Dictionary of kwargs to forward to BlockTransformer.
        token_embedding_size (int): Dimension of the token embeddings (default: 512)
        max_horizon (int): The maximum number of timesteps that the transformer can be run with.
    """

    observation_tokenizers: Dict[str, nn.Module]
    task_tokenizers: Dict[str, nn.Module]
    readouts: Dict[str, int]
    transformer_kwargs: Dict
    token_embedding_size: int = 512
    max_horizon: int = 1

    @nn.compact
    def __call__(
        self,
        observations: Data,
        tasks: Data,
        pad_mask: jax.Array,
        readouts: Optional[Sequence[str]] = None,
        train: bool = False,
        verbose: bool = False,
    ) -> Dict[str, TokenGroup]:
        """
        Args:
            observations: A dictionary containing observation data for a batch of trajectory windows.
                Each entry has shape (batch, horizon, *).
            tasks: A dictionary containing task data for the trajectory windows.
                Each entry has shape (batch, *).
            pad_mask: A boolean mask of shape (batch, horizon) where False indicates a padded timestep.
            readouts: A list of readouts to compute. If None, defaults to all readouts. Must be a subset of the readouts specified in the model config.
            train: Whether model is being trained.
            verbose: If True, prints out the transformer structure.

        Returns:
            transformer_outputs: A dictionary {token_group_name: token_group},
                which contain the transformer embeddings for all observation tokens, task tokens, and readout tokens.
                The special keys "task" and "obs" contain the concatenated embeddings for all task tokens and observation tokens, respectively.

        Note: Horizon can be anything <= max_horizon.
        """
        if readouts is None:
            readouts = list(self.readouts.keys())

        # Check that all inputs are valid
        assert set(readouts).issubset(
            set(self.readouts.keys())
        ), "readouts must be a subset of those specified in the model config"

        batch_size, horizon = jax.tree_util.tree_leaves(observations)[0].shape[:2]
        assert horizon <= self.max_horizon, "horizon must be <= max_horizon"
        assert jax.tree_util.tree_all(
            jax.tree_map(lambda x: x.shape[1] == horizon, observations)
        ), "observations must have the same horizon"

        # Create inputs for the transformer
        all_prefix_groups = []
        all_timestep_groups = []

        # Tasks attend to all other tasks
        task_attention_rules = {"task_*": AttentionRule.CAUSAL}

        # Observations attend to all tasks and previous observations causally
        observation_attention_rules = {
            "task_*": AttentionRule.CAUSAL,
            "obs_*": AttentionRule.CAUSAL,
        }

        # First, add the task tokens
        for name, tok in self.task_tokenizers.items():
            # Receive inputs from tokenizer and cast to embedding size
            group_name = f"task_{name}"
            tokenizer_output: TokenGroup = tok(observations, tasks, train=train)
            if tokenizer_output is None:
                logging.warning(f"Skipping task tokenizer: {group_name}")
                continue

            task_tokens = nn.Dense(
                self.token_embedding_size, name=f"{group_name}_projection"
            )(tokenizer_output.tokens)

            # task_tokens shape is (batch, n_tokens, token_embedding_size)

            # Add positional embedding
            task_tokens += self._create_positional_embedding(group_name, task_tokens)

            all_prefix_groups.append(
                PrefixGroup(
                    tokens=task_tokens,
                    mask=tokenizer_output.mask,
                    name=group_name,
                    attention_rules=task_attention_rules,
                )
            )

        # Next, add the observation tokens
        for name, tok in self.observation_tokenizers.items():
            group_name = f"obs_{name}"
            # Receive inputs from tokenizer and cast to embedding size
            tokenizer_output: TokenGroup = tok(observations, tasks, train=train)
            if tokenizer_output is None:
                logging.warning(f"Skipping observation tokenizer: {name}")
                continue

            obs_tokens = nn.Dense(
                self.token_embedding_size, name=f"obs_{name}_projection"
            )(tokenizer_output.tokens)
            # obs_tokens shape is (batch, horizon, n_tokens, token_embedding_size)

            # Add positional embedding
            obs_tokens += self._create_positional_embedding(group_name, obs_tokens)

            # Update mask to account for which timesteps are padding
            obs_pad_mask = jnp.logical_and(pad_mask[:, :, None], tokenizer_output.mask)

            all_timestep_groups.append(
                TimestepGroup(
                    tokens=obs_tokens,
                    mask=obs_pad_mask,
                    name=group_name,
                    attention_rules=observation_attention_rules,
                )
            )

        # Finally, add the readout tokens
        for readout_name in readouts:
            group_name = f"readout_{readout_name}"
            # Readouts do not correspond to any inputs, so we just create a bunch of zeros
            n_tokens_for_readout = self.readouts[readout_name]
            readout_tokens = jnp.zeros(
                (batch_size, horizon, n_tokens_for_readout, self.token_embedding_size)
            )

            # Add positional embedding
            readout_tokens += self._create_positional_embedding(
                group_name, readout_tokens
            )
            readout_mask = jnp.ones((batch_size, horizon, n_tokens_for_readout))
            readout_attention_rules = {
                "task_*": AttentionRule.CAUSAL,
                "obs_*": AttentionRule.CAUSAL,
                group_name: AttentionRule.CAUSAL,
            }  # Attend to tasks, all previous observations, and *only your own readout*

            all_timestep_groups.append(
                TimestepGroup(
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

        prefix_outputs, timestep_outputs = BlockTransformer(self.transformer_kwargs)(
            all_prefix_groups,
            all_timestep_groups,
            train=train,
            verbose=verbose,
        )
        outputs = {}
        outputs.update(
            {
                group.name: TokenGroup(group.tokens, group.mask)
                for group in prefix_outputs
            }
        )
        outputs.update(
            {
                group.name: TokenGroup(group.tokens, group.mask)
                for group in timestep_outputs
            }
        )

        if len(prefix_outputs) > 0:
            outputs["task"] = TokenGroup.concatenate(
                [TokenGroup(group.tokens, group.mask) for group in prefix_outputs]
            )

        outputs["obs"] = TokenGroup.concatenate(
            [
                TokenGroup(group.tokens, group.mask)
                for group in timestep_outputs
                if group.name.startswith("obs_")
            ],
            axis=-2,
        )

        return outputs

    def _create_positional_embedding(self, name: str, tokens: jax.Array):
        if tokens.ndim == 3:  # for prefixes
            shape = (1, *tokens.shape[-2:])
        elif (
            tokens.ndim == 4
        ):  # for timesteps, create embedding for max_horizon, then truncate
            shape = (1, self.max_horizon, *tokens.shape[-2:])
        else:
            raise ValueError(f"Invalid tokens shape: {tokens.shape}")

        embedding = self.param(
            f"{name}_pos_embedding",
            nn.initializers.normal(stddev=0.02),
            shape,
        )
        if tokens.ndim == 4:
            # Use only the timesteps we receive as input
            embedding = embedding[:, : tokens.shape[1]]
        return jnp.broadcast_to(embedding, tokens.shape)


class OrcaModel(nn.Module):
    """
    Bundles OrcaTransformer with various heads (useful for keeping all parameters in one place).
    """

    orca_transformer: OrcaTransformer
    heads: Dict[str, nn.Module]

    def __call__(self, observations, tasks, pad_mask, train=True, verbose=False):
        """Run transformer and the main method for all heads. Useful for init.

        Args:
            observations: A dictionary containing observation data
                where each element has shape (batch, horizon, *).
            tasks: A dictionary containing task data
                where each element has shape (batch, *).
            pad_mask: A boolean mask of shape (batch, horizon) where False indicates a padded timestep.
            train: Run in training mode
            verbose: If True, prints out the structure of the OrcaTransformer

        Returns:
            transformer_embeddings: See OrcaTransformer.__call__
            head_outputs: dictionary of outputs from heads {head_name: output}
        """
        transformer_embeddings = self.orca_transformer(
            observations, tasks, pad_mask, train=train, verbose=verbose
        )
        head_outputs = {}
        for head_name, head in self.heads.items():
            head_outputs[head_name] = head(transformer_embeddings, train=train)
        return transformer_embeddings, head_outputs

    def run_transformer(self, *args, **kwargs):
        """Run transformer and return embeddings. See OrcaTransformer.__call__"""
        return self.orca_transformer(*args, **kwargs)

    def run_head(
        self,
        *args,
        head_name: str,
        head_method_name: str = "__call__",
        **kwargs,
    ):
        """A convenience utility to run a method on a single head.

        Args:
            head_name: Name of head to run.
            head_method_name: Name of method to run on head. Defaults to "__call__".
            train: Whether model is being trained.
            **kwargs: Keyword arguments to pass to method.
        """
        head = self.heads[head_name]
        method = getattr(head, head_method_name)
        return method(*args, **kwargs)
