# Written by Dibya
from enum import Enum
from fnmatch import fnmatch
import logging
from typing import Any, Dict, Mapping, Sequence, Tuple, Union
from dataclasses import dataclass

import torch
import torch.nn as nn

import einops
import numpy as np

from octo.model.components.base import TokenGroupPt
from octo.model.components.transformer_pt import TransformerPt
from octo.model.components.jax_pt import FromJaxModel

from .block_transformer import AttentionRule


@dataclass
class PrefixGroupPt(TokenGroupPt):
    """A group of tokens that will be at the beginning of the token sequence. (e.g. task tokens)

    Adds a name identifying the group, and a dictionary indicating what groups it should attend to.

    name (str): Name of the group, which other groups will look at when deciding whether to attend to this group.
    attention_rules (Dict[str, AttentionRule]): A dictionary of {pattern: AttentionRule} where the attention rule
        is recovered by fnmatch-ing the name of the other group until a match is found (or the end).
    """

    name: str
    attention_rules: Mapping[str, AttentionRule]

    def __post_init__(self):
        assert (
            len(self.tokens.shape) == 3
        ), "PrefixGroup tokens must be (batch, n_tokens, d)"
        assert len(self.mask.shape) == 2, "PrefixGroup mask must be (batch, n_tokens)"


@dataclass
class TimestepGroupPt(TokenGroupPt):
    """A group of tokens that is repeated for each timestep. (e.g. observation tokens)

    See PrefixGroup for details on the name and attention_rules fields.
    """

    name: str
    attention_rules: Mapping[str, AttentionRule]

    def __post_init__(self):
        assert (
            len(self.tokens.shape) == 4
        ), "TimestepGroup tokens must be (batch, horizon, n_tokens, d)"
        assert (
            len(self.mask.shape) == 3
        ), "TimestepGroup mask must be (batch, horizon, n_tokens)"


def find_match(pattern_dict: Dict[str, Any], name: str, default: Any) -> Any:
    """Find the first matching pattern in the dictionary, or return the default value."""
    for pattern, value in pattern_dict.items():
        if fnmatch(name, pattern):
            return value
    return default


@dataclass
class TokenMetadataPt:
    """Attention mask logic supported by AttentionRule. Note that all tokens within the
    same group at the same timestep always attend to each other unless you explicitly have
    attention_rules[self.name] = AttentionRule.NEVER
    """

    name: str
    timestep: int  # -1 for prefix tokens
    attention_rules: Mapping[str, AttentionRule]

    @classmethod
    def create(cls, group: Union[PrefixGroupPt, TimestepGroupPt], timestep: int):
        return cls(
            timestep=timestep,
            name=group.name,
            attention_rules=group.attention_rules,
        )

    def should_attend_to(self, other_metadata: "TokenMetadataPt") -> bool:
        attention_rule = find_match(
            self.attention_rules, other_metadata.name, AttentionRule.NEVER
        )

        if attention_rule == AttentionRule.CAUSAL:
            return other_metadata.timestep <= self.timestep
        elif attention_rule == AttentionRule.CURRENT:
            return other_metadata.timestep == self.timestep
        elif attention_rule == AttentionRule.STRICT_PAST:
            return other_metadata.timestep < self.timestep
        elif attention_rule == AttentionRule.ALL:
            return True
        elif attention_rule == AttentionRule.NEVER:
            return False
        else:
            raise ValueError(f"Invalid attention rule: {attention_rule}")


def split_tokens(ary: torch.tensor, n_tokens_per_group: Sequence[int], axis: int):
    # cumsum = np.cumsum(n_tokens_per_group)
    return torch.split(ary, n_tokens_per_group, dim=axis)


class BlockTransformerPt(nn.Module, FromJaxModel):
    """A transformer that acts on multiple groups of tokens, which may attend to each other (in complex patterns)."""
    def __init__(self, transformer_kwargs: Dict, enforce_causal: bool = True, use_correct_attention: bool = False):
        super().__init__()
        self.transformer_kwargs = transformer_kwargs
        self.enforce_causal = enforce_causal
        self.use_correct_attention = use_correct_attention
        self.transformer = TransformerPt(**self.transformer_kwargs)
        self.attention_mask = None

    @property
    def pt_to_jax_args_map(self):
        return {
            "transformer": (self.transformer.load_jax_weights, 'Transformer_0')
        }
    
    def forward(
        self,
        prefix_groups: Sequence[PrefixGroupPt],
        timestep_groups: Sequence[TimestepGroupPt],
        train: bool,
        verbose: bool = False,
    ) -> Tuple[Sequence[PrefixGroupPt], Sequence[TimestepGroupPt]]:
        """
        Args:
            prefix_groups: A list of PrefixGroup objects.
                Each group has
                    - tokens with shape (batch, n_tokens, token_embedding_size)
                    - mask with shape (batch, n_tokens) indicating which tokens are padding.
                    - name identifying the group
                    - dictionary of attention patterns dictating which other groups it will attend to.
            timestep_groups: A list of TimestepGroup objects.
                Each group has
                    - tokens with shape (batch, horizon, n_tokens, token_embedding_size)
                    - mask with shape (batch, horizon, n_tokens) indicating which tokens are padding.
                    - name identifying the group
                    - dictionary of attention patterns dictating which other groups it will attend to.
            train: Whether to use dropout.

        Returns:
            prefix_outputs: A list of PrefixGroup objects containing the output embeddings for each token group.
            timestep_outputs: A list of TimestepGroup objects containing the output embeddings for each token group.
        """
        if verbose:
            self.pretty_print_attention_mask(prefix_groups, timestep_groups)

        horizon = timestep_groups[0].tokens.shape[1]
        assert all([group.tokens.shape[1] == horizon for group in timestep_groups])

        token_dim = timestep_groups[0].tokens.shape[-1]
        assert all([group.tokens.shape[-1] == token_dim for group in prefix_groups])
        assert all([group.tokens.shape[-1] == token_dim for group in timestep_groups])

        input_tokens = self.assemble_input_tokens(prefix_groups, timestep_groups)
        
        if self.attention_mask is None:
            attention_mask = self.generate_attention_mask(prefix_groups, timestep_groups)
            attention_mask = attention_mask.repeat(1, self.transformer_kwargs['num_attention_heads'], 1, 1)
            attention_mask = attention_mask.reshape((-1, attention_mask.shape[2], attention_mask.shape[3]))
            attention_mask = ~attention_mask
            self.attention_mask = attention_mask.detach()
        output = self.transformer(input_tokens, self.attention_mask, train=train)

        all_prefix_outputs, all_timestep_outputs = self.split_output_tokens(output, prefix_groups, timestep_groups)
        return all_prefix_outputs, all_timestep_outputs
    
    def assemble_input_tokens(
        self,
        prefix_groups: Sequence[PrefixGroupPt],
        timestep_groups: Sequence[TimestepGroupPt],
    ):
        """
        - Concatenate all timestep tokens together
        - Fold horizon dim into token sequence dim.
        - Prepend task tokens.

        Returns:
            tokens: A tensor of shape (batch, total_tokens, token_embedding_size)
        """
        if len(prefix_groups) > 0:
            all_prefix_tokens = torch.cat([group.tokens for group in prefix_groups], dim=1)
        else:
            all_prefix_tokens = torch.zeros(
                (timestep_groups[0].tokens.shape[0], 0, timestep_groups[0].tokens.shape[-1]),
                dtype=torch.float32, device=timestep_groups[0].tokens.device
            )

        all_timestep_tokens = torch.cat([group.tokens for group in timestep_groups], dim=2)
        all_timestep_tokens = einops.rearrange(all_timestep_tokens, "batch horizon n_tokens d -> batch (horizon n_tokens) d")
        tokens = torch.cat([all_prefix_tokens, all_timestep_tokens], dim=1)
        return tokens

    def split_output_tokens(
        self,
        output_tokens: torch.Tensor,
        prefix_groups: Sequence[PrefixGroupPt],
        timestep_groups: Sequence[TimestepGroupPt],
    ):
        """Reverses the process of assemble_input_tokens."""

        horizon = timestep_groups[0].tokens.shape[1]
        tokens_per_prefix_group = [group.tokens.shape[1] for group in prefix_groups]
        n_prefix_tokens = sum(tokens_per_prefix_group)

        prefix_embeddings, timestep_embeddings = torch.split(output_tokens, [n_prefix_tokens, output_tokens.shape[1] - n_prefix_tokens], dim=1)

        # Process prefix group outputs
        if len(prefix_groups) > 0:
            prefix_embeddings_split = split_tokens(prefix_embeddings, tokens_per_prefix_group, axis=1)
            all_prefix_outputs = [
                PrefixGroupPt(embeddings, group.mask, group.name, group.attention_rules)
                for group, embeddings in zip(prefix_groups, prefix_embeddings_split)
            ]
        else:
            all_prefix_outputs = []

        # Process timestep group outputs
        timestep_embeddings = einops.rearrange(
            timestep_embeddings,
            "batch (horizon n_tokens) d -> batch horizon n_tokens d",
            horizon=horizon,
        )

        tokens_per_timestep_group = [group.tokens.shape[2] for group in timestep_groups]
        timestep_embeddings_split = split_tokens(timestep_embeddings, tokens_per_timestep_group, axis=2)

        all_timestep_outputs = [
            TimestepGroupPt(embeddings, group.mask, group.name, group.attention_rules)
            for group, embeddings in zip(timestep_groups, timestep_embeddings_split)
        ]
        return all_prefix_outputs, all_timestep_outputs

    def generate_attention_mask(
        self,
        prefix_groups: Sequence[PrefixGroupPt],
        timestep_groups: Sequence[TimestepGroupPt],
    ):
        """
        Args:
            prefix_groups: A list of PrefixGroupPt objects.
            timestep_groups: A list of TimestepGroupPt objects.

        Returns:
            attention_mask: A boolean mask of shape (batch, 1, total_tokens, total_tokens)

        We use the attention rules specified by each group to determine the transformer attention mask.
        We then combine this with the padding mask to ensure that padding tokens are not attended to.
        """

        if self.enforce_causal:
            self.verify_causality(prefix_groups, timestep_groups)

        if not self.use_correct_attention:
            # No longer used in new models, but keeping for backward compatibility w/ models released in DEcember
            logging.warning(
                "Using old attention computation from released December models."
            )
            side = "left"
        else:
            side = "right"

        def _get_position(i, tokens_per_elem):
            return np.searchsorted(np.cumsum(tokens_per_elem), i, side=side)

        horizon = timestep_groups[0].tokens.shape[1]
        tokens_per_prefix_group = [group.tokens.shape[1] for group in prefix_groups]
        tokens_per_timestep_group = [group.tokens.shape[2] for group in timestep_groups]

        tokens_for_prefix = sum(tokens_per_prefix_group)
        tokens_per_time_step = sum(tokens_per_timestep_group)

        total_tokens = tokens_for_prefix + tokens_per_time_step * horizon
        attention_mask = torch.zeros((total_tokens, total_tokens), dtype=torch.bool, device=timestep_groups[0].tokens.device)

        def get_token_metadata(i):
            if i < tokens_for_prefix:
                position = _get_position(i, tokens_per_prefix_group)
                return TokenMetadataPt.create(prefix_groups[position], timestep=-1)

            i -= tokens_for_prefix
            timestep, i = divmod(i, tokens_per_time_step)
            position = _get_position(i, tokens_per_timestep_group)
            return TokenMetadataPt.create(timestep_groups[position], timestep)

        for i in range(total_tokens):  # Token attending
            for j in range(total_tokens):  # Token being attended to
                metadata_i = get_token_metadata(i)
                metadata_j = get_token_metadata(j)
                mask = int(metadata_i.should_attend_to(metadata_j))
                attention_mask[i, j] = mask

        pad_attention_mask = self.generate_pad_attention_mask(
            prefix_groups, timestep_groups
        )
        attention_mask = torch.logical_and(torch.tensor(attention_mask), pad_attention_mask)
        return attention_mask

    def generate_pad_attention_mask(
        self,
        prefix_groups: Sequence[PrefixGroupPt],
        timestep_groups: Sequence[TimestepGroupPt],
    ):
        """
        Generate a nn.MultiHeadDotProductAttention mask that ignores padding by masks from all timestep groups,
        unfold the horizon dim, and concatenate with all the prefix group masks.
        We broadcast this (batch, total_tokens) mask to the requisite (batch, 1, total_tokens, total_tokens).
        """
        batch_size, horizon = timestep_groups[0].tokens.shape[:2]
        if len(prefix_groups) > 0:
            prefix_pad_mask = torch.cat([group.mask for group in prefix_groups], dim=1)
        else:
            prefix_pad_mask = torch.zeros((batch_size, 0), dtype=torch.bool, device=timestep_groups[0].tokens.device)
            
        timestep_pad_mask = torch.cat([group.mask for group in timestep_groups], dim=2)
        timestep_pad_mask = einops.rearrange(
            timestep_pad_mask,
            "batch horizon n_tokens -> batch (horizon n_tokens)",
        )
        pad_mask = torch.cat([prefix_pad_mask, timestep_pad_mask], dim=1)
        # pad_mask has shape (batch, total_tokens)
        pad_mask = pad_mask.unsqueeze(1).unsqueeze(2).expand(batch_size, 1, pad_mask.shape[1], pad_mask.shape[1])
        return pad_mask

    def verify_causality(
        self,
        prefix_groups: Sequence[PrefixGroupPt],
        timestep_groups: Sequence[TimestepGroupPt],
    ):
        """Ensures that no token can attend to another token in a future timestep."""
        # First verify that prefix group isn't attending to any timestep group
        for prefix_group in prefix_groups:
            for ts_group in timestep_groups:
                rule = find_match(
                    prefix_group.attention_rules, ts_group.name, AttentionRule.NEVER
                )
                assert (
                    prefix_group.attention_rules.get(ts_group.name, AttentionRule.NEVER)
                    == AttentionRule.NEVER
                ), f"Causality broken! Prefix group {prefix_group.name} is attending to timestep group {ts_group.name}"

        # Next, make sure that nothing is attending to future timesteps
        for group in prefix_groups + timestep_groups:
            for other_group in prefix_groups + timestep_groups:
                rule = find_match(
                    group.attention_rules, other_group.name, AttentionRule.NEVER
                )
                assert (
                    rule != AttentionRule.ALL
                ), "Causality broken! WhenToAttend.ALL attends to future timesteps too."

    def pretty_print_attention_mask(
        self,
        prefix_groups: Sequence[PrefixGroupPt],
        timestep_groups: Sequence[TimestepGroupPt],
    ):
        """
        Visualizes the attention patterns for each token group for debugging purposes.
        """
        logging.warning("Prefix groups:")
        for prefix_group in prefix_groups:
            logging.warning(
                "PrefixGroup(name=%s, shape=%s, attends_to=%s)",
                prefix_group.name,
                prefix_group.tokens.shape,
                str(prefix_group.attention_rules),
            )
        logging.warning("Timestep groups:")
        for timestep_group in timestep_groups:
            logging.warning(
                "TimestepGroup(name=%s, shape=%s, attends_to=%s)",
                timestep_group.name,
                timestep_group.tokens.shape,
                str(timestep_group.attention_rules),
            )

        import rich

        horizon = timestep_groups[0].tokens.shape[1]

        all_metadatas: Sequence[TokenMetadataPt] = []
        column_names = []

        for prefix_group in prefix_groups:
            column_names.append(
                f"{prefix_group.name} ({prefix_group.tokens.shape[1]} tokens)"
            )
            all_metadatas.append(TokenMetadataPt.create(prefix_group, timestep=-1))

        for ts in range(horizon):
            for timestep_group in timestep_groups:
                column_names.append(
                    f"t={ts} {timestep_group.name} ({timestep_group.tokens.shape[2]} tokens) "
                )
                all_metadatas.append(TokenMetadataPt.create(timestep_group, timestep=ts))

        rows = []
        for j in range(len(all_metadatas)):  # Token being attended to
            row = [column_names[j]]
            for i in range(len(all_metadatas)):  # Token attending
                metadata_i = all_metadatas[i]
                metadata_j = all_metadatas[j]
                mask = int(metadata_i.should_attend_to(metadata_j))
                row.append("x" if mask else " ")
            rows.append(row)

        table = rich.table.Table(
            rich.table.Column(no_wrap=True),
            *column_names,
            title="Attention Mask",
            show_header=True,
            show_lines=True,
        )
        for row in rows:
            table.add_row(*row)
        rich.print(table)
