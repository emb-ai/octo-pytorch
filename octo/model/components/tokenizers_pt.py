import logging
import re
from typing import Dict, Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.transforms import ToTensor
import numpy as np

from octo.model.components.base import TokenGroupPt
from octo.utils.spec import ModuleSpec
from octo.model.components.jax_pt import FromJaxModel, LayerNormPt
from octo.model.components.transformer_pt import MAPHeadPt
EPS = 1e-6


def generate_proper_pad_mask(
    tokens: torch.Tensor,
    pad_mask_dict: Optional[Dict[str, torch.Tensor]],
    keys: Sequence[str],
) -> torch.Tensor:
    if pad_mask_dict is None:
        logging.warning("No pad_mask_dict found. Nothing will be masked.")
        return torch.ones(tokens.shape[:-1], dtype=torch.bool, device=tokens.device)
    if not all([key in pad_mask_dict for key in keys]):
        logging.warning(
            f"pad_mask_dict missing keys {set(keys) - set(pad_mask_dict.keys())}."
            "Nothing will be masked."
        )
        return torch.ones(tokens.shape[:-1], dtype=torch.bool, device=tokens.device)

    pad_mask = torch.stack([pad_mask_dict[key] for key in keys], dim=-1)
    pad_mask = torch.any(pad_mask, dim=-1)
    pad_mask = pad_mask.unsqueeze(-1).expand(tokens.shape[:-1])
    return pad_mask

class TokenLearnerPt(nn.Module, FromJaxModel):
    """
    Learns to map fixed-length sequence of tokens into specified number of tokens.

    Args:
        num_tokens (int): Number of output tokens.
        bottleneck_dim (int): Size of the hidden layers of the mapping MLP.
        dropout_rate (float): Rate of dropout applied in the mapping MLP. Defaults to no dropout.
    """

    def __init__(self, num_tokens: int, hid_dim: int, max_len: int = 5000):
        super().__init__()
        self.num_tokens = num_tokens
        self.hid_dim = hid_dim
        self.max_len = max_len

        self.layer_norm = LayerNormPt(hid_dim)
        self.map_head = MAPHeadPt(num_readouts=self.num_tokens)
        pos_embed = torch.randn(max_len, self.hid_dim) * 0.02
        self.register_parameter('pos_embed', pos_embed) # trainable

    def forward(self, inputs: torch.Tensor, train: bool = True):
        # Add positional embedding to inputs
        x = inputs + self.pos_embed
        
        # Apply layer normalization
        x = self.layer_norm(x)
        
        # Apply MAPHead
        return self.map_head(x, train=train)
    
    def load_jax_weights(self, params):
        self.layer_norm.load_jax_weights(params['LayerNorm'])
        self.map_head.load_jax_weights(params['MAPHead'])
        self.pos_embed = self._get_param(params['pos_embed'])
        


def regex_match(regex_keys, x):
    return any([re.match(r_key, x) for r_key in regex_keys])


def regex_filter(regex_keys, xs):
    return list(filter(lambda x: regex_match(regex_keys, x), xs))


class ImageTokenizerPt(nn.Module, FromJaxModel):
    """Image tokenizer that encodes image stack into tokens with optional FiLM conditioning."""

    def __init__(
        self,
        encoder: ModuleSpec,
        use_token_learner: bool = False,
        token_learner_hid_dim: int = 512,
        num_tokens: int = 8,
        conditioning_type: str = "none",
        obs_stack_keys: Sequence[str] = ("image_.*", "depth_.*"),
        task_stack_keys: Sequence[str] = tuple(),
        task_film_keys: Sequence[str] = tuple(),
        proper_pad_mask: bool = True,
    ):
        super().__init__()
        self.encoder = encoder
        self.use_token_learner = use_token_learner
        self.token_learner_hid_dim = token_learner_hid_dim
        self.num_tokens = num_tokens
        self.conditioning_type = conditioning_type
        self.obs_stack_keys = obs_stack_keys
        self.task_stack_keys = task_stack_keys
        self.task_film_keys = task_film_keys
        self.proper_pad_mask = proper_pad_mask

        self.encoder_def = ModuleSpec.instantiate(self.encoder)()
        if self.use_token_learner:
            self.token_learner = TokenLearnerPt(num_tokens=self.num_tokens, hid_dim=self.token_learner_hid_dim)
        
        self.output_dim = self.encoder_def.num_features
        
    def load_jax_weights(self, jax_params):
        self.encoder_def.load_jax_weights(jax_params[list(jax_params.keys())[0]])
        if self.use_token_learner:
            self.token_learner.load_jax_weights(jax_params['TokenLearner'])

    def extract_inputs(self, keys, inputs, check_spatial=False):
        extracted_outputs = []
        for key in keys:
            if check_spatial:
                assert len(inputs[key].shape) >= 4
            extracted_outputs.append(inputs[key])
        return torch.cat(extracted_outputs, dim=-1)

    def forward(
        self,
        observations: Dict[str, torch.Tensor],
        tasks: Optional[Dict[str, torch.Tensor]] = None,
        train: bool = True,
    ):
        obs_stack_keys = regex_filter(self.obs_stack_keys, sorted(observations.keys()))
        if len(obs_stack_keys) == 0:
            logging.info(
                f"No image inputs matching {self.obs_stack_keys} were found. "
                "Skipping tokenizer entirely."
            )
            assert self.proper_pad_mask, "Cannot skip unless using proper_pad_mask."
            return None

        # stack all spatial observation and task inputs
        enc_inputs = self.extract_inputs(obs_stack_keys, observations, check_spatial=True)
        if self.task_stack_keys:
            needed_task_keys = regex_filter(self.task_stack_keys, observations.keys())
            # if any task inputs are missing, replace with zero padding (TODO: be more flexible)
            for k in needed_task_keys:
                if k not in tasks:
                    logging.info(
                        f"No task inputs matching {k} were found. Replacing with zero padding."
                    )
                    tasks[k] = torch.zeros_like(observations[k][:, 0])
            task_stack_keys = regex_filter(self.task_stack_keys, sorted(tasks.keys()))
            if len(task_stack_keys) == 0:
                raise ValueError(
                    f"No task inputs matching {self.task_stack_keys} were found."
                )
            task_inputs = self.extract_inputs(task_stack_keys, tasks, check_spatial=True)
            task_inputs = task_inputs.unsqueeze(1).repeat(1, enc_inputs.shape[1], 1, 1, 1)
            enc_inputs = torch.cat([enc_inputs, task_inputs], dim=-1)
        b, t, h, w, c = enc_inputs.shape
        enc_inputs = enc_inputs.view(b * t, h, w, c)
        enc_inputs = enc_inputs.permute((0, 3, 1, 2))
        
        # extract non-spatial FiLM inputs
        encoder_input_kwargs = {}
        if self.task_film_keys:
            film_inputs = self.extract_inputs(self.task_film_keys, tasks)
            film_inputs = film_inputs.unsqueeze(1).repeat(1, t, 1)
            encoder_input_kwargs.update(
                {"cond_var": film_inputs.view(b * t, -1)}
            )

        # run visual encoder
        image_tokens = self.encoder_def(enc_inputs, **encoder_input_kwargs)
        image_tokens = image_tokens.reshape((image_tokens.shape[0], image_tokens.shape[1], -1))
        image_tokens = image_tokens.reshape((b, t, image_tokens.shape[1], image_tokens.shape[2]))
        image_tokens = image_tokens.permute((0, 1, 3, 2))
        # image_tokens = image_tokens.reshape(b, t, -1, image_tokens.shape[1])

        if self.use_token_learner:
            raise NotImplementedError
            # image_tokens = self.token_learner(image_tokens, train=train)

        if self.proper_pad_mask:
            pad_mask = generate_proper_pad_mask(
                image_tokens,
                observations.get("pad_mask_dict", None),
                obs_stack_keys,
            )
        else:
            pad_mask = torch.ones(image_tokens.shape[:-1], dtype=torch.bool, device=image_tokens.device)
        return TokenGroupPt(image_tokens, pad_mask)

class LanguageTokenizerPt(nn.Module, FromJaxModel):
    """
    Language tokenizer that embeds text input IDs into continuous language embeddings. Supports pre-trained HF models.

    Args:
        num_tokens (int): Number of output tokens (not enforced).
        encoder (str, optional): Optional HuggingFace AutoModel name for encoding input IDs.
        finetune_encoder (bool, optional): Optional finetune last layers of the language model.
    """

    def __init__(self, encoder: Optional[str] = None, finetune_encoder: bool = False, proper_pad_mask: bool = True):
        super().__init__()
        # self.num_tokens = num_tokens
        self.encoder = encoder
        self.finetune_encoder = finetune_encoder
        self.proper_pad_mask = proper_pad_mask
        self.hf_model = None
        if self.encoder is not None:
            from transformers import AutoConfig, AutoModel, T5EncoderModel

            config = AutoConfig.from_pretrained(self.encoder)
            if "t5" in self.encoder:
                self.hf_model = T5EncoderModel.from_pretrained(f"google-t5/{self.encoder}")

            else:
                self.hf_model = AutoModel.from_config(config)
        self.output_dim = self.hf_model.config.d_model

    def load_jax_weights(self, params: None):
        pass
    
    def forward(
        self,
        observations: Dict[str, torch.Tensor],
        tasks: Optional[Dict[str, torch.Tensor]] = None,
        train: bool = True,
    ):
        if tasks is None or "language_instruction" not in tasks:
            logging.warning("No language inputs found. Skipping tokenizer entirely.")
            assert self.proper_pad_mask, "Cannot skip unless using proper pad mask."
            return None

        if not isinstance(tasks["language_instruction"], torch.Tensor):
            assert self.encoder is not None, "Received language tokens but no encoder specified."
            tokens = self.hf_model(**tasks["language_instruction"]).last_hidden_state
        else:
            # add a # tokens dimension to language
            if tasks["language_instruction"].dim() == 2:
                tokens = tasks["language_instruction"].unsqueeze(1)
            else:
                tokens = tasks["language_instruction"]

        if not self.finetune_encoder:
            tokens = tokens.detach()

        # TODO: incorporate padding info from language tokens here too
        if self.proper_pad_mask:
            pad_mask = generate_proper_pad_mask(
                tokens,
                tasks.get("pad_mask_dict", None),
                ("language_instruction",),
            )
        else:
            pad_mask = torch.ones(tokens.shape[:-1], dtype=torch.bool, device=tokens.device)

        return TokenGroupPt(tokens, pad_mask)


# class BinTokenizer(nn.Module):
#     """
#     Tokenizes continuous inputs via dimension-wise binning in given range.

#     Args:
#         n_bins (int): Number of discrete bins per dimension.
#         bin_type (str): Type of binning. ['uniform', 'normal' = Gaussian]
#         low (float): Lower bound for bin range.
#         high (float): Upper bound for bin range.
#     """

#     n_bins: int = 256
#     bin_type: str = "uniform"
#     low: float = 0
#     high: float = 1

#     def setup(self):
#         if self.bin_type == "uniform":
#             self.thresholds = jnp.linspace(self.low, self.high, self.n_bins + 1)
#         elif self.bin_type == "normal":
#             self.thresholds = norm.ppf(jnp.linspace(EPS, 1 - EPS, self.n_bins + 1))
#         else:
#             raise ValueError(
#                 f"Binning type {self.bin_type} not supported in BinTokenizer."
#             )

#     def __call__(self, inputs):
#         if self.bin_type == "uniform":
#             inputs = jnp.clip(inputs, self.low + EPS, self.high - EPS)
#         inputs = inputs[..., None]
#         token_one_hot = (inputs < self.thresholds[1:]) & (
#             inputs >= self.thresholds[:-1]
#         ).astype(jnp.uint8)
#         output_tokens = jnp.argmax(token_one_hot, axis=-1)
#         return output_tokens

#     def decode(self, inputs):
#         one_hot = jax.nn.one_hot(inputs, self.n_bins)
#         bin_avgs = (self.thresholds[1:] + self.thresholds[:-1]) / 2
#         outputs = jnp.sum(one_hot * bin_avgs, axis=-1)
#         return outputs


# class LowdimObsTokenizer(BinTokenizer):
#     """
#     Tokenizer for non-spatial observations. Optionally discretizes into bins per dimension (see BinTokenizer).

#     Args:
#         obs_keys (Sequence[str]): List of non-spatial keys to concatenate & tokenize. Supports regex.
#         discretize (bool): If True, discretizes inputs per dimension, see BinTokenizer.
#     """

#     obs_keys: Sequence[str] = tuple()
#     discretize: bool = False
#     proper_pad_mask: bool = True

#     def __call__(self, observations, *unused_args, **unused_kwargs):
#         assert self.obs_keys, "Need to specify observation keys to tokenize."
#         if len(regex_filter(self.obs_keys, sorted(observations.keys()))) == 0:
#             logging.warning(
#                 f"No observation inputs matching {self.obs_keys} were found."
#                 "Skipping tokenizer entirely."
#             )
#             assert self.proper_pad_mask, "Cannot skip unless using proper pad mask."
#             return None

#         tokenizer_inputs = []
#         for o_key in self.obs_keys:
#             for key in filter(re.compile(o_key).match, sorted(observations.keys())):
#                 assert (
#                     len(observations[key].shape) == 3
#                 ), f"Only supports non-spatial inputs but {key} has shape {observations[key].shape}."
#                 tokenizer_inputs.append(observations[key])
#         tokenizer_inputs = jnp.concatenate(tokenizer_inputs, axis=-1)
#         if self.discretize:
#             tokenized_inputs = super().__call__(tokenizer_inputs)
#             tokens = jax.nn.one_hot(tokenized_inputs, self.n_bins)
#         else:
#             tokens = tokenizer_inputs[..., None]
#         mask = jnp.ones(tokens.shape[:-1])
#         return TokenGroup(tokens, mask)
