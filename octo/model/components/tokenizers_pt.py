import logging
import re
from typing import Dict, Optional, Sequence
from functools import partial
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.transforms import ToTensor
import numpy as np
from scipy.stats import norm

from octo.model.components.base_pt import TokenGroupPt
from octo.utils.spec import ModuleSpec
from octo.model.components.jax_pt import FromJaxModel, LayerNormPt, ParamNode
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

    def __init__(self, num_tokens: int, hid_dim: int, max_len: int = 256):
        super().__init__()
        self.num_tokens = num_tokens
        self.hid_dim = hid_dim

        self.layer_norm = LayerNormPt(hid_dim, eps=1e-6)
        self.map_head = MAPHeadPt(hid_dim, num_readouts=self.num_tokens)
        pos_embed = torch.randn(max_len, self.hid_dim) * 0.02
        self.pos_embed = nn.Parameter(pos_embed) # trainable

    def forward(self, inputs: torch.Tensor, train: bool = True):
        # Add positional embedding to inputs
        bs, t, length = inputs.shape[:3]
        assert length <= self.pos_embed.shape[0]
        
        pos_embed = self.pos_embed[:length]
        pos_embed = pos_embed.unsqueeze(0).unsqueeze(0).repeat(bs, t, 1, 1)
        
        x = inputs + pos_embed
        
        # Apply layer normalization
        x = self.layer_norm(x)
        
        # Apply MAPHead
        return self.map_head(x, train=train)
    
    def pt_to_jax_args_map(self):
        # {
        # pt_module_name: (load_func, jax_param_key),
        # ...
        # }
        return {
            "layer_norm": ParamNode(submodule=self.layer_norm, jax_param_names='LayerNorm_0'),
            "map_head": ParamNode(submodule=self.map_head, jax_param_names='MAPHead_0'), 
            "pos_embed": ParamNode(load_func=self._set_terminal_param, jax_param_names='pos_embed'), 
        }
        


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

    def pt_to_jax_args_map(self):
        # {
        # pt_module_name: (load_func, jax_param_key),
        # ...
        # }
        pt_to_jax_args = {
            'encoder_def': ParamNode(submodule=self.encoder_def, jax_param_names='SmallStem16_0')
        }
        if self.use_token_learner:
            pt_to_jax_args['token_learner'] = ParamNode(submodule=self.token_learner, jax_param_names='TokenLearner_0')
        return pt_to_jax_args
    
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
            enc_inputs = torch.cat([enc_inputs, task_inputs], dim=2)
        b, t, c, h, w = enc_inputs.shape
        enc_inputs = enc_inputs.view(b * t, c, h, w)
        # enc_inputs = enc_inputs.permute((0, 3, 1, 2))
        
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
            image_tokens = self.token_learner(image_tokens, train=train)

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

    def pt_to_jax_args_map(self):
        # {
        # pt_module_name: (load_func, jax_param_key),
        # ...
        # }
        return {}    
    
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


class BinTokenizerPt(nn.Module, FromJaxModel):
    """
    Tokenizes continuous inputs via dimension-wise binning in given range.

    Args:
        n_bins (int): Number of discrete bins per dimension.
        bin_type (str): Type of binning. ['uniform', 'normal' = Gaussian]
        low (float): Lower bound for bin range.
        high (float): Upper bound for bin range.
    """

    def __init__(
        self,
        n_bins: int = 256,
        bin_type: str = "uniform",
        low: float = 0,
        high: float = 1
    ):
        super().__init__()
        self.n_bins = n_bins
        self.bin_type = bin_type
        self.low = low
        self.high = high

        # Initialize thresholds
        if self.bin_type == "uniform":
            thresholds = torch.linspace(self.low, self.high, self.n_bins + 1)
        elif self.bin_type == "normal":
            # Convert numpy array to torch tensor
            thresholds = torch.from_numpy(
                norm.ppf(np.linspace(EPS, 1 - EPS, self.n_bins + 1))
            ).float()
        else:
            raise ValueError(f"Binning type {self.bin_type} not supported in BinTokenizer.")
        
        # Register thresholds as buffer (non-trainable tensor)
        self.register_buffer('thresholds', thresholds)

    def pt_to_jax_args_map(self):
        # {
        # pt_module_name: (load_func, jax_param_key),
        # ...
        # }
        return {}
    
    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        if self.bin_type == "uniform":
            inputs = torch.clamp(inputs, self.low + EPS, self.high - EPS)
        
        inputs = inputs.unsqueeze(-1)
        token_one_hot = (inputs < self.thresholds[1:]) & (inputs >= self.thresholds[:-1])
        output_tokens = torch.argmax(token_one_hot.to(torch.uint8), dim=-1)
        return output_tokens

    def decode(self, inputs: torch.Tensor) -> torch.Tensor:
        one_hot = nn.functional.one_hot(inputs, self.n_bins).float()
        bin_avgs = (self.thresholds[1:] + self.thresholds[:-1]) / 2
        outputs = torch.sum(one_hot * bin_avgs, dim=-1)
        return outputs


    @property
    def output_dim(self) -> int:
        return self.n_bins
    
class LowdimObsTokenizerPt(BinTokenizerPt):
    """
    Tokenizer for non-spatial observations. Optionally discretizes into bins per dimension.

    Args:
        obs_keys (Sequence[str]): List of non-spatial keys to concatenate & tokenize. Supports regex.
        discretize (bool): If True, discretizes inputs per dimension, see BinTokenizer.
        proper_pad_mask (bool): If True, allows skipping tokenizer when no matching keys found.
    """
    def __init__(
        self,
        obs_keys: Sequence[str] = tuple(),
        discretize: bool = False,
        proper_pad_mask: bool = True,
        **bin_tokenizer_kwargs
    ):
        super().__init__(**bin_tokenizer_kwargs)
        self.obs_keys = obs_keys
        self.discretize = discretize
        self.proper_pad_mask = proper_pad_mask

    def pt_to_jax_args_map(self):
        # {
        # pt_module_name: (load_func, jax_param_key),
        # ...
        # }
        return {}
    
    def forward(self, observations, *unused_args, **unused_kwargs) -> TokenGroupPt:
        assert self.obs_keys, "Need to specify observation keys to tokenize."
        
        matching_keys = regex_filter(self.obs_keys, sorted(observations.keys()))
        if len(matching_keys) == 0:
            logging.warning(
                f"No observation inputs matching {self.obs_keys} were found. "
                "Skipping tokenizer entirely."
            )
            assert self.proper_pad_mask, "Cannot skip unless using proper pad mask."
            return None

        tokenizer_inputs = []
        for o_key in self.obs_keys:
            for key in filter(re.compile(o_key).match, sorted(observations.keys())):
                assert len(observations[key].shape) == 3, \
                    f"Only supports non-spatial inputs but {key} has shape {observations[key].shape}."
                tokenizer_inputs.append(observations[key])
        
        tokenizer_inputs = torch.cat(tokenizer_inputs, dim=-1)
        
        if self.discretize:
            tokenized_inputs = super().forward(tokenizer_inputs)
            tokens = nn.functional.one_hot(tokenized_inputs, self.n_bins).float()
        else:
            tokens = tokenizer_inputs.unsqueeze(-1)
        
        mask = torch.ones(tokens.shape[:-1], dtype=torch.bool, device=tokens.device)
        return TokenGroupPt(tokens, mask)

    @property
    def output_dim(self) -> int:
        return self.n_bins if self.discretize else 1