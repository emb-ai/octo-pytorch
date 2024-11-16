from functools import partial
import json
import logging
from typing import Any, Optional, Tuple, Dict
from octo.utils.typing import Config

import torch
import torch.nn as nn
from torch._prims_common import DeviceLikeType

from torchvision.transforms import ToTensor
import numpy as np
import tensorflow as tf
import jax
import jax.numpy as jnp
import flax
import orbax.checkpoint

from pathlib import Path
import pickle

from octo.data.utils.data_utils import NormalizationType
from octo.data.utils.text_processing import TextProcessor
from octo.model.components.action_heads import ActionHead
from octo.model.octo_module_pt import OctoModulePt
from octo.utils.spec import ModuleSpec
from octo.utils.typing import Sequence
from octo.model.octo_module import OctoModule
from octo.utils.train_utils_pt import _flatten_dict, _jax_config_to_pt_config, _np2pt, tree_map


class OctoModelPt(nn.Module):
    """Recommended way of interacting with Octo models.

    TODO

    Usage for finetuning:

       TODO

    Usage for pretraining:

       TODO

    See full usage examples in train.py and finetune.py.

    """
    def __init__(
        self,
        module: OctoModulePt,
        text_processor: TextProcessor,
        config,
        example_batch,
        dataset_statistics
    ):
        super().__init__()
        self.module = module
        self.text_processor = text_processor
        self.config = config
        self.example_batch = example_batch
        self.dataset_statistics = dataset_statistics
        self.to_tensor = ToTensor()
    
    def create_tasks(
        self, 
        goals: Optional[Dict] = None, 
        texts: Optional[Sequence[str]] = None, 
        device: Optional[DeviceLikeType] = None
    ):
        """Creates tasks dict from goals and texts.

        Args:
            goals: if not None, dict of arrays with shape (batch_size, *)
            texts: if not None, list of texts of length batch_size

        Omit images to run the language-conditioned model, and omit texts to run the
        goal-conditioned model.
        """
        
        assert goals is not None or texts is not None
        tasks = {"pad_mask_dict": {}}
        if goals is not None:
            goals_pt = {
                key: torch.stack([
                    self.to_tensor(goal_i).to(device) for goal_i in goal
                ], dim=0)  for key, goal in goals.item()
            }
            
            tasks.update(goals_pt)
            tasks["pad_mask_dict"].update(
                {k: torch.ones(v.shape[:1], dtype=torch.bool, device=device) for k, v in goals.items()}
            )
        else:
            batch_size = len(texts)
            tasks.update(
                {
                    k: torch.zeros((batch_size, *v.shape[1:]), dtype=v.dtype, device=device)
                    for k, v in self.example_batch["task"].items()
                    if k not in ("pad_mask_dict", "language_instruction")
                }
            )
            tasks["pad_mask_dict"].update(
                {
                    k: torch.zeros(batch_size, dtype=torch.bool, device=device)
                    for k in tasks.keys()
                    if k != "pad_mask_dict"
                }
            )

        if texts is not None:
            assert self.text_processor is not None
            tasks["language_instruction"] = texts
            tasks["pad_mask_dict"]["language_instruction"] = torch.ones(
                len(texts), dtype=torch.bool, device=device
            )
        else:
            batch_size = goals[0].shape[0]
            tasks["language_instruction"] = [""] * batch_size
            tasks["pad_mask_dict"]["language_instruction"] = np.zeros(
                batch_size, dtype=bool
            )

        if self.text_processor is not None:
            tasks["language_instruction"] = _np2pt(self.text_processor.encode(
                tasks["language_instruction"]
            ), device)
        else:
            del tasks["language_instruction"]

        _verify_shapes(tasks, "tasks", self.example_batch["task"], starting_dim=1)
        return tasks 
        
    
    @classmethod
    def load_pretrained_from_jax(
        cls,
        checkpoint_path: str,
        step: Optional[int] = None,
    ) -> "OctoModelPt":
        """Loads a model from a checkpoint that was saved via `save_pretrained`.

        Args:
            checkpoint_path (str): A path to either a directory of checkpoints or a single checkpoint.
            step (int, optional): If multiple checkpoints are present, which one to load. Defaults to the latest.
        """
        if checkpoint_path.startswith("hf://"):
            if step:
                raise ValueError(
                    "You can't set config['pretrained_step'] when loading from HuggingFace."
                )
            checkpoint_path = _download_from_huggingface(
                checkpoint_path.removeprefix("hf://")
            )

        # load config
        with tf.io.gfile.GFile(
            tf.io.gfile.join(checkpoint_path, "config.json"), "r"
        ) as f:
            config = json.load(f)

        # shim to support old configs
        if "pred_horizon" in config["model"]["heads"]["action"]["kwargs"]:
            config["model"]["heads"]["action"]["kwargs"]["action_horizon"] = config[
                "model"
            ]["heads"]["action"]["kwargs"].pop("pred_horizon")

        # load dataset statistics
        with tf.io.gfile.GFile(
            tf.io.gfile.join(checkpoint_path, "dataset_statistics.json"), "r"
        ) as f:
            dataset_statistics = json.load(f)
            dataset_statistics = jax.tree_map(
                np.array, dataset_statistics, is_leaf=lambda x: not isinstance(x, dict)
            )

        # create model def (an OctoModule)
        module = OctoModule.create(**config["model"])
        
        config_pt = config
        config_pt['model'] = _jax_config_to_pt_config(config_pt['model'])
        module_pt = OctoModulePt.create(**config_pt["model"])
        # infer params shape without actually doing any computation

        example_batch = load_np_example_batch(checkpoint_path)
        
        init_args = (
            example_batch["observation"],
            example_batch["task"],
            example_batch["observation"]["timestep_pad_mask"],
        )
        params_shape = jax.eval_shape(
            partial(module.init, train=False), jax.random.PRNGKey(0), *init_args
        )["params"]
        # restore params, checking to make sure the shape matches
        checkpointer = orbax.checkpoint.CheckpointManager(
            checkpoint_path, orbax.checkpoint.PyTreeCheckpointer()
        )
        step = step if step is not None else checkpointer.latest_step()
        params = checkpointer.restore(step, params_shape)

        if config["text_processor"] is not None:
            text_processor = ModuleSpec.instantiate(config["text_processor"])()
        else:
            text_processor = None
        
        example_batch = _np2pt(example_batch)
        
        dataset_statistics = _np2pt(dataset_statistics)
        
        
        octo_model = cls(
            module = module_pt,
            text_processor = text_processor,
            config = config_pt,
            example_batch = example_batch,
            dataset_statistics = dataset_statistics
        )
        uninitialized_params, unused_jax_params = octo_model.module.load_jax_weights(params)

        if len(uninitialized_params) == 0:
            logging.warning(f'All parameters were initialized from {checkpoint_path}!')
        else:
            logging.warning(f'Following parameters were not initialized from {checkpoint_path}: {uninitialized_params}')
        
        if len(unused_jax_params) == 0:
            logging.warning(f'All weights in {checkpoint_path} were involved in initialization!')
        else:
            logging.warning(f'Following parameters in {checkpoint_path} were not involved in initialization: {unused_jax_params}')
            
        
        return {
                'octo_model': octo_model,
                'params': params,
                'uninitialized_params': uninitialized_params,
                'unused_jax_params': unused_jax_params 
        }

    @staticmethod
    def load_config_and_meta_from_jax(checkpoint_path, return_jax_meta=False):
        if checkpoint_path.startswith("hf://"):
            checkpoint_path = _download_from_huggingface(
                checkpoint_path.removeprefix("hf://")
            )

        # load config
        with tf.io.gfile.GFile(
            tf.io.gfile.join(checkpoint_path, "config.json"), "r"
        ) as f:
            config = json.load(f)

        # shim to support old configs
        if "pred_horizon" in config["model"]["heads"]["action"]["kwargs"]:
            config["model"]["heads"]["action"]["kwargs"]["action_horizon"] = config[
                "model"
            ]["heads"]["action"]["kwargs"].pop("pred_horizon")

        # load dataset statistics
        dataset_statistics = None
        with tf.io.gfile.GFile(
            tf.io.gfile.join(checkpoint_path, "dataset_statistics.json"), "r"
        ) as f:
            dataset_statistics = json.load(f)
        
        
        example_batch = load_np_example_batch(checkpoint_path)
        
        if config["text_processor"] is not None:
            text_processor = ModuleSpec.instantiate(config["text_processor"])()
        else:
            text_processor = None
        
        if return_jax_meta:
            return {
                'config': config, 
                'example_batch': example_batch, 
                'dataset_statistics': dataset_statistics, 
                'text_processor': text_processor
            }
         
        config_pt = config
        config_pt['model'] = _jax_config_to_pt_config(config_pt['model'])

        example_batch_pt = _np2pt(example_batch)
        
        dataset_statistics_pt = _np2pt(dataset_statistics)
            
        return {
                'config': config_pt, 
                'example_batch': example_batch_pt, 
                'dataset_statistics': dataset_statistics_pt, 
                'text_processor': text_processor
            }
        
    
    def load_weights_from_jax(self, checkpoint_path, step: Optional[int] = None,):
        if checkpoint_path.startswith("hf://"):
            if step:
                raise ValueError(
                    "You can't set config['pretrained_step'] when loading from HuggingFace."
                )
            checkpoint_path = _download_from_huggingface(
                checkpoint_path.removeprefix("hf://")
            )

        # load config
        with tf.io.gfile.GFile(
            tf.io.gfile.join(checkpoint_path, "config.json"), "r"
        ) as f:
            config = json.load(f)

        # shim to support old configs
        if "pred_horizon" in config["model"]["heads"]["action"]["kwargs"]:
            config["model"]["heads"]["action"]["kwargs"]["action_horizon"] = config[
                "model"
            ]["heads"]["action"]["kwargs"].pop("pred_horizon")

        # load example batch
        with tf.io.gfile.GFile(
            tf.io.gfile.join(checkpoint_path, "example_batch.msgpack"), "rb"
        ) as f:
            example_batch = flax.serialization.msgpack_restore(f.read())
        # shim for migrating from "tasks" to "task"
        if "tasks" in example_batch:
            example_batch["task"] = example_batch.pop("tasks")

        logging.debug(
            "Model was trained with observations: %s",
            flax.core.pretty_repr(
                jax.tree_map(jnp.shape, example_batch["observation"])
            ),
        )
        logging.debug(
            "Model was trained with tasks: %s",
            flax.core.pretty_repr(jax.tree_map(jnp.shape, example_batch["task"])),
        )

        # load dataset statistics
        with tf.io.gfile.GFile(
            tf.io.gfile.join(checkpoint_path, "dataset_statistics.json"), "r"
        ) as f:
            dataset_statistics = json.load(f)
            dataset_statistics = jax.tree_map(
                np.array, dataset_statistics, is_leaf=lambda x: not isinstance(x, dict)
            )

        # create model def (an OctoModule)
        module = OctoModule.create(**config["model"])
        # infer params shape without actually doing any computation

        # shim for old checkpoints
        if "timestep_pad_mask" not in example_batch["observation"]:
            example_batch["observation"]["timestep_pad_mask"] = example_batch[
                "observation"
            ]["pad_mask"]

        init_args = (
            example_batch["observation"],
            example_batch["task"],
            example_batch["observation"]["timestep_pad_mask"],
        )
        params_shape = jax.eval_shape(
            partial(module.init, train=False), jax.random.PRNGKey(0), *init_args
        )["params"]
        # restore params, checking to make sure the shape matches
        checkpointer = orbax.checkpoint.CheckpointManager(
            checkpoint_path, orbax.checkpoint.PyTreeCheckpointer()
        )
        step = step if step is not None else checkpointer.latest_step()
        params = checkpointer.restore(step, params_shape)

        uninitialized_params, unused_jax_params = self.module.load_jax_weights(params)
        
        if len(uninitialized_params) == 0:
            logging.warning(f'All parameters were initialized from {checkpoint_path}!')
        else:
            logging.warning(f'Following parameters were not initialized from {checkpoint_path}: {uninitialized_params}')
        
        if len(unused_jax_params) == 0:
            logging.warning(f'All weights in {checkpoint_path} were involved in initialization!')
        else:
            logging.warning(f'Following parameters in {checkpoint_path} were not involved in initialization: {unused_jax_params}')
            
        return uninitialized_params, unused_jax_params
    
    
    def forward(
        self,
        observations: dict,
        tasks: dict,
        timestep_pad_mask: torch.tensor,
        train: bool = False,
        transformer_only=False,
        **kwargs
    ):
        """Runs the transformer, but does shape checking on the inputs.

        Args:
            observations: dictionary of arrays of shape (batch_size, window_size, *shape).
                Shape must be consistent with self.example_batch["observation"]
            tasks: dict of tasks of shape (batch_size, *shape)
                Shape must be consistent with self.example_batch["task"]
            timestep_pad_mask: (batch_size, window_size) Boolean mask that is False when the timestep corresponds to padding
            train: whether to run in train mode
        """
        _verify_shapes(
            observations,
            "observations",
            self.example_batch["observation"],
            starting_dim=2,
        )
        _verify_shapes(tasks, "tasks", self.example_batch["task"], starting_dim=1)

        return self.module(
            observations,
            tasks,
            timestep_pad_mask,
            train=train,
            transformer_only=transformer_only,
            **kwargs
        )
        
    # def to(self, device):
    #     self.device = device
    #     self.module.to(device)
    
    def sample_actions(
        self,
        observations: Dict,
        tasks: Dict,
        unnormalization_statistics: Optional[Dict] = None,
        normalization_type: NormalizationType = NormalizationType.NORMAL,
        timestep_pad_mask: Optional[torch.tensor] = None,
        train: bool = False,
        argmax: bool = False,
        sample_shape: Tuple[int, ...] = (),
        temperature: float = 1.0,
    ):
        """Samples actions from the model. See `action_heads.py` for more info.

        Args:
            TODO
        Returns:
            TODO
        """
        if timestep_pad_mask is None:
            timestep_pad_mask = observations["timestep_pad_mask"]
        transformer_outputs, head_outputs = self(
            observations, tasks, timestep_pad_mask, train=train,
            transformer_only=True
        )
        action_head = self.module.heads["action"]
        action = action_head.predict_action(
            transformer_outputs,
            train=train,
            argmax=argmax,
            sample_shape=sample_shape,
            temperature=temperature,
            embodiment_action_dim=len(unnormalization_statistics["mean"])
            if unnormalization_statistics is not None
            else None,
        )
        action = action.detach().cpu()
        
        if unnormalization_statistics is not None:
            if normalization_type == NormalizationType.NORMAL:
                mask = unnormalization_statistics.get(
                    "mask",
                    torch.ones_like(unnormalization_statistics["mean"].shape, dtype=bool),
                )
                
                unnorm_std = torch.from_numpy(unnormalization_statistics["std"])
                unnorm_mean = torch.from_numpy(unnormalization_statistics["mean"])
                unnorm_p01 = torch.from_numpy(unnormalization_statistics["p01"])
                unnorm_p99 = torch.from_numpy(unnormalization_statistics["p99"])
                
                action = action[..., : len(mask)]
                action = torch.where(
                    mask,
                    (action * unnorm_std)
                    + unnorm_mean,
                    action,
                )
            elif normalization_type == NormalizationType.BOUNDS:
                mask = unnormalization_statistics.get(
                    "mask", jnp.ones_like(unnorm_p01, dtype=bool)
                )
                action = action[..., : len(mask)]
                action = jnp.where(
                    mask,
                    (action + 1) * (unnorm_p99 - unnorm_p01) / 2 + unnorm_p01, 
                    action,
                )
            else:
                raise ValueError(f"Unknown normalization type: {normalization_type}")
        return action
        

    @classmethod
    def load_pretrained(
        cls,
        checkpoint_path: str,
        step: int,
        optimizer: torch.optim.Optimizer = None,
    ) -> "OctoModelPt":
        """Loads a model from a checkpoint that was saved via `save_pretrained`.

        Args:
            checkpoint_path (str): A path to either a directory of checkpoints or a single checkpoint.
            step (int, optional): If multiple checkpoints are present, which one to load. Defaults to the latest.
        """
        checkpoint_path = Path(checkpoint_path)
        with open(checkpoint_path / "config.json", 'r') as f:
            config = json.load(f)
        
        with open(checkpoint_path / "example_batch.pickle", 'rb') as f:
            example_batch = pickle.load(f)
        
        with open(checkpoint_path / "dataset_statistics.json", 'r') as f:
            dataset_statistics = json.load(f)
            dataset_statistics = tree_map(
                np.array, dataset_statistics, is_leaf=lambda x: not isinstance(x, dict)
            )
            dataset_statistics = _np2pt(dataset_statistics)
        if config["text_processor"] is not None:
            text_processor = ModuleSpec.instantiate(config["text_processor"])()
        else:
            text_processor = None
        
        module = OctoModulePt.create(**config["model"])
        
        
        octo_model = cls(
            module = module,
            text_processor = text_processor,
            config = config,
            example_batch = example_batch,
            dataset_statistics = dataset_statistics
        )
        
        weights_path = checkpoint_path / f'{step}' / 'weights.pth'
        checkpoint = torch.load(weights_path, weights_only=True)

        octo_model.load_state_dict(checkpoint['state_dict'])
        ret = {}
        ret['octo_model'] = octo_model
        if optimizer is not None:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            ret['optimizer_state_dict'] = optimizer
        return ret

    def save_pretrained(
        self,
        step: int,
        checkpoint_path: str,
        optimizer = None,
    ):
        # save config
        checkpoint_path = Path(checkpoint_path)
        checkpoint_path.mkdir(exist_ok = True)
        if not Path.exists(checkpoint_path / "config.json"):
            with open(checkpoint_path / "config.json", 'w') as f:
                json.dump(self.config, f)
        
        # save example batch
        if not Path.exists(checkpoint_path / "example_batch.pickle"):
            with open(checkpoint_path / "example_batch.pickle", 'wb') as f:
                pickle.dump(self.example_batch, f)
        
        # save dataset statistics
            
        if not Path.exists(checkpoint_path / "dataset_statistics.json"):
            with open(checkpoint_path / "dataset_statistics.json", 'w') as f:
                json.dump(
                    tree_map(lambda x: x.tolist(), self.dataset_statistics), 
                    f
                )
        
        #save wights
        weights_path = checkpoint_path / f'{step}'
        weights_path.mkdir(exist_ok = True)
        save_dict = {
            'state_dict': self.state_dict(),
        }
        if optimizer:
            save_dict['optimizer_state_dict'] = optimizer.state_dict()
        torch.save(save_dict, weights_path / 'weights.pth')

        
    
    @classmethod
    def from_config(
        cls,
        config: Config,
        example_batch: Dict,
        text_processor: Optional[Any] = None,
        verbose: bool = False,
        dataset_statistics: Optional[Dict] = None,
    ):
        """Initializes a model with a fresh set of weights from a given config + example_batch.

        Args:
            config (Dict[str, Any]): Config dict. The only required key is "model", but other configuration
                may be saved for posterity.
            example_batch (Dict[str, Any]): Example batch.
            text_processor (Any, optional): Preprocessor for text inputs.
            verbose (bool, optional): Whether to print out a summary of the model.
            rng (Optional[PRNGKey], optional): RNG key for initializing the model.
            dataset_statistics (Optional[Dict[str, Any]], optional): Dataset statistics.
        """
        module = OctoModulePt.create(**config["model"])
        
        return cls(
            module=module,
            text_processor=text_processor,
            example_batch=example_batch,
            config=config,
            dataset_statistics=dataset_statistics,
        )


    def get_pretty_spec(self):
        """Brief summary of the model's expected inputs and outputs."""
        # TODO: generalize this to print out proprio when it is being tokenized
        pass

def load_np_example_batch(checkpoint_path):
    if checkpoint_path.startswith("hf://"):
        checkpoint_path = _download_from_huggingface(
            checkpoint_path.removeprefix("hf://")
        )
        
    # load example batch
    with tf.io.gfile.GFile(
        tf.io.gfile.join(checkpoint_path, "example_batch.msgpack"), "rb"
    ) as f:
        example_batch = flax.serialization.msgpack_restore(f.read())
    # shim for migrating from "tasks" to "task"
    if "tasks" in example_batch:
        example_batch["task"] = example_batch.pop("tasks")
    # shim for old checkpoints
    if "timestep_pad_mask" not in example_batch["observation"]:
        example_batch["observation"]["timestep_pad_mask"] = example_batch[
            "observation"
        ]["pad_mask"]
    return example_batch

def _verify_shapes(
    data,
    name: str,
    example_data,
    starting_dim: int = 0,
    strict: bool = False,
    raise_error: bool = True,
    silent: bool = False,
):
    weak_fail, fail = False, False
    data_flat = _flatten_dict(data)
    example_data_flat = _flatten_dict(example_data)

    # Check that all elements are present
    if set(data_flat.keys()) != set(example_data_flat.keys()):
        if not silent:
            extra = set(data_flat.keys()) - set(example_data_flat.keys())
            if extra:
                logging.warning(
                    "'%s' contains extra items compared to example_batch: %s",
                    name,
                    {"/".join(x) for x in extra},
                )
            missing = set(example_data_flat.keys()) - set(data_flat.keys())
            if missing:
                logging.warning(
                    "'%s' is missing items compared to example_batch: %s",
                    name,
                    {"/".join(x) for x in missing},
                )
        weak_fail = True

    mismatched_keys = {
        k: f"{data_flat[k].shape} != {example_data_flat[k].shape}"
        for k in data_flat
        if k in example_data_flat
        and data_flat[k].shape[starting_dim:]
        != example_data_flat[k].shape[starting_dim:]
    }
    if mismatched_keys:
        if not silent:
            logging.error(
                "'%s' contains mismatched shapes compared to example_batch: %s",
                name,
                flax.core.pretty_repr(
                    {"/".join(k): v for k, v in mismatched_keys.items()}
                ),
            )
        fail = True

    if raise_error and (fail or (weak_fail and strict)):
        raise AssertionError(f"{name} does not match example batch.")

    return weak_fail or fail




def _download_from_huggingface(huggingface_repo_id: str):
    import huggingface_hub

    folder = huggingface_hub.snapshot_download(huggingface_repo_id)
    return folder
    return folder

