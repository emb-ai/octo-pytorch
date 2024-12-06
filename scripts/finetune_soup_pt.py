import datetime
from functools import partial
import os
from pathlib import Path
import json


import torch
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import AdamW
from torch.utils.data import DataLoader

from accelerate import PartialState


import numpy as np
from absl import app, flags, logging
import flax
from flax.traverse_util import flatten_dict
import jax
from jax.sharding import Mesh, NamedSharding, PartitionSpec
from ml_collections import config_flags, ConfigDict
import optax
import tensorflow as tf
import tqdm
import wandb

from octo.data.dataset import make_interleaved_dataset
from octo.data.oxe import make_oxe_dataset_kwargs_and_weights
from octo.model.octo_model_pt import OctoModelPt
from octo.utils.jax_utils import initialize_compilation_cache
from octo.utils.spec import ModuleSpec
from octo.utils.train_callbacks import (
    RolloutVisualizationCallback,
    SaveCallback,
    ValidationCallback,
    VisualizationCallback,
)
from octo.utils.train_utils_pt import (
    _jax_config_to_pt_config,
    _np2pt,
    freeze_weights_pt,
    tree_map,
    get_cosine_schedule_with_warmup,
    _flatten_dict
)
from octo.utils.torch_rlds_dataset import TorchRLDSDataset
from octo.utils.train_utils import (
    check_config_diff,
    create_optimizer,
    format_name_with_config,
    merge_params,
    process_text,
    Timer,
    TrainState,
)

try:
    from jax_smi import initialise_tracking  # type: ignore

    initialise_tracking()
except ImportError:
    pass

FLAGS = flags.FLAGS

flags.DEFINE_string("name", "experiment", "Experiment name.")
flags.DEFINE_bool("debug", False, "Debug config (no wandb logging)")

default_config_file = os.path.join(
    os.path.dirname(__file__), "configs/finetune_config.py"
)
config_flags.DEFINE_config_file(
    "config",
    default_config_file,
    "File path to the training hyperparameter configuration.",
    lock_config=False,
)

def _to_device(data, device):
    if isinstance(data, dict):
        return {key: _to_device(val, device) for key, val in data.items()}
    elif isinstance(data, torch.Tensor):
        return data.to(device)



def main(_):
    assert torch.cuda.is_available()
    distributed_state = PartialState()
    
    torch.cuda.set_device(device_id := distributed_state.local_process_index)
    torch.cuda.empty_cache()
    num_devices = distributed_state.num_processes
    
    if distributed_state.is_main_process:
        logging.info(
            f"""
            Octo Finetuning Script
            ======================
            Pretrained model: {FLAGS.config.pretrained_path}
            Finetuning Dataset: {FLAGS.config.dataset_kwargs.oxe_kwargs.data_mix}
            Task Modality: {FLAGS.config.modality}
            Finetuning Mode: {FLAGS.config.finetuning_mode}

            # Devices: {num_devices}
            Batch size: {FLAGS.config.batch_size} ({FLAGS.config.batch_size // num_devices } per device)
            # Steps: {FLAGS.config.num_steps}
        """
        )

    #########
    #
    # Setup Jax Data Parallelism
    #
    #########

    assert (
        FLAGS.config.batch_size % num_devices == 0
    ), f"Batch size ({FLAGS.config.batch_size}) must be divisible by the number of devices ({num_devices})"
    assert (
        FLAGS.config.viz_kwargs.eval_batch_size % num_devices == 0
    ), f"Eval batch size ({FLAGS.config.viz_kwargs.eval_batch_size}) must be divisible by the number of devices ({num_devices})"

    # prevent tensorflow from using GPU memory since it's only used for data loading
    tf.config.set_visible_devices([], "GPU")

    #########
    #
    # Setup WandB
    #
    #########
    if distributed_state.is_main_process:
        name = format_name_with_config(
            FLAGS.name,
            FLAGS.config.to_dict(),
        )
        wandb_id = "{name}_{time}".format(
            name=name,
            time=datetime.datetime.now().strftime("%Y%m%d_%H%M%S"),
        )
        wandb.init(
            config=FLAGS.config.to_dict(),
            id=wandb_id,
            name=name,
            mode="disabled" if FLAGS.debug else None,
            **FLAGS.config.wandb,
        )

    #########
    #
    # Load Pretrained model + optionally modify config
    #
    #########

    meta = OctoModelPt.load_config_and_meta_from_jax(
        FLAGS.config.pretrained_path,
        return_jax_meta=True
    )
    flat_config = flax.traverse_util.flatten_dict(
        meta['config'], keep_empty_nodes=True
    )
    for d_key in flax.traverse_util.flatten_dict(
        FLAGS.config.get("config_delete_keys", ConfigDict()).to_dict()
    ):
        for c_key in list(flat_config.keys()):
            if ".".join(c_key).startswith(".".join(d_key)):
                del flat_config[c_key]

    config = ConfigDict(flax.traverse_util.unflatten_dict(flat_config))
    config.update(FLAGS.config.get("update_config", ConfigDict()))
    config = config.to_dict()
    check_config_diff(config, meta['config'])
    
    meta['config'] = config
    meta['config']['model'] = _jax_config_to_pt_config(meta['config']['model'])
    
    #########
    #
    # Setup Data Loader
    #
    #########
    # TODO

    if "oxe_kwargs" in FLAGS.config.dataset_kwargs:
        # create dataset_kwargs_list from oxe_kwargs
        (
            FLAGS.config.dataset_kwargs["dataset_kwargs_list"],
            FLAGS.config.dataset_kwargs["sample_weights"],
        ) = make_oxe_dataset_kwargs_and_weights(
            **FLAGS.config.dataset_kwargs["oxe_kwargs"]
        )
        del FLAGS.config.dataset_kwargs["oxe_kwargs"]

    
    FLAGS.config.dataset_kwargs.frame_transform_kwargs = FLAGS.config.frame_transform_kwargs
    FLAGS.config.dataset_kwargs.traj_transform_kwargs = FLAGS.config.traj_transform_kwargs
    FLAGS.config.dataset_kwargs.batch_size = None
    dataset = make_interleaved_dataset(**FLAGS.config.dataset_kwargs, train=True)

    # dataset_iter = dataset.repeat().unbatch().shuffle(FLAGS.config.shuffle_buffer_size)
    pytorch_dataset = TorchRLDSDataset(dataset, meta["text_processor"])
    
    dataloader = DataLoader(
            pytorch_dataset,
            batch_size=FLAGS.config.batch_size // num_devices,
            num_workers=0,  # important to keep this to 0 so PyTorch does not mess with the parallelism
        )
    example_batch = next(iter(dataloader))
    
    meta['example_batch'] = example_batch
    meta['dataset_statistics'] = _np2pt(dataset.dataset_statistics)
    #########
    #
    # Load Pretrained Model
    #
    #########

    model = OctoModelPt.from_config(
        **meta
    )
    _, _ = model.load_weights_from_jax(FLAGS.config.pretrained_path,
        step=FLAGS.config.pretrained_step
    )
    
    model = model.to(device_id)
    model = DDP(model, device_ids=[device_id], find_unused_parameters=True, gradient_as_bucket_view=True)
    
    #########
    #
    # Setup Optimizer and Train State
    #
    #########

    if FLAGS.config.optimizer.frozen_keys is None:
        FLAGS.config.optimizer.frozen_keys = meta['config']["optimizer"]["frozen_keys"]
    freeze_weights_pt(model, FLAGS.config.optimizer.frozen_keys)
    
    trainable_params = [param for param in model.parameters() if param.requires_grad]
    optimizer = AdamW(trainable_params, lr=meta['config']["optimizer"]["learning_rate"]["peak_value"])
    lr_sheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=meta['config']["optimizer"]["learning_rate"]["warmup_steps"],
        num_training_steps=int(FLAGS.config.num_steps),
    )
    
    #########
    #
    # Save all metadata
    #
    #########
    if distributed_state.is_main_process:
        if FLAGS.config.save_dir is not None:
            save_dir = Path(FLAGS.config.save_dir)
            group = FLAGS.config.wandb.group or ""
            save_dir = save_dir /  FLAGS.config.wandb.project / group / wandb_id
            save_dir.mkdir(parents = True, exist_ok = True)
            
            wandb.config.update(dict(save_dir=save_dir), allow_val_change=True)
            logging.info("Saving to %s", save_dir)

            # Add window_size to top of config, to make eval easier
            new_config = ConfigDict(meta['config'])
            new_config["window_size"] = example_batch["observation"][
                "timestep_pad_mask"
            ].shape[1]
            model.module.config = new_config

            # Save finetuning config since it's not saved by SaveCallback, i.e. as part of model.save_pretrained()
            with open(save_dir / "finetune_config.json", "w") as f:
                json.dump(json.loads(FLAGS.config.to_json_best_effort()), f)
        else:
            save_dir = None
            logging.warning("save_dir not passed in, not saving checkpoints")

        example_batch_spec = tree_map(
            lambda t: (t.shape, str(t.dtype)), example_batch
        )
        wandb.config.update(
            dict(example_batch_spec=example_batch_spec), allow_val_change=True
        )

    #########
    #
    # Define loss, train_step, and eval_step
    #
    #########

    # Data parallelism
    # Model is replicated across devices, data is split across devices
    
    #########
    #
    # Build validation & visualization callbacks
    #
    #########

    # TODO


    #########
    #
    # Optionally build visualizers for sim env evals
    #
    #########

    # TODO

    #########
    #
    # Train loop
    #
    #########


    for i, batch in tqdm.tqdm(enumerate(dataloader), total=int(FLAGS.config.num_steps), dynamic_ncols=True):
        
        model.train()
        optimizer.zero_grad()
        
        batch = _to_device(batch, device=device_id)
        
        _, head_outputs = model(
                observations=batch['observation'], 
                tasks=batch['task'], 
                timestep_pad_mask=batch["observation"]['timestep_pad_mask'], 
                action_pad_mask=batch['action_pad_mask'],
                gt_actions=batch['action'],
                train=True, 
                verbose=False,
                save_attention_mask=True)
        
        loss = head_outputs['action'][1]['loss']
        
        loss.backward()
        
        if distributed_state.is_main_process and (i + 1) % FLAGS.config.log_interval == 0:
            wandb.log(
                {"train_loss": loss.item(), "learning_rate": lr_sheduler.get_last_lr()[0]}, step=i
            )
        

        if distributed_state.is_main_process and (i + 1) % FLAGS.config.save_interval == 0 and save_dir is not None:
            logging.info("Saving checkpoint...")
            model.module.save_pretrained(step=i, checkpoint_path=save_dir, optimizer=optimizer)

        optimizer.step()
        lr_sheduler.step()

if __name__ == "__main__":
    app.run(main)
