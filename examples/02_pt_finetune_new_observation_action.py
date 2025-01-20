"""
This script demonstrates how to finetune OctoPt to a new observation space (single camera + proprio)
and new action space (bimanual) using a simulated ALOHA cube handover dataset (https://tonyzhaozh.github.io/aloha/).

To run this example, first download and extract the dataset from here: https://rail.eecs.berkeley.edu/datasets/example_sim_data.zip

python examples/02_pt_finetune_new_observation_action.py --pretrained_path=hf://rail-berkeley/octo-small-1.5 --data_dir=...
"""
from absl import app, flags, logging
import torch
from torch.optim.lr_scheduler import SequentialLR, LinearLR, ConstantLR
from torch.optim import AdamW 
from torch.utils.data import DataLoader

import tensorflow as tf
import tqdm
import wandb
import numpy as np

from octo.data.dataset import make_single_dataset
from octo.model.components.action_heads_pt import L1ActionHeadPt
from octo.model.components.tokenizers_pt import LowdimObsTokenizerPt
from octo.model.octo_model_pt import OctoModelPt, _np2pt
from octo.utils.spec import ModuleSpec
from octo.utils.train_utils import process_text

from octo.utils.train_utils_pt import freeze_weights_pt

tf.config.set_visible_devices([], "GPU")

class TorchRLDSDataset(torch.utils.data.IterableDataset):
    """Thin wrapper around RLDS dataset for use with PyTorch dataloaders."""

    def __init__(
        self,
        rlds_dataset,
        text_processor,
        train=True,
    ):
        self._rlds_dataset = rlds_dataset
        self._text_processor = text_processor
        self._is_train = train

    def __iter__(self):
        for sample in self._rlds_dataset.as_numpy_iterator():
            sample["task"]["language_instruction"] = np.array([sample["task"]["language_instruction"]])
            sample["task"]["pad_mask_dict"]['language_instruction'] = np.array([sample["task"]["pad_mask_dict"]['language_instruction']])
            sample = process_text(sample, self._text_processor)
            
            # remove extra dim
            sample["task"]["language_instruction"]['input_ids'] = sample["task"]["language_instruction"]['input_ids'][0]
            sample["task"]["language_instruction"]['attention_mask'] = sample["task"]["language_instruction"]['attention_mask'][0]
            
            del sample["dataset_name"]
            sample = _np2pt(sample)
            yield sample

def _to_device(data, device):
    if isinstance(data, dict):
        return {key: _to_device(val, device) for key, val in data.items()}
    elif isinstance(data, torch.Tensor):
        return data.to(device)

FLAGS = flags.FLAGS

flags.DEFINE_string(
    "pretrained_path", None, "Path to pre-trained Octo checkpoint directory."
)
flags.DEFINE_string("data_dir", None, "Path to finetuning dataset, in RLDS format.")
flags.DEFINE_string("save_dir", None, "Directory for saving finetuning checkpoints.")
flags.DEFINE_integer("batch_size", 128, "Batch size for finetuning.")

flags.DEFINE_bool(
    "freeze_transformer",
    False,
    "Whether pre-trained transformer weights should be frozen.",
)


def main(_):
    # setup wandb for logging
    wandb.init(name="finetune_aloha_pt", project="octo")

    logging.info("Loading pre-trained model...")
    logging.set_verbosity(logging.INFO)
    
    # load meta information for pretrained model
    meta = OctoModelPt.load_config_and_meta_from_jax(FLAGS.pretrained_path)
    
    text_processor = meta['text_processor']
    device = 'cuda:0'
    
    # make finetuning dataset
    # apply Gaussian normalization, load chunks of 50 actions since we'll train with action chunking
    # delete goal images in the data loader since we will train a language-conditioned-only policy
    
    logging.info("Loading finetuning dataset...")
    dataset = make_single_dataset(
        dataset_kwargs=dict(
            name="aloha_sim_cube_scripted_dataset",
            data_dir=FLAGS.data_dir,
            image_obs_keys={"primary": "top"},
            proprio_obs_key="state",
            language_key="language_instruction",
        ),
        traj_transform_kwargs=dict(
            window_size=2,
            action_horizon=50,
        ),
        frame_transform_kwargs=dict(
            resize_size={"primary": (256, 256)},
        ),
        train=True,
    )
    dataset_statistics = dataset.dataset_statistics
    
    dataset = dataset.repeat().unbatch().shuffle(10000)
    
    pytorch_dataset = TorchRLDSDataset(dataset, text_processor)
    dataloader = DataLoader(
        pytorch_dataset,
        batch_size=FLAGS.batch_size,
        num_workers=0,  # important to keep this to 0 so PyTorch does not mess with the parallelism
    )

    
    example_batch = next(iter(dataloader))

    # modify config --> remove wrist cam, add proprio input, change action head
    # following Zhao et al. we use "action chunks" of length 50 and L1 loss for ALOHA
    
    del meta["config"]["model"]["observation_tokenizers"]["wrist"]
    ###
    meta["config"]["model"]["observation_tokenizers"]["proprio"] = ModuleSpec.create(
        LowdimObsTokenizerPt,
        n_bins=256,
        bin_type="normal",
        low=-2.0,
        high=2.0,
        obs_keys=["proprio"],
    )
    
    # we have to explicitly specify number of tokens for proprio and observations
    meta["config"]["model"]["num_tokens_dict"] = {
        'primary': 256,
        'language': 16,
        'proprio': 14,
        'action': 1
    }
    
    # Fully override the old action head with a new one (for smaller changes, you can use update_config)
    meta["config"]["model"]["heads"]["action"] = ModuleSpec.create(
        L1ActionHeadPt,
        input_dim=384,
        action_horizon=50,
        action_dim=14,
        readout_key="readout_action",
    )
    meta['example_batch'] = example_batch
    meta['dataset_statistics'] = dataset_statistics
    
    # initialize new OctoPt model from modified config
    logging.info("Updating model for new observation & action space...")
    model = OctoModelPt.from_config(
        **meta,
        verbose=True,
    )
    
    # load weights from JAX model
    _,  _ = model.load_weights_from_jax(FLAGS.pretrained_path, skip_keys_regex=".*hf_model")
    model.to(device)

    # create optimizer, optionally freeze keys for pre-trained transformer
   
    frozen_keys = model.config["optimizer"]["frozen_keys"]
    if FLAGS.freeze_transformer:
        frozen_keys.append("BlockTransformer_0")
    freeze_weights_pt(model.module, frozen_keys)
    
    model.train()
    
    trainable_params = [param for param in model.module.parameters() if param.requires_grad]
    optimizer = AdamW(trainable_params, lr=3e-5)
    linear_part  = LinearLR(optimizer, start_factor=1e-2, total_iters=100)
    constant_part = ConstantLR(optimizer, factor=1.0, total_iters=1)
    scheduler = SequentialLR(optimizer, schedulers=[linear_part, constant_part], milestones=[100])

    example_batch["task"]["pad_mask_dict"]['language_instruction'] = example_batch["task"]["pad_mask_dict"]['language_instruction'][:, 0]
    
    # run finetuning loop
    logging.info("Starting finetuning...")
   
    for i, batch in tqdm.tqdm(enumerate(dataloader), total=5000, dynamic_ncols=True):
        if i == 5000:
            break
       
        batch["task"]["pad_mask_dict"]['language_instruction'] = batch["task"]["pad_mask_dict"]['language_instruction'][:, 0]
        batch = _to_device(batch, device=device)
        optimizer.zero_grad()
        
        _, head_outputs = model(
                observations=batch['observation'], 
                tasks=batch['task'], 
                timestep_pad_mask=batch["observation"]['timestep_pad_mask'], 
                action_pad_mask=batch['action_pad_mask'],
                gt_actions=batch['action'],
                train=True, 
                verbose=False,
                save_attention_mask=True)
        
        loss = head_outputs['action'][0]
        info = head_outputs['action'][1]
        
        logging.info(f'Loss: {info["loss"]:.2f}; Cur LR = {scheduler.get_last_lr()}')
        
        loss.backward()
        optimizer.step()
        scheduler.step()
        
        if (i + 1) % 10 == 0:
            wandb.log({
                "train_loss": info['loss'], 
                "mse": info['mse'], 
                "learning_rate": scheduler.get_last_lr()[0]
                }, step=i
            )
        
        if (i + 1) % 1000 == 0:
            # save checkpoint
            model.save_pretrained(step=i, checkpoint_path=FLAGS.save_dir, optimizer=optimizer)


if __name__ == "__main__":
    app.run(main)
