# Octo-PyTorch

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://githubtocolab.com/emb-ai/octo-pytorch/blob/main/examples/01_pt_inference_pretrained.ipynb)

This repo contains code for training and finetuning [Octo](https://octo-models.github.io/) (Team, Octo Model, et al. "Octo: An open-source generalist robot policy.") generalist robotic policies (GRPs) using PyTorch framework.
Base architecture and fine-tuning procedure were reimplemented from original repo, which can be useful for further research related to robotic manipulation.

## Get Started

Follow the installation instructions, then load a pretrained Octo model!
See examples for guides to zero-shot [evaluation](examples/01_pt_inference_pretrained.ipynb) and [finetuning](examples/02_pt_finetune_new_observation_action.ipynb) example.

You can load original JAX weights directly from [HuggingFace](https://huggingface.co/rail-berkeley) or any provided local checkpoint and use it for PyTorch module initialization. 

```python
from octo.model.octo_model_pt import OctoModelPt
model = OctoModelPt.load_pretrained_from_jax("hf://rail-berkeley/octo-small-1.5")['octo_model']
```

Or from PyTorch checkpoint:

```python
from octo.model.octo_model_pt import OctoModelPt
model = OctoModelPt.load_pretrained(checkpoint_path)['octo_model']
```

![Octo model](docs/assets/teaser.jpg)


## Installation
```bash
conda create -n octo_pt python=3.10
conda activate octo_pt
pip install -e .
pip install -r requirements.txt
pip install torch torchvision
pip install --upgrade "jax==0.4.20" -f https://storage.googleapis.com/jax-releases/jax_releases.html
pip install accelerate
```

Note: since we use JAX only for weights loading, we do not need JAX with CUDA support.

See the [Jax Github page](https://github.com/google/jax) for more details on installing Jax.

Test the installation by finetuning on the debug dataset:
```bash
python scripts/finetune_pt.py --config.pretrained_path=hf://rail-berkeley/octo-small-1.5 --debug
```

## Checkpoints

You can find original pretrained Octo checkpoints [here](https://huggingface.co/rail-berkeley).
At the moment authors provide the following model versions:

| Model                                                         | Inference on 1x NVIDIA 4090 | Size       |
|---------------------------------------------------------------|-----------------------------|------------|
| [Octo-Base](https://huggingface.co/rail-berkeley/octo-base)   | 13 it/sec                   | 93M Params |
| [Octo-Small](https://huggingface.co/rail-berkeley/octo-small) | 17 it/sec                   | 27M Params |


## Examples

We provide simple [example scripts](examples) that demonstrate how to use and finetune Octo models,
as well as how to use our data loader independently. We provide the following examples:

|                                                                      |                                                                                                                    |
|----------------------------------------------------------------------|--------------------------------------------------------------------------------------------------------------------|
| [OctoPt Inference](examples/01_pt_inference_pretrained.ipynb)             | Minimal example for loading and running a pretrained Octo model                                                    |
| [OctoPt Finetuning](examples/02_pt_finetune_new_observation_action.py)    | Minimal example for finetuning a pretrained Octo models on a small dataset with a new observation and action space |
| [OctoPt Rollout](examples/03_pt_eval_finetuned.py)                        | Run a rollout of a pretrained Octo policy in a Gym environment                                                     |
| 


## OctoPt Finetuning

We provide a [minimal example](examples/02_pt_finetune_new_observation_action.py) for finetuning with a new observation and action space.

We also provide a more advanced finetuning script that allows you to change hyperparameters via a config file and logs finetuning
metrics. To run advanced finetuning, use:
```bash
python scripts/finetune_pt.py --config.pretrained_path=hf://rail-berkeley/octo-small-1.5
```

More details can be found in the original [repo](https://octo-models.github.io/).

