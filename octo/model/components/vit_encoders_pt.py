"""
Encoders more suitable for ViT architectures.

- PatchEncoderPt: Just patchifies the image
- SmallStemPt: 3 conv layers, then patchifies the image (from xiao et al. 2021)
- ViTResnetPt: ResNetv2, followed by patchification (from google-research/vision_transformer)
"""

import functools as ft
from typing import Callable, Sequence, TypeVar
from abc import ABC, abstractmethod

import torch
import torch.nn as nn
import torch.nn.functional as F
# from octo.model.components.film_conditioning_layer import FilmConditioning
from octo.model.components.jax_pt import FromJaxModel, ConvPt, GroupNormPt, StdConvPt

T = TypeVar("T")


def normalize_images(img, img_norm_type="default"):
    if img_norm_type == "default":
        # put pixels in [-1, 1]
        return img.float() / 127.5 - 1.0
    elif img_norm_type == "imagenet":
        raise NotImplementedError
        # put pixels in [0,1]
        img = img.float() / 255
        assert img.shape[-1] % 3 == 0, "images should have rgb channels!"

        # define pixel-wise mean/std stats calculated from ImageNet
        mean = torch.tensor([0.485, 0.456, 0.406]).reshape((1, 1, 1, 3))
        std = torch.tensor([0.229, 0.224, 0.225]).reshape((1, 1, 1, 3))

        # tile mean and std (to account for stacked early_fusion images)
        num_tile = (1, 1, 1, int(img.shape[-1] / 3))
        mean_tile = mean.repeat(*num_tile)
        std_tile = std.repeat(*num_tile)

        # tile the mean/std, normalize image, and return
        return (img - mean_tile.to(img.device)) / std_tile.to(img.device)
    raise ValueError()


def weight_standardize(w, axis, eps):
    """Subtracts mean and divides by standard deviation."""
    w = w - w.mean(dim=axis, keepdim=True)
    w = w / (w.std(dim=axis, keepdim=True) + eps)
    return w

class PatchEncoderPt(nn.Module, FromJaxModel):
    """Takes an image and breaks it up into patches of size (patch_size x patch_size),
    applying a fully connected network to each patch individually.

    The default "encoder" used by most ViTs in practice.
    """

    def __init__(self, use_film=False, patch_size=32, num_features=512, img_norm_type="default"):
        super().__init__()
        self.use_film = use_film
        self.patch_size = patch_size
        self.num_features = num_features
        self.img_norm_type = img_norm_type

        self.embedding = ConvPt(3, num_features, kernel_size=patch_size, stride=patch_size)
        if use_film:
            raise NotImplementedError
        
    def load_jax_weights(self, jax_params):
        self.embedding.load_jax_weights(jax_params['embedding'])

    def forward(self, observations: torch.Tensor, train: bool = True, cond_var=None):
        expecting_cond_var = self.use_film
        received_cond_var = cond_var is not None
        assert (
            expecting_cond_var == received_cond_var
        ), "Only pass in cond var iff model expecting cond var"
        x = normalize_images(observations, self.img_norm_type)
        x = self.embedding(x)
        if self.use_film:
            assert cond_var is not None, "Cond var is None, nothing to condition on"
            x = self.film(x, cond_var)
        return x


class SmallStemPt(nn.Module, FromJaxModel):
    """Passes the image through a few light-weight convolutional layers,
    before patchifying the image. Empirically useful for many computer vision tasks.

    See Xiao et al: Early Convolutions Help Transformers See Better
    """

    def __init__(self, use_film=False, patch_size=32, kernel_sizes=(3, 3, 3, 3),
                 strides=(2, 2, 2, 2), features=(32, 96, 192, 384), padding=(1, 1, 1, 1),
                 num_features=512, img_norm_type="default"):
        super().__init__()
        self.use_film = use_film
        self.patch_size = patch_size
        self.img_norm_type = img_norm_type
        self.num_features = num_features

        self.layers = nn.ModuleList()
        for n, (kernel_size, stride, feature, pad) in enumerate(zip(kernel_sizes, strides, features, padding)):
            self.layers.append(nn.Sequential(
                StdConvPt(6 if n == 0 else features[n-1], feature, kernel_size=kernel_size, stride=stride, padding=pad),
                GroupNormPt(32, feature),
                nn.ReLU()
            ))

        self.embedding = ConvPt(features[-1], num_features, kernel_size=patch_size // 16, stride=patch_size // 16)
        if use_film:
            raise NotImplementedError
            # self.film = FilmConditioning()
    
    def load_jax_weights(self, jax_params):
        self.embedding.load_jax_weights(jax_params['embedding'])
    
        for i in range(len(self.layers)):
            self.layers[i][0].load_jax_weights(jax_params[f'StdConv_{i}'])
            self.layers[i][1].load_jax_weights(jax_params[f'GroupNorm_{i}'])
            
            
            
    def forward(self, observations: torch.Tensor, train: bool = True, cond_var=None):
        expecting_cond_var = self.use_film
        received_cond_var = cond_var is not None
        assert (
            expecting_cond_var == received_cond_var
        ), "Only pass in cond var iff model expecting cond var"

        x = normalize_images(observations, self.img_norm_type)
        
        for layer in self.layers:
            x = layer(x)
            
        x = self.embedding(x)
        if self.use_film:
            assert cond_var is not None, "Cond var is None, nothing to condition on"
            x = self.film(x, cond_var)
        return x


class ResidualUnit(nn.Module):
    """Bottleneck ResNet block."""

    def __init__(self, in_features, features, strides=(1, 1)):
        super().__init__()
        self.features = features
        self.strides = strides

        self.conv1 = StdConvPt(in_features, features, kernel_size=1, bias=False)
        self.gn1 = GroupNormPt(32, features)
        self.conv2 = StdConvPt(features, features, kernel_size=3, stride=strides, padding=1, bias=False)
        self.gn2 = GroupNormPt(32, features)
        self.conv3 = StdConvPt(features, features * 4, kernel_size=1, bias=False)
        self.gn3 = GroupNormPt(32, features * 4)

        if in_features != features * 4 or strides != (1, 1):
            self.proj_conv = StdConvPt(in_features, features * 4, kernel_size=1, stride=strides, bias=False)
            self.proj_gn = GroupNormPt(32, features * 4)
        else:
            self.proj_conv = None

    def forward(self, x):
        residual = x
        if self.proj_conv is not None:
            residual = self.proj_conv(residual)
            residual = self.proj_gn(residual)

        y = self.conv1(x)
        y = self.gn1(y)
        y = F.relu(y)
        y = self.conv2(y)
        y = self.gn2(y)
        y = F.relu(y)
        y = self.conv3(y)
        y = self.gn3(y)

        return F.relu(residual + y)


class ResNetStage(nn.Module):
    """A ResNet stage."""

    def __init__(self, in_features, nout, block_size, first_stride):
        super().__init__()
        self.blocks = nn.ModuleList([
            ResidualUnit(in_features if i == 0 else nout * 4, nout, first_stride if i == 0 else (1, 1))
            for i in range(block_size)
        ])

    def forward(self, x):
        for block in self.blocks:
            x = block(x)
        return x


class ViTResnetPt(nn.Module):
    """Resnet-v2 architecture used in the original ViT paper for hybrid (Resnet+ViT) architectures

    Mostly copied from https://github.com/google-research/vision_transformer/blob/main/vit_jax/models_vit.py

    There exist pre-trained parameters here: github.com/google-research/vision_transformer/
    """

    def __init__(self, use_film=False, width=1, num_layers=tuple(), img_norm_type="default"):
        super().__init__()
        self.use_film = use_film
        self.width = width
        self.num_layers = num_layers
        self.img_norm_type = img_norm_type

        width = int(64 * self.width)
        self.root = nn.Sequential(
            StdConvPt(3, width, kernel_size=7, stride=2, padding=3, bias=False),
            GroupNormPt(32, width),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )

        if self.num_layers:
            self.stages = nn.ModuleList()
            in_features = width
            for i, block_size in enumerate(self.num_layers):
                out_features = width * 2**i
                self.stages.append(ResNetStage(in_features, out_features, block_size, (2, 2) if i > 0 else (1, 1)))
                in_features = out_features * 4

        if use_film:
            raise NotImplementedError

    def forward(self, observations: torch.Tensor, train: bool = True, cond_var=None):
        expecting_cond_var = self.use_film
        received_cond_var = cond_var is not None
        assert (
            expecting_cond_var == received_cond_var
        ), "Only pass in cond var iff model expecting cond var"

        x = normalize_images(observations, self.img_norm_type)
        x = self.root(x)

        if self.num_layers:
            for stage in self.stages:
                x = stage(x)
                if self.use_film:
                    assert cond_var is not None, "Cond var is None, nothing to condition on"
                    x = self.film(x, cond_var)
        else:
            if self.use_film:
                assert cond_var is not None, "Cond var is None, nothing to condition on"
                x = self.film(x, cond_var)

        return x


class SmallStemPt16(SmallStemPt):
    def __init__(self, **kwargs):
        super().__init__(patch_size=16, **kwargs)


class SmallStemPt32(SmallStemPt):
    def __init__(self, **kwargs):
        super().__init__(patch_size=32, **kwargs)


class ResNet26FILMPt(ViTResnetPt):
    def __init__(self, **kwargs):
        super().__init__(use_film=True, num_layers=(2, 2, 2, 2), **kwargs)


vit_encoder_configs_pt = {
    "patchify-32-film": ft.partial(
        PatchEncoderPt,
        use_film=True,
        patch_size=32,
    ),
    "patchify-16-film": ft.partial(
        PatchEncoderPt,
        use_film=True,
        patch_size=16,
    ),
    "small-stem-8-film": ft.partial(
        SmallStemPt,
        use_film=True,
        patch_size=16,
        kernel_sizes=(3, 3, 3),
        strides=(2, 2, 2),
        features=(32, 96, 192),
        padding=(1, 1, 1),
    ),
    "small-stem-16": ft.partial(
        SmallStemPt,
        patch_size=16,
    ),
    "small-stem-16-film": ft.partial(
        SmallStemPt,
        use_film=True,
        patch_size=16,
    ),
    "small-stem-32-film": ft.partial(
        SmallStemPt,
        use_film=True,
        patch_size=32,
    ),
    "resnetv2-26-film": ft.partial(
        ViTResnetPt,
        use_film=True,
        num_layers=(2, 2, 2, 2),
    ),
    "resnetv2-50-film": ft.partial(
        ViTResnetPt,
        use_film=True,
        num_layers=(3, 4, 6, 3),
    ),
}