from collections import OrderedDict
from omegaconf import DictConfig
import torch.nn as nn
from typing import Optional
from omegaconf import OmegaConf

from o2grad.modules.o2layer.o2reshape import Reshape
from o2grad.modules.o2layer.o2pooling import O2MaxPool2d


def get_activation(name):
    return {"sigmoid": nn.Sigmoid, "relu": nn.ReLU}[name]()


def get_mlp(inter_dim: int, activation: str, **kwargs):
    return nn.Sequential(
        Reshape(256),
        nn.Linear(256, inter_dim),
        get_activation(activation),
        nn.Linear(inter_dim, 10),
    )


def get_cnn(
    base_channels: int,
    block_cnt: int,
    activation: str,
    batch_norm: bool = False,
    **kwargs,
):
    head = [
        (
            "conv1",
            nn.Conv2d(
                in_channels=1, out_channels=base_channels, kernel_size=3, padding=1
            ),
        ),
        ("relu1", get_activation(activation)),
    ]
    if batch_norm:
        head += [("bn1", nn.BatchNorm2d(base_channels))]
    # Constant dimension part
    body = []
    for i in range(1, block_cnt):
        body.append(
            (
                f"conv{i+1}",
                nn.Conv2d(
                    in_channels=base_channels,
                    out_channels=base_channels,
                    kernel_size=3,
                    padding=1,
                ),
            )
        )
        body.append((f"relu{i+1}", get_activation(activation)))
        if batch_norm:
            body.append((f"bn{i+1}", nn.BatchNorm2d(base_channels)))
    # Downsampling part
    tail = [
        ("tail_pool1", O2MaxPool2d(kernel_size=2, stride=2)),
        (
            "tail_conv2",
            nn.Conv2d(
                in_channels=base_channels,
                out_channels=base_channels,
                kernel_size=3,
                padding=1,
            ),
        ),
        ("tail_fn2", get_activation(activation)),
    ]
    if batch_norm:
        tail += [("bn2", nn.BatchNorm2d(base_channels))]
    tail += [
        ("tail_pool2", O2MaxPool2d(kernel_size=2, stride=2)),
        (
            "tail_conv3",
            nn.Conv2d(
                in_channels=base_channels,
                out_channels=2 * base_channels,
                kernel_size=3,
                padding=1,
            ),
        ),
        ("tail_fn3", get_activation(activation)),
    ]
    if batch_norm:
        tail += [("bn3", nn.BatchNorm2d(2 * base_channels))]
    tail += [
        ("tail_pool3", O2MaxPool2d(kernel_size=2, stride=2)),
        (
            "tail_conv4",
            nn.Conv2d(
                in_channels=2 * base_channels,
                out_channels=2 * base_channels,
                kernel_size=3,
                padding=1,
            ),
        ),
        ("tail_fn4", get_activation(activation)),
    ]
    if batch_norm:
        tail += [("bn4", nn.BatchNorm2d(2 * base_channels))]
    tail += [
        ("tail_pool4", O2MaxPool2d(kernel_size=2, stride=2)),
        ("tail_reshape", Reshape([-1, 2 * base_channels])),
        ("tail_fc1", nn.Linear(in_features=2 * base_channels, out_features=10)),
    ]
    return nn.Sequential(OrderedDict([*head, *body, *tail]))


def get_model(config: DictConfig, name: Optional[str] = None):
    model_config = config.model if "model" in config else config
    name = name or model_config.name
    kwargs = OmegaConf.to_container(model_config)
    return {"mlp": get_mlp, "cnn": get_cnn,}[
        name
    ](**kwargs)
