import torch.nn as nn

from .o2parametric.o2linear import O2Linear
from .o2parametric.o2conv1d import O2Conv1d
from .o2parametric.o2conv2d import O2Conv2d
from .o2parametric.o2batchnorm import O2BatchNorm1d, O2BatchNorm2d
from .o2layer.activations import O2Sigmoid, O2ReLU
from .o2layer.o2pooling import O2MaxPool1d, O2MaxPool2d, O2AvgPool1d, O2AvgPool2d
from .o2module import O2Module
from .o2layer.o2reshape import Reshape, O2Reshape
from .o2container.o2sequential import O2Sequential
from .o2container.o2residual import Residual, O2Residual

MODULE_MAPPING = {
    nn.Linear: O2Linear,
    nn.Conv1d: O2Conv1d,
    nn.Conv2d: O2Conv2d,
    nn.MaxPool1d: O2MaxPool1d,
    nn.MaxPool2d: O2MaxPool2d,
    nn.AvgPool1d: O2AvgPool1d,
    nn.AvgPool2d: O2AvgPool2d,
    nn.BatchNorm1d: O2BatchNorm1d,
    nn.BatchNorm2d: O2BatchNorm2d,
    nn.Sigmoid: O2Sigmoid,
    nn.ReLU: O2ReLU,
    Reshape: O2Reshape,
    nn.Sequential: O2Sequential,
    Residual: O2Residual,
}


def is_module_supported(module: nn.Module, custom_layers={}) -> bool:
    """Given a module, checks if a respective module with O2 support is available.

    Parameters:
    -----------
    layer: nn.Module
        The module to be replaced with a supported module.
    custom_layers: dict
        A dictionary of layers mapping nn.Module -> O2Module.

    Returns:
    --------
    True if supported, False otherwise.
    """
    return type(module) in MODULE_MAPPING or type(module) in custom_layers


def map_class(C, custom_layers={}) -> nn.Module:
    if C in MODULE_MAPPING:
        return MODULE_MAPPING[C]
    elif C in custom_layers:
        return custom_layers[C]
    else:
        return C


def map_module(module: nn.Module, custom_layers={}) -> nn.Module:
    """Returns a parametric module supporting 2nd order differentiation if available, otherwise returns the module wrapped in O2Module.

    Parameters:
    -----------
    layer: nn.Module
        The module to be replaced with a supported module.
    custom_layers: dict
        A dictionary of layers mapping nn.Module -> O2Module.

    Returns:
    --------
    True if supported, False otherwise.
    """
    C = type(module)
    if C in MODULE_MAPPING:
        return MODULE_MAPPING[C].from_module(module)
    elif C in custom_layers:
        return custom_layers[C].from_module(module)
    elif isinstance(module, O2Module):
        return module
    else:
        return O2Module(module)
