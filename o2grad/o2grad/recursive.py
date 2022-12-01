import warnings
from typing import Type, Union, Dict, Callable
from collections import OrderedDict
import torch.nn as nn

from o2grad.modules.mapping import (
    MODULE_MAPPING,
)
from o2grad.modules.o2layer.o2layer import O2Layer
from o2grad.modules.o2container.o2container import O2Container


def set_nesting_level(module, nesting_level=0) -> None:
    if isinstance(module, O2Container):
        module.nesting_level = nesting_level
        for m in module.o2children():
            set_nesting_level(m, nesting_level=nesting_level + 1)
    elif isinstance(module, O2Layer):
        module.nesting_level = nesting_level


def set_next_layer(module, layer) -> None:
    if isinstance(module, O2Layer):
        module.next_layer = layer
    elif isinstance(module, O2Container):
        module.next_layer = layer
        children = [*module.o2children(), layer]
        for i in range(len(children) - 1):
            set_next_layer(children[i], children[i + 1])


def add_callbacks(
    module,
    callbacks: Dict[str, Callable[[nn.Module], None]],
    cond: Callable[[nn.Module], bool] = lambda m: True,
):
    """Starting from module provided, recursively adds callback of name provided to all children if condition is met."""
    if isinstance(module, O2Layer):
        if cond(module):
            module.add_callbacks(callbacks)
    elif isinstance(module, O2Container):
        for m in module.o2children():
            add_callbacks(m, callbacks, cond)
        if cond(module):
            module.add_callbacks(callbacks)


def add_default_callbacks(module):
    if isinstance(module, O2Layer):
        module.add_default_callbacks()
    elif isinstance(module, O2Container):
        module.add_default_callbacks()
        for m in module.o2children():
            add_default_callbacks(m)


def try_clear_cache(module, force=False) -> None:
    if isinstance(module, O2Layer):
        module.try_clear_cache(force=force)
    elif isinstance(module, O2Container):
        module.try_clear_cache(force=force)
        for m in module.o2children():
            try_clear_cache(m, force=force)


def get_o2modules(module, prefix="", sep=".") -> OrderedDict[str, O2Layer]:
    if isinstance(module, O2Layer):
        return OrderedDict({prefix: module})
    elif isinstance(module, O2Container):
        module_dict = OrderedDict({})
        for key, m in module.named_o2children().items():
            prefix_ = f"{prefix}{sep}{key}"
            o2modules = get_o2modules(m, prefix_)
            module_dict.update(o2modules)
        return module_dict


def _replace_with_o2modules(
    module: nn.Module, mapping: Dict[Type[nn.Module], Type[Union[O2Layer, O2Container]]]
) -> Union[O2Layer, O2Container]:
    if isinstance(module, nn.Module):
        if type(module) in mapping:
            o2class = mapping[type(module)]
            if issubclass(o2class, O2Layer):
                o2module = o2class.from_module(module)
                return o2module
            elif issubclass(o2class, O2Container):
                for key, m in module.named_children():
                    module._modules[key] = _replace_with_o2modules(m, mapping)
                o2container = o2class.from_module(module)
                return o2container
        elif isinstance(module, O2Layer) or isinstance(module, O2Container):
            return module
        else:
            WARNING = f"""Encountered module of type {type(module)} that cannot be replaced by O2Module! This will likely result in calculation errors."""
            warnings.warn(WARNING)
    else:
        raise ValueError(
            f"Encountered an instance of {type(module)}, but expected torch.nn.Module"
        )


def replace_with_o2modules(
    module: nn.Module, custom_objects={}
) -> Union[O2Layer, O2Container]:
    mapping = {**MODULE_MAPPING, **custom_objects}
    return _replace_with_o2modules(module, mapping)


def remove_o2modules(module: nn.Module) -> nn.Module:
    if isinstance(module, nn.Module):
        if isinstance(module, O2Layer):
            return module.module
        else:
            if isinstance(module, O2Container):
                module = module.module
            for key, m in module.named_children():
                module._modules[key] = remove_o2modules(m)
            return module
    else:
        raise ValueError(
            f"Encountered an instance of {type(module)}, but expected torch.nn.Module"
        )
