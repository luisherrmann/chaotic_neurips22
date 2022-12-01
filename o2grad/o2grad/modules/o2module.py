import torch
import torch.nn as nn
from abc import ABC, abstractmethod
from typing import Callable, Sequence, Union, List, Dict
from addict import Dict as AttrDict

from o2grad.sparse import SparseSymmetricMatrix
from o2grad.utils import CallbackSequence


class O2Module(nn.Module, ABC):
    def __init__(self, module: nn.Module):
        super().__init__()
        self.module = module
        # The Hessian of the loss L w.r.t. input x
        self.dL2dx2 = None
        # A list of all the tensors that can be computed and cached by this layer
        self.TENSOR_NAMES = []
        # Pointer to next connected layer
        self.next_layer = None
        self.settings = AttrDict({"save_intermediate": [], "return_layout": {}})
        self._callbacks = AttrDict({"on_complete": CallbackSequence()})

    def forward(self, x: torch.Tensor):
        y = self.module(x)
        self.input = x.clone()
        self.output = y.clone()
        return y

    def try_cache(self, tname, tensor):
        if tname in self.settings.save_intermediate and tname in self.TENSOR_NAMES:
            setattr(self, tname, tensor)
            return 1
        else:
            return 0

    def try_clear_cache(self, *tnames, force=False) -> bool:
        if len(tnames) == 0:
            tnames = self.TENSOR_NAMES
        elif len(tnames) == 1:
            if isinstance(tnames, str):
                tnames = [tnames]
            elif isinstance(tnames, Sequence):
                tnames = [*tnames]
        SUCCESS = 1
        for tn in tnames:
            if tn in self.TENSOR_NAMES:
                if force or tn not in self.settings.save_intermediate:
                    delattr(self, tn)
                    setattr(self, tn, None)
                else:
                    SUCCESS = 0
        return SUCCESS

    def add_callbacks(self, callbacks: Dict[str, Callable[["O2Module"], None]]):
        for key, cb in callbacks.items():
            if key in self._callbacks:
                self._callbacks[key].add(lambda: cb(self))

    def clear_callbacks(self, names: List[str]):
        for key in names:
            if key in self._callbacks:
                self._callbacks[key].clear()

    @abstractmethod
    def add_default_callbacks():
        """A method that adds callbacks to be used in any case."""

    @abstractmethod
    def get_loss_input_hessian(
        self, *args, **kwargs
    ) -> Union[torch.Tensor, SparseSymmetricMatrix]:
        pass

    def get_loss_input_hessian_cached(
        self, *args, **kwargs
    ) -> Union[torch.Tensor, SparseSymmetricMatrix]:
        return self.dL2dx2
