from collections import OrderedDict
import torch
import torch.nn as nn
from abc import ABC, abstractmethod
from typing import Union, List
from addict import Dict as AttrDict

from o2grad.utils import CallbackSequence
from o2grad.sparse import SparseSymmetricMatrix
from ..o2module import O2Module


class O2Container(O2Module, ABC):
    def __init__(self, module: nn.Module, save_intermediate: List[str] = []):
        super().__init__(module)
        self.module = module
        self.TENSOR_NAMES = ["dL2dx2", "dyydx"]
        self.settings = AttrDict(
            {"save_intermediate": [], "per_batch": False, "elem": 0}
        )
        save_intermediate_f = [
            tname for tname in save_intermediate if tname in self.TENSOR_NAMES
        ]
        self.settings.save_intermediate.append(save_intermediate_f)
        self.input = None
        self.output = None
        self.dL2dx2 = None
        self.next_layer = None
        # Chained Jacobian outputs w.r.t inputs
        self.dyydx = None
        self.chain_dydx = False
        self.chain_end_dydx = False
        self._callbacks = AttrDict(
            {
                "on_child_complete": CallbackSequence(),
                "on_children_complete": CallbackSequence(),
                "on_complete": CallbackSequence(),
            }
        )
        self.nesting_level = 0

    @abstractmethod
    def o2children(self) -> List["O2Module"]:
        """Returns a list of all O2Modules contained, in execution order."""

    def named_o2children(self) -> OrderedDict[str, "O2Module"]:
        """Returns an OrderedDict of all O2Modules contained, in execution order.
        NOTE: It is recommended to override the default implementation."""
        items = OrderedDict([(i, m) for i, m in enumerate(self.o2children())])
        return items

    @abstractmethod
    def get_output_input_jacobian(self) -> torch.Tensor:
        """Returns the Jacobian of the module output y w.r.t. an input x.

        Returns:
        --------
        torch.Tensor with shape [m, n], where
        (1) m is the total dimension of the output shape, i.e. y.numel()
        (2) n is the total dimension of the input shape, i.e. x.numel()
        """

    def get_output_input_jacobian_cached(self) -> torch.Tensor:
        return self.dydx

    @abstractmethod
    def get_loss_input_hessian(
        self, dL2dy2: Union[torch.Tensor, SparseSymmetricMatrix]
    ) -> torch.Tensor:
        """Given the LIH of the next layer, returns this layer's LIH.

        Returns:
        --------
        torch.Tensor with shape [n, n], where n is the number of input parameters, i.e. x.numel().
        """

    def get_loss_input_hessian_cached(
        self,
    ) -> Union[torch.Tensor, SparseSymmetricMatrix]:
        return self.dL2dx2

    @abstractmethod
    def set_chain_output_input_jacobian(self, value: bool) -> None:
        """Sets this layer as intermediate point for chaining the Output-Input-Jacobian."""

    @abstractmethod
    def set_chain_end_output_input_jacobian(self, value: bool) -> None:
        """Sets this layers as endpoint for chaining the Output-Input-Jacobian."""

    def get_chained_output_input_jacobian_cached(self):
        """Retrieves the cached chained Output-Input-Jacobian (cOIJ)"""
        return self.dyydx

    @classmethod
    def from_module(cls, module):
        o2_pmodule = cls.__new__(cls)
        super(cls, o2_pmodule).__init__(module)
        return o2_pmodule
