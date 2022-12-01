from collections import OrderedDict
import torch
import torch.nn as nn
from typing import overload, Iterable, Union

from ..o2module import O2Module
from .o2container import O2Container
from o2grad.sparse import SparseSymmetricMatrix


class O2Sequential(O2Container):
    @overload
    def __init__(self, *args: O2Module) -> None:
        ...

    @overload
    def __init__(self, arg: OrderedDict[str, O2Module]) -> None:
        ...

    def __init__(self, *args, **kwargs):
        if len(args) == 0:
            raise ValueError(f"Expects at least one module, but you provided none!")
        sequential = nn.Sequential(*args, **kwargs)
        are_o2 = [
            isinstance(m, O2Module) or isinstance(m, O2Container)
            for m in sequential.children()
        ]
        if not all(are_o2):
            wrong_types = [
                type(m) for m, is_o2 in zip(sequential.children(), are_o2) if is_o2
            ]
            wrong_types_str = ", ".join([str(x) for x in wrong_types])
            raise TypeError(
                f"Not all of the modules provided are O2Module instances! Encountered {wrong_types_str}!"
            )
        super().__init__(sequential)

    def __getitem__(self, idx: int):
        return self.module.__getitem__(idx)

    def __setitem__(self, idx: int, module: Union[O2Module, O2Container]) -> None:
        assert isinstance(module, O2Module) or isinstance(module, O2Container)
        self.module.__setitem__(idx)

    def __delitem__(self, idx: int) -> None:
        self.module.__delitem__(idx)

    def o2children(self) -> Iterable[O2Module]:
        return self.module.children()

    def named_o2children(self) -> OrderedDict[str, O2Module]:
        return OrderedDict(self.module.named_children())

    def get_loss_input_hessian(
        self, dL2dy2: Union[torch.Tensor, SparseSymmetricMatrix]
    ) -> torch.Tensor:
        dL2dx2 = self.module[0].dL2dx2
        return dL2dx2

    def set_chain_output_input_jacobian(self, value: bool) -> None:
        self.chain_dydx = value
        for m in self.module:
            m.set_chain_output_input_jacobian(value)
        self.module[-1].set_chain_end_output_input_jacobian(value)

    def set_chain_end_output_input_jacobian(self, value: bool) -> None:
        self.chain_end_dydx = value
        self.module[-1].set_chain_end_output_input_jacobian(value)

    def get_output_input_jacobian(self, x: torch.Tensor) -> torch.Tensor:
        dydx = self.module[0].get_chained_output_input_jacobian_cached()
        return dydx

    def add_default_callbacks(self) -> None:
        self.module[0]._callbacks.on_complete.add(self._callbacks.on_child_complete)
        self._callbacks.on_child_complete.add(self._callbacks.on_children_complete)
        self._callbacks.on_children_complete.add(
            lambda: self.module[0].try_clear_cache("dyydx")
        )
        self._callbacks.on_children_complete.add(
            lambda: self.module[0].try_clear_cache("dL2dx2")
        )
        self._callbacks.on_children_complete.add(self._callbacks.on_complete)

    @classmethod
    def from_module(cls, module: nn.Sequential):
        assert isinstance(module, nn.Sequential)
        return super().from_module(module)
