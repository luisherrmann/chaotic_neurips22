from typing import overload, Union, List
from collections import OrderedDict
import torch
import torch.nn as nn

from ..o2layer.o2layer import O2Layer
from .o2sequential import O2Sequential
from .o2container import O2Container
from o2grad.sparse import SparseSymmetricMatrix
from o2grad.linalg import matmul_mixed, sum_mixed


class Residual(nn.Module):
    @overload
    def __init__(self, *args: nn.Module) -> None:
        ...

    @overload
    def __init__(self, arg: "OrderedDict[str, nn.Module]") -> None:
        ...

    def __init__(self, *args):
        super().__init__()
        self.sequence = self.preprocess_args(*args)

    def __repr__(self) -> str:
        return f"""Residual({repr(self.sequence)})"""

    @staticmethod
    def preprocess_args(*args):
        if len(args) == 0:
            raise ValueError(f"Expects at least one module, but you provided none!")
        elif len(args) == 1:
            if isinstance(args[0], nn.Module):
                modules = args[0]
            elif isinstance(args[0], OrderedDict):
                modules = nn.Sequential(args[0])
            else:
                raise TypeError(
                    f"Passed an instance of {type(args[0])}, but expected nn.Module or OrderedDict!"
                )
        else:
            modules = nn.Sequential(*args)
        return modules

    def forward(self, x):
        return self.sequence(x) + x


class O2Residual(O2Container):
    @overload
    def __init__(self, *args: nn.Module) -> None:
        ...

    @overload
    def __init__(self, arg: "OrderedDict[str, nn.Module]") -> None:
        ...

    def __init__(self, *args):
        module = O2Residual.preprocess_args(*args)
        residual = Residual(module)
        super().__init__(residual)
        # These are set by default as they are required to compute the Loss-Input-Hessian (LIH)
        self.module.sequence.set_chain_output_input_jacobian(True)
        self.module.sequence.set_chain_end_output_input_jacobian(True)

    @staticmethod
    def assert_o2modules(*args) -> None:
        are_o2 = [isinstance(m, O2Layer) or isinstance(m, O2Container) for m in args]
        if not all(are_o2):
            wrong_types = [type(m) for m, is_o2 in zip(args, are_o2) if is_o2]
            wrong_types_str = ", ".join([str(x) for x in wrong_types])
            raise TypeError(
                f"Not all of the modules provided are O2Module instances! Encountered {wrong_types_str}!"
            )

    @staticmethod
    def preprocess_args(*args):
        if len(args) == 0:
            raise ValueError(f"Expects at least one module, but you provided none!")
        elif len(args) == 1:
            if isinstance(args[0], O2Layer) or isinstance(args[0], O2Container):
                module = args[0]
            elif isinstance(args[0], OrderedDict):
                O2Residual.assert_o2modules(*args[0].values())
                module = O2Sequential(*args)
            else:
                raise TypeError(
                    f"Passed an instance of {type(args[0])}, but expected nn.Module or OrderedDict!"
                )
        else:
            O2Residual.assert_o2modules(*args)
            module = O2Sequential(*args)
        return module

    def o2children(self) -> List[Union["O2Layer", O2Container]]:
        return [self.module.sequence]

    def named_o2children(self) -> OrderedDict[str, "O2Layer"]:
        return OrderedDict([("sequence", self.module.sequence)])

    def get_loss_input_hessian(
        self, dL2dy2: Union[torch.Tensor, SparseSymmetricMatrix]
    ) -> torch.Tensor:
        dL2dx2_term1 = self.module.sequence.get_loss_input_hessian_cached()
        U = self.module.sequence.dyydx
        U.shape[0]
        if isinstance(dL2dy2, SparseSymmetricMatrix):
            dL2dx2_term2 = dL2dy2.matmul(U)
        else:
            dL2dx2_term2 = matmul_mixed(dL2dy2, U)
        dL2dx2 = dL2dy2 + dL2dx2_term1 + dL2dx2_term2 + dL2dx2_term2.t()
        return dL2dx2

    def set_chain_output_input_jacobian(self, value: bool) -> None:
        self.chain_dydx = value
        self.module.sequence.set_chain_output_input_jacobian(value)
        self.module.sequence.set_chain_end_output_input_jacobian(value)

    def set_chain_end_output_input_jacobian(self, value: bool) -> None:
        self.chain_end_dydx = value
        self.module.sequence.set_chain_end_output_input_jacobian(value)

    def get_output_input_jacobian(self, x: torch.Tensor):
        dydx = self.module.sequence.get_chained_output_input_jacobian_cached()
        output_dim, input_dim = dydx.shape
        device = dydx.device
        assert output_dim == input_dim
        idx = torch.tensor(
            [(i, i) for i in range(output_dim)], dtype=torch.int64, device=device
        ).T
        val = torch.empty(output_dim, device=device)
        val[:] = 1
        identity = torch.sparse_coo_tensor(idx, val, dydx.shape)
        dydx = sum_mixed(identity, dydx)
        return dydx

    def add_default_callbacks(self) -> None:
        self.module.sequence._callbacks.on_complete.add(
            self._callbacks.on_child_complete
        )
        self._callbacks.on_child_complete.add(self._callbacks.on_children_complete)
        self._callbacks.on_children_complete.add(
            lambda: self.module.sequence.try_clear_cache("dyydx")
        )
        self._callbacks.on_children_complete.add(
            lambda: self.module.sequence.try_clear_cache("dL2dx2")
        )
        self._callbacks.on_children_complete.add(self._callbacks.on_complete)

    @classmethod
    def from_module(cls, module: Residual):
        assert isinstance(module, Residual)
        o2residual = super().from_module(module)
        o2residual.module.sequence.set_chain_output_input_jacobian(True)
        o2residual.module.sequence.set_chain_end_output_input_jacobian(True)
        return o2residual
