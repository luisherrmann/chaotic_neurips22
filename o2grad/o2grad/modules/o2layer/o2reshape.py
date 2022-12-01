from collections.abc import Sequence
import numpy as np
import torch
import torch.nn as nn

from .o2layer import O2Layer


class Reshape(nn.Module):
    def __init__(self, *args):
        super().__init__()
        if len(args) == 1:
            if isinstance(args[0], Sequence):
                shape = list(args[0])
            else:
                shape = [args[0]]
        elif len(args) > 1:
            are_ints = [isinstance(x, int) for x in args]
            all_ints = np.all(are_ints)
            if not all_ints:
                positions = np.where(np.logical_not(are_ints))[0]
                positions_str = ", ".join([str(pos) for pos in positions])
                raise ValueError(
                    f"Arguments {positions_str} of the input are not integers."
                )
            shape = list(args)
        if shape[0] != -1:
            shape = [-1, *shape]
        self.shape = shape

    def __repr__(self):
        dimstr = ", ".join([str(x) for x in self.shape])
        return f"Reshape({dimstr})"

    def forward(self, x):
        return x.reshape(self.shape)


class O2Reshape(O2Layer):
    def __init__(self, *args, **kwargs):
        module = Reshape(*args, **kwargs)
        super().__init__(module)

    def get_output_input_jacobian(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        fan_in = x.numel()
        device = x.device
        indices = torch.tensor(
            [[i, i] for i in range(fan_in)], device=device, dtype=torch.int64
        ).T
        values = torch.empty(fan_in, device=device)
        values[:] = 1
        return torch.sparse_coo_tensor(indices, values, (fan_in, fan_in))

    def get_output_input_hessian(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        indices, values = torch.tensor([], device=x.device, dtype=torch.int64).reshape(
            2, 0
        ), torch.tensor([], device=x.device)
        return torch.sparse_coo_tensor(
            indices, values, (x.numel(), x.numel() * x.numel())
        )

    def get_loss_input_hessian(
        self,
        dLdy: torch.Tensor,
        dL2dy2: torch.Tensor,
        dydx: torch.Tensor,
        dy2dx2: torch.Tensor,
        **kwargs,
    ) -> torch.Tensor:
        return dL2dy2
