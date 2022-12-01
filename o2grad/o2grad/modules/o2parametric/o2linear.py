import torch
import torch.nn as nn
from typing import Union, Tuple

from .o2parametric import O2ParametricLayer
from o2grad.utils import flatten_multiindex


def o2method(tname):
    def _o2method(func):
        def __o2method(self, x, sparse=None):
            if sparse is None:
                if tname in self.settings.return_layout:
                    sparse = self.settings.return_layout[tname] in [
                        torch.sparse_coo,
                        "sparse",
                    ]
                    return func(self, x, sparse=sparse)
                else:
                    return func(self, x)
            else:
                return func(self, x, sparse=sparse)

        return __o2method

    return _o2method


class O2Linear(O2ParametricLayer):
    def __init__(self, *args, **kwargs):
        module = nn.Linear(*args, **kwargs)
        super().__init__(module)

    @o2method("dydx")
    def get_output_input_jacobian(self, x: torch.Tensor, sparse=False):
        linear = self.module
        N = x.shape[0]
        w = linear.weight
        output_dim, input_dim = w.shape
        device = w.device
        kronecker_N = torch.eye(N, device=device)
        if not sparse:
            dydx = torch.einsum("ij,kl->ikjl", kronecker_N, torch.clone(w).detach())
            dydx = dydx.reshape(N * output_dim, N * input_dim)
        else:
            idx = torch.cartesian_prod(
                torch.arange(N, dtype=torch.int64, device=device),
                torch.arange(output_dim, dtype=torch.int64, device=device),
                torch.arange(input_dim, dtype=torch.int64, device=device),
            )
            idx_y = flatten_multiindex(idx[:, :2], [N, output_dim])
            idx_x = flatten_multiindex(
                idx[:, torch.tensor([0, 2], dtype=torch.int64, device=device)],
                [N, input_dim],
            )
            idx_w = flatten_multiindex(idx[:, 1:], [output_dim, input_dim])
            val = w.reshape(-1)[idx_w]
            # Combine indices
            idx = torch.empty([2, len(idx)], dtype=torch.int64, device=device)
            idx[0, :] = idx_y
            idx[1, :] = idx_x
            dydx = torch.sparse_coo_tensor(
                idx, val, [N * output_dim, N * input_dim]
            ).coalesce()
        return dydx

    @o2method("dy2dx2")
    def get_output_input_hessian(self, x: torch.Tensor, sparse=True):
        linear = self.module
        N = x.shape[0]
        output_dim, input_dim = linear.weight.shape
        device = linear.weight.device
        idx, val = torch.tensor([], dtype=torch.int64, device=device).reshape(
            2, 0
        ), torch.tensor([], device=device)
        dy2dx2 = torch.sparse_coo_tensor(
            idx, val, (N * output_dim, x.numel() * x.numel())
        ).coalesce()
        if not sparse:
            dy2dx2 = dy2dx2.to_dense()
        return dy2dx2

    @o2method("dydp")
    def get_output_param_jacobian(
        self, x: torch.Tensor, sparse=False
    ) -> Union[torch.Tensor, Tuple[torch.Tensor]]:
        # Retrieve input, output, parameter dimensions
        linear = self.module
        w, b = linear.weight.data, linear.bias.data
        output_dim, input_dim = w.shape
        nw = input_dim * output_dim
        nb = output_dim
        n = nw + nb
        N = x.shape[0]
        device = x.device
        # Jacobian wrt weights, i.e. dydw
        if not sparse:
            kronecker_x = torch.eye(output_dim, device=device)
            dydw = torch.einsum("nk,ij->nijk", x, kronecker_x)
            dydw = dydw.reshape(N, output_dim, nw)
            # Jacobian wrt bias, i.e. dydb
            dydb = kronecker_x.repeat([N, 1]).reshape(N, output_dim, nb)
            # Combine OPJ for weights and bias
            dydp = torch.empty([N, output_dim, n], device=device)
            dydp[:, :, :nw] = dydw
            dydp[:, :, nw:] = dydb
            dydp = dydp.reshape(N * output_dim, n)
        else:
            # Generate indices for weights:
            idx = torch.cartesian_prod(
                torch.arange(N, dtype=torch.int64, device=device),
                torch.arange(output_dim, dtype=torch.int64, device=device),
                torch.arange(input_dim, dtype=torch.int64, device=device),
            )
            idx_wy = flatten_multiindex(idx[:, :2], [N, output_dim])
            idx_wx = flatten_multiindex(
                idx[:, torch.tensor([0, 2], dtype=torch.int64, device=device)],
                [N, input_dim],
            )
            idx_ww = flatten_multiindex(idx[:, 1:], [output_dim, input_dim])
            val_ww = x.reshape(-1)[idx_wx]
            # Generate indices for bias:
            idx = torch.cartesian_prod(
                torch.arange(N, dtype=torch.int64, device=device),
                torch.arange(output_dim, dtype=torch.int64, device=device),
            )
            idx_by = flatten_multiindex(idx, [N, output_dim])
            idx_bb = nw + idx[:, 1]
            # Combine indices:
            idx = torch.empty(
                [2, len(idx_wy) + len(idx_by)], dtype=torch.int64, device=device
            )
            idx[0, : len(idx_wy)] = idx_wy
            idx[1, : len(idx_wy)] = idx_ww
            idx[0, len(idx_wy) :] = idx_by
            idx[1, len(idx_wy) :] = idx_bb
            val = torch.empty(len(idx_wy) + len(idx_by), dtype=x.dtype, device=device)
            val[:] = 0
            val[: len(val_ww)] = val_ww
            val[len(val_ww) :] = 1.0
            dydp = torch.sparse_coo_tensor(idx, val, [N * output_dim, n]).coalesce()
        return dydp

    @o2method("dy2dp2")
    def get_output_param_hessian(self, x: torch.Tensor, sparse=True) -> torch.Tensor:
        # Retrieve input, output, parameter dimensions
        linear = self.module
        w, b = linear.weight.data, linear.bias.data
        output_dim, input_dim = w.shape
        nw = input_dim * output_dim
        nb = output_dim
        n = nw + nb
        N = x.shape[0]
        device = w.device
        idx, val = torch.tensor([], device=device).reshape(2, 0), torch.tensor(
            [], device=device
        )
        dy2dp2 = torch.sparse_coo_tensor(idx, val, (N * output_dim, n * n)).coalesce()
        if not sparse:
            dy2dp2 = dy2dp2.to_dense()
        return dy2dp2

    @o2method("dy2dxdp")
    def get_mixed_output_param_hessian(
        self, x: torch.Tensor, sparse=True
    ) -> torch.Tensor:
        # Retrieve input, output, parameter dimensions
        linear = self.module
        w, b = linear.weight.data, linear.bias.data
        output_dim, input_dim = w.shape
        nw = input_dim * output_dim
        nb = output_dim
        n = nw + nb
        N = x.shape[0]
        device = w.device
        if not sparse:
            kronecker_y = torch.eye(output_dim, device=device)
            kronecker_xw = torch.eye(input_dim, device=device)
            kronecker_b = torch.eye(N, device=device)
            dy2dxdw = torch.einsum(
                "ijkl,mn->ijklmn",
                torch.einsum("ij,kl->ijkl", kronecker_y, kronecker_xw),
                kronecker_b,
            ).permute([4, 0, 5, 2, 1, 3])
            dy2dxdb = torch.zeros([N, output_dim, N * input_dim, nb], device=device)
            # Combine matrix for weights and parameters
            dy2dxdp = torch.empty([N, output_dim, N * input_dim, n], device=device)
            dy2dxdp[:, :, :, :nw] = dy2dxdw.reshape(N, output_dim, N * input_dim, nw)
            dy2dxdp[:, :, :, nw:] = dy2dxdb
            dy2dxdp = dy2dxdp.reshape(N * output_dim, N * input_dim * n)
        else:
            idx = torch.cartesian_prod(
                torch.arange(N, dtype=torch.int64, device=device),
                torch.arange(output_dim, dtype=torch.int64, device=device),
                torch.arange(input_dim, dtype=torch.int64, device=device),
            )
            # Transform indices to match output shape
            idx_y = flatten_multiindex(idx[:, :2], [N, output_dim])
            idx_w = flatten_multiindex(idx[:, 1:], [output_dim, input_dim])
            idx[:, 1] = idx[:, 2]
            idx[:, 2] = idx_w
            idx_xw = flatten_multiindex(idx[:, :], [N, input_dim, n])
            # Combine indices
            idx = torch.empty(
                [2, N * input_dim * output_dim], dtype=torch.int64, device=device
            )
            idx[0, :] = idx_y
            idx[1, :] = idx_xw
            val = torch.empty([N * input_dim * output_dim], device=device)
            val[:] = 1
            dy2dxdp = torch.sparse_coo_tensor(
                idx, val, (N * output_dim, N * input_dim * n)
            ).coalesce()
        return dy2dxdp
