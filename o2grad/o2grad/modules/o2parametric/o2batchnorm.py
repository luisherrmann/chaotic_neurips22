from typing import Tuple
import torch
import torch.nn as nn

from .o2parametric import O2ParametricLayer
from o2grad.utils import flatten_multiindex


class O2BatchNorm1d(O2ParametricLayer):
    def __init__(self, *args, **kwargs):
        module = nn.BatchNorm1d(*args, **kwargs)
        super().__init__(module)

    def _get_pre_output_input_jacobian(self, x) -> Tuple[torch.Tensor, torch.Tensor]:
        bn = self.module
        bn.weight.data
        if len(x.shape) == 2:
            N, c = x.shape
            if bn.train:
                mean = torch.mean(x, dim=0)
                var = torch.var(x, dim=0, unbiased=False)
            else:
                mean = bn.running_mean
                var = bn.running_var
            range_N = torch.arange(N, device=x.device, dtype=torch.int64)
            range_c = torch.arange(c, device=x.device, dtype=torch.int64)
            idx = torch.cartesian_prod(range_N, range_N, range_c)
            idx_xi = flatten_multiindex(
                idx[:, torch.tensor([0, 2], device=x.device, dtype=torch.int64)], [N, c]
            )
            idx_xj = flatten_multiindex(idx[:, 1:], [N, c])
            # Get values for xi, xj
            xi = x.reshape(-1)[idx_xi]
            xj = x.reshape(-1)[idx_xj]
            mean_c = mean[idx[:, 2]]
            a = torch.sqrt(var[idx[:, 2]] + bn.eps)
            delta_ij = (idx[:, 0] == idx[:, 1]).to(x.dtype)
            val = (delta_ij - 1 / N) * 1 / a
            val += -1 / N * (xi - mean_c) * (xj - mean_c) / a**3
            idx[:, 0] = idx_xi
            idx[:, 1] = idx_xj
        elif len(x.shape) == 3:
            N, c, input_dim = x.shape
            if bn.train:
                mean = torch.mean(x, dim=[0, 2])
                var = torch.var(x, dim=[0, 2], unbiased=False)
            else:
                mean = bn.running_mean
                var = bn.running_var
            B = N * input_dim
            range_N = torch.arange(N, device=x.device, dtype=torch.int64)
            range_c = torch.arange(c, device=x.device, dtype=torch.int64)
            range_input_dim = torch.arange(
                input_dim, device=x.device, dtype=torch.int64
            )
            idx = torch.cartesian_prod(
                range_N, range_input_dim, range_N, range_c, range_input_dim
            )
            idx_c = idx[:, 3]
            idx_xi = flatten_multiindex(
                idx[:, torch.tensor([0, 3, 1], device=x.device, dtype=torch.int64)],
                [N, c, input_dim],
            )
            idx_xj = flatten_multiindex(idx[:, 2:], [N, c, input_dim])
            xi = x.reshape(-1)[idx_xi]
            xj = x.reshape(-1)[idx_xj]
            mean_c = mean[idx_c]
            a = torch.sqrt(var[idx_c] + bn.eps)
            delta_ij = (idx_xi == idx_xj).to(torch.int64)
            val = (delta_ij - 1 / B) * 1 / a
            val += -1 / B * (xi - mean_c) * (xj - mean_c) / a**3
            idx[:, 0] = idx_xi
            idx[:, 1] = idx_xj
            idx[:, 2] = idx[:, 3]
        return idx, val

    def get_output_input_jacobian(self, x: torch.Tensor, sparse=True) -> torch.Tensor:
        bn = self.module
        w = bn.weight.data
        idx, val = self._get_pre_output_input_jacobian(x)
        idx_c = idx[:, 2]
        w = w[idx_c]
        idx = idx[:, :2].T
        val = w * val
        dydx = torch.sparse_coo_tensor(idx, val, (x.numel(), x.numel())).coalesce()
        if not sparse:
            dydx = dydx.to_dense()
        return dydx

    def get_output_input_hessian(self, x: torch.Tensor, sparse=True) -> torch.Tensor:
        bn = self.module
        if len(x.shape) == 2:
            N, c = x.shape
            if bn.train:
                mean = torch.mean(x, dim=0)
                var = torch.var(x, dim=0, unbiased=False)
            else:
                mean = bn.running_mean
                var = bn.running_var
            # for i, j, k in itertools.product(range(2), range(2), range(2)):
            #     b = - (input[k] - mean) / B * a_inv**3
            #     c1 = ((input[j] - mean) * (k == i) + (input[i] - mean) * (k == j)) / B * a_inv**3
            #     c2 = - ((input[i] - mean) + (input[j] - mean)) / B**2 * a_inv**3
            #     c3 = - 3 * (input[i] - mean) * (input[j] - mean) * (input[k] - mean) / B**2 * a_inv**5
            #     H_[i, j, k] = b * (i == j) -b/B - (c1 + c2 + c3)
            range_N = torch.arange(N, dtype=torch.int64, device=x.device)
            range_c = torch.arange(c, dtype=torch.int64, device=x.device)
            idx = torch.cartesian_prod(range_N, range_N, range_N, range_c)
            # Get flattened indices for xi, xj, xk
            idx_xi = flatten_multiindex(
                idx[:, torch.tensor([0, 3], device=x.device, dtype=torch.int64)], [N, c]
            )
            idx_xj = flatten_multiindex(
                idx[:, torch.tensor([1, 3], device=x.device, dtype=torch.int64)], [N, c]
            )
            idx_xk = flatten_multiindex(idx[:, 2:], [N, c])
            # Get values for xi, xj, xk
            xi = x.reshape(-1)[idx_xi]
            xj = x.reshape(-1)[idx_xj]
            xk = x.reshape(-1)[idx_xk]
            mean_c = mean[idx[:, 3]]
            a = var[idx[:, 3]] + bn.eps
            b = -1 / N * (xk - mean_c) * torch.sqrt((1 / a) ** 3)
            c1 = (
                1
                / N
                * (
                    (xi - mean_c) * (idx[:, 1] == idx[:, 2])
                    + (xj - mean_c) * (idx[:, 0] == idx[:, 2])
                )
                * torch.sqrt((1 / a) ** 3)
            )
            c2 = (
                -1 / N**2 * ((xi - mean_c) + (xj - mean_c)) * torch.sqrt((1 / a) ** 3)
            )
            c3 = (
                -3
                / N**2
                * (xi - mean_c)
                * (xj - mean_c)
                * (xk - mean_c)
                * torch.sqrt((1 / a) ** 5)
            )
            val = b * (idx[:, 0] == idx[:, 1]) - b / N - (c1 + c2 + c3)
            # Prepare indices
            idx[:, 0] = idx_xi
            idx[:, 1] = idx_xj
            idx[:, 2] = idx_xk
            idx[:, 1] = flatten_multiindex(idx[:, 1:3], [N * c, N * c])
            idx = idx[:, :2].T
        elif len(x.shape) == 3:
            N, c, input_dim = x.shape
            if bn.train:
                mean = torch.mean(x, dim=[0, 2])
                var = torch.var(x, dim=[0, 2], unbiased=False)
            else:
                mean = bn.running_mean
                var = bn.running_var
            B = N * input_dim
            range_N = torch.arange(N, dtype=torch.int64, device=x.device)
            range_c = torch.arange(c, dtype=torch.int64, device=x.device)
            range_input_dim = torch.arange(
                input_dim, dtype=torch.int64, device=x.device
            )
            idx = torch.cartesian_prod(
                range_N,
                range_input_dim,
                range_N,
                range_input_dim,
                range_N,
                range_c,
                range_input_dim,
            )
            # Get flattened indices for xi, xj, xk
            idx_xi = flatten_multiindex(
                idx[:, torch.tensor([0, 5, 1], device=x.device, dtype=torch.int64)],
                [N, c, input_dim],
            )
            idx_xj = flatten_multiindex(
                idx[:, torch.tensor([2, 5, 3], device=x.device, dtype=torch.int64)],
                [N, c, input_dim],
            )
            idx_xk = flatten_multiindex(idx[:, 4:], [N, c, input_dim])
            # Get values for xi, xj, xk
            xi = x.reshape(-1)[idx_xi]
            xj = x.reshape(-1)[idx_xj]
            xk = x.reshape(-1)[idx_xk]
            mean_c = mean[idx[:, 5]]
            a = var[idx[:, 5]] + bn.eps
            b = -1 / B * (xk - mean_c) * torch.sqrt((1 / a) ** 3)
            delta_jk = idx_xj == idx_xk
            delta_ik = idx_xi == idx_xk
            c1 = (
                1
                / B
                * ((xi - mean_c) * delta_jk + (xj - mean_c) * delta_ik)
                * torch.sqrt((1 / a) ** 3)
            )
            c2 = (
                -1 / B**2 * ((xi - mean_c) + (xj - mean_c)) * torch.sqrt((1 / a) ** 3)
            )
            c3 = (
                -3
                / B**2
                * (xi - mean_c)
                * (xj - mean_c)
                * (xk - mean_c)
                * torch.sqrt((1 / a) ** 5)
            )
            delta_ij = (idx_xi == idx_xj).to(torch.int64)
            val = b * delta_ij - b / B - (c1 + c2 + c3)
            # Prepare indices
            idx[:, 0] = idx_xi
            idx[:, 1] = idx_xj
            idx[:, 2] = idx_xk
            idx[:, 1] = flatten_multiindex(idx[:, 1:3], [B * c, B * c])
            idx = idx[:, :2].T
        dy2dx2 = torch.sparse_coo_tensor(
            idx, val, (x.numel(), x.numel() * x.numel())
        ).coalesce()
        if not sparse:
            dy2dx2 = dy2dx2.to_dense()
        return dy2dx2

    def get_output_param_jacobian(self, x: torch.Tensor, sparse=True) -> torch.Tensor:
        bn = self.module
        y = self.get_output(x)
        w, b = bn.weight.data, bn.bias.data
        nw, nb = w.numel(), b.numel()
        n = nw + nb
        device = x.device
        if len(x.shape) == 2:
            N, c = x.shape
            x = x.reshape(N, c, 1)
            y = y.reshape(N, c, 1)
        N, c, input_dim = x.shape
        x_hat = (y - b.reshape(1, c, 1)) / w.reshape(1, c, 1)
        idx_x = torch.cartesian_prod(
            torch.arange(N, dtype=torch.int64, device=device),
            torch.arange(c, dtype=torch.int64, device=device),
            torch.arange(input_dim, dtype=torch.int64, device=device),
        )
        count = len(idx_x)
        idx = torch.empty([2 * count, 2], device=device, dtype=torch.int64)
        idx[:count, 0] = flatten_multiindex(idx_x, [N, c, input_dim])
        idx[count:, 0] = flatten_multiindex(idx_x, [N, c, input_dim])
        idx[:count, 1] = idx_x[:, 1]
        idx[count:, 1] = idx_x[:, 1] + nw
        idx = idx.T
        val = torch.empty([2 * count], device=device)
        val[:count] = x_hat.reshape(-1)[idx[0, :count]]
        val[count:] = 1
        dydp = torch.sparse_coo_tensor(idx, val, (y.numel(), n)).coalesce()
        if not sparse:
            dydp = dydp.to_dense()
        return dydp

    def get_output_param_hessian(
        self, x: torch.Tensor, sparse=True
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        bn = self.module
        w, b = bn.weight.data, bn.bias.data
        nw, nb = w.numel(), b.numel()
        n = nw + nb
        indices, val = torch.tensor([], device=x.device, dtype=torch.int64).reshape(
            2, 0
        ), torch.tensor([], device=x.device)
        dy2dp2 = torch.sparse_coo_tensor(indices, val, (x.numel(), n * n))
        if not sparse:
            dy2dp2 = dy2dp2.to_dense()
        return dy2dp2

    def get_mixed_output_param_hessian(
        self, x: torch.Tensor, sparse=True
    ) -> torch.Tensor:
        bn = self.module
        w, b = bn.weight.data, bn.bias.data
        nw, nb = w.numel(), b.numel()
        n = nw + nb
        idx, val = self._get_pre_output_input_jacobian(x)
        idx[:, 1] = flatten_multiindex(
            idx[:, torch.tensor([1, 2], device=x.device, dtype=torch.int64)],
            [x.numel(), n],
        )
        idx = idx[:, :2].T
        dy2dxdp = torch.sparse_coo_tensor(
            idx, val, (x.numel(), x.numel() * n)
        ).coalesce()
        if not sparse:
            dy2dxdp = dy2dxdp.to_dense()
        return dy2dxdp


class O2BatchNorm2d(O2ParametricLayer):
    def __init__(self, *args, **kwargs):
        module = nn.BatchNorm2d(*args, **kwargs)
        super().__init__(module)

    def _get_pre_output_input_jacobian(self, x) -> Tuple[torch.Tensor, torch.Tensor]:
        bn = self.module
        if len(x.shape) == 4:
            N, c, h, w = x.shape
            if bn.train:
                mean = torch.mean(x, dim=[0, 2, 3])
                var = torch.var(x, dim=[0, 2, 3], unbiased=False)
            else:
                mean = bn.running_mean
                var = bn.running_var
            B = N * h * w
            range_N = torch.arange(N, device=x.device, dtype=torch.int64)
            range_c = torch.arange(c, device=x.device, dtype=torch.int64)
            range_h = torch.arange(h, device=x.device, dtype=torch.int64)
            range_w = torch.arange(w, device=x.device, dtype=torch.int64)
            idx = torch.cartesian_prod(
                range_N, range_h, range_w, range_N, range_c, range_h, range_w
            )
            idx_c = idx[:, 4]
            idx_xi = flatten_multiindex(
                idx[:, torch.tensor([0, 4, 1, 2], device=x.device, dtype=torch.int64)],
                [N, c, h, w],
            )
            idx_xj = flatten_multiindex(idx[:, 3:], [N, c, h, w])
            xi = x.reshape(-1)[idx_xi]
            xj = x.reshape(-1)[idx_xj]
            mean_c = mean[idx_c]
            a = torch.sqrt(var[idx_c] + bn.eps)
            delta_ij = (idx_xi == idx_xj).to(torch.int64)
            val = (delta_ij - 1 / B) * 1 / a
            val += -1 / B * (xi - mean_c) * (xj - mean_c) / a**3
            idx[:, 0] = idx_xi
            idx[:, 1] = idx_xj
            idx[:, 2] = idx[:, 4]
        return idx, val

    def get_output_input_jacobian(self, x: torch.Tensor, sparse=True) -> torch.Tensor:
        bn = self.module
        w = bn.weight.data
        idx, val = self._get_pre_output_input_jacobian(x)
        idx_c = idx[:, 2]
        w = w[idx_c]
        idx = idx[:, :2].T
        val = w * val
        dydx = torch.sparse_coo_tensor(idx, val, (x.numel(), x.numel())).coalesce()
        if not sparse:
            dydx = dydx.to_dense()
        return dydx

    def get_output_input_hessian(self, x: torch.Tensor, sparse=True) -> torch.Tensor:
        bn = self.module
        N, c, h, w = x.shape
        if bn.train:
            mean = torch.mean(x, dim=[0, 2, 3])
            var = torch.var(x, dim=[0, 2, 3], unbiased=False)
        else:
            mean = bn.running_mean
            var = bn.running_var
        B = N * h * w
        range_N = torch.arange(N, dtype=torch.int64, device=x.device)
        range_c = torch.arange(c, dtype=torch.int64, device=x.device)
        range_h = torch.arange(h, dtype=torch.int64, device=x.device)
        range_w = torch.arange(w, dtype=torch.int64, device=x.device)
        idx = torch.cartesian_prod(
            range_N,
            range_h,
            range_w,
            range_N,
            range_h,
            range_w,
            range_N,
            range_c,
            range_h,
            range_w,
        )
        # Get flattened indices for xi, xj, xk
        idx_xi = flatten_multiindex(
            idx[:, torch.tensor([0, 7, 1, 2], device=x.device, dtype=torch.int64)],
            [N, c, h, w],
        )
        idx_xj = flatten_multiindex(
            idx[:, torch.tensor([3, 7, 4, 5], device=x.device, dtype=torch.int64)],
            [N, c, h, w],
        )
        idx_xk = flatten_multiindex(idx[:, 6:], [N, c, h, w])
        # Get values for xi, xj, xk
        xi = x.reshape(-1)[idx_xi]
        xj = x.reshape(-1)[idx_xj]
        xk = x.reshape(-1)[idx_xk]
        mean_c = mean[idx[:, 7]]
        a = var[idx[:, 7]] + bn.eps
        b = -1 / B * (xk - mean_c) * torch.sqrt((1 / a) ** 3)
        delta_jk = idx_xj == idx_xk
        delta_ik = idx_xi == idx_xk
        c1 = (
            1
            / B
            * ((xi - mean_c) * delta_jk + (xj - mean_c) * delta_ik)
            * torch.sqrt((1 / a) ** 3)
        )
        c2 = -1 / B**2 * ((xi - mean_c) + (xj - mean_c)) * torch.sqrt((1 / a) ** 3)
        c3 = (
            -3
            / B**2
            * (xi - mean_c)
            * (xj - mean_c)
            * (xk - mean_c)
            * torch.sqrt((1 / a) ** 5)
        )
        delta_ij = (idx_xi == idx_xj).to(torch.int64)
        val = b * delta_ij - b / B - (c1 + c2 + c3)
        # Prepare indices
        idx[:, 0] = idx_xi
        idx[:, 1] = idx_xj
        idx[:, 2] = idx_xk
        idx[:, 1] = flatten_multiindex(idx[:, 1:3], [B * c, B * c])
        idx = idx[:, :2].T
        dy2dx2 = torch.sparse_coo_tensor(
            idx, val, (x.numel(), x.numel() * x.numel())
        ).coalesce()
        if not sparse:
            dy2dx2 = dy2dx2.to_dense()
        return dy2dx2

    def get_output_param_jacobian(self, x: torch.Tensor, sparse=True) -> torch.Tensor:
        bn = self.module
        y = self.get_output(x)
        w, b = bn.weight.data, bn.bias.data
        nw, nb = w.numel(), b.numel()
        n = nw + nb
        # x_hat = (y - b) / w
        # Adding over channel dimension
        N, c, _, _ = x.shape
        x_hat = (y - b.reshape(1, c, 1, 1)) / w.reshape(1, c, 1, 1)
        _, _, h, w = x.shape
        device = x.device
        idx_x = torch.cartesian_prod(
            torch.arange(N, dtype=torch.int64, device=device),
            torch.arange(c, dtype=torch.int64, device=device),
            torch.arange(h, dtype=torch.int64, device=device),
            torch.arange(w, dtype=torch.int64, device=device),
        )
        count = len(idx_x)
        idx = torch.empty([2 * count, 2], device=device, dtype=torch.int64)
        idx[:count, 0] = flatten_multiindex(idx_x, [N, c, h, w])
        idx[count:, 0] = flatten_multiindex(idx_x, [N, c, h, w])
        idx[:count, 1] = idx_x[:, 1]
        idx[count:, 1] = idx_x[:, 1] + nw
        idx = idx.T
        val = torch.empty([2 * count], device=device)
        val[:count] = x_hat.reshape(-1)[idx[0, :count]]
        val[count:] = 1
        dydp = torch.sparse_coo_tensor(idx, val, (y.numel(), n)).coalesce()
        if not sparse:
            dydp = dydp.to_dense()
        return dydp

    def get_output_param_hessian(
        self, x: torch.Tensor, sparse=True
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        bn = self.module
        w, b = bn.weight.data, bn.bias.data
        nw, nb = w.numel(), b.numel()
        n = nw + nb
        indices, val = torch.tensor([], device=x.device, dtype=torch.int64).reshape(
            2, 0
        ), torch.tensor([], device=x.device)
        dy2dp2 = torch.sparse_coo_tensor(indices, val, (x.numel(), n * n))
        if not sparse:
            dy2dp2 = dy2dp2.to_dense()
        return dy2dp2

    def get_mixed_output_param_hessian(
        self, x: torch.Tensor, sparse=True
    ) -> torch.Tensor:
        bn = self.module
        w, b = bn.weight.data, bn.bias.data
        nw, nb = w.numel(), b.numel()
        n = nw + nb
        idx, val = self._get_pre_output_input_jacobian(x)
        idx[:, 1] = flatten_multiindex(
            idx[:, torch.tensor([1, 2], device=x.device, dtype=torch.int64)],
            [x.numel(), n],
        )
        idx = idx[:, :2].T
        dy2dxdp = torch.sparse_coo_tensor(
            idx, val, (x.numel(), x.numel() * n)
        ).coalesce()
        if not sparse:
            dy2dxdp = dy2dxdp.to_dense()
        return dy2dxdp
