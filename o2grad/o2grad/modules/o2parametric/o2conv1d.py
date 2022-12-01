from typing import Union, Tuple
import torch
import torch.nn as nn
from .o2parametric import O2ParametricLayer
from o2grad.utils import (
    cartesian_prod_2d,
    get_conv_output_shape,
    get_tconv_output_shape,
    flatten_multiindex,
)


class O2Conv1d(O2ParametricLayer):
    def __init__(self, *args, **kwargs):
        module = nn.Conv1d(*args, **kwargs)
        super().__init__(module)

    def get_output_input_jacobian(self, x: torch.Tensor, sparse=True) -> torch.Tensor:
        # Retrieve input, output, parameter dimensions
        conv1d = self.module
        stride = conv1d.stride[0]
        padding = conv1d.padding[0]
        dilation = conv1d.dilation[0]
        w = conv1d.weight.data
        N, _, input_dim = x.shape
        c_out, c_in, K = w.shape
        output_dim = get_conv_output_shape(input_dim, K, stride, padding, dilation)
        device = x.device
        # Equivalent code using Python lists (slow):
        # idx1 = [[i, j] for i, j in itertools.product(range(output_dim), range(input_dim)) if \
        #     (0 <= j - i * stride + padding and j - i * stride + padding < dilation * K and (j - i * stride + padding) % dilation == 0)]
        # idx1 = torch.tensor(idx1, dtype=torch.int64, device=device)
        idx1 = torch.cartesian_prod(
            torch.arange(output_dim, dtype=torch.int64, device=device),
            torch.arange(input_dim, dtype=torch.int64, device=device),
        )
        i, j = idx1[:, 0], idx1[:, 1]
        mask = (
            (0 <= j - i * stride + padding)
            & (j - i * stride + padding < dilation * K)
            & ((j - i * stride + padding) % dilation == 0)
        )
        idx1 = idx1[mask, :]
        # Cartesian product with c_out, c_in, b
        idx2 = torch.cartesian_prod(
            torch.arange(c_out, dtype=torch.int64, device=device),
            torch.arange(c_in, dtype=torch.int64, device=device),
            torch.arange(N, dtype=torch.int64, device=device),
        )
        idx = cartesian_prod_2d(idx1, idx2)
        # Index order: input_dim, output_dim, c_out, c_in, N
        # Flatten weight indices
        idx_w = torch.empty([len(idx), 3], dtype=torch.int64, device=device)
        idx_w[:, 0] = idx[:, 2]  # output channel idx
        idx_w[:, 1] = idx[:, 3]  # input channel idx
        idx_w[:, 2] = (
            idx[:, 1] - idx[:, 0] * stride + padding
        ) / dilation  # kernel idx
        idx_w = flatten_multiindex(idx_w, w.shape)
        val = w.view(-1)[idx_w]
        # dydx indices
        idx[:, 0] = flatten_multiindex(
            idx[:, torch.tensor([4, 2, 0], dtype=torch.int64, device=device)],
            [N, c_out, output_dim],
        )
        idx[:, 1] = flatten_multiindex(
            idx[:, torch.tensor([4, 3, 1], dtype=torch.int64, device=device)], x.shape
        )
        idx = idx[:, :2].T
        dydx = torch.sparse_coo_tensor(
            idx, val, (N * c_out * output_dim, N * c_in * input_dim)
        ).coalesce()
        if not sparse:
            dydx = dydx.to_dense()
        return dydx

    def get_output_input_hessian(self, x: torch.Tensor, sparse=True) -> torch.Tensor:
        # Retrieve input, output, parameter dimensions
        conv1d = self.module
        stride = conv1d.stride[0]
        padding = conv1d.padding[0]
        dilation = conv1d.dilation[0]
        w = conv1d.weight.data
        c_out, _, K = w.shape
        N, _, input_dim = x.shape
        output_dim = get_conv_output_shape(input_dim, K, stride, padding, dilation)
        device = x.device
        idx, val = torch.tensor([], dtype=torch.int64, device=device).reshape(
            2, 0
        ), torch.tensor([], device=device)
        dy2dx2 = torch.sparse_coo_tensor(
            idx, val, (N * c_out * output_dim, x.numel() * x.numel())
        ).coalesce()
        if not sparse:
            dy2dx2 = dy2dx2.to_dense()
        return dy2dx2

    def get_output_param_jacobian(
        self, x: torch.Tensor, sparse=True
    ) -> Union[torch.Tensor, Tuple[torch.Tensor]]:
        # Retrieve input, output, parameter dimensions
        conv1d = self.module
        stride = conv1d.stride[0]
        padding = conv1d.padding[0]
        dilation = conv1d.dilation[0]
        w, b = conv1d.weight.data, conv1d.bias.data
        c_out, c_in, K = w.shape
        (
            N,
            _,
            input_dim,
        ) = x.shape
        nw, nb = w.numel(), b.numel()
        n = nw + nb
        output_dim = get_conv_output_shape(input_dim, K, stride, padding, dilation)
        device = x.device
        # Equivalent code using Python lists (slow):
        # idx_w1 = [[i, k] for i, k in itertools.product(range(output_dim), range(K)) if \
        #     (0 <= k * dilation + i * stride - padding and k * dilation + i * stride  - padding < input_dim)]
        # idx_w1 = torch.tensor(idx_w1, dtype=torch.int64, device=device)
        idx1 = torch.cartesian_prod(
            torch.arange(output_dim, dtype=torch.int64, device=device),
            torch.arange(K, dtype=torch.int64, device=device),
        )
        i, k = idx1[:, 0], idx1[:, 1]
        mask = (0 <= k * dilation + i * stride - padding) & (
            k * dilation + i * stride - padding < input_dim
        )
        idx1 = idx1[mask, :]
        # Cartesian product with c_out, c_in, b
        idx2 = torch.cartesian_prod(
            torch.arange(c_out, dtype=torch.int64, device=device),
            torch.arange(c_in, dtype=torch.int64, device=device),
            torch.arange(N, dtype=torch.int64, device=device),
        )
        idx_w = cartesian_prod_2d(idx1, idx2)
        # Index order: output_dim, K, c_out, c_in, N
        # Flattened input indices
        idx_x = torch.empty([len(idx_w), 3], dtype=torch.int64, device=device)
        idx_x[:, 0] = idx_w[:, 4]
        idx_x[:, 1] = idx_w[:, 3]
        idx_x[:, 2] = idx_w[:, 1] * dilation + idx_w[:, 0] * stride - padding
        idx_x_flat = flatten_multiindex(idx_x, x.shape)
        val_w = x.reshape(-1)[idx_x_flat]
        # Flattened weight indices
        idx_selection = torch.tensor([4, 2, 0], dtype=torch.int64, device=device)
        idx_w[:, 0] = flatten_multiindex(
            idx_w[:, idx_selection], [N, c_out, output_dim]
        )
        idx_selection = torch.tensor([2, 3, 1], dtype=torch.int64, device=device)
        idx_w[:, 1] = flatten_multiindex(idx_w[:, idx_selection], [c_out, c_in, K])
        # Flattened bias indices
        # iterator_b = itertools.product(range(c_out), range(output_dim), range(N))
        # idx_b = [[b, c, i, nw + c] for c, i, b in iterator_b]
        # idx_b = torch.tensor(idx_b, dtype=torch.int64, device=device).reshape(-1, 4)
        idx_b = torch.cartesian_prod(
            torch.arange(N, dtype=torch.int64, device=device),
            torch.arange(c_out, dtype=torch.int64, device=device),
            torch.arange(output_dim, dtype=torch.int64, device=device),
        )
        idx_b[:, 0] = flatten_multiindex(idx_b[:, :3], [N, c_out, output_dim])
        # Add offset of weight parameter count (to avoid indices overlapping)
        idx_b[:, 1] += nw
        # dydp indices
        idx = torch.empty(
            [len(idx_w) + len(idx_b), 2], dtype=torch.int64, device=w.device
        )
        idx[: len(idx_w), :] = idx_w[:, :2]
        idx[len(idx_w) :, :] = idx_b[:, :2]
        # Combined parameter values
        val = torch.empty([len(idx_w) + len(idx_b)], dtype=w.dtype, device=w.device)
        val[: len(val_w)] = val_w
        val[len(val_w) :] = 1
        dydp = torch.sparse_coo_tensor(
            idx.T, val, (N * c_out * output_dim, n)
        ).coalesce()
        if not sparse:
            dydp = dydp.to_dense()
        return dydp

    def get_output_param_hessian(self, x: torch.Tensor, sparse=True) -> torch.Tensor:
        # Retrieve input, output, parameter dimensions
        conv1d = self.module
        stride = conv1d.stride[0]
        padding = conv1d.padding[0]
        dilation = conv1d.dilation[0]
        w, b = conv1d.weight.data, conv1d.bias.data
        c_out, c_in, K = w.shape
        (
            N,
            _,
            input_dim,
        ) = x.shape
        nw, nb = w.numel(), b.numel()
        n = nw + nb
        output_dim = get_conv_output_shape(input_dim, K, stride, padding, dilation)
        device = x.device
        # Output-Parameter-Hessian (OPH):
        indices, values = torch.tensor([], dtype=torch.int64, device=device).reshape(
            2, 0
        ), torch.tensor([], device=device)
        dy2dp2 = torch.sparse_coo_tensor(
            indices, values, (N * c_out * output_dim, n * n)
        ).coalesce()
        if not sparse:
            dy2dp2 = dy2dp2.to_dense()
        return dy2dp2

    def get_mixed_output_param_hessian(
        self, x: torch.Tensor, sparse=True
    ) -> torch.Tensor:
        # Retrieve input, output, parameter dimensions
        conv1d = self.module
        stride = conv1d.stride[0]
        padding = conv1d.padding[0]
        dilation = conv1d.dilation[0]
        w, b = conv1d.weight.data, conv1d.bias.data
        c_out, c_in, K = w.shape
        (
            N,
            _,
            input_dim,
        ) = x.shape
        nw, nb = w.numel(), b.numel()
        n = nw + nb
        output_dim = get_conv_output_shape(input_dim, K, stride, padding, dilation)
        device = x.device
        # mixed Output-Parameter-Hessian (mOPH):
        # Output, input, kernel indices
        # iterator = itertools.product(range(output_dim), range(input_dim), range(K))
        # idx1 = [[i, j, k] for i, j, k in iterator if (j - i * stride + padding == dilation * k)]
        # idx1 = torch.tensor(idx1, dtype=torch.int64, device=device)
        idx1 = torch.cartesian_prod(
            torch.arange(output_dim, dtype=torch.int64, device=device),
            torch.arange(input_dim, dtype=torch.int64, device=device),
            torch.arange(K, dtype=torch.int64, device=device),
        )
        i, j, k = idx1[:, 0], idx1[:, 1], idx1[:, 2]
        mask = j - i * stride + padding == dilation * k
        idx1 = idx1[mask, :]
        # Cartesian product with c_out, c_in, b
        idx2 = torch.cartesian_prod(
            torch.arange(c_out, dtype=torch.int64, device=device),
            torch.arange(c_in, dtype=torch.int64, device=device),
            torch.arange(N, dtype=torch.int64, device=device),
        )
        idx = cartesian_prod_2d(idx1, idx2)
        # Flattened output indices
        idx_y = idx[:, torch.tensor([5, 3, 0], dtype=torch.int64, device=device)]
        idx_y = flatten_multiindex(idx_y, [N, c_out, output_dim])
        # Flattened weight indices
        idx_w = idx[:, torch.tensor([3, 4, 2], dtype=torch.int64, device=device)]
        idx_w = flatten_multiindex(idx_w, w.shape)
        # Plattened parameter + weight indices
        idx_xw = torch.empty([len(idx), 4], device=device)
        idx_xw[:, :3] = idx[
            :, torch.tensor([5, 4, 1], dtype=torch.int64, device=device)
        ]
        idx_xw[:, 3] = idx_w
        idx_xw = flatten_multiindex(idx_xw, [*x.shape, n])
        # dy2dxdp indices
        idx = torch.empty([2, len(idx_y)], dtype=torch.int64, device=device)
        idx[0, :] = idx_y
        idx[1, :] = idx_xw
        val = torch.empty([len(idx_y)], device=device)
        val[:] = 1
        dy2dxdp = torch.sparse_coo_tensor(
            idx, val, [N * c_out * output_dim, x.numel() * n]
        ).coalesce()
        if not sparse:
            dy2dxdp = dy2dxdp.to_dense()
        return dy2dxdp


class O2ConvTranspose1d(O2ParametricLayer):
    def __init__(self, *args, **kwargs):
        module = nn.ConvTranspose1d(*args, **kwargs)
        super().__init__(module)

    def get_output_input_jacobian(self, x: torch.Tensor, sparse=True) -> torch.Tensor:
        # Retrieve input, output, parameter dimensions
        tconv1d = self.module
        stride = tconv1d.stride[0]
        padding = tconv1d.padding[0]
        dilation = tconv1d.dilation[0]
        w = tconv1d.weight.data
        N, _, input_dim = x.shape
        c_in, c_out, K = w.shape
        output_dim = get_tconv_output_shape(input_dim, K, stride, padding, dilation)
        device = x.device
        # Equivalent code using Python lists (slow):
        # idx1 = [[i, j] for i, j in itertools.product(range(output_dim), range(input_dim)) if \
        #     (0 <= i - j * stride + padding and i - j * stride + padding < dilation * K and (i - j * stride + padding) % dilation == 0)]
        # idx1 = torch.tensor(idx1, dtype=torch.int64, device=device)
        idx1 = torch.cartesian_prod(
            torch.arange(output_dim, dtype=torch.int64, device=device),
            torch.arange(input_dim, dtype=torch.int64, device=device),
        )
        i, j = idx1[:, 0], idx1[:, 1]
        mask = (
            (0 <= i - j * stride + padding)
            & (i - j * stride + padding < dilation * K)
            & ((i - j * stride + padding) % dilation == 0)
        )
        idx1 = idx1[mask, :]
        # Cartesian product with c_out, c_in, b
        idx2 = torch.cartesian_prod(
            torch.arange(c_out, dtype=torch.int64, device=device),
            torch.arange(c_in, dtype=torch.int64, device=device),
            torch.arange(N, dtype=torch.int64, device=device),
        )
        idx = cartesian_prod_2d(idx1, idx2)
        # Index order: input_dim, output_dim, c_out, c_in, N
        # Flatten weight indices
        idx_w = torch.empty([len(idx), 3], dtype=torch.int64, device=device)
        idx_w[:, 0] = idx[:, 3]  # input channel idx
        idx_w[:, 1] = idx[:, 2]  # output channel idx
        idx_w[:, 2] = (
            idx[:, 0] - idx[:, 1] * stride + padding
        ) / dilation  # kernel idx
        idx_w = flatten_multiindex(idx_w, w.shape)
        val = w.view(-1)[idx_w]
        # dydx indices
        idx[:, 0] = flatten_multiindex(
            idx[:, torch.tensor([4, 2, 0], dtype=torch.int64, device=device)],
            [N, c_out, output_dim],
        )
        idx[:, 1] = flatten_multiindex(
            idx[:, torch.tensor([4, 3, 1], dtype=torch.int64, device=device)], x.shape
        )
        idx = idx[:, :2].T
        dydx = torch.sparse_coo_tensor(
            idx, val, (N * c_out * output_dim, N * c_in * input_dim)
        ).coalesce()
        if not sparse:
            dydx = dydx.to_dense()
        return dydx

    def get_output_input_hessian(self, x: torch.Tensor, sparse=True) -> torch.Tensor:
        # Retrieve input, output, parameter dimensions
        convt1d = self.module
        stride = convt1d.stride[0]
        padding = convt1d.padding[0]
        dilation = convt1d.dilation[0]
        w = convt1d.weight.data
        c_in, c_out, K = w.shape
        N, _, input_dim = x.shape
        output_dim = get_tconv_output_shape(input_dim, K, stride, padding, dilation)
        device = x.device
        idx, val = torch.tensor([], dtype=torch.int64, device=device).reshape(
            2, 0
        ), torch.tensor([], device=device)
        dy2dx2 = torch.sparse_coo_tensor(
            idx, val, (N * c_out * output_dim, x.numel() * x.numel())
        ).coalesce()
        if not sparse:
            dy2dx2 = dy2dx2.to_dense()
        return dy2dx2

    def get_output_param_jacobian(
        self, x: torch.Tensor, sparse=True
    ) -> Union[torch.Tensor, Tuple[torch.Tensor]]:
        # Retrieve input, output, parameter dimensions
        convt1d = self.module
        stride = convt1d.stride[0]
        padding = convt1d.padding[0]
        dilation = convt1d.dilation[0]
        w, b = convt1d.weight.data, convt1d.bias.data
        c_in, c_out, K = w.shape
        (
            N,
            _,
            input_dim,
        ) = x.shape
        nw, nb = w.numel(), b.numel()
        n = nw + nb
        output_dim = get_tconv_output_shape(input_dim, K, stride, padding, dilation)
        device = x.device
        # Equivalent code using Python lists (slow):
        # idx_w1 = [[i, k] for i, k in itertools.product(range(output_dim), range(K)) if \
        #     (0 <= i + padding - k * dilation and i + padding - k * dilation < input_dim * stride) and (i + padding - k * dilation) % stride == 0]
        # idx_w1 = torch.tensor(idx_w1, dtype=torch.int64, device=device)
        idx1 = torch.cartesian_prod(
            torch.arange(output_dim, dtype=torch.int64, device=device),
            torch.arange(K, dtype=torch.int64, device=device),
        )
        i, k = idx1[:, 0], idx1[:, 1]
        mask = (
            (0 <= i + padding - k * dilation)
            & (i + padding - k * dilation < stride * input_dim)
            & ((i + padding - k * dilation) % stride == 0)
        )
        idx1 = idx1[mask, :]
        # Cartesian product with c_out, c_in, b
        idx2 = torch.cartesian_prod(
            torch.arange(c_out, dtype=torch.int64, device=device),
            torch.arange(c_in, dtype=torch.int64, device=device),
            torch.arange(N, dtype=torch.int64, device=device),
        )
        idx_w = cartesian_prod_2d(idx1, idx2)
        # Index order: output_dim, K, c_out, c_in, N
        # Flattened input indices
        idx_x = torch.empty([len(idx_w), 3], dtype=torch.int64, device=device)
        idx_x[:, 0] = idx_w[:, 4]
        idx_x[:, 1] = idx_w[:, 3]
        idx_x[:, 2] = (idx_w[:, 0] + padding - idx_w[:, 1] * dilation) / stride
        idx_x_flat = flatten_multiindex(idx_x, x.shape)
        val_w = x.reshape(-1)[idx_x_flat]
        # Flattened weight indices
        idx_selection = torch.tensor([4, 2, 0], dtype=torch.int64, device=device)
        idx_w[:, 0] = flatten_multiindex(
            idx_w[:, idx_selection], [N, c_out, output_dim]
        )
        idx_selection = torch.tensor([2, 3, 1], dtype=torch.int64, device=device)
        idx_w[:, 1] = flatten_multiindex(idx_w[:, idx_selection], [c_out, c_in, K])
        # Flattened bias indices
        # iterator_b = itertools.product(range(c_out), range(output_dim), range(N))
        # idx_b = [[b, c, i, nw + c] for c, i, b in iterator_b]
        # idx_b = torch.tensor(idx_b, dtype=torch.int64, device=device).reshape(-1, 4)
        idx_b = torch.cartesian_prod(
            torch.arange(N, dtype=torch.int64, device=device),
            torch.arange(c_out, dtype=torch.int64, device=device),
            torch.arange(output_dim, dtype=torch.int64, device=device),
        )
        idx_b[:, 0] = flatten_multiindex(idx_b[:, :3], [N, c_out, output_dim])
        # Add offset of weight parameter count (to avoid indices overlapping)
        idx_b[:, 1] += nw
        # dydp indices
        idx = torch.empty(
            [len(idx_w) + len(idx_b), 2], dtype=torch.int64, device=w.device
        )
        idx[: len(idx_w), :] = idx_w[:, :2]
        idx[len(idx_w) :, :] = idx_b[:, :2]
        # Combined parameter values
        val = torch.empty([len(idx_w) + len(idx_b)], dtype=w.dtype, device=w.device)
        val[: len(val_w)] = val_w
        val[len(val_w) :] = 1
        dydp = torch.sparse_coo_tensor(
            idx.T, val, (N * c_out * output_dim, n)
        ).coalesce()
        if not sparse:
            dydp = dydp.to_dense()
        return dydp

    def get_output_param_hessian(self, x: torch.Tensor, sparse=True) -> torch.Tensor:
        # Retrieve input, output, parameter dimensions
        conv1d = self.module
        stride = conv1d.stride[0]
        padding = conv1d.padding[0]
        dilation = conv1d.dilation[0]
        w, b = conv1d.weight.data, conv1d.bias.data
        c_in, c_out, K = w.shape
        (
            N,
            _,
            input_dim,
        ) = x.shape
        nw, nb = w.numel(), b.numel()
        n = nw + nb
        output_dim = get_tconv_output_shape(input_dim, K, stride, padding, dilation)
        device = x.device
        indices, values = torch.tensor([], dtype=torch.int64, device=device).reshape(
            2, 0
        ), torch.tensor([], device=device)
        dy2dp2 = torch.sparse_coo_tensor(
            indices, values, (N * c_out * output_dim, n * n)
        ).coalesce()
        if not sparse:
            dy2dp2 = dy2dp2.to_dense()
        return dy2dp2

    def get_mixed_output_param_hessian(
        self, x: torch.Tensor, sparse=True
    ) -> torch.Tensor:
        conv1d = self.module
        stride = conv1d.stride[0]
        padding = conv1d.padding[0]
        dilation = conv1d.dilation[0]
        w, b = conv1d.weight.data, conv1d.bias.data
        c_in, c_out, K = w.shape
        (
            N,
            _,
            input_dim,
        ) = x.shape
        nw, nb = w.numel(), b.numel()
        n = nw + nb
        output_dim = get_tconv_output_shape(input_dim, K, stride, padding, dilation)
        device = x.device
        # Output, input, kernel indices
        # iterator = itertools.product(range(output_dim), range(input_dim), range(K))
        # idx1 = [[i, j, k] for i, j, k in iterator if (i - j * stride + padding == dilation * k)]
        # idx1 = torch.tensor(idx1, dtype=torch.int64, device=device)
        idx1 = torch.cartesian_prod(
            torch.arange(output_dim, dtype=torch.int64, device=device),
            torch.arange(input_dim, dtype=torch.int64, device=device),
            torch.arange(K, dtype=torch.int64, device=device),
        )
        i, j, k = idx1[:, 0], idx1[:, 1], idx1[:, 2]
        mask = i - j * stride + padding == dilation * k
        idx1 = idx1[mask, :]
        # Cartesian product with c_out, c_in, b
        idx2 = torch.cartesian_prod(
            torch.arange(c_out, dtype=torch.int64, device=device),
            torch.arange(c_in, dtype=torch.int64, device=device),
            torch.arange(N, dtype=torch.int64, device=device),
        )
        idx = cartesian_prod_2d(idx1, idx2)
        # Flattened output indices
        idx_y = idx[:, torch.tensor([5, 3, 0], dtype=torch.int64, device=device)]
        idx_y = flatten_multiindex(idx_y, [N, c_out, output_dim])
        # Flattened weight indices
        idx_w = idx[:, torch.tensor([4, 3, 2], dtype=torch.int64, device=device)]
        idx_w = flatten_multiindex(idx_w, w.shape)
        # Flattened parameter + weight indices
        idx_xw = torch.empty([len(idx), 4], device=device)
        idx_xw[:, :3] = idx[
            :, torch.tensor([5, 4, 1], dtype=torch.int64, device=device)
        ]
        idx_xw[:, 3] = idx_w
        idx_xw = flatten_multiindex(idx_xw, [*x.shape, n])
        # dy2dxdp indices
        idx = torch.empty([2, len(idx_y)], dtype=torch.int64, device=device)
        idx[0, :] = idx_y
        idx[1, :] = idx_xw
        val = torch.empty([len(idx_y)], device=device)
        val[:] = 1
        dy2dxdp = torch.sparse_coo_tensor(
            idx, val, [N * c_out * output_dim, x.numel() * n]
        ).coalesce()
        if not sparse:
            dy2dxdp = dy2dxdp.to_dense()
        return dy2dxdp
