from typing import Tuple, Union
import torch
import torch.nn as nn

from .o2layer import O2Layer
from o2grad.utils import cartesian_prod_2d, flatten_multiindex, get_conv_output_shape


class O2AvgPool1d(O2Layer):
    def __init__(self, *args, **kwargs):
        module = nn.AvgPool1d(*args, **kwargs)
        super().__init__(module)

    def get_output_input_jacobian(self, x: torch.Tensor) -> torch.Tensor:
        avg = self.module
        stride = avg.stride[0]
        padding = avg.padding[0]
        K = avg.kernel_size[0]
        N, c_in, input_dim = x.shape
        output_dim = get_conv_output_shape(input_dim, K, stride, padding, dilation=1)
        device = x.device
        idx1 = torch.cartesian_prod(
            torch.arange(output_dim, dtype=torch.int64, device=device),
            torch.arange(input_dim, dtype=torch.int64, device=device),
        )
        i, j = idx1[:, 0], idx1[:, 1]
        mask = (0 <= j - i * stride + padding) & (j - i * stride + padding < K)
        idx1 = idx1[mask, :]
        idx2 = torch.cartesian_prod(
            torch.arange(N, dtype=torch.int64, device=device),
            torch.arange(c_in, dtype=torch.int64, device=device),
        )
        idx = cartesian_prod_2d(idx1, idx2)
        idx[:, 0] = flatten_multiindex(
            idx[:, torch.tensor([2, 3, 0], dtype=torch.int64, device=device)],
            [N, c_in, output_dim],
        )
        idx[:, 1] = flatten_multiindex(
            idx[:, torch.tensor([2, 3, 1], dtype=torch.int64, device=device)],
            [N, c_in, input_dim],
        )
        idx = idx[:, :2].T
        val = torch.empty(idx.shape[1], device=device)
        val[:] = 1 / K
        dydx = torch.sparse_coo_tensor(
            idx, val, (N * c_in * output_dim, x.numel())
        ).coalesce()
        return dydx

    def get_output_input_hessian(
        self, x: torch.Tensor
    ) -> Union[torch.Tensor, Tuple[torch.Tensor]]:
        y = self.get_output(x)
        idx, val = torch.tensor([], device=x.device, dtype=torch.int64).reshape(
            2, 0
        ), torch.tensor([], device=x.device)
        dy2dx2 = torch.sparse_coo_tensor(
            idx, val, (y.numel(), x.numel() * x.numel())
        ).coalesce()
        return dy2dx2


class O2AvgPool2d(O2Layer):
    def __init__(self, *args, **kwargs):
        module = nn.AvgPool2d(*args, **kwargs)
        super().__init__(module)

    def get_output_input_jacobian(self, x: torch.Tensor) -> torch.Tensor:
        avg = self.module
        stride_h, stride_w = (
            avg.stride if isinstance(avg.stride, tuple) else avg.stride,
            avg.stride,
        )
        padding_h, padding_w = (
            avg.padding if isinstance(avg.padding, tuple) else avg.padding,
            avg.padding,
        )
        Kh, Kw = (
            avg.kernel_size if isinstance(avg.kernel_size, tuple) else avg.kernel_size,
            avg.kernel_size,
        )
        N, c_in, input_dim_h, input_dim_w = x.shape
        output_dim_h = get_conv_output_shape(
            input_dim_h, Kh, stride_h, padding_h, dilation=1
        )
        output_dim_w = get_conv_output_shape(
            input_dim_w, Kw, stride_w, padding_w, dilation=1
        )
        device = x.device
        idx1 = torch.cartesian_prod(
            torch.arange(output_dim_h, dtype=torch.int64, device=device),
            torch.arange(input_dim_h, dtype=torch.int64, device=device),
        )
        ih, jh = idx1[:, 0], idx1[:, 1]
        mask = (0 <= jh - ih * stride_h + padding_h) & (
            jh - ih * stride_h + padding_h < Kh
        )
        idx1 = idx1[mask, :]
        idx2 = torch.cartesian_prod(
            torch.arange(output_dim_w, dtype=torch.int64, device=device),
            torch.arange(input_dim_w, dtype=torch.int64, device=device),
        )
        iw, jw = idx2[:, 0], idx2[:, 1]
        mask = (0 <= jw - iw * stride_w + padding_w) & (
            jw - iw * stride_w + padding_w < Kw
        )
        idx2 = idx2[mask, :]
        idx3 = torch.cartesian_prod(
            torch.arange(N, dtype=torch.int64, device=device),
            torch.arange(c_in, dtype=torch.int64, device=device),
        )
        idx = cartesian_prod_2d(idx1, idx2, idx3)
        idx[:, 0] = flatten_multiindex(
            idx[:, torch.tensor([4, 5, 0, 2], dtype=torch.int64, device=device)],
            [N, c_in, output_dim_h, output_dim_w],
        )
        idx[:, 1] = flatten_multiindex(
            idx[:, torch.tensor([4, 5, 1, 3], dtype=torch.int64, device=device)],
            [N, c_in, input_dim_h, input_dim_w],
        )
        idx = idx[:, :2].T
        val = torch.empty(idx.shape[1], device=device)
        val[:] = 1 / (Kh * Kw)
        dydx = torch.sparse_coo_tensor(
            idx, val, (N * c_in * output_dim_h * output_dim_w, x.numel())
        ).coalesce()
        return dydx

    def get_output_input_hessian(
        self, x: torch.Tensor
    ) -> Union[torch.Tensor, Tuple[torch.Tensor]]:
        y = self.get_output(x)
        idx, val = torch.tensor([], device=x.device, dtype=torch.int64).reshape(
            2, 0
        ), torch.tensor([], device=x.device)
        dy2dx2 = torch.sparse_coo_tensor(
            idx, val, (y.numel(), x.numel() * x.numel())
        ).coalesce()
        return dy2dx2


class O2MaxPool1d(O2Layer):
    def __init__(self, *args, **kwargs):
        self.return_indices = (
            kwargs["return_indices"] if "return_indices" in kwargs else False
        )
        kwargs["return_indices"] = True
        module = nn.MaxPool1d(*args, **kwargs)
        super().__init__(module)
        self.indices = None

    def forward(self, x: torch.Tensor):
        full_output = self.module(x)
        y, idx = full_output
        self.input = x.clone()
        self.output = y.clone()
        self.indices = idx.clone()
        if self.return_indices:
            return full_output
        else:
            return y

    def get_output(self, x: torch.Tensor):
        if (self.input is not None) and (self.output is not None) and (x is self.input):
            return self.output
        else:
            y, idx = self.module(x)
            return y

    def get_indices(self, x: torch.Tensor):
        if (
            (self.input is not None)
            and (self.return_indices is not None)
            and (x is self.input)
        ):
            return self.indices
        else:
            y, idx = self.module(x)
            return idx

    def get_output_indices(self, x: torch.Tensor):
        if (
            (self.input is not None)
            and (self.output is not None)
            and (self.return_indices is not None)
            and (x is self.input)
        ):
            return self.output, self.indices
        else:
            return self.module(x)

    def get_output_input_jacobian(self, x: torch.Tensor) -> torch.Tensor:
        y, pool_idx = self.get_output_indices(x)
        count = y.numel()
        idx = torch.empty([count, 4], device=x.device, dtype=torch.int64)
        idx[:, 0] = torch.arange(0, count, 1)
        idx[:, 1:3] = torch.nonzero(pool_idx != -1)[:, :2]
        idx[:, 3] = pool_idx.reshape(-1)
        idx[:, 1] = flatten_multiindex(idx[:, 1:4], [*x.shape])
        idx = idx[:, :2].T
        val = torch.empty([count], device=x.device)
        val[:] = 1
        dydx = torch.sparse_coo_tensor(idx, val, (y.numel(), x.numel())).coalesce()
        return dydx

    def get_output_input_hessian(
        self, x: torch.Tensor
    ) -> Union[torch.Tensor, Tuple[torch.Tensor]]:
        y = self.get_output(x)
        idx, val = torch.tensor([], device=x.device, dtype=torch.int64).reshape(
            2, 0
        ), torch.tensor([], device=x.device)
        dy2dx2 = torch.sparse_coo_tensor(
            idx, val, (y.numel(), x.numel() * x.numel())
        ).coalesce()
        return dy2dx2


class O2MaxPool2d(O2Layer):
    def __init__(self, *args, **kwargs):
        self.return_indices = (
            kwargs["return_indices"] if "return_indices" in kwargs else False
        )
        kwargs["return_indices"] = True
        module = nn.MaxPool2d(*args, **kwargs)
        super().__init__(module)
        self.indices = None

    def forward(self, x: torch.Tensor):
        output = self.module(x)
        y, idx = output
        self.input = x.clone()
        self.output = y.clone()
        self.indices = idx.clone()
        if self.return_indices:
            return output
        else:
            return y

    def get_output(self, x: torch.Tensor):
        if (self.input is not None) and (self.output is not None) and (x is self.input):
            return self.output
        else:
            y, idx = self.module(x)
            return y

    def get_indices(self, x: torch.Tensor):
        if (
            (self.input is not None)
            and (self.return_indices is not None)
            and (x is self.input)
        ):
            return self.indices
        else:
            y, idx = self.module(x)
            return idx

    def get_output_indices(self, x: torch.Tensor):
        if (
            (self.input is not None)
            and (self.output is not None)
            and (self.return_indices is not None)
            and (x is self.input)
        ):
            return self.output, self.indices
        else:
            return self.module(x)

    def get_output_input_jacobian(self, x: torch.Tensor) -> torch.Tensor:
        y, pool_idx = self.get_output_indices(x)
        pool_idx = pool_idx.reshape(*x.shape[:2], -1)
        N, c, input_height, input_width = x.shape
        input_size = input_height * input_width
        count = y.numel()
        idx = torch.empty([count, 4], device=x.device, dtype=torch.int64)
        idx[:, 0] = torch.arange(0, count, 1)
        idx[:, 1:3] = torch.nonzero(pool_idx != -1)[:, :2]
        idx[:, 3] = pool_idx.reshape(-1)
        idx[:, 1] = flatten_multiindex(idx[:, 1:4], [N, c, input_size])
        idx = idx[:, :2].T
        val = torch.empty([count], device=x.device)
        val[:] = 1
        dydx = torch.sparse_coo_tensor(idx, val, (y.numel(), x.numel())).coalesce()
        return dydx

    def get_output_input_hessian(
        self, x: torch.Tensor
    ) -> Union[torch.Tensor, Tuple[torch.Tensor]]:
        y = self.get_output(x)
        idx, values = torch.tensor([], device=x.device, dtype=torch.int64).reshape(
            2, 0
        ), torch.tensor([], device=x.device)
        dy2dx2 = torch.sparse_coo_tensor(
            idx, values, (y.numel(), x.numel() * x.numel())
        ).coalesce()
        return dy2dx2
