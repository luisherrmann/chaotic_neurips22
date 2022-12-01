from typing import Union, Tuple
import torch
import torch.nn as nn

from .o2parametric import O2ParametricLayer
from o2grad.utils import (
    cartesian_prod_2d,
    flatten_multiindex,
    get_conv_output_shape,
    get_tconv_output_shape,
)


class O2Conv2d(O2ParametricLayer):
    def __init__(self, *args, **kwargs):
        module = nn.Conv2d(*args, **kwargs)
        super().__init__(module)

    def get_output_input_jacobian(self, x: torch.Tensor, sparse=True) -> torch.Tensor:
        # Retrieve input, output, parameter dimensions
        conv2d = self.module
        stride_h, stride_w = conv2d.stride
        padding_h, padding_w = conv2d.padding
        dilation_h, dilation_w = conv2d.dilation
        w = conv2d.weight.data
        N, _, input_dim_h, input_dim_w = x.shape
        c_out, c_in, Kh, Kw = w.shape
        output_dim_h = get_conv_output_shape(
            input_dim_h, Kh, stride_h, padding_h, dilation_h
        )
        output_dim_w = get_conv_output_shape(
            input_dim_w, Kw, stride_w, padding_w, dilation_w
        )
        device = x.device
        # Output, input indices for height dimension
        # Equivalent code using Python lists (slow):
        # idx_h = [[ih, jh] for ih, jh in itertools.product(range(output_dim_h), range(input_dim_h)) if \
        #     (0 <= jh - ih * stride_h + padding_h and jh - ih * stride_h + padding_h < dilation_h * Kh and (jh - ih * stride_h + padding_h) % dilation_h == 0)]
        # idx_h = torch.tensor(idx_h, dtype=torch.int64, device=device)
        idx1 = torch.cartesian_prod(
            torch.arange(output_dim_h, dtype=torch.int64, device=device),
            torch.arange(input_dim_h, dtype=torch.int64, device=device),
        )
        ih, jh = idx1[:, 0], idx1[:, 1]
        mask = (
            (0 <= jh - ih * stride_h + padding_h)
            & (jh - ih * stride_h + padding_h < dilation_h * Kh)
            & ((jh - ih * stride_h + padding_h) % dilation_h == 0)
        )
        idx1 = idx1[mask, :]
        # Output, input indices for width dimension
        # Equivalent code using Python lists (slow):
        # idx_w = [[iw, jw] for iw, jw in itertools.product(range(output_dim_w), range(input_dim_w)) if \
        #     (0 <= jw - iw * stride_w + padding_w and jw - iw * stride_w + padding_w < dilation_w * Kw and (jw - iw * stride_w + padding_w) % dilation_w == 0)]
        # idx_w = torch.tensor(idx_w, dtype=torch.int64, device=device)
        idx2 = torch.cartesian_prod(
            torch.arange(output_dim_w, dtype=torch.int64, device=device),
            torch.arange(input_dim_w, dtype=torch.int64, device=device),
        )
        iw, jw = idx2[:, 0], idx2[:, 1]
        mask = (
            (0 <= jw - iw * stride_w + padding_w)
            & (jw - iw * stride_w + padding_w < dilation_w * Kw)
            & ((jw - iw * stride_w + padding_w) % dilation_w == 0)
        )
        idx2 = idx2[mask, :]
        # Cartesian product with c_out, c_in, b
        idx3 = torch.cartesian_prod(
            torch.arange(c_out, dtype=torch.int64, device=device),
            torch.arange(c_in, dtype=torch.int64, device=device),
            torch.arange(N, dtype=torch.int64, device=device),
        )
        idx = cartesian_prod_2d(idx1, idx2, idx3)
        # Channel order: ih, jh, iw, jw, c_out, c_in, b
        # Flattened weight indices
        idx_w = torch.empty([len(idx), 4], dtype=torch.int64, device=device)
        idx_w[:, 0] = idx[:, 4]  # out_channel idx
        idx_w[:, 1] = idx[:, 5]  # in_channel idx
        idx_w[:, 2] = (
            idx[:, 1] - idx[:, 0] * stride_h + padding_h
        ) / dilation_h  # 1st kernel idx
        idx_w[:, 3] = (
            idx[:, 3] - idx[:, 2] * stride_w + padding_w
        ) / dilation_w  # 2nd kernel idx
        idx_w = flatten_multiindex(idx_w, w.shape)
        val = w.reshape(-1)[idx_w]
        # dydx indices
        idx_selection = torch.tensor([6, 4, 0, 2], dtype=torch.int64, device=device)
        idx[:, 0] = flatten_multiindex(
            idx.index_select(1, idx_selection), [N, c_out, output_dim_h, output_dim_w]
        )
        idx_selection = torch.tensor([6, 5, 1, 3], dtype=torch.int64, device=device)
        idx[:, 1] = flatten_multiindex(idx.index_select(1, idx_selection), x.shape)
        idx = idx[:, :2].T
        dydx = torch.sparse_coo_tensor(
            idx, val, (N * c_out * output_dim_h * output_dim_w, x.numel())
        ).coalesce()
        if not sparse:
            dydx = dydx.to_dense()
        return dydx

    def get_output_input_hessian(self, x: torch.Tensor, sparse=True) -> torch.Tensor:
        # Retrieve input, output, parameter dimensions
        conv2d = self.module
        stride_h, stride_w = conv2d.stride
        padding_h, padding_w = conv2d.padding
        dilation_h, dilation_w = conv2d.dilation
        w = conv2d.weight.data
        N, _, input_dim_h, input_dim_w = x.shape
        c_out, c_in, Kh, Kw = w.shape
        output_dim_h = get_conv_output_shape(
            input_dim_h, Kh, stride_h, padding_h, dilation_h
        )
        output_dim_w = get_conv_output_shape(
            input_dim_w, Kw, stride_w, padding_w, dilation_w
        )
        device = x.device
        idx, val = torch.tensor([], dtype=torch.int64, device=device).reshape(
            2, 0
        ), torch.tensor([], device=device)
        dy2dx2 = torch.sparse_coo_tensor(
            idx, val, (N * c_out * output_dim_h * output_dim_w, x.numel() * x.numel())
        ).coalesce()
        if not sparse:
            dy2dx2 = dy2dx2.to_dense()
        return dy2dx2

    def get_output_param_jacobian(
        self, x: torch.Tensor, sparse=True
    ) -> Union[torch.Tensor, Tuple[torch.Tensor]]:
        # Retrieve input, output, parameter dimensions
        conv2d = self.module
        stride_h, stride_w = conv2d.stride
        padding_h, padding_w = conv2d.padding
        dilation_h, dilation_w = conv2d.dilation
        w = conv2d.weight.data
        b = conv2d.bias.data
        nw, nb = w.numel(), b.numel()
        n = nw + nb
        N, _, input_dim_h, input_dim_w = x.shape
        c_out, c_in, Kh, Kw = w.shape
        output_dim_h = get_conv_output_shape(
            input_dim_h, Kh, stride_h, padding_h, dilation_h
        )
        output_dim_w = get_conv_output_shape(
            input_dim_w, Kw, stride_w, padding_w, dilation_w
        )
        device = x.device
        # Output and kernel indices along height dimension
        # Equivalent code using Python lists (slow):
        # idx1 = [[ih, kh] for ih, kh in itertools.product(range(output_dim_h), range(Kh)) if \
        #     (0 <= kh * dilation_h + ih * stride_h - padding_h and kh * dilation_h + ih * stride_h  - padding_h < input_dim_h)]
        # idx1 = torch.tensor(idx1, dtype=torch.int64, device=device)
        idx1 = torch.cartesian_prod(
            torch.arange(output_dim_h, dtype=torch.int64, device=device),
            torch.arange(Kh, dtype=torch.int64, device=device),
        )
        ih, kh = idx1[:, 0], idx1[:, 1]
        mask = (0 <= kh * dilation_h + ih * stride_h - padding_h) & (
            kh * dilation_h + ih * stride_h - padding_h < input_dim_h
        )
        idx1 = idx1[mask, :]
        # Output and kernel indices along width dimension
        # Equivalent code using Python lists (slow):
        # idx2 = [[iw, kw] for iw, kw in itertools.product(range(output_dim_w), range(Kw)) if \
        #     (0 <= kw * dilation_w + iw * stride_w - padding_w and kw * dilation_w + iw * stride_w  - padding_w < input_dim_w)]
        # idx2 = torch.tensor(idx2, dtype=torch.int64, device=device)
        idx2 = torch.cartesian_prod(
            torch.arange(output_dim_w, dtype=torch.int64, device=device),
            torch.arange(Kw, dtype=torch.int64, device=device),
        )
        iw, kw = idx2[:, 0], idx2[:, 1]
        mask = (0 <= kw * dilation_w + iw * stride_w - padding_w) & (
            kw * dilation_w + iw * stride_w - padding_w < input_dim_w
        )
        idx2 = idx2[mask, :]
        # Cartesian product with c_out, c_in, N
        idx3 = torch.cartesian_prod(
            torch.arange(c_out, dtype=torch.int64, device=device),
            torch.arange(c_in, dtype=torch.int64, device=device),
            torch.arange(N, dtype=torch.int64, device=device),
        )
        idx_w = cartesian_prod_2d(idx1, idx2, idx3)
        # Index order: ih, kh, iw, kw, c_out, c_in, b
        # Flattened input indices
        idx_x = torch.empty([len(idx_w), 4], dtype=torch.int64, device=device)
        idx_x[:, 0] = idx_w[:, 6]
        idx_x[:, 1] = idx_w[:, 5]
        idx_x[:, 2] = idx_w[:, 1] * dilation_h + idx_w[:, 0] * stride_h - padding_h
        idx_x[:, 3] = idx_w[:, 3] * dilation_w + idx_w[:, 2] * stride_w - padding_w
        idx_x_flat = flatten_multiindex(idx_x, x.shape)
        val_w = x.reshape(-1)[idx_x_flat]
        # Flattened weight indices
        idx_selection = torch.tensor([6, 4, 0, 2], dtype=torch.int64, device=device)
        idx_w[:, 0] = flatten_multiindex(
            idx_w.index_select(1, idx_selection), [N, c_out, output_dim_h, output_dim_w]
        )
        idx_selection = torch.tensor([4, 5, 1, 3], dtype=torch.int64, device=device)
        idx_w[:, 1] = flatten_multiindex(
            idx_w.index_select(1, idx_selection), [c_out, c_in, Kh, Kw]
        )
        # Flattened bias indices
        # Equivalent code using Python lists (slow):
        # iterator_b = itertools.product(range(c_out), range(output_dim_h), range(output_dim_w), range(N))
        # idx_b = [[b, c, ih, iw, nw + c] for c, ih, iw, b in iterator_b]
        # idx_b = torch.tensor(idx_b, dtype=torch.int64, device=device).reshape(-1, 5)
        idx_b = torch.cartesian_prod(
            torch.arange(N, dtype=torch.int64, device=device),
            torch.arange(c_out, dtype=torch.int64, device=device),
            torch.arange(output_dim_h, dtype=torch.int64, device=device),
            torch.arange(output_dim_w, dtype=torch.int64, device=device),
        )
        idx_b[:, 0] = flatten_multiindex(
            idx_b[:, :4], [N, c_out, output_dim_h, output_dim_w]
        )
        # Add offset of weight parameter count (to avoid overlapping indices)
        idx_b[:, 1] += nw
        val_b = torch.empty(idx_b.shape[0], dtype=w.dtype, device=device)
        val_b[:] = 1
        # Combined flattened parameter indices
        idx = torch.empty(
            [len(idx_w) + len(idx_b), 2], dtype=torch.int64, device=device
        )
        idx[: len(idx_w), :] = idx_w[:, :2]
        idx[len(idx_w) :, :] = idx_b[:, :2]
        # Combined parameter values
        val = torch.empty([len(val_w) + len(val_b)], dtype=w.dtype, device=device)
        val[: len(val_w)] = val_w
        val[len(val_w) :] = val_b
        dydp = torch.sparse_coo_tensor(
            idx.T, val, (N * c_out * output_dim_h * output_dim_w, n)
        ).coalesce()
        if not sparse:
            dydp = dydp.to_dense()
        return dydp

    def get_output_param_hessian(self, x: torch.Tensor, sparse=True) -> torch.Tensor:
        # Retrieve input, output, parameter dimensions
        conv2d = self.module
        stride_h, stride_w = conv2d.stride
        padding_h, padding_w = conv2d.padding
        dilation_h, dilation_w = conv2d.dilation
        w = conv2d.weight.data
        b = conv2d.bias.data
        nw, nb = w.numel(), b.numel()
        n = nw + nb
        N, _, input_dim_h, input_dim_w = x.shape
        c_out, c_in, Kh, Kw = w.shape
        output_dim_h = get_conv_output_shape(
            input_dim_h, Kh, stride_h, padding_h, dilation_h
        )
        output_dim_w = get_conv_output_shape(
            input_dim_w, Kw, stride_w, padding_w, dilation_w
        )
        device = x.device
        idx, val = torch.tensor([], dtype=torch.int64, device=device).reshape(
            2, 0
        ), torch.tensor([], device=device)
        dy2dp2 = torch.sparse_coo_tensor(
            idx, val, (N * c_out * output_dim_h * output_dim_w, n * n)
        ).coalesce()
        if not sparse:
            dy2dp2 = dy2dp2.to_dense()
        return dy2dp2

    def get_mixed_output_param_hessian(
        self, x: torch.Tensor, sparse=True
    ) -> torch.Tensor:
        # Retrieve input, output, parameter dimensions
        conv2d = self.module
        stride_h, stride_w = conv2d.stride
        padding_h, padding_w = conv2d.padding
        dilation_h, dilation_w = conv2d.dilation
        w = conv2d.weight.data
        b = conv2d.bias.data
        nw, nb = w.numel(), b.numel()
        n = nw + nb
        N, _, input_dim_h, input_dim_w = x.shape
        c_out, c_in, Kh, Kw = w.shape
        output_dim_h = get_conv_output_shape(
            input_dim_h, Kh, stride_h, padding_h, dilation_h
        )
        output_dim_w = get_conv_output_shape(
            input_dim_w, Kw, stride_w, padding_w, dilation_w
        )
        device = x.device
        # Output, input, kernel indices over height dimension
        # iterator = itertools.product(range(output_dim_h), range(input_dim_h), range(Kh))
        # idx1 = [[ih, jh, kh] for ih, jh, kh in iterator if (jh - ih * stride_h + padding_h == dilation_h * kh)]
        # idx1 = torch.tensor(idx1, dtype=torch.int64, device=device)
        idx1 = torch.cartesian_prod(
            torch.arange(output_dim_h, dtype=torch.int64, device=device),
            torch.arange(input_dim_h, dtype=torch.int64, device=device),
            torch.arange(Kh, dtype=torch.int64, device=device),
        )
        ih, jh, kh = idx1[:, 0], idx1[:, 1], idx1[:, 2]
        mask = jh - ih * stride_h + padding_h == dilation_h * kh
        idx1 = idx1[mask, :]
        # Output, input, kernel indices over width dimension
        # iterator = itertools.product(range(output_dim_w), range(input_dim_w), range(Kw))
        # idx2 = [[iw, jw, kw] for iw, jw, kw in iterator if (jw - iw * stride_w + padding_w == dilation_w * kw)]
        # idx2 = torch.tensor(idx1, dtype=torch.int64, device=device)
        idx2 = torch.cartesian_prod(
            torch.arange(output_dim_w, dtype=torch.int64, device=device),
            torch.arange(input_dim_w, dtype=torch.int64, device=device),
            torch.arange(Kw, dtype=torch.int64, device=device),
        )
        iw, jw, kw = idx2[:, 0], idx2[:, 1], idx2[:, 2]
        mask = jw - iw * stride_w + padding_w == dilation_w * kw
        idx2 = idx2[mask, :]
        # Cartesian product with c_out, c_in, b
        idx3 = torch.cartesian_prod(
            torch.arange(c_out, dtype=torch.int64, device=device),
            torch.arange(c_in, dtype=torch.int64, device=device),
            torch.arange(N, dtype=torch.int64, device=device),
        )
        idx = cartesian_prod_2d(idx1, idx2, idx3)
        # Channel order: ih, jh, kh, iw, jw, kw, c, d, b
        # Flattened output indices
        idx_y = torch.index_select(
            idx, 1, torch.tensor([8, 6, 0, 3], dtype=torch.int64, device=device)
        )
        idx_y = flatten_multiindex(idx_y, [N, c_out, output_dim_h, output_dim_w])
        # Flattened weight indices
        idx_w = torch.index_select(
            idx, 1, torch.tensor([6, 7, 2, 5], dtype=torch.int64, device=device)
        )  # c_out, c_in, kh, kw
        idx_w = flatten_multiindex(idx_w, w.shape)
        # Flattened input + weight indices
        idx_xw = torch.empty([len(idx), 5], dtype=torch.int64, device=device)
        idx_xw[:, :4] = torch.index_select(
            idx, 1, torch.tensor([8, 7, 1, 4], dtype=torch.int64, device=device)
        )  # N, c_in, input_h, input_w
        idx_xw[:, 4] = idx_w
        idx_xw = flatten_multiindex(idx_xw, [*x.shape, n])
        # dy2dxdp indices
        idx = torch.empty([2, len(idx_y)], dtype=torch.int64, device=device)
        idx[0, :] = idx_y
        idx[1, :] = idx_xw
        val = torch.empty([len(idx_y)], dtype=w.dtype, device=device)
        val[:] = 1
        dy2dxdp = torch.sparse_coo_tensor(
            idx, val, [N * c_out * output_dim_h * output_dim_w, x.numel() * n]
        ).coalesce()
        if not sparse:
            dy2dxdp = dy2dxdp.to_dense()
        return dy2dxdp


class O2ConvTranspose2d(O2ParametricLayer):
    def __init__(self, *args, **kwargs):
        module = nn.ConvTranspose2d(*args, **kwargs)
        super().__init__(module)

    def get_output_input_jacobian(self, x: torch.Tensor, sparse=True) -> torch.Tensor:
        # Retrieve input, output, parameter dimensions
        convt2d = self.module
        stride_h, stride_w = convt2d.stride
        padding_h, padding_w = convt2d.padding
        dilation_h, dilation_w = convt2d.dilation
        w = convt2d.weight.data
        N, _, input_dim_h, input_dim_w = x.shape
        c_in, c_out, Kh, Kw = w.shape
        output_dim_h = get_tconv_output_shape(
            input_dim_h, Kh, stride_h, padding_h, dilation_h
        )
        output_dim_w = get_tconv_output_shape(
            input_dim_w, Kw, stride_w, padding_w, dilation_w
        )
        device = x.device
        # Output, input indices for height dimension
        # Equivalent code using Python lists (slow):
        # idx_h = [[ih, jh] for ih, jh in itertools.product(range(output_dim_h), range(input_dim_h)) if \
        #     (0 <= ih - jh * stride_h + padding_h and ih - jh * stride_h + padding_h < dilation_h * Kh and (ih - jh * stride_h + padding_h) % dilation_h == 0)]
        # idx_h = torch.tensor(idx_h, dtype=torch.int64, device=device)
        idx1 = torch.cartesian_prod(
            torch.arange(output_dim_h, dtype=torch.int64, device=device),
            torch.arange(input_dim_h, dtype=torch.int64, device=device),
        )
        ih, jh = idx1[:, 0], idx1[:, 1]
        mask = (
            (0 <= ih - jh * stride_h + padding_h)
            & (ih - jh * stride_h + padding_h < dilation_h * Kh)
            & ((ih - jh * stride_h + padding_h) % dilation_h == 0)
        )
        idx1 = idx1[mask, :]
        # Output, input indices for width dimension
        # Equivalent code using Python lists (slow):
        # idx_w = [[iw, jw] for iw, jw in itertools.product(range(output_dim_w), range(input_dim_w)) if \
        #     (0 <= iw - jw * stride_w + padding_w and iw - jw * stride_w + padding_w < dilation_w * Kw and (iw - jw * stride_w + padding_w) % dilation_w == 0)]
        # idx_w = torch.tensor(idx_w, dtype=torch.int64, device=device)
        idx2 = torch.cartesian_prod(
            torch.arange(output_dim_w, dtype=torch.int64, device=device),
            torch.arange(input_dim_w, dtype=torch.int64, device=device),
        )
        iw, jw = idx2[:, 0], idx2[:, 1]
        mask = (
            (0 <= iw - jw * stride_w + padding_w)
            & (iw - jw * stride_w + padding_w < dilation_w * Kw)
            & ((iw - jw * stride_w + padding_w) % dilation_w == 0)
        )
        idx2 = idx2[mask, :]
        # Cartesian product with c_out, c_in, b
        idx3 = torch.cartesian_prod(
            torch.arange(c_out, dtype=torch.int64, device=device),
            torch.arange(c_in, dtype=torch.int64, device=device),
            torch.arange(N, dtype=torch.int64, device=device),
        )
        idx = cartesian_prod_2d(idx1, idx2, idx3)
        # Channel order: ih, jh, iw, jw, c_out, c_in, b
        # Flattened weight indices
        idx_w = torch.empty([len(idx), 4], dtype=torch.int64, device=device)
        idx_w[:, 0] = idx[:, 5]  # in_channel idx
        idx_w[:, 1] = idx[:, 4]  # out_channel idx
        idx_w[:, 2] = (
            idx[:, 0] - idx[:, 1] * stride_h + padding_h
        ) / dilation_h  # 1st kernel idx
        idx_w[:, 3] = (
            idx[:, 2] - idx[:, 3] * stride_w + padding_w
        ) / dilation_w  # 2nd kernel idx
        idx_w = flatten_multiindex(idx_w, w.shape)
        val = w.reshape(-1)[idx_w]
        # dydx indices
        idx_selection = torch.tensor([6, 4, 0, 2], dtype=torch.int64, device=device)
        idx[:, 0] = flatten_multiindex(
            idx.index_select(1, idx_selection), [N, c_out, output_dim_h, output_dim_w]
        )
        idx_selection = torch.tensor([6, 5, 1, 3], dtype=torch.int64, device=device)
        idx[:, 1] = flatten_multiindex(idx.index_select(1, idx_selection), x.shape)
        idx = idx[:, :2].T
        dydx = torch.sparse_coo_tensor(
            idx, val, (N * c_out * output_dim_h * output_dim_w, x.numel())
        ).coalesce()
        if not sparse:
            dydx = dydx.to_dense()
        return dydx

    def get_output_input_hessian(self, x: torch.Tensor, sparse=True) -> torch.Tensor:
        # Retrieve input, output, parameter dimensions
        conv2d = self.module
        stride_h, stride_w = conv2d.stride
        padding_h, padding_w = conv2d.padding
        dilation_h, dilation_w = conv2d.dilation
        w = conv2d.weight.data
        N, _, input_dim_h, input_dim_w = x.shape
        c_in, c_out, Kh, Kw = w.shape
        output_dim_h = get_tconv_output_shape(
            input_dim_h, Kh, stride_h, padding_h, dilation_h
        )
        output_dim_w = get_tconv_output_shape(
            input_dim_w, Kw, stride_w, padding_w, dilation_w
        )
        device = x.device
        idx, val = torch.tensor([], dtype=torch.int64, device=device).reshape(
            2, 0
        ), torch.tensor([], device=device)
        dy2dx2 = torch.sparse_coo_tensor(
            idx, val, (N * c_out * output_dim_h * output_dim_w, x.numel() * x.numel())
        ).coalesce()
        if not sparse:
            dy2dx2 = dy2dx2.to_dense()
        return dy2dx2

    def get_output_param_jacobian(
        self, x: torch.Tensor, sparse=True
    ) -> Union[torch.Tensor, Tuple[torch.Tensor]]:
        # Retrieve input, output, parameter dimensions
        conv2d = self.module
        stride_h, stride_w = conv2d.stride
        padding_h, padding_w = conv2d.padding
        dilation_h, dilation_w = conv2d.dilation
        w = conv2d.weight.data
        b = conv2d.bias.data
        nw, nb = w.numel(), b.numel()
        n = nw + nb
        N, _, input_dim_h, input_dim_w = x.shape
        c_in, c_out, Kh, Kw = w.shape
        output_dim_h = get_tconv_output_shape(
            input_dim_h, Kh, stride_h, padding_h, dilation_h
        )
        output_dim_w = get_tconv_output_shape(
            input_dim_w, Kw, stride_w, padding_w, dilation_w
        )
        device = x.device
        # Output and kernel indices along height dimension
        # Equivalent code using Python lists (slow):
        # idx1 = [[ih, kh] for ih, kh in itertools.product(range(output_dim_h), range(Kh)) if \
        #     (0 <= ih + padding_h - kh * dilation_h and ih + padding_h - kh * dilation_h < input_dim_h * stride_h) and \
        #     ((ih + padding_h - kh * dilation_h) % stride_h == 0)]
        # idx1 = torch.tensor(idx1, dtype=torch.int64, device=device)
        idx1 = torch.cartesian_prod(
            torch.arange(output_dim_h, dtype=torch.int64, device=device),
            torch.arange(Kh, dtype=torch.int64, device=device),
        )
        ih, kh = idx1[:, 0], idx1[:, 1]
        mask = (
            (0 <= ih + padding_h - kh * dilation_h)
            & (ih + padding_h - kh * dilation_h < input_dim_h * stride_h)
            & ((ih + padding_h - kh * dilation_h) % stride_h == 0)
        )
        idx1 = idx1[mask, :]
        # Output and kernel indices along width dimension
        # Equivalent code using Python lists (slow):
        # idx2 = [[iw, kw] for iw, kw in itertools.product(range(output_dim_w), range(Kw)) if \
        #     (0 <= iw + padding_w - kw * dilation_w and iw + padding_w - kw * dilation_w < input_dim_w * stride_w) and \
        #     ((iw + padding_w - kw * dilation_w) % stride_w == 0)]
        # idx2 = torch.tensor(idx2, dtype=torch.int64, device=device)
        idx2 = torch.cartesian_prod(
            torch.arange(output_dim_w, dtype=torch.int64, device=device),
            torch.arange(Kw, dtype=torch.int64, device=device),
        )
        iw, kw = idx2[:, 0], idx2[:, 1]
        mask = (
            (0 <= iw + padding_w - kw * dilation_w)
            & (iw + padding_w - kw * dilation_w < input_dim_w * stride_w)
            & ((iw + padding_w - kw * dilation_w) % stride_w == 0)
        )
        idx2 = idx2[mask, :]
        # Cartesian product with c_out, c_in, N
        idx3 = torch.cartesian_prod(
            torch.arange(c_out, dtype=torch.int64, device=device),
            torch.arange(c_in, dtype=torch.int64, device=device),
            torch.arange(N, dtype=torch.int64, device=device),
        )
        idx_w = cartesian_prod_2d(idx1, idx2, idx3)
        # Index order: ih, kh, iw, kw, c_out, c_in, b
        # Flattened input indices
        idx_x = torch.empty([len(idx_w), 4], dtype=torch.int64, device=device)
        idx_x[:, 0] = idx_w[:, 6]
        idx_x[:, 1] = idx_w[:, 5]
        idx_x[:, 2] = (idx_w[:, 0] + padding_h - idx_w[:, 1] * dilation_h) / stride_h
        idx_x[:, 3] = (idx_w[:, 2] + padding_w - idx_w[:, 3] * dilation_w) / stride_w
        idx_x_flat = flatten_multiindex(idx_x, x.shape)
        val_w = x.reshape(-1)[idx_x_flat]
        # Flattened weight indices
        idx_selection = torch.tensor([6, 4, 0, 2], dtype=torch.int64, device=device)
        idx_w[:, 0] = flatten_multiindex(
            idx_w.index_select(1, idx_selection), [N, c_out, output_dim_h, output_dim_w]
        )
        idx_selection = torch.tensor([5, 4, 1, 3], dtype=torch.int64, device=device)
        idx_w[:, 1] = flatten_multiindex(
            idx_w.index_select(1, idx_selection), [c_in, c_out, Kh, Kw]
        )
        # Flattened bias indices
        # Equivalent code using Python lists (slow):
        # iterator_b = itertools.product(range(c_out), range(output_dim_h), range(output_dim_w), range(N))
        # idx_b = [[b, c, ih, iw, nw + c] for c, ih, iw, b in iterator_b]
        # idx_b = torch.tensor(idx_b, dtype=torch.int64, device=device).reshape(-1, 5)
        idx_b = torch.cartesian_prod(
            torch.arange(N, dtype=torch.int64, device=device),
            torch.arange(c_out, dtype=torch.int64, device=device),
            torch.arange(output_dim_h, dtype=torch.int64, device=device),
            torch.arange(output_dim_w, dtype=torch.int64, device=device),
        )
        idx_b[:, 0] = flatten_multiindex(
            idx_b[:, :4], [N, c_out, output_dim_h, output_dim_w]
        )
        # Add offset of weight parameter count (to avoid overlapping indices)
        idx_b[:, 1] += nw
        val_b = torch.empty(idx_b.shape[0], dtype=w.dtype, device=device)
        val_b[:] = 1
        # Combined flattened parameter indices
        idx = torch.empty(
            [len(idx_w) + len(idx_b), 2], dtype=torch.int64, device=device
        )
        idx[: len(idx_w), :] = idx_w[:, :2]
        idx[len(idx_w) :, :] = idx_b[:, :2]
        # Combined parameter values
        val = torch.empty([len(val_w) + len(val_b)], dtype=w.dtype, device=device)
        val[: len(val_w)] = val_w
        val[len(val_w) :] = val_b
        dydp = torch.sparse_coo_tensor(
            idx.T, val, (N * c_out * output_dim_h * output_dim_w, n)
        ).coalesce()
        if not sparse:
            dydp = dydp.to_dense()
        return dydp

    def get_output_param_hessian(self, x: torch.Tensor, sparse=True) -> torch.Tensor:
        # Retrieve input, output, parameter dimensions
        conv2d = self.module
        stride_h, stride_w = conv2d.stride
        padding_h, padding_w = conv2d.padding
        dilation_h, dilation_w = conv2d.dilation
        w = conv2d.weight.data
        b = conv2d.bias.data
        nw, nb = w.numel(), b.numel()
        n = nw + nb
        N, _, input_dim_h, input_dim_w = x.shape
        c_in, c_out, Kh, Kw = w.shape
        output_dim_h = get_tconv_output_shape(
            input_dim_h, Kh, stride_h, padding_h, dilation_h
        )
        output_dim_w = get_tconv_output_shape(
            input_dim_w, Kw, stride_w, padding_w, dilation_w
        )
        device = x.device
        idx, val = torch.tensor([], dtype=torch.int64, device=device).reshape(
            2, 0
        ), torch.tensor([], device=device)
        dy2dp2 = torch.sparse_coo_tensor(
            idx, val, (N * c_out * output_dim_h * output_dim_w, n * n)
        ).coalesce()
        if not sparse:
            dy2dp2 = dy2dp2.to_dense()
        return dy2dp2

    def get_mixed_output_param_hessian(
        self, x: torch.Tensor, sparse=True
    ) -> torch.Tensor:
        # Retrieve input, output, parameter dimensions
        conv2d = self.module
        stride_h, stride_w = conv2d.stride
        padding_h, padding_w = conv2d.padding
        dilation_h, dilation_w = conv2d.dilation
        w = conv2d.weight.data
        b = conv2d.bias.data
        nw, nb = w.numel(), b.numel()
        n = nw + nb
        N, _, input_dim_h, input_dim_w = x.shape
        c_in, c_out, Kh, Kw = w.shape
        output_dim_h = get_tconv_output_shape(
            input_dim_h, Kh, stride_h, padding_h, dilation_h
        )
        output_dim_w = get_tconv_output_shape(
            input_dim_w, Kw, stride_w, padding_w, dilation_w
        )
        device = x.device
        # Output, input, kernel indices over height dimension
        # iterator = itertools.product(range(output_dim_h), range(input_dim_h), range(Kh))
        # idx1 = [[ih, jh, kh] for ih, jh, kh in iterator if (ih - jh * stride_h + padding_h == dilation_h * kh)]
        # idx1 = torch.tensor(idx1, dtype=torch.int64, device=device)
        idx1 = torch.cartesian_prod(
            torch.arange(output_dim_h, dtype=torch.int64, device=device),
            torch.arange(input_dim_h, dtype=torch.int64, device=device),
            torch.arange(Kh, dtype=torch.int64, device=device),
        )
        ih, jh, kh = idx1[:, 0], idx1[:, 1], idx1[:, 2]
        mask = ih - jh * stride_h + padding_h == dilation_h * kh
        idx1 = idx1[mask, :]
        # Output, input, kernel indices over width dimension
        # iterator = itertools.product(range(output_dim_w), range(input_dim_w), range(Kw))
        # idx2 = [[iw, jw, kw] for iw, jw, kw in iterator if (iw - jw * stride_w + padding_w == dilation_w * kw)]
        # idx2 = torch.tensor(idx1, dtype=torch.int64, device=device)
        idx2 = torch.cartesian_prod(
            torch.arange(output_dim_w, dtype=torch.int64, device=device),
            torch.arange(input_dim_w, dtype=torch.int64, device=device),
            torch.arange(Kw, dtype=torch.int64, device=device),
        )
        iw, jw, kw = idx2[:, 0], idx2[:, 1], idx2[:, 2]
        mask = iw - jw * stride_w + padding_w == dilation_w * kw
        idx2 = idx2[mask, :]
        # Cartesian product with c_out, c_in, b
        idx3 = torch.cartesian_prod(
            torch.arange(c_out, dtype=torch.int64, device=device),
            torch.arange(c_in, dtype=torch.int64, device=device),
            torch.arange(N, dtype=torch.int64, device=device),
        )
        idx = cartesian_prod_2d(idx1, idx2, idx3)
        # Channel order: ih, jh, kh, iw, jw, kw, c, d, b
        # Flattened output indices
        idx_y = torch.index_select(
            idx, 1, torch.tensor([8, 6, 0, 3], dtype=torch.int64, device=device)
        )
        idx_y = flatten_multiindex(idx_y, [N, c_out, output_dim_h, output_dim_w])
        # Flattened weight indices
        idx_w = torch.index_select(
            idx, 1, torch.tensor([7, 6, 2, 5], dtype=torch.int64, device=device)
        )  # c_in, c_out, kh, kw
        idx_w = flatten_multiindex(idx_w, w.shape)
        # Flattened input + weight indices
        idx_xw = torch.empty([len(idx), 5], dtype=torch.int64, device=device)
        idx_xw[:, :4] = torch.index_select(
            idx, 1, torch.tensor([8, 7, 1, 4], dtype=torch.int64, device=device)
        )  # N, c_in, input_h, input_w
        idx_xw[:, 4] = idx_w
        idx_xw = flatten_multiindex(idx_xw, [*x.shape, n])
        # dy2dxdp indices
        idx = torch.empty([2, len(idx_y)], dtype=torch.int64, device=device)
        idx[0, :] = idx_y
        idx[1, :] = idx_xw
        val = torch.empty([len(idx_y)], dtype=w.dtype, device=device)
        val[:] = 1
        dy2dxdp = torch.sparse_coo_tensor(
            idx, val, [N * c_out * output_dim_h * output_dim_w, x.numel() * n]
        ).coalesce()
        if not sparse:
            dy2dxdp = dy2dxdp.to_dense()
        return dy2dxdp
