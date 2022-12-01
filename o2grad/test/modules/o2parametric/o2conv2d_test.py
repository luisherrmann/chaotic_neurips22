import unittest
import timeit
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd.functional import jacobian, hessian

from o2grad.modules.o2parametric.o2conv2d import O2Conv2d, O2ConvTranspose2d
from o2grad.backprop import get_hessian


class O2Conv2dTest(unittest.TestCase):
    def _setup(self, create_graph=False):
        torch.manual_seed(0)
        x = torch.rand(2, 2, 4, 4)
        criterion = nn.MSELoss()
        o2conv2d = O2Conv2d(
            in_channels=2,
            out_channels=3,
            kernel_size=2,
            stride=2,
            padding=1,
            dilation=2,
        )
        y = o2conv2d(x)
        target = torch.rand(y.shape)
        y.retain_grad()
        y.requires_grad_()
        loss = criterion(y, target)
        loss.backward(create_graph=create_graph)
        dLdy = y.grad

        def _criterion(y):
            return criterion(y, target)

        dL2dy2 = hessian(_criterion, y)

        return x, y, o2conv2d, dLdy, dL2dy2

    def test_forward(self):
        x = torch.rand(10, 3, 28, 28)
        torch.manual_seed(0)
        o2conv2d = O2Conv2d(in_channels=3, out_channels=8, kernel_size=2, stride=2)
        torch.manual_seed(0)
        conv2d = nn.Conv2d(in_channels=3, out_channels=8, kernel_size=2, stride=2)
        yo2 = o2conv2d(x)
        y = conv2d(x)
        self.assertEqual(yo2.shape, y.shape)
        self.assertTrue(torch.equal(yo2, y))

    def test_get_output_input_jacobian(self):
        x, y, o2conv2d, dLdy, _ = self._setup()
        conv2d = o2conv2d.module
        dydx = o2conv2d.get_output_input_jacobian(x, sparse=False)

        def _jac(x):
            return jacobian(conv2d.forward, x, create_graph=True)

        dydx_ = _jac(x)

        x.shape[2]
        dydx = dydx.reshape(y.numel(), x.numel())
        dydx_ = dydx.reshape(y.numel(), x.numel())
        self.assertEqual(dydx.shape, dydx_.shape)
        self.assertTrue(torch.allclose(dydx, dydx_))
        # Benchmark
        def _o2grad_code():
            o2conv2d.get_output_input_jacobian(x)

        o2grad_timer = timeit.Timer(stmt=_o2grad_code, setup=_o2grad_code)
        reps = 4
        time_w = o2grad_timer.timeit(number=reps) / reps
        print(f"Average time: {time_w}s")

    def test_get_output_input_hessian(self):
        x, y, o2conv2d, dLdy, _ = self._setup()
        conv2d = o2conv2d.module
        dy2dx2 = o2conv2d.get_output_input_hessian(x)
        if dy2dx2.layout == torch.sparse_coo:
            dy2dx2 = dy2dx2.to_dense()

        def _jac(x):
            return jacobian(conv2d.forward, x, create_graph=True)

        x = x.clone()
        dy2dx2_ = jacobian(_jac, x)
        self.assertEqual(dy2dx2.numel(), dy2dx2_.numel())
        dy2dx2 = dy2dx2.reshape(dy2dx2_.shape)
        self.assertTrue(torch.allclose(dy2dx2, dy2dx2_))

    def test_get_output_param_jacobian(self):
        x, y, o2conv2d, dLdy, _ = self._setup()
        conv2d = o2conv2d.module
        nw, nb = conv2d.weight.data.numel(), conv2d.bias.data.numel()
        dydp = o2conv2d.get_output_param_jacobian(x, sparse=False)
        dydp = dydp.reshape(y.numel(), -1)
        # First check: Compare loss Jacobians
        dLdw, dLdb = conv2d.weight.grad, conv2d.bias.grad
        dLdp = torch.cat([dLdw.reshape(-1), dLdb.reshape(-1)])
        dLdp_ = torch.einsum("i,ip->p", dLdy.reshape(-1), dydp)
        self.assertEqual(dLdp.shape, dLdp_.shape)
        self.assertTrue(torch.allclose(dLdp, dLdp_))
        # Second check: Compare output Jacobians

        def _conv2d_weight(w):
            return F.conv2d(
                x,
                w,
                conv2d.bias,
                conv2d.stride,
                conv2d.padding,
                conv2d.dilation,
                conv2d.groups,
            )

        def _conv2d_bias(b):
            return F.conv2d(
                x,
                conv2d.weight,
                b,
                conv2d.stride,
                conv2d.padding,
                conv2d.dilation,
                conv2d.groups,
            )

        dydw = jacobian(_conv2d_weight, conv2d.weight)
        dydb = jacobian(_conv2d_bias, conv2d.bias)
        dydp_ = torch.empty([y.numel(), nw + nb])
        dydp_[:, :nw] = dydw.reshape(y.numel(), -1)
        dydp_[:, nw:] = dydb.reshape(y.numel(), -1)
        dydp = dydp.reshape(y.numel(), -1)
        dydp_ = dydp_.reshape(y.numel(), -1)
        self.assertEqual(dydp.shape, dydp_.shape)
        self.assertTrue(torch.allclose(dydp, dydp_))

        # Benchmark:
        def _o2grad_code():
            o2conv2d.get_output_param_jacobian(x)

        o2grad_timer = timeit.Timer(stmt=_o2grad_code, setup=_o2grad_code)

        reps = 4
        time_w = o2grad_timer.timeit(number=reps) / reps
        print(f"Average time: {time_w}s")

    def test_get_output_param_hessian(self):
        x, y, o2conv2d, dLdy, dL2dy2 = self._setup(create_graph=True)
        conv2d = o2conv2d.module
        dy2dp2 = o2conv2d.get_output_param_hessian(x, sparse=False)

        dL2dp2 = get_hessian(conv2d)
        dydp = o2conv2d.get_output_param_jacobian(x)
        if dydp.layout == torch.sparse_coo:
            dydp = dydp.to_dense()
        dydp = dydp.reshape(y.numel(), -1)
        dL2dp2_term1 = torch.matmul(dLdy.reshape(-1), dy2dp2)
        dL2dp2_term2 = torch.matmul(
            dydp.T, torch.matmul(dL2dy2.reshape(y.numel(), y.numel()), dydp)
        )
        self.assertEqual(dL2dp2_term1.numel(), dL2dp2_term2.numel())
        dL2dp2_term1 = dL2dp2_term1.reshape(dL2dp2_term2.shape)
        dL2dp2_ = dL2dp2_term1 + dL2dp2_term2
        self.assertEqual(dL2dp2.shape, dL2dp2_.shape)
        self.assertTrue(torch.allclose(dL2dp2, dL2dp2_))

    def test_get_mixed_output_param_hessian(self):
        x, y, o2conv2d, dLdy, dL2dy2 = self._setup(create_graph=True)
        conv2d = o2conv2d.module
        nw, nb = conv2d.weight.data.numel(), conv2d.bias.data.numel()

        dy2dxdp = o2conv2d.get_mixed_output_param_hessian(x, sparse=False)
        dy2dxdp = dy2dxdp.reshape(y.numel(), -1)
        # Second check: Compare mixed Hessians
        def _conv2d(x, w, b):
            return F.conv2d(
                x, w, b, conv2d.stride, conv2d.padding, conv2d.dilation, conv2d.groups
            )

        def _jac_w(w):
            _conv1d_weight = lambda x: _conv2d(x, w, conv2d.bias)
            return jacobian(_conv1d_weight, x, create_graph=True)

        def _jac_b(b):
            _conv1d_bias = lambda x: _conv2d(x, conv2d.weight, b)
            return jacobian(_conv1d_bias, x, create_graph=True)

        dy2dxdw = jacobian(_jac_w, conv2d.weight)
        dy2dxdb = jacobian(_jac_b, conv2d.bias)
        dy2dxdp_ = torch.empty([y.numel(), x.numel(), nw + nb])
        dy2dxdp_[:, :, :nw] = dy2dxdw.reshape(y.numel(), x.numel(), -1)
        dy2dxdp_[:, :, nw:] = dy2dxdb.reshape(y.numel(), x.numel(), -1)
        dy2dxdp_ = dy2dxdp_.reshape(y.numel(), -1)
        self.assertEqual(dy2dxdp.shape, dy2dxdp_.shape)
        self.assertTrue(torch.allclose(dy2dxdp, dy2dxdp_))


class O2ConvTranspose2dTest(unittest.TestCase):
    def _setup(self, create_graph=False):
        torch.manual_seed(0)
        x = torch.rand(2, 1, 4, 4)
        criterion = nn.MSELoss()
        o2conv2d = O2ConvTranspose2d(
            in_channels=1,
            out_channels=3,
            kernel_size=2,
            stride=2,
            padding=1,
            dilation=2,
        )
        y = o2conv2d(x)
        target = torch.rand(y.shape)
        y.retain_grad()
        y.requires_grad_()
        loss = criterion(y, target)
        loss.backward(create_graph=create_graph)
        dLdy = y.grad

        def _criterion(y):
            return criterion(y, target)

        dL2dy2 = hessian(_criterion, y)

        return x, y, o2conv2d, dLdy, dL2dy2

    def test_forward(self):
        x = torch.rand(10, 3, 14, 14)
        torch.manual_seed(0)
        o2convt2d = O2ConvTranspose2d(
            in_channels=3, out_channels=8, kernel_size=2, stride=2
        )
        torch.manual_seed(0)
        convt2d = nn.ConvTranspose2d(
            in_channels=3, out_channels=8, kernel_size=2, stride=2
        )
        yo2 = o2convt2d(x)
        y = convt2d(x)
        self.assertEqual(yo2.shape, y.shape)
        self.assertTrue(torch.equal(yo2, y))

    def test_get_output_input_jacobian(self):
        x, y, o2convt2d, dLdy, _ = self._setup()
        conv1d = o2convt2d.module
        dydx = o2convt2d.get_output_input_jacobian(x, sparse=False)

        def _jac(x):
            return jacobian(conv1d.forward, x, create_graph=True)

        dydx_ = _jac(x)

        x.shape[2]
        dydx = dydx.reshape(y.numel(), x.numel())
        dydx_ = dydx.reshape(y.numel(), x.numel())
        self.assertEqual(dydx.shape, dydx_.shape)
        self.assertTrue(torch.allclose(dydx, dydx_))
        # Benchmark
        def _o2grad_code():
            o2convt2d.get_output_input_jacobian(x)

        o2grad_timer = timeit.Timer(stmt=_o2grad_code, setup=_o2grad_code)
        reps = 4
        time_w = o2grad_timer.timeit(number=reps) / reps
        print(f"Average time: {time_w}s")

    def test_get_output_input_hessian(self):
        x, y, o2convt2d, dLdy, _ = self._setup()
        convt2d = o2convt2d.module
        dy2dx2 = o2convt2d.get_output_input_hessian(x)
        if dy2dx2.layout == torch.sparse_coo:
            dy2dx2 = dy2dx2.to_dense()

        def _jac(x):
            return jacobian(convt2d.forward, x, create_graph=True)

        x = x.clone()
        dy2dx2_ = jacobian(_jac, x)
        self.assertEqual(dy2dx2.numel(), dy2dx2_.numel())
        dy2dx2 = dy2dx2.reshape(dy2dx2_.shape)
        self.assertTrue(torch.allclose(dy2dx2, dy2dx2_))

    def test_get_output_param_jacobian(self):
        x, y, o2convt2d, dLdy, _ = self._setup()
        convt2d = o2convt2d.module
        nw, nb = convt2d.weight.data.numel(), convt2d.bias.data.numel()
        dydp = o2convt2d.get_output_param_jacobian(x, sparse=False)
        dydp = dydp.reshape(y.numel(), -1)
        # First check: Compare loss Jacobians
        dLdw, dLdb = convt2d.weight.grad, convt2d.bias.grad
        dLdp = torch.cat([dLdw.reshape(-1), dLdb.reshape(-1)])
        dLdp_ = torch.einsum("i,ip->p", dLdy.reshape(-1), dydp)
        self.assertEqual(dLdp.shape, dLdp_.shape)
        self.assertTrue(torch.allclose(dLdp, dLdp_))
        # Second check: Compare output Jacobians

        def _conv2d_weight(w):
            output_padding = convt2d._output_padding(x, None, convt2d.stride, convt2d.padding, convt2d.kernel_size, convt2d.dilation)  # type: ignore[arg-type]
            return F.conv_transpose2d(
                x,
                w,
                convt2d.bias,
                convt2d.stride,
                convt2d.padding,
                output_padding,
                convt2d.groups,
                convt2d.dilation,
            )

        def _conv2d_bias(b):
            output_padding = convt2d._output_padding(x, None, convt2d.stride, convt2d.padding, convt2d.kernel_size, convt2d.dilation)  # type: ignore[arg-type]
            return F.conv_transpose2d(
                x,
                convt2d.weight,
                b,
                convt2d.stride,
                convt2d.padding,
                output_padding,
                convt2d.groups,
                convt2d.dilation,
            )

        dydw = jacobian(_conv2d_weight, convt2d.weight)
        dydb = jacobian(_conv2d_bias, convt2d.bias)
        dydp_ = torch.empty([y.numel(), nw + nb])
        dydp_[:, :nw] = dydw.reshape(y.numel(), -1)
        dydp_[:, nw:] = dydb.reshape(y.numel(), -1)
        dydp = dydp.reshape(y.numel(), -1)
        dydp_ = dydp_.reshape(y.numel(), -1)
        self.assertEqual(dydp.shape, dydp_.shape)
        self.assertTrue(torch.allclose(dydp, dydp_))

        # Benchmark:
        def _o2grad_code():
            o2convt2d.get_output_param_jacobian(x)

        o2grad_timer = timeit.Timer(stmt=_o2grad_code, setup=_o2grad_code)

        reps = 4
        time_w = o2grad_timer.timeit(number=reps) / reps
        print(f"Average time: {time_w}s")

    def test_get_output_param_hessians(self):
        x, y, o2convt2d, dLdy, dL2dy2 = self._setup(create_graph=True)
        convt2d = o2convt2d.module
        dy2dp2 = o2convt2d.get_output_param_hessian(x, sparse=False)

        dL2dp2 = get_hessian(convt2d)
        dydp = o2convt2d.get_output_param_jacobian(x)
        if dydp.layout == torch.sparse_coo:
            dydp = dydp.to_dense()
        dydp = dydp.reshape(y.numel(), -1)
        dL2dp2_term1 = torch.matmul(dLdy.reshape(-1), dy2dp2)
        dL2dp2_term2 = torch.matmul(
            dydp.T, torch.matmul(dL2dy2.reshape(y.numel(), y.numel()), dydp)
        )
        self.assertEqual(dL2dp2_term1.numel(), dL2dp2_term2.numel())
        dL2dp2_term1 = dL2dp2_term1.reshape(dL2dp2_term2.shape)
        dL2dp2_ = dL2dp2_term1 + dL2dp2_term2
        self.assertEqual(dL2dp2.shape, dL2dp2_.shape)
        self.assertTrue(torch.allclose(dL2dp2, dL2dp2_))

    def test_get_mixed_output_param_hessian(self):
        x, y, o2convt2d, dLdy, dL2dy2 = self._setup(create_graph=True)
        convt2d = o2convt2d.module
        nw, nb = convt2d.weight.data.numel(), convt2d.bias.data.numel()

        dy2dxdp = o2convt2d.get_mixed_output_param_hessian(x, sparse=False)
        dy2dxdp = dy2dxdp.reshape(y.numel(), -1)
        # Second check: Compare mixed Hessians
        def _conv2d(x, w, b):
            output_padding = convt2d._output_padding(x, None, convt2d.stride, convt2d.padding, convt2d.kernel_size, convt2d.dilation)  # type: ignore[arg-type]
            return F.conv_transpose2d(
                x,
                w,
                b,
                convt2d.stride,
                convt2d.padding,
                output_padding,
                convt2d.groups,
                convt2d.dilation,
            )

        def _jac_w(w):
            _conv1d_weight = lambda x: _conv2d(x, w, convt2d.bias)
            return jacobian(_conv1d_weight, x, create_graph=True)

        def _jac_b(b):
            _conv1d_bias = lambda x: _conv2d(x, convt2d.weight, b)
            return jacobian(_conv1d_bias, x, create_graph=True)

        dy2dxdw = jacobian(_jac_w, convt2d.weight)
        dy2dxdb = jacobian(_jac_b, convt2d.bias)
        dy2dxdp_ = torch.empty([y.numel(), x.numel(), nw + nb])
        dy2dxdp_[:, :, :nw] = dy2dxdw.reshape(y.numel(), x.numel(), -1)
        dy2dxdp_[:, :, nw:] = dy2dxdb.reshape(y.numel(), x.numel(), -1)
        dy2dxdp_ = dy2dxdp_.reshape(y.numel(), -1)
        self.assertEqual(dy2dxdp.shape, dy2dxdp_.shape)
        self.assertTrue(torch.allclose(dy2dxdp, dy2dxdp_))


if __name__ == "__main__":
    unittest.main()
