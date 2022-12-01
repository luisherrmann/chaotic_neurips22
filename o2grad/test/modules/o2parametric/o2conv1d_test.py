import unittest
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd.functional import jacobian, hessian

from o2grad.modules.o2parametric.o2conv1d import O2Conv1d, O2ConvTranspose1d
from o2grad.backprop import get_hessian


class O2Conv1dTest(unittest.TestCase):
    def _setup(self, create_graph=False):
        torch.manual_seed(0)
        x = torch.rand(2, 2, 10)
        criterion = nn.MSELoss()
        o2conv1d = O2Conv1d(
            in_channels=2,
            out_channels=3,
            kernel_size=2,
            stride=2,
            padding=2,
            dilation=2,
        )
        # o2conv1d = O2Conv1d(in_channels=1, out_channels=1, kernel_size=2, stride=1, padding=2, dilation=2)
        y = o2conv1d(x)
        target = torch.rand(y.shape)
        y.retain_grad()
        y.requires_grad_()
        loss = criterion(y, target)
        loss.backward(create_graph=create_graph)
        dLdy = y.grad

        def _criterion(y):
            return criterion(y, target)

        dL2dy2 = hessian(_criterion, y)

        return x, y, o2conv1d, dLdy, dL2dy2

    def test_forward(self):
        x = torch.rand(10, 3, 28)
        torch.manual_seed(0)
        o2conv1d = O2Conv1d(in_channels=3, out_channels=8, kernel_size=2, stride=2)
        torch.manual_seed(0)
        conv1d = nn.Conv1d(in_channels=3, out_channels=8, kernel_size=2, stride=2)
        yo2 = o2conv1d(x)
        y = conv1d(x)
        self.assertEqual(yo2.shape, y.shape)
        self.assertTrue(torch.equal(yo2, y))

    def test_get_output_input_jacobian(self):
        x, y, o2conv1d, dLdy, _ = self._setup()
        conv1d = o2conv1d.module
        dydx = o2conv1d.get_output_input_jacobian(x, sparse=False)

        def _jac(x):
            return jacobian(conv1d.forward, x, create_graph=True)

        dydx_ = _jac(x)

        dydx = dydx.reshape(y.numel(), x.numel())
        dydx_ = dydx.reshape(y.numel(), x.numel())
        self.assertEqual(dydx.shape, dydx_.shape)
        self.assertTrue(torch.allclose(dydx, dydx_))

    def test_get_output_input_hessian(self):
        x, y, o2conv1d, dLdy, _ = self._setup()
        conv1d = o2conv1d.module
        dy2dx2 = o2conv1d.get_output_input_hessian(x)
        if dy2dx2.layout == torch.sparse_coo:
            dy2dx2 = dy2dx2.to_dense()

        def _jac(x):
            return jacobian(conv1d.forward, x, create_graph=True)

        x = x.clone()
        dy2dx2_ = jacobian(_jac, x)
        self.assertEqual(dy2dx2.numel(), dy2dx2_.numel())
        dy2dx2 = dy2dx2.reshape(dy2dx2_.shape)
        self.assertTrue(torch.allclose(dy2dx2, dy2dx2_))

    def test_get_output_param_jacobian(self):
        x, y, o2conv1d, dLdy, _ = self._setup()
        conv1d = o2conv1d.module
        nw, nb = conv1d.weight.data.numel(), conv1d.bias.data.numel()
        dydp = o2conv1d.get_output_param_jacobian(x, sparse=False)
        dydp = dydp.reshape(y.numel(), -1)
        # First check: Compare loss Jacobians
        dLdw, dLdb = conv1d.weight.grad, conv1d.bias.grad
        dLdp = torch.cat([dLdw.reshape(-1), dLdb.reshape(-1)])
        dLdp_ = torch.einsum("i,ip->p", dLdy.reshape(-1), dydp)
        self.assertEqual(dLdp.shape, dLdp_.shape)
        self.assertTrue(torch.allclose(dLdp, dLdp_))
        # Second check: Compare output Jacobians

        def _conv1d_weight(w):
            return F.conv1d(
                x,
                w,
                conv1d.bias,
                conv1d.stride,
                conv1d.padding,
                conv1d.dilation,
                conv1d.groups,
            )

        def _conv1d_bias(b):
            return F.conv1d(
                x,
                conv1d.weight,
                b,
                conv1d.stride,
                conv1d.padding,
                conv1d.dilation,
                conv1d.groups,
            )

        dydw = jacobian(_conv1d_weight, conv1d.weight)
        dydb = jacobian(_conv1d_bias, conv1d.bias)
        dydp_ = torch.empty([*y.shape, nw + nb])
        dydp_[:, :, :, :nw] = dydw.reshape(*y.shape, -1)
        dydp_[:, :, :, nw:] = dydb.reshape(*y.shape, -1)
        dydp = dydp.reshape(y.numel(), -1)
        dydp_ = dydp_.reshape(y.numel(), -1)
        self.assertEqual(dydp.shape, dydp_.shape)
        self.assertTrue(torch.allclose(dydp, dydp_))

    def test_get_output_param_hessian(self):
        x, y, o2conv1d, dLdy, dL2dy2 = self._setup(create_graph=True)
        conv1d = o2conv1d.module
        nw, nb = conv1d.weight.data.numel(), conv1d.bias.data.numel()
        dy2dp2 = o2conv1d.get_output_param_hessian(x, sparse=False)
        dL2dp2 = get_hessian(conv1d)
        dydp = o2conv1d.get_output_param_jacobian(x, sparse=False)
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

    def test_get_output_param_hessians(self):
        x, y, o2conv1d, dLdy, dL2dy2 = self._setup(create_graph=True)
        conv1d = o2conv1d.module
        nw, nb = conv1d.weight.data.numel(), conv1d.bias.data.numel()
        dy2dxdp = o2conv1d.get_mixed_output_param_hessian(x, sparse=False)
        dy2dxdp = dy2dxdp.reshape(y.numel(), -1)

        def _conv1d(x, w, b):
            return F.conv1d(
                x, w, b, conv1d.stride, conv1d.padding, conv1d.dilation, conv1d.groups
            )

        def _jac_w(w):
            _conv1d_weight = lambda x: _conv1d(x, w, conv1d.bias)
            return jacobian(_conv1d_weight, x, create_graph=True)

        def _jac_b(b):
            _conv1d_bias = lambda x: _conv1d(x, conv1d.weight, b)
            return jacobian(_conv1d_bias, x, create_graph=True)

        dy2dxdw = jacobian(_jac_w, conv1d.weight)
        dy2dxdb = jacobian(_jac_b, conv1d.bias)
        dy2dxdp_ = torch.empty([y.numel(), x.numel(), nw + nb])
        dy2dxdp_[:, :, :nw] = dy2dxdw.reshape(y.numel(), x.numel(), -1)
        dy2dxdp_[:, :, nw:] = dy2dxdb.reshape(y.numel(), x.numel(), -1)
        dy2dxdp_ = dy2dxdp_.reshape(y.numel(), -1)
        self.assertEqual(dy2dxdp.shape, dy2dxdp_.shape)
        self.assertTrue(torch.allclose(dy2dxdp, dy2dxdp_))


class O2ConvTranspose1dTest(unittest.TestCase):
    def _setup(self, create_graph=False):
        torch.manual_seed(0)
        x = torch.rand(2, 1, 10)
        criterion = nn.MSELoss()
        o2tconv1d = O2ConvTranspose1d(
            in_channels=1,
            out_channels=3,
            kernel_size=2,
            stride=2,
            padding=2,
            dilation=2,
        )
        # o2conv1d = O2Conv1d(in_channels=1, out_channels=1, kernel_size=2, stride=1, padding=2, dilation=2)
        y = o2tconv1d(x)
        target = torch.rand(y.shape)
        y.retain_grad()
        y.requires_grad_()
        loss = criterion(y, target)
        loss.backward(create_graph=create_graph)
        dLdy = y.grad

        def _criterion(y):
            return criterion(y, target)

        dL2dy2 = hessian(_criterion, y)

        return x, y, o2tconv1d, dLdy, dL2dy2

    def test_forward(self):
        x = torch.rand(10, 3, 28)
        torch.manual_seed(0)
        o2conv1d = O2ConvTranspose1d(
            in_channels=3, out_channels=8, kernel_size=3, stride=2
        )
        torch.manual_seed(0)
        conv1d = nn.ConvTranspose1d(
            in_channels=3, out_channels=8, kernel_size=3, stride=2
        )
        yo2 = o2conv1d(x)
        y = conv1d(x)
        self.assertEqual(yo2.shape, y.shape)
        self.assertTrue(torch.equal(yo2, y))

    def test_get_output_input_jacobian(self):
        x, y, o2convt1d, dLdy, _ = self._setup()
        convt1d = o2convt1d.module
        dydx = o2convt1d.get_output_input_jacobian(x, sparse=False)

        def _jac(x):
            return jacobian(convt1d.forward, x, create_graph=True)

        dydx_ = _jac(x)

        dydx = dydx.reshape(y.numel(), x.numel())
        dydx_ = dydx.reshape(y.numel(), x.numel())
        self.assertEqual(dydx.shape, dydx_.shape)
        self.assertTrue(torch.allclose(dydx, dydx_))

    def test_get_output_input_hessian(self):
        x, y, o2tconv1d, dLdy, _ = self._setup()
        convt1d = o2tconv1d.module
        dy2dx2 = o2tconv1d.get_output_input_hessian(x)
        if dy2dx2.layout == torch.sparse_coo:
            dy2dx2 = dy2dx2.to_dense()

        def _jac(x):
            return jacobian(convt1d.forward, x, create_graph=True)

        x = x.clone()
        dy2dx2_ = jacobian(_jac, x)
        print(y.shape, x.shape)
        print(dy2dx2.shape, dy2dx2_.shape)
        self.assertEqual(dy2dx2.numel(), dy2dx2_.numel())
        dy2dx2 = dy2dx2.reshape(dy2dx2_.shape)
        self.assertTrue(torch.allclose(dy2dx2, dy2dx2_))

    def test_get_output_param_jacobian(self):
        x, y, o2convt1d, dLdy, _ = self._setup()
        convt1d = o2convt1d.module
        nw, nb = convt1d.weight.data.numel(), convt1d.bias.data.numel()
        dydp = o2convt1d.get_output_param_jacobian(x, sparse=False)
        dydp = dydp.reshape(y.numel(), -1)
        # First check: Compare loss Jacobians
        dLdw, dLdb = convt1d.weight.grad, convt1d.bias.grad
        dLdp = torch.cat([dLdw.reshape(-1), dLdb.reshape(-1)])
        dLdp_ = torch.einsum("i,ip->p", dLdy.reshape(-1), dydp)
        self.assertEqual(dLdp.shape, dLdp_.shape)
        self.assertTrue(torch.allclose(dLdp, dLdp_))
        # Second check: Compare output Jacobians

        def _conv1d_weight(w):
            output_padding = convt1d._output_padding(
                x,
                None,
                convt1d.stride,
                convt1d.padding,
                convt1d.kernel_size,
                convt1d.dilation,
            )
            return F.conv_transpose1d(
                x,
                w,
                convt1d.bias,
                convt1d.stride,
                convt1d.padding,
                output_padding,
                convt1d.groups,
                convt1d.dilation,
            )

        def _conv1d_bias(b):
            output_padding = convt1d._output_padding(
                x,
                None,
                convt1d.stride,
                convt1d.padding,
                convt1d.kernel_size,
                convt1d.dilation,
            )
            return F.conv_transpose1d(
                x,
                convt1d.weight,
                b,
                convt1d.stride,
                convt1d.padding,
                output_padding,
                convt1d.groups,
                convt1d.dilation,
            )

        dydw = jacobian(_conv1d_weight, convt1d.weight)
        dydb = jacobian(_conv1d_bias, convt1d.bias)
        dydp_ = torch.empty([*y.shape, nw + nb])
        dydp_[:, :, :, :nw] = dydw.reshape(*y.shape, -1)
        dydp_[:, :, :, nw:] = dydb.reshape(*y.shape, -1)
        dydp = dydp.reshape(y.numel(), -1)
        dydp_ = dydp_.reshape(y.numel(), -1)
        self.assertEqual(dydp.shape, dydp_.shape)
        self.assertTrue(torch.allclose(dydp, dydp_))

    def test_get_output_param_hessian(self):
        x, y, o2convt1d, dLdy, dL2dy2 = self._setup(create_graph=True)
        convt1d = o2convt1d.module
        nw, nb = convt1d.weight.data.numel(), convt1d.bias.data.numel()
        dy2dp2 = o2convt1d.get_output_param_hessian(x, sparse=False)
        # First check: Compare loss Hessians
        dL2dp2 = get_hessian(convt1d)
        dydp = o2convt1d.get_output_param_jacobian(x, sparse=False)
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
        x, y, o2convt1d, dLdy, dL2dy2 = self._setup(create_graph=True)
        convt1d = o2convt1d.module
        nw, nb = convt1d.weight.data.numel(), convt1d.bias.data.numel()
        dy2dxdp = o2convt1d.get_mixed_output_param_hessian(x, sparse=False)
        dy2dxdp = dy2dxdp.reshape(y.numel(), -1)

        def _conv1d(x, w, b):
            output_padding = convt1d._output_padding(
                x,
                None,
                convt1d.stride,
                convt1d.padding,
                convt1d.kernel_size,
                convt1d.dilation,
            )
            return F.conv_transpose1d(
                x,
                w,
                b,
                convt1d.stride,
                convt1d.padding,
                output_padding,
                convt1d.groups,
                convt1d.dilation,
            )

        def _jac_w(w):
            _convt1d_weight = lambda x: _conv1d(x, w, convt1d.bias)
            return jacobian(_convt1d_weight, x, create_graph=True)

        def _jac_b(b):
            _convt1d_bias = lambda x: _conv1d(x, convt1d.weight, b)
            return jacobian(_convt1d_bias, x, create_graph=True)

        dy2dxdw = jacobian(_jac_w, convt1d.weight)
        dy2dxdb = jacobian(_jac_b, convt1d.bias)
        dy2dxdp_ = torch.empty([y.numel(), x.numel(), nw + nb])
        dy2dxdp_[:, :, :nw] = dy2dxdw.reshape(y.numel(), x.numel(), -1)
        dy2dxdp_[:, :, nw:] = dy2dxdb.reshape(y.numel(), x.numel(), -1)
        dy2dxdp_ = dy2dxdp_.reshape(y.numel(), -1)
        self.assertEqual(dy2dxdp.shape, dy2dxdp_.shape)
        self.assertTrue(torch.allclose(dy2dxdp, dy2dxdp_))


if __name__ == "__main__":
    unittest.main()
