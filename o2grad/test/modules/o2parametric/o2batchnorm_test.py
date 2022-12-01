import unittest
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd.functional import jacobian, hessian

from o2grad.modules.o2parametric.o2batchnorm import O2BatchNorm1d, O2BatchNorm2d
from o2grad.backprop.backprop import get_hessian

"""
Due to the eps used in the root of the BatchNorm layer, the results of this layer match those produced
by autograd only up to a certain delta. Hence, we check if the calculated local derivatives are equal
with a certain precision. """
PRECISION = 1e-5
torch.set_printoptions(precision=10)


class BatchNorm1dTest(unittest.TestCase):
    def _setup(self, channels=False, create_graph=False):
        torch.manual_seed(0)
        if channels:
            x = torch.rand(2, 3, 4)
        else:
            x = torch.rand(2, 3)
        criterion = nn.MSELoss()
        o2bn1d = O2BatchNorm1d(num_features=3)
        y = o2bn1d(x)
        target = torch.rand(y.shape)
        y.retain_grad()
        y.requires_grad_()
        loss = criterion(y, target)
        loss.backward(create_graph=create_graph)
        dLdy = y.grad

        def _criterion(y):
            return criterion(y, target)

        dL2dy2 = hessian(_criterion, y)

        return x, y, o2bn1d, dLdy, dL2dy2

    def test_forward(self):
        x = torch.rand(10, 3, 28)
        torch.manual_seed(0)
        o2bn1d = O2BatchNorm1d(num_features=3, momentum=0.1)
        torch.manual_seed(0)
        bn1d = nn.BatchNorm1d(num_features=3, momentum=0.1)
        yo2 = o2bn1d(x)
        y = bn1d(x)
        self.assertEqual(yo2.shape, y.shape)
        self.assertTrue(torch.equal(yo2, y))

    def _test_get_output_input_jacobian(self, channels):
        x, y, o2bn1d, dLdy, dL2dy2 = self._setup(channels=channels)
        bn1d = o2bn1d.module

        def _jac(x):
            return jacobian(bn1d.forward, x, create_graph=True)

        dydx_ = _jac(x)

        dydx = o2bn1d.get_output_input_jacobian(x)
        if dydx.layout == torch.sparse_coo:
            dydx = dydx.to_dense()

        dydx = dydx.reshape(y.numel(), x.numel())
        dydx_ = dydx_.reshape(y.numel(), x.numel())
        self.assertEqual(dydx.shape, dydx_.shape)
        self.assertTrue(torch.all(torch.abs(dydx - dydx_) < PRECISION))

    def test_get_output_input_jacobian(self):
        self._test_get_output_input_jacobian(False)
        self._test_get_output_input_jacobian(True)

    def _test_get_output_input_hessian(self, channels):
        x, y, o2bn1d, dLdy, _ = self._setup(channels=channels)
        bn1d = o2bn1d.module
        bn1d.train()
        dy2dx2 = o2bn1d.get_output_input_hessian(x)
        if dy2dx2.layout == torch.sparse_coo:
            dy2dx2 = dy2dx2.to_dense()

        def _jac(x):
            return jacobian(bn1d.forward, x, create_graph=True)

        x = x.clone()
        dy2dx2_ = jacobian(_jac, x)

        self.assertEqual(dy2dx2.numel(), dy2dx2_.numel())
        dy2dx2 = dy2dx2.reshape(dy2dx2_.shape)
        # print(torch.abs(dy2dx2 - dy2dx2_))
        tmp = (dy2dx2 - dy2dx2_).reshape(-1)
        print(torch.abs(tmp) < PRECISION)
        print(tmp[torch.abs(tmp) >= PRECISION])
        self.assertTrue(torch.all(torch.abs(dy2dx2 - dy2dx2_) < PRECISION))

    def test_get_output_input_hessian(self):
        self._test_get_output_input_hessian(False)
        self._test_get_output_input_hessian(True)

    def _test_get_output_param_jacobian(self, channels):
        x, y, o2bn1d, dLdy, _ = self._setup(channels=channels)
        bn1d = o2bn1d.module
        bn1d.train()
        nw, nb = bn1d.weight.data.numel(), bn1d.bias.data.numel()

        dydp = o2bn1d.get_output_param_jacobian(x)
        if dydp.layout == torch.sparse_coo:
            dydp = dydp.to_dense()
        dydp = dydp.reshape(y.numel(), -1)

        # First check: Compare loss Jacobians
        dLdw, dLdb = bn1d.weight.grad, bn1d.bias.grad
        dLdp = torch.cat([dLdw.reshape(-1), dLdb.reshape(-1)])

        dLdp_ = torch.einsum("i,ip->p", dLdy.reshape(-1), dydp)

        self.assertEqual(dLdp.shape, dLdp_.shape)
        self.assertTrue(torch.allclose(dLdp, dLdp_))

        # Second check: Compare output Jacobians
        def _bn1d_weight(w):
            return F.batch_norm(
                x,
                bn1d.running_mean
                if not bn1d.training or bn1d.track_running_stats
                else None,
                bn1d.running_var
                if not bn1d.training or bn1d.track_running_stats
                else None,
                w,
                bn1d.bias,
                bn1d.training,
                bn1d.momentum,
                bn1d.eps,
            )

        def _bn1d_bias(b):
            return F.batch_norm(
                x,
                bn1d.running_mean
                if not bn1d.training or bn1d.track_running_stats
                else None,
                bn1d.running_var
                if not bn1d.training or bn1d.track_running_stats
                else None,
                bn1d.weight,
                b,
                bn1d.training,
                bn1d.momentum,
                bn1d.eps,
            )

        dydw = jacobian(_bn1d_weight, bn1d.weight)
        dydb = jacobian(_bn1d_bias, bn1d.bias)
        dydp_ = torch.empty([y.numel(), nw + nb])
        dydp_[:, :nw] = dydw.reshape(y.numel(), -1)
        dydp_[:, nw:] = dydb.reshape(y.numel(), -1)
        # Compare results
        dydp = dydp.reshape(y.numel(), -1)
        dydp_ = dydp_.reshape(y.numel(), -1)
        self.assertEqual(dydp.shape, dydp_.shape)
        self.assertTrue(torch.allclose(dydp, dydp_))

    def test_get_output_param_jacobian(self):
        self._test_get_output_param_jacobian(False)
        self._test_get_output_param_jacobian(True)

    def _test_get_output_param_hessian(self, channels):
        x, y, o2bn1d, dLdy, dL2dy2 = self._setup(channels=channels, create_graph=True)
        bn1d = o2bn1d.module
        nw, nb = bn1d.weight.data.numel(), bn1d.bias.data.numel()
        n = nw + nb

        dy2dp2 = o2bn1d.get_output_param_hessian(x)
        if dy2dp2.layout == torch.sparse_coo:
            dy2dp2 = dy2dp2.to_dense()

        # First check: Compare loss Hessians
        dL2dp2 = get_hessian(bn1d)
        dydp = o2bn1d.get_output_param_jacobian(x)
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
        errors = torch.abs(dL2dp2 - dL2dp2_)
        self.assertTrue(torch.all(errors < PRECISION))

        def _bn1d(x, w, b):
            return F.batch_norm(
                x,
                torch.clone(bn1d.running_mean)
                if not bn1d.training or bn1d.track_running_stats
                else None,
                torch.clone(bn1d.running_var)
                if not bn1d.training or bn1d.track_running_stats
                else None,
                w,
                b,
                bn1d.training,
                bn1d.momentum,
                bn1d.eps,
            )

        def _jac_w(w):
            _bn1d_weight = lambda w: _bn1d(x, w, bn1d.bias)
            return jacobian(_bn1d_weight, w, create_graph=True)

        def _jac_b(b):
            _bn1d_bias = lambda b: _bn1d(x, bn1d.weight, b)
            return jacobian(_bn1d_bias, b, create_graph=True)

        def _jac_wb(b):
            _bn1d_bias = lambda w: _bn1d(x, w, b)
            return jacobian(_bn1d_bias, bn1d.weight, create_graph=True)

        dy2dw2 = jacobian(_jac_w, bn1d.weight).reshape(y.numel(), nw, nw)
        dy2db2 = jacobian(_jac_b, bn1d.bias).reshape(y.numel(), nb, nb)
        dy2dwdb = jacobian(_jac_wb, bn1d.bias).reshape(y.numel(), nw, nb)
        dy2dp2_ = torch.empty(y.numel(), n, n)
        dy2dp2_[:, :nw, :nw] = dy2dw2
        dy2dp2_[:, nw:, nw:] = dy2db2
        dy2dp2_[:, :nw, nw:] = dy2dwdb
        dy2dp2_[:, nw:, :nw] = dy2dwdb.permute([0, 2, 1])
        dy2dp2_ = dy2dp2_.reshape(dy2dp2.shape)
        self.assertTrue(torch.allclose(dy2dp2, dy2dp2_))

    def test_get_output_param_hessian(self):
        self._test_get_output_param_hessian(False)
        self._test_get_output_param_hessian(False)

    def _test_get_mixed_output_param_hessian(self, channels):
        x, y, o2bn1d, dLdy, dL2dy2 = self._setup(channels=channels, create_graph=True)
        bn1d = o2bn1d.module
        nw, nb = bn1d.weight.data.numel(), bn1d.bias.data.numel()
        nw + nb
        dy2dxdp = o2bn1d.get_mixed_output_param_hessian(x)
        if dy2dxdp.layout == torch.sparse_coo:
            dy2dxdp = dy2dxdp.to_dense()
        dy2dxdp = dy2dxdp.reshape(y.numel(), -1)

        def _bn1d(x, w, b):
            return F.batch_norm(
                x,
                torch.clone(bn1d.running_mean)
                if not bn1d.training or bn1d.track_running_stats
                else None,
                torch.clone(bn1d.running_var)
                if not bn1d.training or bn1d.track_running_stats
                else None,
                w,
                b,
                bn1d.training,
                bn1d.momentum,
                bn1d.eps,
            )

        def _jac_w(w):
            _bn1d_weight = lambda x: _bn1d(x, w, bn1d.bias)
            return jacobian(_bn1d_weight, x, create_graph=True)

        def _jac_b(b):
            _bn1d_bias = lambda x: _bn1d(x, bn1d.weight, b)
            return jacobian(_bn1d_bias, x, create_graph=True)

        dy2dxdw = jacobian(_jac_w, bn1d.weight)
        dy2dxdb = jacobian(_jac_b, bn1d.bias)
        dy2dxdp_ = torch.empty([y.numel(), x.numel(), nw + nb])
        dy2dxdp_[:, :, :nw] = dy2dxdw.reshape(y.numel(), x.numel(), -1)
        dy2dxdp_[:, :, nw:] = dy2dxdb.reshape(y.numel(), x.numel(), -1)
        dy2dxdp_ = dy2dxdp_.reshape(y.numel(), -1)

        self.assertEqual(dy2dxdp.shape, dy2dxdp_.shape)
        errors = torch.abs(dy2dxdp - dy2dxdp_)
        self.assertTrue(torch.all(errors < PRECISION))

    def test_get_mixed_output_param_hessian(self):
        self._test_get_mixed_output_param_hessian(False)
        self._test_get_mixed_output_param_hessian(False)


class BatchNorm2dTest(unittest.TestCase):
    def _setup(self, create_graph=False):
        torch.manual_seed(0)
        x = torch.rand(3, 3, 2, 2)
        criterion = nn.MSELoss()
        o2bn1d = O2BatchNorm2d(num_features=3)
        y = o2bn1d(x)
        target = torch.rand(y.shape)
        y.retain_grad()
        y.requires_grad_()
        loss = criterion(y, target)
        loss.backward(create_graph=create_graph)
        dLdy = y.grad

        def _criterion(y):
            return criterion(y, target)

        dL2dy2 = hessian(_criterion, y)

        return x, y, o2bn1d, dLdy, dL2dy2

    def test_forward(self):
        # x = torch.rand(10, 3, 28, 28)
        x = torch.rand(2, 3, 4, 5)
        torch.manual_seed(0)
        o2bn2d = O2BatchNorm2d(num_features=3, momentum=0.1)
        # o2bn2d = O2BatchNorm2d(num_features=3, momentum=0.1)
        torch.manual_seed(0)
        bn2d = nn.BatchNorm2d(num_features=3, momentum=0.1)
        # bn2d = nn.BatchNorm2d(num_features=3, momentum=0.1)
        yo2 = o2bn2d(x)
        y = bn2d(x)
        self.assertEqual(yo2.shape, y.shape)
        self.assertTrue(torch.equal(yo2, y))
        mean = torch.mean(x, dim=[0, 2, 3]).reshape(1, 3, 1, 1)
        var = torch.var(x, dim=[0, 2, 3], unbiased=False).reshape(1, 3, 1, 1)
        xhat = (x - mean) / torch.sqrt(var + bn2d.eps)
        w, b = bn2d.weight.data, bn2d.bias.data
        w = w.reshape(1, 3, 1, 1)
        b = b.reshape(1, 3, 1, 1)
        y_ = xhat * w + b
        print(y)
        print(y_)
        print(y - y_)

    def test_get_output_input_jacobian(self):
        x, y, o2bn2d, dLdy, dL2dy2 = self._setup()
        bn2d = o2bn2d.module

        def _jac(x):
            return jacobian(bn2d.forward, x, create_graph=True)

        dydx_ = _jac(x)

        dydx = o2bn2d.get_output_input_jacobian(x)
        if dydx.layout == torch.sparse_coo:
            dydx = dydx.to_dense()

        dydx = dydx.reshape(y.numel(), x.numel())
        dydx_ = dydx_.reshape(y.numel(), x.numel())
        self.assertEqual(dydx.shape, dydx_.shape)
        self.assertTrue(torch.all(torch.abs(dydx - dydx_) < PRECISION))

    def test_get_output_input_hessian(self):
        x, y, o2bn2d, dLdy, _ = self._setup()
        bn2d = o2bn2d.module
        bn2d.train()
        dy2dx2 = o2bn2d.get_output_input_hessian(x)
        if dy2dx2.layout == torch.sparse_coo:
            dy2dx2 = dy2dx2.to_dense()

        def _jac(x):
            return jacobian(bn2d.forward, x, create_graph=True)

        x = x.clone()
        dy2dx2_ = jacobian(_jac, x)

        self.assertEqual(dy2dx2.numel(), dy2dx2_.numel())
        dy2dx2 = dy2dx2.reshape(dy2dx2_.shape)
        # print(torch.abs(dy2dx2 - dy2dx2_))
        tmp = (dy2dx2 - dy2dx2_).reshape(-1)
        print(torch.abs(tmp) < PRECISION)
        print(tmp[torch.abs(tmp) >= PRECISION])
        self.assertTrue(torch.all(torch.abs(dy2dx2 - dy2dx2_) < PRECISION))

    def test_get_output_param_jacobian(self):
        x, y, o2bn1d, dLdy, _ = self._setup()
        bn1d = o2bn1d.module
        bn1d.train()
        nw, nb = bn1d.weight.data.numel(), bn1d.bias.data.numel()

        dydp = o2bn1d.get_output_param_jacobian(x)
        if dydp.layout == torch.sparse_coo:
            dydp = dydp.to_dense()
        dydp = dydp.reshape(y.numel(), -1)

        # First check: Compare loss Jacobians
        dLdw, dLdb = bn1d.weight.grad, bn1d.bias.grad
        dLdp = torch.cat([dLdw.reshape(-1), dLdb.reshape(-1)])

        dLdp_ = torch.einsum("i,ip->p", dLdy.reshape(-1), dydp)

        self.assertEqual(dLdp.shape, dLdp_.shape)
        self.assertTrue(torch.allclose(dLdp, dLdp_))

        # Second check: Compare output Jacobians
        def _bn1d_weight(w):
            return F.batch_norm(
                x,
                bn1d.running_mean
                if not bn1d.training or bn1d.track_running_stats
                else None,
                bn1d.running_var
                if not bn1d.training or bn1d.track_running_stats
                else None,
                w,
                bn1d.bias,
                bn1d.training,
                bn1d.momentum,
                bn1d.eps,
            )

        def _bn1d_bias(b):
            return F.batch_norm(
                x,
                bn1d.running_mean
                if not bn1d.training or bn1d.track_running_stats
                else None,
                bn1d.running_var
                if not bn1d.training or bn1d.track_running_stats
                else None,
                bn1d.weight,
                b,
                bn1d.training,
                bn1d.momentum,
                bn1d.eps,
            )

        dydw = jacobian(_bn1d_weight, bn1d.weight)
        dydb = jacobian(_bn1d_bias, bn1d.bias)
        dydp_ = torch.empty([y.numel(), nw + nb])
        dydp_[:, :nw] = dydw.reshape(y.numel(), -1)
        dydp_[:, nw:] = dydb.reshape(y.numel(), -1)
        # Compare results
        dydp = dydp.reshape(y.numel(), -1)
        dydp_ = dydp_.reshape(y.numel(), -1)
        self.assertEqual(dydp.shape, dydp_.shape)
        self.assertTrue(torch.allclose(dydp, dydp_))

    def test_get_output_param_hessian(self):
        x, y, o2bn1d, dLdy, dL2dy2 = self._setup(create_graph=True)
        bn1d = o2bn1d.module
        nw, nb = bn1d.weight.data.numel(), bn1d.bias.data.numel()
        n = nw + nb

        dy2dp2 = o2bn1d.get_output_param_hessian(x)
        if dy2dp2.layout == torch.sparse_coo:
            dy2dp2 = dy2dp2.to_dense()

        # First check: Compare loss Hessians
        dL2dp2 = get_hessian(bn1d)
        dydp = o2bn1d.get_output_param_jacobian(x)
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
        errors = torch.abs(dL2dp2 - dL2dp2_)
        self.assertTrue(torch.all(errors < PRECISION))

        def _bn1d(x, w, b):
            return F.batch_norm(
                x,
                torch.clone(bn1d.running_mean)
                if not bn1d.training or bn1d.track_running_stats
                else None,
                torch.clone(bn1d.running_var)
                if not bn1d.training or bn1d.track_running_stats
                else None,
                w,
                b,
                bn1d.training,
                bn1d.momentum,
                bn1d.eps,
            )

        def _jac_w(w):
            _bn1d_weight = lambda w: _bn1d(x, w, bn1d.bias)
            return jacobian(_bn1d_weight, w, create_graph=True)

        def _jac_b(b):
            _bn1d_bias = lambda b: _bn1d(x, bn1d.weight, b)
            return jacobian(_bn1d_bias, b, create_graph=True)

        def _jac_wb(b):
            _bn1d_bias = lambda w: _bn1d(x, w, b)
            return jacobian(_bn1d_bias, bn1d.weight, create_graph=True)

        dy2dw2 = jacobian(_jac_w, bn1d.weight).reshape(y.numel(), nw, nw)
        dy2db2 = jacobian(_jac_b, bn1d.bias).reshape(y.numel(), nb, nb)
        dy2dwdb = jacobian(_jac_wb, bn1d.bias).reshape(y.numel(), nw, nb)
        dy2dp2_ = torch.empty(y.numel(), n, n)
        dy2dp2_[:, :nw, :nw] = dy2dw2
        dy2dp2_[:, nw:, nw:] = dy2db2
        dy2dp2_[:, :nw, nw:] = dy2dwdb
        dy2dp2_[:, nw:, :nw] = dy2dwdb.permute([0, 2, 1])
        dy2dp2_ = dy2dp2_.reshape(dy2dp2.shape)
        self.assertTrue(torch.allclose(dy2dp2, dy2dp2_))

    def test_get_mixed_output_param_hessian(self):
        x, y, o2bn2d, dLdy, dL2dy2 = self._setup(create_graph=True)
        bn2d = o2bn2d.module
        nw, nb = bn2d.weight.data.numel(), bn2d.bias.data.numel()
        nw + nb
        dy2dxdp = o2bn2d.get_mixed_output_param_hessian(x)
        if dy2dxdp.layout == torch.sparse_coo:
            dy2dxdp = dy2dxdp.to_dense()
        dy2dxdp = dy2dxdp.reshape(y.numel(), -1)

        def _bn2d(x, w, b):
            return F.batch_norm(
                x,
                torch.clone(bn2d.running_mean)
                if not bn2d.training or bn2d.track_running_stats
                else None,
                torch.clone(bn2d.running_var)
                if not bn2d.training or bn2d.track_running_stats
                else None,
                w,
                b,
                bn2d.training,
                bn2d.momentum,
                bn2d.eps,
            )

        def _jac_w(w):
            _bn1d_weight = lambda x: _bn2d(x, w, bn2d.bias)
            return jacobian(_bn1d_weight, x, create_graph=True)

        def _jac_b(b):
            _bn1d_bias = lambda x: _bn2d(x, bn2d.weight, b)
            return jacobian(_bn1d_bias, x, create_graph=True)

        dy2dxdw = jacobian(_jac_w, bn2d.weight)
        dy2dxdb = jacobian(_jac_b, bn2d.bias)
        dy2dxdp_ = torch.empty([y.numel(), x.numel(), nw + nb])
        dy2dxdp_[:, :, :nw] = dy2dxdw.reshape(y.numel(), x.numel(), -1)
        dy2dxdp_[:, :, nw:] = dy2dxdb.reshape(y.numel(), x.numel(), -1)
        dy2dxdp_ = dy2dxdp_.reshape(y.numel(), -1)

        self.assertEqual(dy2dxdp.shape, dy2dxdp_.shape)
        errors = torch.abs(dy2dxdp - dy2dxdp_)
        self.assertTrue(torch.all(errors < PRECISION))


if __name__ == "__main__":
    unittest.main()
