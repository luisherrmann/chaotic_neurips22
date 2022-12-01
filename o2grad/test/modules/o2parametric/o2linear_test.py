import unittest
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd.functional import jacobian, hessian

from o2grad.backprop import get_hessian
from o2grad.modules.o2parametric.o2linear import O2Linear


class O2LinearTest(unittest.TestCase):
    def _setup(self, create_graph=False):
        torch.manual_seed(0)
        x = torch.rand(2, 3)
        target = torch.rand(2, 5)
        criterion = nn.MSELoss()
        o2linear = O2Linear(3, 5)
        # x = torch.rand(64, 1000)
        # target = torch.rand(64, 1)
        # criterion = nn.MSELoss()
        # o2linear = O2Linear(1000, 1)
        y = o2linear(x)
        y.retain_grad()
        y.requires_grad_()
        loss = criterion(y, target)
        loss.backward(create_graph=create_graph)
        dLdy = y.grad

        def _criterion(y):
            return criterion(y, target)

        dL2dy2 = hessian(_criterion, y)
        return x, y, o2linear, dLdy, dL2dy2

    def test_forward(self):
        x = torch.rand(1, 3)
        torch.manual_seed(0)
        o2linear = O2Linear(3, 2)
        torch.manual_seed(0)
        linear = nn.Linear(3, 2)
        yo2 = o2linear(x)
        y = linear(x)
        self.assertTrue(torch.equal(yo2, y))

    def test_get_output_input_jacobian(self):
        x, y, o2linear, dLdy, _ = self._setup()
        linear = o2linear.module
        dydx = o2linear.get_output_input_jacobian(x)
        dydx_sparse = o2linear.get_output_input_jacobian(x, sparse=True)
        self.assertTrue(torch.all(dydx == dydx_sparse.to_dense()).item())

        def _jac(x):
            return jacobian(linear.forward, x, create_graph=True)

        dydx_ = _jac(x)

        self.assertEqual(dydx.numel(), dydx_.numel())
        dydx = dydx.reshape(dydx_.shape)
        self.assertTrue(torch.allclose(dydx, dydx_))

    def test_get_output_input_hessian(self):
        x, y, o2linear, dLdy, _ = self._setup()
        linear = o2linear.module
        dy2dx2 = o2linear.get_output_input_hessian(x)
        if dy2dx2.layout == torch.sparse_coo:
            dy2dx2 = dy2dx2.to_dense()

        def _jac(x):
            return jacobian(linear.forward, x, create_graph=True)

        x = x.clone()
        dy2dx2_ = jacobian(_jac, x)
        self.assertEqual(dy2dx2.numel(), dy2dx2_.numel())
        dy2dx2 = dy2dx2.reshape(dy2dx2_.shape)
        self.assertTrue(torch.allclose(dy2dx2, dy2dx2_))

    def test_get_output_param_jacobian(self):
        x, y, o2linear, dLdy, _ = self._setup()
        linear = o2linear.module
        weight, bias = linear.weight.data, linear.bias.data
        dydp = o2linear.get_output_param_jacobian(x)
        dydp_sparse = o2linear.get_output_param_jacobian(x, sparse=True)
        self.assertTrue(torch.all(dydp == dydp_sparse.to_dense()).item())
        dLdw, dLdb = linear.weight.grad, linear.bias.grad
        dLdp = torch.cat([dLdw.reshape(-1), dLdb.reshape(-1)])
        dLdp_ = torch.matmul(dLdy.reshape(-1), dydp)
        self.assertEqual(dLdp.shape, dLdp_.shape)
        self.assertTrue(torch.allclose(dLdp, dLdp_))
        # Get dimensions
        output_dim, input_dim = weight.shape
        N, _ = x.shape
        nw, nb = weight.numel(), bias.numel()
        n = nw + nb

        def _linear_w(w):
            return F.linear(x, w, linear.bias)

        dydw = jacobian(_linear_w, linear.weight)
        dydw = torch.reshape(dydw, [N * output_dim, nw])

        def _linear_b(b):
            return F.linear(x, linear.weight, b)

        dydb = jacobian(_linear_b, linear.bias)
        dydb = torch.reshape(dydb, [N * output_dim, nb])
        dydp_ = torch.empty([N * output_dim, n])
        dydp_[:, :nw] = dydw
        dydp_[:, nw:] = dydb
        self.assertEqual(dydp.shape, dydp_.shape)
        self.assertTrue(torch.allclose(dydp, dydp_))

    def test_get_output_param_hessian(self):
        x, y, o2linear, dLdy, dL2dy2 = self._setup(create_graph=True)
        linear = o2linear.module
        dy2dp2 = o2linear.get_output_param_hessian(x, sparse=True)
        dy2dp2 = dy2dp2.to_dense()
        # First check: Compare loss Hessians
        dL2dp2 = get_hessian(linear)
        dydp = o2linear.get_output_param_jacobian(x)
        if dydp.layout == torch.sparse_coo:
            dydp = dydp.to_dense()
        dL2dp2_term1 = torch.matmul(dLdy.reshape(-1), dy2dp2)
        dL2dp2_term2 = torch.matmul(
            dydp.T, torch.matmul(dL2dy2.reshape(y.numel(), y.numel()), dydp)
        )
        self.assertEqual(dL2dp2_term1.numel(), dL2dp2_term2.numel())
        dL2dp2_term1 = dL2dp2_term1.reshape(dL2dp2_term2.shape)
        dL2dp2_ = dL2dp2_term1 + dL2dp2_term2
        self.assertEqual(dL2dp2.numel(), dL2dp2_.numel())
        dL2dp2 = dL2dp2.reshape(dL2dp2_.shape)
        self.assertTrue(torch.allclose(dL2dp2, dL2dp2_))

    def test_get_mixed_output_param_hessian(self):
        # Second check: Compare mixed Hessians
        x, y, o2linear, dLdy, dL2dy2 = self._setup(create_graph=True)
        linear = o2linear.module
        dy2dxdp = o2linear.get_mixed_output_param_hessian(x, sparse=True)
        dy2dxdp = dy2dxdp.to_dense()
        linear = o2linear.module
        N, input_dim = x.shape
        _, output_dim = y.shape
        nw = input_dim * output_dim
        nb = output_dim
        n = nw + nb

        def _jac_linear_w(w):
            W = lambda x: F.linear(x, w, linear.bias)
            return jacobian(W, x, create_graph=True)

        dy2dxdw = jacobian(_jac_linear_w, linear.weight)

        def _jac_linear_b(b):
            B = lambda x: F.linear(x, linear.weight, b)
            return jacobian(B, x, create_graph=True)

        dy2dxdb = jacobian(_jac_linear_b, linear.bias)
        dy2dxdp_ = torch.empty([N, output_dim, N * input_dim, n])
        dy2dxdp_[:, :, :, :nw] = dy2dxdw.reshape(N, output_dim, N * input_dim, nw)
        dy2dxdp_[:, :, :, nw:] = dy2dxdb.reshape(N, output_dim, N * input_dim, nb)
        self.assertEqual(dy2dxdp.numel(), dy2dxdp_.numel())
        dy2dxdp = dy2dxdp.reshape(dy2dxdp_.shape)
        self.assertTrue(torch.allclose(dy2dxdp, dy2dxdp_))


if __name__ == "__main__":
    unittest.main()
