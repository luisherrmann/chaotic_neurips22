import unittest
import torch
import torch.nn as nn
from torch.autograd.functional import jacobian, hessian

from o2grad.modules.o2layer.o2reshape import O2Reshape, Reshape
from o2grad.modules.o2parametric.o2linear import O2Linear
from o2grad.backprop.o2model import O2Model


class ReshapeTest(unittest.TestCase):
    def test_init_forward(self):
        x = torch.rand(2, 3, 5, 2)
        reshape = Reshape(30)
        y = reshape(x)
        all_match = torch.all(y == x.reshape(2, 30)).item()
        self.assertTrue(all_match)
        # Initialize by passing shape as list
        reshape = Reshape([30])
        y = reshape(x)
        all_match = torch.all(y == x.reshape(2, 30)).item()
        self.assertTrue(all_match)
        # Initialize by passing shape as tuple
        reshape = Reshape((30,))
        y = reshape(x)
        all_match = torch.all(y == x.reshape(2, 30)).item()
        self.assertTrue(all_match)
        # Initialize by passing several shape dimensions
        reshape = Reshape(3, 10)
        y = reshape(x)
        all_match = torch.all(y == x.reshape(2, 3, 10)).item()
        self.assertTrue(all_match)
        # Initialize by passing several shape dimensions as list
        reshape = Reshape([3, 10])
        y = reshape(x)
        all_match = torch.all(y == x.reshape(2, 3, 10)).item()
        self.assertTrue(all_match)
        # Initialize by passing several shape dimensions as tuple
        reshape = Reshape((3, 10))
        y = reshape(x)
        all_match = torch.all(y == x.reshape(2, 3, 10)).item()
        self.assertTrue(all_match)
        # Check for error if initialized with several lists:
        def _reshape():
            Reshape([2, 2], [1])

        self.assertRaises(ValueError, _reshape)


class O2ReshapeTest(unittest.TestCase):
    def _setup(self, create_graph=False):
        torch.manual_seed(0)
        x = torch.rand(2, 3, 5)
        target = torch.rand(2, 15)
        criterion = nn.MSELoss()
        o2reshape = O2Reshape([15])
        y = o2reshape(x)
        y.requires_grad_()
        y.retain_grad()
        loss = criterion(y, target)
        loss.backward(create_graph=create_graph)
        dLdy = y.grad

        def _criterion(y):
            return criterion(y, target)

        dL2dy2 = hessian(_criterion, y)

        return x, y, o2reshape, dLdy, dL2dy2

    def test_forward(self):
        x = torch.rand(1, 3, 5)
        torch.manual_seed(0)
        o2reshape = O2Reshape([15])
        torch.manual_seed(0)
        yo2 = o2reshape(x)
        y = x.reshape(15)
        self.assertTrue(torch.all(yo2 == y).item())

    def test_get_output_input_jacobian(self):
        x, y, o2reshape, _, _ = self._setup()
        linear = o2reshape.module
        dydx = o2reshape.get_output_input_jacobian(x)
        if dydx.layout == torch.sparse_coo:
            dydx = dydx.to_dense()
        # Obtain jacobian of module
        def _jac(x):
            return jacobian(linear.forward, x, create_graph=True)

        dydx_ = _jac(x)

        self.assertEqual(dydx.numel(), dydx_.numel())
        dydx = dydx.reshape(dydx_.shape)
        self.assertTrue(torch.allclose(dydx, dydx_))

    def test_get_output_input_hessian(self):
        x, y, o2reshape, _, _ = self._setup()
        linear = o2reshape.module
        dy2dx2 = o2reshape.get_output_input_hessian(x)
        if dy2dx2.layout == torch.sparse_coo:
            dy2dx2 = dy2dx2.to_dense()
        # Obtain Hessian of module
        def _jac(x):
            return jacobian(linear.forward, x, create_graph=True)

        x = x.clone()
        dy2dx2_ = jacobian(_jac, x)

        self.assertEqual(dy2dx2.numel(), dy2dx2_.numel())
        dy2dx2 = dy2dx2.reshape(dy2dx2_.shape)
        self.assertTrue(torch.allclose(dy2dx2, dy2dx2_))

    def test_get_loss_input_hessian(self):
        x = torch.rand(2, 3, 5)
        x.requires_grad = True
        y = torch.rand(2, 3)
        o2reshape = O2Reshape([15])
        o2linear = O2Linear(15, 3)
        model = nn.Sequential(o2reshape, o2linear)
        o2model = O2Model(model, nn.MSELoss(), save_intermediate=["dL2dx2"])
        y_hat = o2model(x)
        criterion = o2model.criterion
        print(criterion.input)
        loss = criterion(y_hat, y)
        loss.backward(create_graph=True)
        print(o2linear.settings, o2linear.dL2dx2)
        print(o2reshape.settings, o2reshape.dL2dx2)
        dL2dx2 = o2reshape.dL2dx2
        if dL2dx2.layout == torch.sparse_coo:
            dL2dx2 = dL2dx2.to_dense()
        dL2dy2 = o2linear.dL2dx2
        if dL2dy2.layout == torch.sparse_coo:
            dL2dy2 == dL2dy2.to_dense()

        all_match = torch.all(dL2dx2 == dL2dy2).item()
        self.assertTrue(all_match)


if __name__ == "__main__":
    unittest.main()
