import unittest
import torch
import torch.nn as nn
from torch.autograd.functional import jacobian, hessian

from o2grad.modules.o2layer.o2pooling import (
    O2AvgPool1d,
    O2AvgPool2d,
    O2MaxPool1d,
    O2MaxPool2d,
)


class O2AvgPool1dTest(unittest.TestCase):
    def _setup(self, create_graph=False):
        torch.manual_seed(0)
        x = torch.rand(2, 3, 6)
        criterion = nn.MSELoss()
        o2avgpool1d = O2AvgPool1d(kernel_size=2, stride=2)
        output = o2avgpool1d(x)
        y, idx = output
        y.requires_grad_()
        y.retain_grad()
        target = torch.rand(y.shape)
        loss = criterion(y, target)
        loss.backward(create_graph=create_graph)
        dLdy = y.grad

        def _criterion(y):
            return criterion(y, target)

        dL2dy2 = hessian(_criterion, y)
        return x, output, o2avgpool1d, dLdy, dL2dy2

    def test_forward(self):
        x = torch.rand(2, 3, 5)
        torch.manual_seed(0)
        o2avgpool1d = O2AvgPool1d(kernel_size=2, stride=2)
        torch.manual_seed(0)
        avgpool1d = nn.AvgPool1d(kernel_size=2, stride=2)
        o2y, o2idx = o2avgpool1d(x)
        y, idx = avgpool1d(x)
        self.assertTrue(all((o2y == y).flatten()))
        self.assertTrue(all((o2idx == idx).flatten()))

    def test_get_output_input_jacobian(self):
        x, y, o2avgpool1d, dLdy, _ = self._setup()
        avgpool1d = o2avgpool1d.module
        dydx = o2avgpool1d.get_output_input_jacobian(x)
        if dydx.layout == torch.sparse_coo:
            dydx = dydx.to_dense()

        def _o2avgpool1d(x):
            y = avgpool1d(x)
            return y

        def _jac(x):
            return jacobian(_o2avgpool1d, x, create_graph=True)

        dydx_ = _jac(x)

        self.assertEqual(dydx.numel(), dydx_.numel())
        dydx = dydx.reshape(dydx_.shape)
        y_, idx = y
        self.assertTrue(torch.allclose(dydx, dydx_))

    def test_get_output_input_hessian(self):
        x, y, o2avgpool1d, dLdy, _ = self._setup()
        avgpool1d = o2avgpool1d.module

        dy2dx2 = o2avgpool1d.get_output_input_hessian(x)
        if dy2dx2.layout == torch.sparse_coo:
            dy2dx2 = dy2dx2.to_dense()

        def _avgpool1d(x):
            y = avgpool1d(x)
            return y

        def _jac(x):
            return jacobian(_avgpool1d, x, create_graph=True)

        x = x.clone()
        dy2dx2_ = jacobian(_jac, x)
        self.assertEqual(dy2dx2.numel(), dy2dx2_.numel())
        dy2dx2 = dy2dx2.reshape(dy2dx2_.shape)
        self.assertTrue(torch.allclose(dy2dx2, dy2dx2_))


class O2AvgPool2dTest(unittest.TestCase):
    def _setup(self, create_graph=False):
        torch.manual_seed(0)
        x = torch.rand(2, 3, 6, 6)
        criterion = nn.MSELoss()
        o2avgpool2d = O2AvgPool2d(kernel_size=2, stride=2)
        output = o2avgpool2d(x)
        y, idx = output
        y.requires_grad_()
        y.retain_grad()
        target = torch.rand(y.shape)
        loss = criterion(y, target)
        loss.backward(create_graph=create_graph)
        dLdy = y.grad

        def _criterion(y):
            return criterion(y, target)

        dL2dy2 = hessian(_criterion, y)
        return x, output, o2avgpool2d, dLdy, dL2dy2

    def test_forward(self):
        x = torch.rand(2, 3, 5)
        torch.manual_seed(0)
        o2avgpool2d = O2AvgPool2d(kernel_size=2, stride=2)
        torch.manual_seed(0)
        avgpool1d = nn.AvgPool2d(kernel_size=2, stride=2)
        o2y, o2idx = o2avgpool2d(x)
        y, idx = avgpool1d(x)
        self.assertTrue(all((o2y == y).flatten()))
        self.assertTrue(all((o2idx == idx).flatten()))

    def test_get_output_input_jacobian(self):
        x, y, o2avgpool2d, dLdy, _ = self._setup()
        avgpool2d = o2avgpool2d.module
        dydx = o2avgpool2d.get_output_input_jacobian(x)
        if dydx.layout == torch.sparse_coo:
            dydx = dydx.to_dense()

        def _o2avgpool1d(x):
            y = avgpool2d(x)
            return y

        def _jac(x):
            return jacobian(_o2avgpool1d, x, create_graph=True)

        dydx_ = _jac(x)

        self.assertEqual(dydx.numel(), dydx_.numel())
        dydx = dydx.reshape(dydx_.shape)
        y_, idx = y
        self.assertTrue(torch.allclose(dydx, dydx_))

    def test_get_output_input_hessian(self):
        x, y, o2avgpool2d, dLdy, _ = self._setup()
        avgpool2d = o2avgpool2d.module

        dy2dx2 = o2avgpool2d.get_output_input_hessian(x)
        if dy2dx2.layout == torch.sparse_coo:
            dy2dx2 = dy2dx2.to_dense()

        def _avgpool2d(x):
            y = avgpool2d(x)
            return y

        def _jac(x):
            return jacobian(_avgpool2d, x, create_graph=True)

        x = x.clone()
        dy2dx2_ = jacobian(_jac, x)
        self.assertEqual(dy2dx2.numel(), dy2dx2_.numel())
        dy2dx2 = dy2dx2.reshape(dy2dx2_.shape)
        self.assertTrue(torch.allclose(dy2dx2, dy2dx2_))


class O2MaxPool1dTest(unittest.TestCase):
    def _setup(self, create_graph=False):
        torch.manual_seed(0)
        x = torch.rand(2, 3, 6)
        criterion = nn.MSELoss()
        o2maxpool1d = O2MaxPool1d(kernel_size=2, stride=2, return_indices=True)
        output = o2maxpool1d(x)
        y, idx = output
        y.requires_grad_()
        y.retain_grad()
        target = torch.rand(y.shape)
        loss = criterion(y, target)
        loss.backward(create_graph=create_graph)
        dLdy = y.grad

        def _criterion(y):
            return criterion(y, target)

        dL2dy2 = hessian(_criterion, y)
        return x, output, o2maxpool1d, dLdy, dL2dy2

    def test_forward(self):
        x = torch.rand(2, 3, 5)
        torch.manual_seed(0)
        o2maxpool1d = O2MaxPool1d(kernel_size=2, stride=2, return_indices=True)
        torch.manual_seed(0)
        maxpool1d = nn.MaxPool1d(kernel_size=2, stride=2, return_indices=True)
        o2y, o2idx = o2maxpool1d(x)
        y, idx = maxpool1d(x)
        self.assertTrue(all((o2y == y).flatten()))
        self.assertTrue(all((o2idx == idx).flatten()))

    def test_get_output_input_jacobian(self):
        x, y, o2maxpool1d, dLdy, _ = self._setup()
        maxpool1d = o2maxpool1d.module
        dydx = o2maxpool1d.get_output_input_jacobian(x)
        if dydx.layout == torch.sparse_coo:
            dydx = dydx.to_dense()

        def _o2pool1d(x):
            y, idx = maxpool1d(x)
            return y

        def _jac(x):
            return jacobian(_o2pool1d, x, create_graph=True)

        dydx_ = _jac(x)

        self.assertEqual(dydx.numel(), dydx_.numel())
        dydx = dydx.reshape(dydx_.shape)
        y_, idx = y
        self.assertTrue(torch.allclose(dydx, dydx_))

    def test_get_output_input_hessian(self):
        x, y, o2maxpool1d, dLdy, _ = self._setup()
        maxpool1d = o2maxpool1d.module

        dy2dx2 = o2maxpool1d.get_output_input_hessian(x)
        if dy2dx2.layout == torch.sparse_coo:
            dy2dx2 = dy2dx2.to_dense()

        def _maxpool1d(x):
            y, idx = maxpool1d(x)
            return y

        def _jac(x):
            return jacobian(_maxpool1d, x, create_graph=True)

        x = x.clone()
        dy2dx2_ = jacobian(_jac, x)

        self.assertEqual(dy2dx2.numel(), dy2dx2_.numel())
        dy2dx2 = dy2dx2.reshape(dy2dx2_.shape)
        self.assertTrue(torch.allclose(dy2dx2, dy2dx2_))


class O2MaxPool2dTest(unittest.TestCase):
    def _setup(self, create_graph=False):
        torch.manual_seed(0)
        x = torch.rand(1, 3, 6, 6)
        criterion = nn.MSELoss()

        o2maxpool2d = O2MaxPool2d(kernel_size=2, stride=2, return_indices=True)
        output = o2maxpool2d(x)
        y, idx = output
        y.requires_grad_()
        y.retain_grad()
        target = torch.rand(y.shape)
        loss = criterion(y, target)
        loss.backward(create_graph=create_graph)

        dLdy = y.grad

        def _criterion(y):
            return criterion(y, target)

        dL2dy2 = hessian(_criterion, y)

        return x, output, o2maxpool2d, dLdy, dL2dy2

    def test_forward(self):
        x = torch.rand(2, 3, 5)
        torch.manual_seed(0)
        o2maxpool2d = O2MaxPool2d(kernel_size=2, stride=2, return_indices=True)
        torch.manual_seed(0)
        maxpool1d = nn.MaxPool2d(kernel_size=2, stride=2, return_indices=True)

        o2y, o2idx = o2maxpool2d(x)
        y, idx = maxpool1d(x)
        self.assertTrue(all((o2y == y).flatten()))
        self.assertTrue(all((o2idx == idx).flatten()))

    def test_get_output_input_jacobian(self):
        x, y, o2maxpool2d, dLdy, _ = self._setup()
        maxpool2d = o2maxpool2d.module

        dydx = o2maxpool2d.get_output_input_jacobian(x)
        if dydx.layout == torch.sparse_coo:
            dydx = dydx.to_dense()

        def _o2pool2d(x):
            y, idx = maxpool2d(x)
            return y

        def _jac(x):
            return jacobian(_o2pool2d, x, create_graph=True)

        dydx_ = _jac(x)

        self.assertEqual(dydx.numel(), dydx_.numel())
        dydx = dydx.reshape(dydx_.shape)
        y_, idx = y
        self.assertTrue(torch.allclose(dydx, dydx_))

    def test_get_output_input_hessian(self):
        x, y, o2maxpool2d, dLdy, _ = self._setup()
        maxpool2d = o2maxpool2d.module

        dy2dx2 = o2maxpool2d.get_output_input_hessian(x)
        if dy2dx2.layout == torch.sparse_coo:
            dy2dx2 = dy2dx2.to_dense()

        def _maxpool2d(x):
            y, idx = maxpool2d(x)
            return y

        def _jac(x):
            return jacobian(_maxpool2d, x, create_graph=True)

        x = x.clone()
        dy2dx2_ = jacobian(_jac, x)

        self.assertEqual(dy2dx2.numel(), dy2dx2_.numel())
        dy2dx2 = dy2dx2.reshape(dy2dx2_.shape)
        self.assertTrue(torch.allclose(dy2dx2, dy2dx2_))


if __name__ == "__main__":
    unittest.main()
