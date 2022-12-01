import unittest
from collections import OrderedDict
import torch
import torch.nn as nn
from torch.autograd.functional import jacobian, hessian

from o2grad.linalg import matmul_mixed
from o2grad.recursive import (
    add_callbacks,
    add_default_callbacks,
    set_nesting_level,
    set_next_layer,
    get_o2modules,
)
from o2grad.backprop.o2backprop import backprop_step
from o2grad.modules.o2loss import O2Loss
from o2grad.modules.o2layer.o2layer import O2Layer
from o2grad.modules.o2parametric.o2linear import O2Linear
from o2grad.modules.o2layer.activations import O2ReLU
from o2grad.modules.o2layer.o2reshape import O2Reshape
from o2grad.modules.o2container.o2container import O2Container
from o2grad.modules.o2container.o2residual import O2Residual, Residual


class Square(nn.Module):
    def forward(self, x):
        return x**2


class O2Square(O2Layer):
    def __init__(self):
        module = Square()
        super().__init__(module)

    def forward(self, x):
        self.input = x.clone().detach()
        return self.module(x)

    def get_output_input_jacobian(self, x: torch.Tensor) -> torch.Tensor:
        return 2 * torch.diag(x.reshape(-1))

    def get_output_input_hessian(self, x: torch.Tensor) -> torch.Tensor:
        n = x.numel()
        idx = torch.tensor([(i, i, i) for i in range(n)], dtype=torch.int64).T
        val = torch.empty(n)
        val[:] = 2
        return (
            torch.sparse_coo_tensor(idx, val, (n, n, n)).to_dense().reshape((n, n * n))
        )


class Triple(nn.Module):
    def forward(self, x):
        return 3 * x


class O2Triple(O2Layer):
    def __init__(self):
        module = Triple()
        super().__init__(module)

    def forward(self, x):
        self.input = x.clone().detach()
        return self.module(x)

    def get_output_input_jacobian(self, x: torch.Tensor) -> torch.Tensor:
        n = x.numel()
        return 3 * torch.eye(n).to_sparse()

    def get_output_input_hessian(self, x: torch.Tensor) -> torch.Tensor:
        n = x.numel()
        return torch.zeros(n, n * n).to_sparse()


class ResidualTest(unittest.TestCase):
    def _setup_modules(self):
        modules = OrderedDict(
            [
                ("fc1", nn.Linear(20, 30)),
                ("relu", nn.ReLU()),
                ("fc2", nn.Linear(30, 20)),
            ]
        )
        return modules

    def test_init(self):
        modules = self._setup_modules()
        Residual(modules)
        Residual(*modules.values())

        def _try_empty():
            Residual()

        self.assertRaises(ValueError, _try_empty)

    def test_forward(self):
        torch.manual_seed(0)
        modules = self._setup_modules()
        res = nn.Sequential(modules)
        torch.manual_seed(0)
        modules = self._setup_modules()
        o2res = Residual(modules)
        x = torch.rand(2, 20)
        y = x + res(x)
        yo2 = o2res(x)
        self.assertTrue(torch.all(y == yo2).item())


class O2ResidualTest(unittest.TestCase):
    def _setup_modules(self):
        modules = OrderedDict(
            [("fc1", O2Linear(20, 30)), ("relu", O2ReLU()), ("fc2", O2Linear(30, 20))]
        )
        return modules

    def test_init(self):
        modules = self._setup_modules()
        O2Residual(modules)
        O2Residual(*modules.values())

        def _try_empty():
            O2Residual()

        self.assertRaises(ValueError, _try_empty)

    def test_forward(self):
        torch.manual_seed(0)
        modules_ = self._setup_modules()
        res = nn.Sequential(modules_)
        torch.manual_seed(0)
        modules = self._setup_modules()
        o2res = O2Residual(modules)
        x = torch.rand(2, 20)
        y = x + res(x)
        yo2 = o2res(x)
        self.assertTrue(torch.all(y == yo2).item())

    def test_set_chain_output_input_jacobian(self):
        o2res = O2Residual(O2Square(), O2Residual(O2Square(), O2Square()), O2Square())
        # Check that chaining is set for all submodules
        o2res.set_chain_output_input_jacobian(True)
        self.assertTrue(o2res.chain_dydx)
        self.assertTrue(o2res.module.sequence[0].chain_dydx)
        self.assertTrue(o2res.module.sequence[1].chain_dydx)
        self.assertTrue(o2res.module.sequence[2].chain_dydx)
        self.assertTrue(o2res.module.sequence[1].module.sequence[0].chain_dydx)
        self.assertTrue(o2res.module.sequence[1].module.sequence[1].chain_dydx)
        # Last submodules should be set as chaining ends
        self.assertFalse(o2res.chain_end_dydx)
        self.assertFalse(o2res.module.sequence[0].chain_end_dydx)
        self.assertFalse(o2res.module.sequence[1].chain_end_dydx)
        self.assertTrue(o2res.module.sequence[2].chain_end_dydx)
        self.assertFalse(o2res.module.sequence[1].module.sequence[0].chain_end_dydx)
        self.assertTrue(o2res.module.sequence[1].module.sequence[1].chain_end_dydx)
        # Check that chaining is unset for all submodules
        o2res.set_chain_output_input_jacobian(False)
        self.assertFalse(o2res.chain_dydx)
        self.assertFalse(o2res.module.sequence[0].chain_dydx)
        self.assertFalse(o2res.module.sequence[1].chain_dydx)
        self.assertFalse(o2res.module.sequence[2].chain_dydx)
        self.assertFalse(o2res.module.sequence[1].module.sequence[0].chain_dydx)
        self.assertFalse(o2res.module.sequence[1].module.sequence[1].chain_dydx)
        # Chaining ends should also be unset
        self.assertFalse(o2res.chain_end_dydx)
        self.assertFalse(o2res.module.sequence[0].chain_end_dydx)
        self.assertFalse(o2res.module.sequence[1].chain_end_dydx)
        self.assertFalse(o2res.module.sequence[2].chain_end_dydx)
        self.assertFalse(o2res.module.sequence[1].module.sequence[0].chain_end_dydx)
        self.assertFalse(o2res.module.sequence[1].module.sequence[1].chain_end_dydx)
        # Check that chaining is only set for submodules of targeted layer
        o2res.module.sequence[1].set_chain_output_input_jacobian(True)
        self.assertFalse(o2res.chain_dydx)
        self.assertFalse(o2res.module.sequence[0].chain_dydx)
        self.assertTrue(o2res.module.sequence[1].chain_dydx)
        self.assertFalse(o2res.module.sequence[2].chain_dydx)
        self.assertTrue(o2res.module.sequence[1].module.sequence[0].chain_dydx)
        self.assertTrue(o2res.module.sequence[1].module.sequence[1].chain_dydx)
        # Check that chaining end is only set for last submodule of targeted layer
        self.assertFalse(o2res.chain_end_dydx)
        self.assertFalse(o2res.module.sequence[0].chain_end_dydx)
        self.assertFalse(o2res.module.sequence[1].chain_end_dydx)
        self.assertFalse(o2res.module.sequence[2].chain_end_dydx)
        self.assertFalse(o2res.module.sequence[1].module.sequence[0].chain_end_dydx)
        self.assertTrue(o2res.module.sequence[1].module.sequence[1].chain_end_dydx)

    def test_set_chain_end_output_input_jacobian(self):
        o2res = O2Residual(O2Square(), O2Residual(O2Square(), O2Square()))
        # Check that chain end is only set for this end module
        o2res.set_chain_end_output_input_jacobian(True)
        self.assertTrue(o2res.chain_end_dydx)
        self.assertTrue(o2res.module.sequence.chain_end_dydx)
        self.assertFalse(o2res.module.sequence[0].chain_end_dydx)
        self.assertTrue(o2res.module.sequence[1].chain_end_dydx)
        self.assertFalse(o2res.module.sequence[1].module.sequence[0].chain_end_dydx)
        self.assertTrue(o2res.module.sequence[1].module.sequence[1].chain_end_dydx)
        # Check that chain end is unset for this end module
        o2res.set_chain_end_output_input_jacobian(False)
        self.assertFalse(o2res.chain_end_dydx)
        self.assertFalse(o2res.module.sequence[0].chain_end_dydx)
        self.assertFalse(o2res.module.sequence[1].chain_end_dydx)
        self.assertFalse(o2res.module.sequence[1].module.sequence[0].chain_end_dydx)
        self.assertFalse(o2res.module.sequence[1].module.sequence[1].chain_end_dydx)

    def test_get_chain_output_input_jacobian_cached(self):
        o2res = O2Residual(
            O2Square(),
            O2Residual(O2Square(), O2Residual(O2Square(), O2Square()), O2Square()),
            O2Square(),
        )
        o2criterion = O2Loss(nn.MSELoss())
        set_next_layer(o2res, o2criterion)
        set_nesting_level(o2res)
        o2res.set_chain_output_input_jacobian(True)
        o2res.set_chain_end_output_input_jacobian(True)

        def _capture_backprops(layer, grad_input, grad_output):
            x = layer.input
            if layer.chain_dydx:
                dydx = layer.get_output_input_jacobian(x)
                if layer.chain_end_dydx:
                    layer.dyydx = dydx
                else:
                    dzdy = layer.next_layer.get_chained_output_input_jacobian_cached()
                    layer.dyydx = matmul_mixed(dzdy, dydx)
            if isinstance(layer, O2Layer):
                layer._callbacks.on_complete()

        for m in get_o2modules(o2res).values():
            m.register_full_backward_hook(_capture_backprops)
        add_callbacks(
            o2res,
            dict(
                on_children_complete=lambda layer: _capture_backprops(layer, None, None)
            ),
        )
        add_default_callbacks(o2res)

        x = torch.rand(2, 5)
        x.requires_grad_()
        t = torch.rand(2, 5)
        y = o2res(x)
        loss = o2criterion(y, t)
        loss.backward(create_graph=True)
        o2res.zero_grad()
        dydx = o2res.get_chained_output_input_jacobian_cached()
        dydx_ = jacobian(o2res, x)
        if not dydx.layout == torch.strided:
            dydx = dydx.to_dense()
        if not dydx_.layout == torch.strided:
            dydx_ = dydx_.to_dense()
        self.assertTrue(dydx.numel() == dydx_.numel())
        dydx_ = dydx_.reshape(dydx.shape)
        self.assertTrue(torch.allclose(dydx, dydx_))

        # # Remove chaining and apply only to inner sequential layer
        o2res.set_chain_output_input_jacobian(False)
        o2res.set_chain_end_output_input_jacobian(False)
        o2res.module.sequence[1].set_chain_output_input_jacobian(True)
        o2res.module.sequence[1].set_chain_end_output_input_jacobian(True)
        o2res.module.sequence[1].settings.save_intermediate = ["dyydx"]
        x = torch.rand(2, 5)
        x.requires_grad_()
        t = torch.rand(2, 5)
        y = o2res(x)
        loss = o2criterion(y, t)
        loss.backward(create_graph=True)
        o2res.zero_grad()
        dyydx = o2res.module.sequence[1].get_chained_output_input_jacobian_cached()
        xx = o2res.module.sequence[1].input
        dyydx_ = jacobian(o2res.module.sequence[1], xx)
        if not dyydx.layout == torch.strided:
            dyydx = dyydx.to_dense()
        if not dyydx_.layout == torch.strided:
            dyydx_ = dyydx_.to_dense()
        self.assertTrue(dyydx.numel() == dyydx_.numel())
        print(dyydx)
        print(dyydx_)
        dyydx_ = dyydx_.reshape(dyydx.shape)
        self.assertTrue(torch.allclose(dyydx, dyydx_))

    def test_backward_identity(self):
        torch.manual_seed(0)
        x = torch.rand(2, 6)
        x.requires_grad_()
        t = torch.rand(2, 6)
        o2seq = O2Residual(O2Reshape(2, 3), O2Reshape(6))

        o2criterion = O2Loss(nn.MSELoss())
        set_next_layer(o2seq, o2criterion)
        set_nesting_level(o2seq)
        o2seq.settings.save_intermediate = ["dL2dx2"]
        o2seq.set_chain_end_output_input_jacobian(True)

        def _capture_backprops(layer, grad_input, grad_output):
            x = layer.input
            dL2dy2 = layer.next_layer.get_loss_input_hessian_cached()
            if isinstance(layer, O2Layer):
                dydx = layer.get_output_input_jacobian(x)
                if layer.chain_dydx:
                    if layer.chain_end_dydx:
                        layer.dyydx = dydx
                    else:
                        dzdy = (
                            layer.next_layer.get_chained_output_input_jacobian_cached()
                        )
                        layer.dyydx = matmul_mixed(dzdy, dydx)
                layer.dL2dx2 = layer.get_loss_input_hessian(
                    None, dL2dy2, None, None
                ).clone()
            if isinstance(layer.next_layer, O2Layer) or isinstance(
                layer.next_layer, O2Container
            ):
                if layer.nesting_level == layer.next_layer.nesting_level:
                    layer.next_layer.try_clear_cache("dL2dx2")
            layer._callbacks.on_complete()

        for m in get_o2modules(o2seq).values():
            m.register_full_backward_hook(_capture_backprops)
        # add_default_callbacks(o2seq)
        add_callbacks(
            o2seq, dict(on_children_complete=lambda layer: backprop_step(layer))
        )
        add_callbacks(o2seq, dict(on_capture=lambda layer: backprop_step(layer)))
        add_default_callbacks(o2seq)

        def _loss(x):
            y = o2seq(x)
            return o2criterion(y, t)

        loss = _loss(x)
        loss.backward()
        dL2dx2_ = o2seq.get_loss_input_hessian_cached().to_dense()
        o2seq.zero_grad()
        dL2dx2 = hessian(_loss, x).reshape(x.numel(), x.numel())
        self.assertTrue(torch.allclose(dL2dx2, dL2dx2_))

    def test_backward(self):
        # Test residual layer with one single x^2
        torch.manual_seed(0)
        x = torch.rand(2, 6)
        x.requires_grad_()
        t = torch.rand(2, 6)
        o2seq = O2Residual(
            O2Residual(O2Square(), O2Square()), O2Residual(O2Residual(O2Square()))
        )

        o2criterion = O2Loss(nn.MSELoss())
        set_next_layer(o2seq, o2criterion)
        set_nesting_level(o2seq)
        o2seq.settings.save_intermediate = ["dL2dx2"]
        o2seq.set_chain_end_output_input_jacobian(True)

        def _capture_backprops(layer, grad_input, grad_output):
            x = layer.input
            dL2dy2 = layer.next_layer.get_loss_input_hessian_cached()
            if isinstance(layer, O2Layer):
                dydx = layer.get_output_input_jacobian(x)
                if layer.chain_dydx:
                    if layer.chain_end_dydx:
                        layer.dyydx = dydx
                    else:
                        dzdy = (
                            layer.next_layer.get_chained_output_input_jacobian_cached()
                        )
                        layer.dyydx = matmul_mixed(dzdy, dydx)
                dLdy = grad_output[0].clone().detach().reshape(-1).to_sparse()
                dydx = layer.get_output_input_jacobian(x)
                dy2dx2 = layer.get_output_input_hessian(x)
                layer.dL2dx2 = layer.get_loss_input_hessian(
                    dLdy, dL2dy2, dydx, dy2dx2
                ).clone()
            if isinstance(layer.next_layer, O2Layer) or isinstance(
                layer.next_layer, O2Container
            ):
                if layer.nesting_level == layer.next_layer.nesting_level:
                    layer.next_layer.try_clear_cache("dL2dx2")
            layer._callbacks.on_complete()

        for m in get_o2modules(o2seq).values():
            m.register_full_backward_hook(_capture_backprops)
        # add_o2container_default_callbacks(o2seq)
        add_callbacks(
            o2seq, dict(on_children_complete=lambda layer: backprop_step(layer))
        )
        add_callbacks(o2seq, dict(on_capture=lambda layer: backprop_step(layer)))
        add_default_callbacks(o2seq)

        def _loss(x):
            y = o2seq(x)
            return o2criterion(y, t)

        loss = _loss(x)
        loss.backward()
        dL2dx2_ = o2seq.get_loss_input_hessian_cached()
        if dL2dx2_.layout == torch.sparse_coo:
            dL2dx2_ = dL2dx2_.to_dense()
        o2seq.zero_grad()
        dL2dx2 = hessian(_loss, x)
        dL2dx2 = dL2dx2.reshape(x.numel(), x.numel())
        self.assertTrue(torch.allclose(dL2dx2, dL2dx2_))

    def test_backward_2layer(self):
        # Test residual layer with one single x^2
        torch.manual_seed(0)
        x = torch.rand(2, 6)
        x.requires_grad_()
        t = torch.rand(2, 6)
        o2seq = O2Residual(O2ReLU(), O2ReLU())

        o2criterion = O2Loss(nn.MSELoss())
        set_next_layer(o2seq, o2criterion)
        set_nesting_level(o2seq)
        o2seq.settings.save_intermediate = ["dL2dx2"]
        o2seq.set_chain_end_output_input_jacobian(True)

        def _capture_backprops(layer: nn.Module, grad_input, grad_output):
            if isinstance(grad_output, tuple) and len(grad_output) == 1:
                grad_output = grad_output[0].clone()
            layer.dLdy = grad_output
            layer._callbacks.on_capture()

        for m in get_o2modules(o2seq).values():
            m.register_full_backward_hook(_capture_backprops)
        add_callbacks(
            o2seq, dict(on_children_complete=lambda layer: backprop_step(layer))
        )
        add_callbacks(o2seq, dict(on_capture=lambda layer: backprop_step(layer)))
        add_default_callbacks(o2seq)

        def _loss(x):
            y = o2seq(x)
            return o2criterion(y, t)

        loss = _loss(x)
        loss.backward()
        dL2dx2_ = o2seq.get_loss_input_hessian_cached()
        if dL2dx2_.layout == torch.sparse_coo:
            dL2dx2_ = dL2dx2_.to_dense()
        o2seq.zero_grad()
        dL2dx2 = hessian(_loss, x)
        dL2dx2 = dL2dx2.reshape(x.numel(), x.numel())
        self.assertTrue(torch.allclose(dL2dx2, dL2dx2_))
