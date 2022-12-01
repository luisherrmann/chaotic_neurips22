import unittest
import os
from os.path import dirname
from collections import OrderedDict
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import grad
from torch.autograd.functional import hessian

from o2grad.linalg import sum_mixed, matmul_mixed
from o2grad.sparse import SparseSymmetricMatrix
from o2grad.modules.o2parametric.o2linear import O2Linear
from o2grad.modules.o2parametric.o2conv1d import O2Conv1d
from o2grad.backprop.o2model import O2Model
from o2grad.backprop.backprop import get_hessian


class O2ModelTest(unittest.TestCase):
    def setUp(cls):
        cls.MODEL_INIT = {
            "default": cls._setup_default,
            "linear": cls._setup_linear,
            "conv1d": cls._setup_conv1d,
            "conv2d": cls._setup_conv2d,
            "2linear": cls._setup_2linear,
            "2conv1d": cls._setup_2conv1d,
            "2conv2d": cls._setup_2conv2d,
        }

    @classmethod
    def _setup_default(cls):
        torch.manual_seed(0)
        x = torch.rand(10, 2)
        target = torch.rand(10, 2)
        model = nn.Sequential(
            OrderedDict(
                [
                    ("fc1", O2Linear(2, 3)),
                    ("relu1", nn.ReLU()),
                    ("fc2", O2Linear(3, 3)),
                    ("relu2", nn.Sigmoid()),
                    ("fc3", O2Linear(3, 2)),
                    ("fc4", O2Linear(2, 2)),
                ]
            )
        )
        wrapped_model = O2Model(model, nn.MSELoss(), save_intermediate="all")
        return x, target, wrapped_model

    @classmethod
    def _setup_linear(cls):
        torch.manual_seed(0)
        x = torch.rand(10, 2)
        model = nn.Sequential(
            OrderedDict(
                [
                    ("layer1", O2Linear(2, 3)),
                ]
            )
        )
        output = model(x)
        target = torch.rand(output.shape)
        wrapped_model = O2Model(model, nn.MSELoss(), save_intermediate="all")
        return x, target, wrapped_model

    @classmethod
    def _setup_conv1d(cls):
        torch.manual_seed(0)
        x = torch.rand(1, 2, 6)
        model = nn.Sequential(
            OrderedDict(
                [
                    (
                        "layer1",
                        nn.Conv1d(
                            in_channels=2,
                            out_channels=3,
                            kernel_size=3,
                            stride=1,
                            padding=1,
                            dilation=2,
                        ),
                    ),
                ]
            )
        )
        output = model(x)
        target = torch.rand(output.shape)
        wrapped_model = O2Model(model, nn.MSELoss(), save_intermediate="all")
        return x, target, wrapped_model

    @classmethod
    def _setup_conv2d(cls):
        torch.manual_seed(0)
        x = torch.rand(1, 2, 4, 4)
        model = nn.Sequential(
            OrderedDict(
                [
                    (
                        "layer1",
                        nn.Conv2d(
                            in_channels=2,
                            out_channels=3,
                            kernel_size=3,
                            stride=1,
                            padding=1,
                        ),
                    ),
                ]
            )
        )
        output = model(x)
        target = torch.rand(output.shape)
        wrapped_model = O2Model(model, nn.MSELoss(), save_intermediate="all")
        return x, target, wrapped_model

    @classmethod
    def _setup_2linear(cls):
        torch.manual_seed(0)
        x = torch.rand(10, 2)
        target = torch.rand(10, 2)
        model = nn.Sequential(
            OrderedDict([("layer1", O2Linear(2, 3)), ("layer2", O2Linear(3, 2))])
        )
        wrapped_model = O2Model(model, nn.MSELoss(), save_intermediate="all")
        return x, target, wrapped_model

    @classmethod
    def _setup_2conv1d(cls):
        torch.manual_seed(0)
        x = torch.rand(1, 1, 6)
        # model = nn.Sequential(OrderedDict([
        #     ('layer1', nn.Conv1d(in_channels=1, out_channels=1, kernel_size=3, stride=1, padding=1)),
        #     ('layer2', nn.Conv1d(in_channels=1, out_channels=1, kernel_size=2, stride=2, padding=2, dilation=2)),
        # ]))

        model = nn.Sequential(
            OrderedDict(
                [
                    (
                        "layer1",
                        O2Conv1d(
                            in_channels=1,
                            out_channels=1,
                            kernel_size=3,
                            stride=1,
                            padding=1,
                        ),
                    ),
                    (
                        "layer2",
                        O2Conv1d(
                            in_channels=1,
                            out_channels=1,
                            kernel_size=3,
                            stride=1,
                            padding=1,
                        ),
                    ),
                ]
            )
        )

        output = model(x)
        target = torch.rand(output.shape)
        # target = torch.rand(1, 4, 3)
        wrapped_model = O2Model(model, nn.MSELoss(), save_intermediate="all")
        print(wrapped_model)
        return x, target, wrapped_model

    @classmethod
    def _setup_2conv2d(cls):
        torch.manual_seed(0)
        x = torch.rand(1, 2, 4, 4)
        model = nn.Sequential(
            OrderedDict(
                [
                    (
                        "layer1",
                        nn.Conv2d(
                            in_channels=2,
                            out_channels=3,
                            kernel_size=3,
                            stride=1,
                            padding=1,
                        ),
                    ),
                    (
                        "layer2",
                        nn.Conv2d(
                            in_channels=3,
                            out_channels=4,
                            kernel_size=2,
                            stride=2,
                            padding=2,
                            dilation=2,
                        ),
                    ),
                ]
            )
        )
        output = model(x)
        target = torch.rand(output.shape)
        wrapped_model = O2Model(model, nn.MSELoss(), save_intermediate="all")
        return x, target, wrapped_model

    def _setup(self, model_name="default", cuda=False):
        x, target, model = self.MODEL_INIT[model_name]()
        if cuda == True:
            x = x.cuda()
            target = target.cuda()
            model = model.cuda()
        return x, target, model

    def test_wrap(self):
        x, y, model = self._setup()
        self.assertTrue(True)

    def test_add_connections(self):
        x, y, model = self._setup()
        next_layers = [layer.next_layer for layer in model.name_to_layer.values()]
        self.assertListEqual(
            next_layers, [*list(model.name_to_layer.values())[1:], model.criterion]
        )

    def test_capture_backprops(self):
        x, y, model = self._setup(model_name="2conv1d")
        x.requires_grad = True
        criterion = model.criterion
        output = model(x)
        loss = criterion(output, y)
        loss.backward()
        print("Name to layer", model.name_to_layer)
        first = model.name_to_layer[".layer1"]
        # Get dL2dx2, dL2dw2 using o2grad
        dLdx2_o2grad = first.dL2dx2
        if isinstance(dLdx2_o2grad, SparseSymmetricMatrix) or (
            dLdx2_o2grad.layout == torch.sparse_coo
        ):
            dLdx2_o2grad = dLdx2_o2grad.to_dense()
        dLdw2_o2grad = first.dL2dw2[".layer1"][".layer1"]
        if isinstance(dLdw2_o2grad, SparseSymmetricMatrix) or (
            dLdw2_o2grad.layout == torch.sparse_coo
        ):
            dLdw2_o2grad = dLdw2_o2grad.to_dense()
        # Get dL2dx2 using autograd
        model.disable_o2backprop()
        model.zero_grad()

        def _loss(x):
            output = model(x)
            loss = criterion(output, y)
            return loss

        fan_in = x.numel()
        dLdx2_autograd = hessian(_loss, x).reshape(fan_in, fan_in)
        # Get dL2dw2 using autograd
        model.zero_grad()
        loss = _loss(x)
        loss.backward(create_graph=True)
        dLdw = torch.cat([p.grad.view(-1) for p in first.module.parameters()])
        p_count = dLdw.numel()
        dLdw2_autograd = torch.ones(p_count, p_count)
        for i, dLdwi in enumerate(dLdw):
            grads2nd = grad(
                dLdwi, first.module.parameters(), create_graph=True, allow_unused=True
            )
            dLdw2_autograd[i, :] = torch.cat([g.view(-1) for g in grads2nd]).reshape(
                1, -1
            )
        # Make sure results match
        self.assertTrue(torch.allclose(dLdx2_o2grad, dLdx2_autograd))
        self.assertTrue(torch.allclose(dLdw2_o2grad, dLdw2_autograd))

    def test_get_hessian_2layer(self):
        x, y, model = self._setup(model_name="2conv1d")
        x.requires_grad = True
        criterion = model.criterion
        model.zero_grad()
        output = model(x)
        loss = criterion(output, y)
        loss.backward()
        # Fetch layers
        layer_dict = {**model.name_to_layer}
        fc1 = layer_dict[".layer1"]
        fc2 = layer_dict[".layer2"]
        # Calculate dL2dw2 from stored intermediate terms
        fan_in = fc2.input.numel()
        dL2dy2 = fc2.next_layer.get_loss_input_hessian_cached()
        if isinstance(dL2dy2, SparseSymmetricMatrix):
            dL2dy2 = dL2dy2.to_dense()
        dL2dydw = matmul_mixed(dL2dy2, fc2.dydw, is_symmetric1=True)
        V_term1 = matmul_mixed(fc2.dydx, dL2dydw, is_transposed1=True)
        layer_t_dLdy = fc2.dLdy
        if isinstance(layer_t_dLdy, SparseSymmetricMatrix) or (
            layer_t_dLdy.layout == torch.sparse_coo
        ):
            layer_t_dLdy = layer_t_dLdy.to_dense()
        V_term2 = matmul_mixed(layer_t_dLdy, fc2.dy2dxdw).reshape(fan_in, -1)
        V = sum_mixed(V_term1, V_term2)
        U = fc1.dydw
        dLdw2_o2grad = matmul_mixed(U, V, is_transposed1=True)
        # Calculate dL2dw2 using autograd
        model.disable_o2backprop()
        model.zero_grad()
        output = model(x)
        loss = criterion(output, y)
        loss.backward(create_graph=True)
        dLdw2_autograd = get_hessian(model)
        np1, np2 = dLdw2_o2grad.shape
        # Asser results match
        fc1_dL2dw2 = fc1.dL2dw2[".layer1"][".layer1"]
        fc2_dL2dw2 = fc2.dL2dw2[".layer2"][".layer2"]
        if (
            isinstance(fc1_dL2dw2, SparseSymmetricMatrix)
            or fc1_dL2dw2.layout == torch.sparse_coo
        ):
            fc1_dL2dw2 = fc1_dL2dw2.to_dense()
        if (
            isinstance(fc2_dL2dw2, SparseSymmetricMatrix)
            or fc2_dL2dw2.layout == torch.sparse_coo
        ):
            fc2_dL2dw2 = fc2_dL2dw2.to_dense()
        self.assertTrue(torch.allclose(fc1_dL2dw2, dLdw2_autograd[:np1, :np1]))
        self.assertTrue(torch.allclose(fc2_dL2dw2, dLdw2_autograd[np1:, np1:]))
        self.assertTrue(torch.allclose(dLdw2_o2grad, dLdw2_autograd[:np1, np1:]))

    def _setup_hessian(self):
        dL2dw2 = {
            "layer1": {
                "layer1": torch.rand(2, 2),
                "layer2": torch.rand(2, 3),
                "layer3": torch.rand(2, 4),
            },
            "layer2": {"layer2": torch.rand(3, 3), "layer3": torch.rand(3, 4)},
            "layer3": {"layer3": torch.rand(4, 4)},
        }
        for s in dL2dw2:
            dL2dw2[s][s] = dL2dw2[s][s] + dL2dw2[s][s].T
        return dL2dw2

    def test__get_hessian_as_dict(self):
        dL2dw2 = self._setup_hessian()
        dL2dw2_ = O2Model._get_hessian_as_dict(dL2dw2)
        for s in dL2dw2_:
            for t in dL2dw2_[s]:
                self.assertTrue(dL2dw2[s][t] is dL2dw2_[s][t])
        dL2dw2_ = O2Model._get_hessian_as_dict(dL2dw2, diagonal_blocks=True)
        for s in dL2dw2_:
            self.assertEqual(len(dL2dw2_[s]), 1)
            self.assertTrue(dL2dw2[s][s] is dL2dw2_[s][s])
        dL2dw2_ = O2Model._get_hessian_as_dict(dL2dw2, as_type="dense")
        for s in dL2dw2_:
            for t in dL2dw2_[s]:
                self.assertTrue(
                    isinstance(dL2dw2_[s][t], torch.Tensor)
                    and dL2dw2_[s][t].layout == torch.strided
                )
                self.assertTrue(torch.all(dL2dw2[s][t] == dL2dw2_[s][t]).item())
        dL2dw2_ = O2Model._get_hessian_as_dict(dL2dw2, as_type="sparse")
        for s in dL2dw2_:
            for t in dL2dw2_[s]:
                self.assertTrue(
                    isinstance(dL2dw2_[s][t], torch.Tensor)
                    and dL2dw2_[s][t].layout == torch.sparse_coo
                )
                self.assertTrue(
                    torch.all(dL2dw2[s][t] == dL2dw2_[s][t].to_dense()).item()
                )

    def test__get_hessian(self):
        dL2dw2 = self._setup_hessian()
        for s in dL2dw2:
            dL2dw2[s][s] = dL2dw2[s][s] + dL2dw2[s][s].T
        layers = ["layer1", "layer2", "layer3"]
        pcounts = [dL2dw2[s][s].shape[0] for s in layers]
        total_param_cnt = sum(pcounts)
        offsets = [0] + np.cumsum(pcounts).tolist()
        hessian = torch.zeros([total_param_cnt, total_param_cnt])
        hessian_diagonal = torch.zeros([total_param_cnt, total_param_cnt])
        for s in range(len(layers)):
            for t in range(s, len(layers)):
                s_low, s_high = offsets[s], offsets[s + 1]
                t_low, t_high = offsets[t], offsets[t + 1]
                s_name, t_name = layers[s], layers[t]
                hessian[s_low:s_high, t_low:t_high] = dL2dw2[s_name][t_name]
                if s != t:
                    hessian[t_low:t_high, s_low:s_high] = dL2dw2[s_name][t_name].T
                if s == t:
                    hessian_diagonal[s_low:s_high, t_low:t_high] = dL2dw2[s_name][
                        t_name
                    ]
        basepath = dirname(__file__)
        tmp_path = os.path.join(basepath, "tmp")
        if not os.path.exists(tmp_path):
            os.mkdir(tmp_path)
        # torch.Tensor - dense
        hessian_ = O2Model._get_hessian(
            layers,
            dL2dw2,
            as_tensor=True,
            as_type="dense",
            as_file=False,
            diagonal_blocks=False,
        )
        self.assertTrue(
            isinstance(hessian_, torch.Tensor) and hessian_.layout == torch.strided
        )
        self.assertTrue(torch.all(hessian == hessian_).item())
        # torch.Tensor - dense - blocks
        hessian_diagonal_ = O2Model._get_hessian(
            layers,
            dL2dw2,
            as_tensor=True,
            as_type="dense",
            as_file=False,
            diagonal_blocks=True,
        )
        self.assertTrue(
            isinstance(hessian_diagonal_, torch.Tensor)
            and hessian_diagonal_.layout == torch.strided
        )
        self.assertTrue(torch.all(hessian_diagonal == hessian_diagonal_).item())
        # torch.Tensor - dense - file
        filepath = os.path.join(tmp_path, "dense")
        hessian_ = O2Model._get_hessian(
            layers,
            dL2dw2,
            as_tensor=True,
            as_type="dense",
            as_file=filepath,
            diagonal_blocks=False,
        )
        self.assertTrue(
            isinstance(hessian_, torch.Tensor) and hessian_.layout == torch.strided
        )
        self.assertTrue(torch.all(hessian == hessian_).item())
        del hessian_
        hstorage = torch.DoubleStorage.from_file(
            filepath, shared=True, size=total_param_cnt**2
        )
        hessian_ = torch.DoubleTensor(hstorage)
        hessian_ = hessian_.reshape(total_param_cnt, total_param_cnt)
        os.remove(filepath)
        # torch.Tensor - sparse
        hessian_ = O2Model._get_hessian(
            layers,
            dL2dw2,
            as_tensor=True,
            as_type="sparse",
            as_file=False,
            diagonal_blocks=False,
        )
        self.assertTrue(
            isinstance(hessian_, torch.Tensor) and hessian_.layout == torch.sparse_coo
        )
        self.assertTrue(torch.all(hessian == hessian_.to_dense()).item())
        # torch.Tensor - sparse - blocks
        hessian_diagonal_ = O2Model._get_hessian(
            layers,
            dL2dw2,
            as_tensor=True,
            as_type="sparse",
            as_file=False,
            diagonal_blocks=True,
        )
        self.assertTrue(
            isinstance(hessian_diagonal_, torch.Tensor)
            and hessian_diagonal_.layout == torch.sparse_coo
        )
        self.assertTrue(
            torch.all(hessian_diagonal == hessian_diagonal_.to_dense()).item()
        )
        # SparseSymmeticMatrix
        hessian_ = O2Model._get_hessian(
            layers,
            dL2dw2,
            as_tensor=True,
            as_type="symmetric",
            as_file=False,
            diagonal_blocks=False,
        )
        self.assertTrue(isinstance(hessian_, SparseSymmetricMatrix))
        self.assertTrue(torch.all(hessian == hessian_.to_dense()).item())
        # SparseSymmetricMatrix - blocks
        hessian_diagonal_ = O2Model._get_hessian(
            layers,
            dL2dw2,
            as_tensor=True,
            as_type="symmetric",
            as_file=False,
            diagonal_blocks=True,
        )
        self.assertTrue(isinstance(hessian_, SparseSymmetricMatrix))
        self.assertTrue(
            torch.all(hessian_diagonal == hessian_diagonal_.to_dense()).item()
        )
        # TODO: Finish SparseStorageTensor implementation
        # torch.Tensor - sparse - file
        # filepath = os.path.join(tmp_path, 'sparse')
        # hessian_ = O2Model._get_hessian(layers, dL2dw2, as_tensor=True, as_file=filepath, diagonal_blocks=False, sparse=True)
        # hessian_.persist()
        # self.assertTrue(isinstance(hessian_, SparseStorageTensor) and hessian_.layout == torch.sparse_coo)
        # self.assertTrue(torch.all(hessian == hessian_.to_dense()).item())
        # del hessian_
        # hessian_ = SparseStorageTensor.from_file(filepath)
        # self.assertTrue(isinstance(hessian_, SparseStorageTensor) and hessian_.layout == torch.sparse_coo)
        # self.assertTrue(torch.all(hessian == hessian_.to_dense()).item())
        # del hessian_
        # shutil.rmtree(tmp_path)

    def test_get_hessian(self):
        x, y, model = self._setup("2conv1d", cuda=True)
        x.requires_grad = True
        criterion = model.criterion
        # Calculate o2grad hessian
        model.zero_grad()
        output = model(x)
        loss = criterion(output, y)
        loss.backward()
        dL2dw2_o2grad = model.get_hessian()
        # Calculate autograd hessian
        model.disable_o2backprop()
        model.zero_grad()
        output = model(x)
        loss = criterion(output, y)
        loss.backward(create_graph=True)
        dL2dw2_autograd = get_hessian(model)
        # Compare results
        self.assertTrue(torch.allclose(dL2dw2_o2grad, dL2dw2_autograd))
        layer_dict = {**model.name_to_layer}
        fc1 = layer_dict[".layer1"]
        fc2 = layer_dict[".layer2"]
        np1 = sum([p.numel() for p in fc1.module.parameters()])
        sum([p.numel() for p in fc2.module.parameters()])
        #
        # Test that Hessians match when they are returned as file storage on disk
        #
        x, y, model = self._setup("2conv1d", cuda=True)
        x.requires_grad = True
        criterion = model.criterion
        # Calculate o2grad hessian
        model.zero_grad()
        output = model(x)
        loss = criterion(output, y)
        loss.backward()
        dL2dw2_o2grad_file = model.get_hessian(as_file="tmp/o2grad")
        # Calculate autograd hessian
        model.disable_o2backprop()
        model.zero_grad()
        output = model(x)
        loss = criterion(output, y)
        loss.backward(create_graph=True)
        dL2dw2_autograd_file = get_hessian(model, as_file="tmp/autograd")
        # Compare results
        self.assertTrue(all((dL2dw2_o2grad == dL2dw2_o2grad_file).flatten()))
        self.assertTrue(all((dL2dw2_autograd == dL2dw2_autograd_file).flatten()))
        self.assertTrue(torch.allclose(dL2dw2_o2grad_file, dL2dw2_autograd_file))
        os.remove("tmp/o2grad")
        os.remove("tmp/autograd")
        #
        # Test that Hessians match if they are returned as sparse matrices:
        #
        x, y, model = self._setup("2conv1d", cuda=True)
        x.requires_grad = True
        criterion = model.criterion
        # Calculate o2grad hessian
        model.zero_grad()
        output = model(x)
        loss = criterion(output, y)
        loss.backward()
        dL2dw2_o2grad_sparse = model.get_hessian(as_type="sparse")
        # Calculate autograd hessian
        model.disable_o2backprop()
        model.zero_grad()
        output = model(x)
        loss = criterion(output, y)
        loss.backward(create_graph=True)
        dL2dw2_autograd_sparse = get_hessian(model, sparse=True)
        # Compare results
        self.assertTrue(dL2dw2_o2grad_sparse.layout == torch.sparse_coo)
        self.assertTrue(dL2dw2_autograd_sparse.layout == torch.sparse_coo)
        tensors_match = torch.allclose(dL2dw2_o2grad, dL2dw2_o2grad_sparse.to_dense())
        self.assertTrue(tensors_match)
        tensors_match = torch.allclose(
            dL2dw2_autograd, dL2dw2_autograd_sparse.to_dense()
        )
        self.assertTrue(tensors_match)
        tensors_match = torch.allclose(
            dL2dw2_o2grad_sparse.to_dense(), dL2dw2_autograd_sparse.to_dense()
        )
        self.assertTrue(tensors_match)
        #
        # Test that diagonal blocks match if only these are computed.
        #
        x, y, model = self._setup("2conv1d", cuda=True)
        x.requires_grad = True
        criterion = model.criterion
        # Calculate o2grad hessian
        model.zero_grad()
        output = model(x)
        loss = criterion(output, y)
        loss.backward()
        dL2dw2_o2grad = model.get_hessian(diagonal_blocks=True)
        # Calculate autograd hessian
        model.disable_o2backprop()
        model.zero_grad()
        output = model(x)
        loss = criterion(output, y)
        loss.backward(create_graph=True)
        dL2dw2_autograd = get_hessian(model.module.module, diagonal_blocks=True)
        # Compare results
        is_offdiagonal_zero = (dL2dw2_o2grad[:np1, np1:] == 0).flatten()
        self.assertTrue(all(is_offdiagonal_zero))
        is_offdiagonal_zero = (dL2dw2_autograd[:np1, np1:] == 0).flatten()
        self.assertTrue(all(is_offdiagonal_zero))
        tensors_match = torch.allclose(dL2dw2_o2grad, dL2dw2_autograd)
        self.assertTrue(tensors_match)
        #
        # Test that diagonal blocks match if only these are computed as sparse tensors.
        #
        x, y, model = self._setup("2conv1d", cuda=True)
        x.requires_grad = True
        criterion = model.criterion
        # Calculate o2grad hessian
        model.zero_grad()
        output = model(x)
        loss = criterion(output, y)
        loss.backward()
        dL2dw2_o2grad_sparse = model.get_hessian(diagonal_blocks=True, as_type="sparse")
        # Calculate autograd hessian
        model.disable_o2backprop()
        model.zero_grad()
        output = model(x)
        loss = criterion(output, y)
        loss.backward(create_graph=True)
        dL2dw2_autograd_sparse = get_hessian(
            model.module.module, diagonal_blocks=True, sparse=True
        )
        # Compare results
        self.assertTrue(dL2dw2_o2grad_sparse.layout == torch.sparse_coo)
        self.assertTrue(dL2dw2_autograd_sparse.layout == torch.sparse_coo)
        tensors_match = torch.allclose(dL2dw2_o2grad, dL2dw2_o2grad_sparse.to_dense())
        self.assertTrue(tensors_match)
        tensors_match = torch.allclose(
            dL2dw2_autograd, dL2dw2_autograd_sparse.to_dense()
        )
        self.assertTrue(tensors_match)
        tensors_match = torch.allclose(
            dL2dw2_o2grad_sparse.to_dense(), dL2dw2_autograd_sparse.to_dense()
        )
        self.assertTrue(tensors_match)
        # Compare results
        is_offdiagonal_zero = (dL2dw2_o2grad[:np1, np1:] == 0).flatten()
        self.assertTrue(all(is_offdiagonal_zero))
        is_offdiagonal_zero = (dL2dw2_autograd[:np1, np1:] == 0).flatten()
        self.assertTrue(all(is_offdiagonal_zero))
        tensors_match = torch.allclose(dL2dw2_o2grad, dL2dw2_autograd)
        self.assertTrue(tensors_match)
        #
        # Test that diagonal blocks match when sparse and returned as file storage on disk
        #
        x, y, model = self._setup("2conv1d", cuda=True)
        x.requires_grad = True
        criterion = model.criterion
        # Calculate o2grad hessian
        model.zero_grad()
        output = model(x)
        loss = criterion(output, y)
        loss.backward()
        dL2dw2_o2grad_sparse = model.get_hessian(
            as_file="tmp/o2grad", diagonal_blocks=True, as_type="sparse"
        )
        # Calculate autograd hessian
        model.disable_o2backprop()
        model.zero_grad()
        output = model(x)
        loss = criterion(output, y)
        loss.backward(create_graph=True)
        dL2dw2_autograd_sparse = get_hessian(
            model.module.module,
            as_file="tmp/autograd",
            diagonal_blocks=True,
            sparse=True,
        )
        # Compare results
        tensors_match = torch.allclose(dL2dw2_o2grad, dL2dw2_o2grad_sparse.to_dense())
        self.assertTrue(tensors_match)
        tensors_match = torch.allclose(
            dL2dw2_autograd, dL2dw2_autograd_sparse.to_dense()
        )
        self.assertTrue(tensors_match)
        tensors_match = torch.allclose(
            dL2dw2_o2grad_sparse.to_dense(), dL2dw2_autograd_sparse.to_dense()
        )
        self.assertTrue(tensors_match)
        del dL2dw2_autograd, dL2dw2_o2grad_sparse

    def test_get_hessian_eigs_from_input(self):
        x, y, model = self._setup("2conv1d", cuda=False)
        x.requires_grad = True
        eigs, vals = model.get_hessian_eigs_from_input(x, y, diagonal_blocks=True)


if __name__ == "__main__":
    unittest.main()
