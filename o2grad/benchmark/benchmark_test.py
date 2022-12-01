import itertools
import unittest
import os
from os.path import dirname, abspath
import gc
import time
import tqdm
import pandas as pd
import timeit
from collections import OrderedDict
import math
import numpy as np
import torch
import torch.nn as nn

# Profiling
from memory_profiler import memory_usage

# O2Grad
from o2grad.backprop.backprop import get_hessian
from o2grad.backprop.o2model import O2Model
from o2grad.modules.o2layer.o2reshape import O2Reshape
from o2grad.modules.o2layer.o2pooling import O2MaxPool1d, O2MaxPool2d
from o2grad.modules.o2container.o2residual import Residual


class BenchmarkTest(unittest.TestCase):
    @classmethod
    def setUp(cls):
        cls.MODEL_INIT = {
            "big_mlp": cls._setup_big_mlp,
            "conv1d": cls._setup_conv1d,
            "conv2d": cls._setup_conv2d,
            "residual": cls._setup_residual,
        }

    @classmethod
    def _setup_big_mlp(cls):
        torch.manual_seed(0)
        model = nn.Sequential(
            OrderedDict(
                [
                    ("fc1", nn.Linear(100, 100)),
                    ("sigmoid1", nn.Sigmoid()),
                    ("fc2", nn.Linear(100, 20)),
                    ("sigmoid2", nn.Sigmoid()),
                    ("fc3", nn.Linear(20, 2)),
                ]
            )
        )

        x = torch.rand(16, 100)
        x.requires_grad = True
        target = torch.rand(16, 2)
        return model, x, target

    @classmethod
    def _setup_conv1d(cls):
        torch.manual_seed(0)
        model = nn.Sequential(
            OrderedDict(
                [
                    (
                        "conv1",
                        nn.Conv1d(
                            in_channels=3,
                            out_channels=10,
                            kernel_size=3,
                            stride=1,
                            padding=1,
                        ),
                    ),
                    ("relu1", nn.ReLU()),
                    (
                        "conv2",
                        nn.Conv1d(
                            in_channels=10,
                            out_channels=10,
                            kernel_size=3,
                            stride=1,
                            padding=1,
                        ),
                    ),
                    ("relu2", nn.ReLU()),
                    (
                        "conv3",
                        nn.Conv1d(
                            in_channels=10,
                            out_channels=10,
                            kernel_size=3,
                            stride=1,
                            padding=1,
                        ),
                    ),
                    ("relu3", nn.ReLU()),
                    ("pool3", O2MaxPool1d(kernel_size=2, stride=2)),
                    (
                        "conv4",
                        nn.Conv1d(
                            in_channels=10,
                            out_channels=3,
                            kernel_size=3,
                            stride=1,
                            padding=1,
                        ),
                    ),
                    ("relu4", nn.ReLU()),
                    ("pool4", O2MaxPool1d(kernel_size=2, stride=2)),
                ]
            )
        )
        x = torch.rand(16, 3, 128)
        x.requires_grad = True
        y = model(x)
        target = torch.rand(*y.shape)
        return model, x, target

    @classmethod
    def _setup_conv2d(cls):
        torch.manual_seed(0)
        model = nn.Sequential(
            OrderedDict(
                [
                    (
                        "conv1",
                        nn.Conv2d(
                            in_channels=1,
                            out_channels=10,
                            kernel_size=3,
                            stride=1,
                            padding=1,
                        ),
                    ),
                    ("relu1", nn.ReLU()),
                    (
                        "conv2",
                        nn.Conv2d(
                            in_channels=10,
                            out_channels=10,
                            kernel_size=3,
                            stride=1,
                            padding=1,
                        ),
                    ),
                    ("relu2", nn.ReLU()),
                    (
                        "conv3",
                        nn.Conv2d(
                            in_channels=10,
                            out_channels=10,
                            kernel_size=3,
                            stride=1,
                            padding=1,
                        ),
                    ),
                    ("relu3", nn.ReLU()),
                    ("pool3", O2MaxPool2d(kernel_size=2, stride=2)),
                    (
                        "conv4",
                        nn.Conv2d(
                            in_channels=10,
                            out_channels=10,
                            kernel_size=3,
                            stride=1,
                            padding=1,
                        ),
                    ),
                    ("relu4", nn.ReLU()),
                    ("pool4", O2MaxPool2d(kernel_size=2, stride=2)),
                ]
            )
        )

        x = torch.rand(2, 1, 28, 28)
        x.requires_grad = True
        y = model(x)
        target = torch.rand(*y.shape)
        return model, x, target

    @classmethod
    def _setup_residual(cls):
        torch.manual_seed(0)
        residual = nn.Sequential(
            Residual(
                OrderedDict(
                    [
                        ("fc1", nn.Linear(3, 4)),
                        ("relu1", nn.ReLU()),
                        ("fc2", nn.Linear(4, 3)),
                        ("relu2", nn.ReLU()),
                    ]
                )
            ),
            Residual(
                OrderedDict(
                    [
                        ("fc1", nn.Linear(3, 4)),
                        ("relu1", nn.ReLU()),
                        ("fc2", nn.Linear(4, 3)),
                        ("relu2", nn.ReLU()),
                    ]
                )
            ),
        )
        model = residual
        x = torch.rand(2, 3)
        x.requires_grad = True
        y = model(x)
        target = torch.rand(*y.shape)
        return model, x, target

    def setup_model(self, model_name, wrapped=False, cuda=False):
        model, x, t = self.MODEL_INIT[model_name]()
        if wrapped:
            model = O2Model(model, nn.MSELoss())
        if cuda:
            model = model.cuda()
            x = x.cuda()
            t = t.cuda()

        return model, x, t

    def run_benchmark(self, model_name):
        model, x, t = self.setup_model(model_name, wrapped=False, cuda=True)
        criterion = nn.MSELoss()
        print(model)
        loss = criterion(model(x), t)
        start = time.time()
        loss.backward(create_graph=True)
        autograd_hessian = get_hessian(model, progress=True)
        end = time.time()
        autograd_time = end - start
        model.zero_grad()
        del model
        # O2Grad
        model_w, x_w, t_w = self.setup_model(model_name, wrapped=True, cuda=True)
        o2start = time.time()
        o2grad_hessian = model_w.get_hessian_from_input(x_w, t_w)
        o2end = time.time()
        o2time = o2end - o2start
        print("Benchmark complete!")
        print(f"Autograd: {autograd_time:.6f}s")
        print(f"O2Grad: {o2time:.6f}s")
        self.assertTrue(torch.allclose(autograd_hessian, o2grad_hessian))

    def test_big_mlp(self):
        header_str = f"""
        \rMLP Benchmark
        \r-------------"""
        print(header_str)
        self.run_benchmark("big_mlp")

    def test_conv1d(self):
        header_str = f"""
        \rCNN 1D Benchmark
        \r----------------"""
        print(header_str)
        self.run_benchmark("conv1d")

    def test_conv2d(self):
        header_str = f"""
        \rCNN 2D Benchmark
        \r----------------"""
        print(header_str)
        self.run_benchmark("conv2d")

    def test_residual(self):
        header_str = f"""
        \rResidual Benchmark
        \r------------------"""
        print(header_str)
        self.run_benchmark("residual")


class MLPGridBenchmarkTest(unittest.TestCase):
    @staticmethod
    def setup_mlp(
        input_dim=100,
        inter_dim=100,
        output_dim=10,
        batch_size=16,
        wrapped=True,
        device="cpu",
    ):
        torch.manual_seed(0)
        model = nn.Sequential(
            OrderedDict(
                [
                    ("fc1", nn.Linear(input_dim, inter_dim)),
                    ("sigmoid1", nn.Sigmoid()),
                    ("fc2", nn.Linear(inter_dim, output_dim)),
                ]
            )
        )
        x = torch.rand(batch_size, input_dim)
        x.requires_grad = True
        t = torch.rand(batch_size, output_dim)
        if wrapped:
            model = O2Model(model, nn.MSELoss())
        if device != "cpu":
            model = model.to(device)
            x = x.to(device)
            t = t.to(device)
        return model, x, t

    def save_results(self, df: pd.DataFrame, name: str):
        module_path = abspath(__file__)
        root_path = dirname(dirname(module_path))
        benchmark_path = os.path.join(root_path, "results/benchmark_test")
        if not os.path.exists(benchmark_path):
            os.makedirs(benchmark_path)
        file_path = os.path.join(benchmark_path, f"{name}.csv")
        df.to_csv(file_path)

    def run_results(self, grid, o2grad, devices, reps=5):
        print(grid)
        grid = [(*x, y, z) for x, y, z in itertools.product(grid, o2grad, devices)]
        df = pd.DataFrame(
            grid,
            columns=[
                "input_dim",
                "inter_dim",
                "output_dim",
                "batch_size",
                "o2grad",
                "device",
            ],
        )
        time_list = []
        mem_list = []
        err_list = []
        cached = [None, None]
        for element in tqdm.tqdm(grid):
            input_dim, inter_dim, output_dim, batch_size, wrapped, device = element
            model, input, target = self.setup_mlp(
                input_dim,
                inter_dim,
                output_dim,
                batch_size,
                wrapped=wrapped,
                device=device,
            )
            try:
                err_tmp = []
                if wrapped:

                    def _get_hessian_o2():
                        gc.enable()
                        model.zero_grad()
                        model.get_hessian_from_input(input, target)
                        model.clear_cache()
                        del hessian
                        gc.collect()

                    timer = timeit.Timer(_get_hessian_o2)
                    t = timer.timeit(number=reps) / reps
                    gc.collect()
                    mem_usg = memory_usage(
                        (_get_hessian_o2, [], {}), max_iterations=reps
                    )
                    gc.collect()

                    def _get_hessian_o2err():
                        model.zero_grad()
                        hessian = model.get_hessian_from_input(input, target)
                        if cached[0] != element[:4]:
                            cached[0] = element[:4]
                            cached[1] = hessian
                            err_tmp.append(0)
                        else:
                            err = (cached[1] - hessian).detach().abs().max().item()
                            err_tmp.append(err)
                        model.clear_cache()
                        del hessian

                    _get_hessian_o2err()
                    gc.collect()
                else:

                    def _get_hessian():
                        gc.enable()
                        model.zero_grad()
                        hessian = get_hessian(
                            model,
                            input=input,
                            target=target,
                            criterion=nn.MSELoss(),
                            progress=True,
                        )
                        del hessian
                        gc.collect()

                    timer = timeit.Timer(_get_hessian)
                    t = timer.timeit(number=reps) / reps
                    gc.collect()
                    mem_usg = memory_usage((_get_hessian, [], {}), max_iterations=reps)
                    gc.collect()

                    def _get_hessian_err():
                        model.zero_grad()
                        hessian = get_hessian(
                            model,
                            input=input,
                            target=target,
                            criterion=nn.MSELoss(),
                            progress=True,
                        )
                        if cached[0] != element[:4]:
                            cached[0] = element[:4]
                            cached[1] = hessian
                            err_tmp.append(0)
                        else:
                            err = (cached[1] - hessian).detach().abs().max().item()
                            err_tmp.append(err)
                        del hessian

                    _get_hessian_err()
                    gc.collect()

                mem_usg = np.average(mem_usg)
                err = np.average(err_tmp)
                time_list.append(t)
                mem_list.append(mem_usg)
                err_list.append(err)
            except Exception as e:
                print(f"FAILED: {e}")
                time_list.append(None)
                mem_list.append(None)
                err_list.append(None)
            finally:
                del model, input, target
                torch.cuda.empty_cache()
        df["time"] = time_list
        df["mem"] = mem_list
        df["err"] = err_list
        print(df)
        return df

    def test_grid(self):
        input_dims = [5, 10, 50, 100]
        batch_sizes = [1, 8, 16, 32]
        o2grad = [False, True]
        devices = ["cpu"]
        if torch.cuda.is_available():
            devices += ["cuda:2"]
        # grid = [*itertools.product(input_dims, inter_dims, output_dims, batch_sizes)]
        grid = [*itertools.product(input_dims, batch_sizes)]
        grid = [
            (input_dim, input_dim, input_dim, batch_size)
            for input_dim, batch_size in grid
        ]
        header_str = f"""
        \rMLP Grid Benchmark
        \r------------------"""
        print(header_str)
        df = self.run_results(grid, o2grad, devices)
        self.save_results(df, "mlp_grid")

    def test_return_layout(self):
        model, input, target = self.setup_mlp(
            input_dim=100, inter_dim=20, output_dim=10, batch_size=16
        )
        reps = 5
        devices = ["cpu"]
        if torch.cuda.is_available():
            devices += ["cuda:0"]
        methods = ["dydx", "dy2dx2", "dydp", "dy2dp2", "dy2dxdp"]
        grid = [[torch.sparse_coo, torch.strided]] * len(methods)
        grid = [*itertools.product(*grid)]
        df = pd.DataFrame(grid, columns=methods)
        time_list = []
        mem_list = []
        header_str = f"""
        \rMLP Layout Benchmark
        \r----------------------"""
        print(header_str)
        for layouts in grid:
            oij_layout, oih_layout, opj_layout, oph_layout, moph_layout = layouts
            settings = {
                "return_layout": {
                    "dydx": oij_layout,
                    "dy2dx2": oih_layout,
                    "dydp": opj_layout,
                    "dy2dp2": oph_layout,
                    "dy2dxdp": moph_layout,
                }
            }
            model.distribute_settings(settings)

            def _get_hessian():
                model.get_hessian_from_input_as_dict(input, target)

            timer = timeit.Timer(_get_hessian)
            t = timer.timeit(number=reps) / reps
            time_list.append(t)
            mem_usg = memory_usage(_get_hessian, max_iterations=reps)
            mem = np.average(mem_usg)
            mem_list.append(mem)

        df["time"] = time_list
        df["mem"] = mem_list
        print(df)
        min_t_idx = df["time"].idxmin()
        min_t_config = df.iloc[min_t_idx].to_dict()
        min_mem_idx = df["mem"].idxmin()
        min_mem_config = df.iloc[min_mem_idx].to_dict()
        print(f"Minimum time layout configuration : {min_t_config}")
        print(f"Minimum memory layout configuration: {min_mem_config}")
        self.save_results(df, "mlp_layout")

    def test_mnist(self):
        reps = 1
        input_dims = [28 * 28]
        inter_dims = [10, 20, 30]
        output_dims = [10]
        batch_sizes = [1, 8, 16, 32]
        o2grad = [False, True]
        devices = ["cpu"]
        if torch.cuda.is_available():
            devices += ["cuda:2"]
        grid = [*itertools.product(input_dims, inter_dims, output_dims, batch_sizes)]
        header_str = f"""
        \rMLP MNIST Benchmark
        \r-------------------"""
        print(header_str)
        df = self.run_results(grid, o2grad=o2grad, devices=devices, reps=reps)
        self.save_results(df, "mlp_mnist")


class CNN1dGridBenchmarkTest(unittest.TestCase):
    @staticmethod
    def setup_conv1d(
        input_dim=32,
        in_channels=3,
        out_channels=10,
        block_cnt=3,
        batch_size=16,
        wrapped=True,
        device="cpu",
        end_shape=None,
    ):
        torch.manual_seed(0)
        end_cnt = math.floor(np.log2(input_dim)) - 1
        counter = 0

        def _block(in_channels, out_channels, counter=0, pool=False):
            block = [
                (
                    f"conv{counter}",
                    nn.Conv1d(
                        in_channels=in_channels,
                        out_channels=out_channels,
                        kernel_size=3,
                        stride=1,
                        padding=1,
                    ),
                ),
                (f"relu{counter}", nn.ReLU()),
            ]
            if pool:
                block += [(f"pool{counter}", O2MaxPool1d(kernel_size=2, stride=2))]
            return block

        layers = _block(in_channels, out_channels)
        for i in range(block_cnt):
            counter += 1
            layers += _block(out_channels, out_channels, counter)
        for i in range(end_cnt):
            counter += 1
            layers += _block(out_channels, out_channels, counter, pool=True)
        if end_shape:
            layers += [("reshape_end", O2Reshape(out_channels))]
        model = nn.Sequential(OrderedDict(layers))
        x = torch.rand(batch_size, in_channels, input_dim)
        x.requires_grad = False
        t = model(x)
        t = torch.rand(t.shape)
        if wrapped:
            model = O2Model(model, nn.MSELoss())
        if device != "cpu":
            model = model.to(device)
            x = x.to(device)
            t = t.to(device)
        return model, x, t

    def save_results(self, df: pd.DataFrame, name: str):
        module_path = abspath(__file__)
        root_path = dirname(dirname(module_path))
        benchmark_path = os.path.join(root_path, "results/benchmark_test")
        if not os.path.exists(benchmark_path):
            os.makedirs(benchmark_path)
        file_path = os.path.join(benchmark_path, f"{name}.csv")
        df.to_csv(file_path)

    def run_results(self, grid, reps=3):
        df = pd.DataFrame(
            grid,
            columns=[
                "input_dim",
                "in_channels",
                "out_channels",
                "block_cnt",
                "batch_size",
                "o2grad",
                "device",
            ],
        )
        time_list = []
        mem_list = []
        err_list = []
        cached = [None, None]
        for element in tqdm.tqdm(grid):
            (
                input_dim,
                in_channels,
                out_channels,
                block_cnt,
                batch_size,
                wrapped,
                device,
            ) = element
            model, input, target = self.setup_conv1d(
                input_dim,
                in_channels,
                out_channels,
                block_cnt,
                batch_size,
                wrapped=wrapped,
                device=device,
            )
            print(model)
            try:
                err_tmp = []
                if wrapped:

                    def _get_hessian_o2():
                        gc.enable()
                        model.zero_grad()
                        model.get_hessian_from_input(input, target)
                        model.clear_cache()
                        del hessian
                        gc.collect()

                    timer = timeit.Timer(_get_hessian_o2)
                    t = timer.timeit(number=reps) / reps
                    gc.collect()
                    mem_usg = memory_usage(
                        (_get_hessian_o2, [], {}), max_iterations=reps
                    )
                    gc.collect()

                    def _get_hessian_o2err():
                        model.zero_grad()
                        hessian = model.get_hessian_from_input(input, target)
                        if cached[0] != element[:5]:
                            cached[0] = element[:5]
                            cached[1] = hessian
                            err_tmp.append(0)
                        else:
                            err = (cached[1] - hessian).detach().abs().max().item()
                            err_tmp.append(err)
                        model.clear_cache()
                        del hessian

                    _get_hessian_o2err()
                    gc.collect()
                else:

                    def _get_hessian():
                        gc.enable()
                        model.zero_grad()
                        hessian = get_hessian(
                            model,
                            input=input,
                            target=target,
                            criterion=nn.MSELoss(),
                            progress=True,
                        )
                        del hessian
                        gc.collect()

                    timer = timeit.Timer(_get_hessian)
                    t = timer.timeit(number=reps) / reps
                    gc.collect()
                    mem_usg = memory_usage((_get_hessian, [], {}), max_iterations=reps)
                    gc.collect()

                    def _get_hessian_err():
                        model.zero_grad()
                        hessian = get_hessian(
                            model,
                            input=input,
                            target=target,
                            criterion=nn.MSELoss(),
                            progress=True,
                        )
                        if cached[0] != element[:5]:
                            cached[0] = element[:5]
                            cached[1] = hessian
                            err_tmp.append(0)
                        else:
                            err = (cached[1] - hessian).detach().abs().max().item()
                            err_tmp.append(err)
                        del hessian

                    _get_hessian_err()
                    gc.collect()

                mem_usg = np.average(mem_usg)
                err = np.average(err_tmp)
                time_list.append(t)
                mem_list.append(mem_usg)
                err_list.append(err)
            except Exception as e:
                print(f"FAILED: {e}")
                time_list.append(None)
                mem_list.append(None)
                err_list.append(None)
            finally:
                del model, input, target
                torch.cuda.empty_cache()
        df["time"] = time_list
        df["mem"] = mem_list
        df["err"] = err_list
        print(df)
        return df

    def test_grid(self):
        input_dims = [16, 32, 64, 100]
        in_channels = [3]
        out_channels = [1, 5, 10, 15]
        block_cnts = [1, 5, 10, 20]
        batch_sizes = [1, 4, 8, 16]
        o2grad = [False, True]
        devices = ["cpu"]
        if torch.cuda.is_available():
            devices += ["cuda:2"]
        grid = [
            *itertools.product(
                input_dims,
                in_channels,
                out_channels,
                block_cnts,
                batch_sizes,
                o2grad,
                devices,
            )
        ]
        param_sets = [input_dims, in_channels, out_channels, block_cnts, batch_sizes]
        baseline = [p[-1] for p in param_sets]
        grid = [baseline]
        for i, pset in enumerate(param_sets):
            for p in pset[:-1]:
                config = (
                    [p[-1] for p in param_sets[:i]]
                    + [p]
                    + [p[-1] for p in param_sets[i + 1 :]]
                )
                grid.append(config)
        grid = [(*x, y, z) for x, y, z in itertools.product(grid, o2grad, devices)]
        print(grid)
        header_str = f"""
        \rConv1d Grid Benchmark
        \r------------------"""
        print(header_str)
        df = self.run_results(grid, reps=3)
        self.save_results(df, "conv1d_input")


class CNN2dGridBenchmarkTest(unittest.TestCase):
    @staticmethod
    def setup_conv2d(
        input_dim=32,
        in_channels=3,
        out_channels=10,
        block_cnt=3,
        batch_size=16,
        wrapped=True,
        device="cpu",
    ):
        torch.manual_seed(0)
        end_cnt = math.floor(np.log2(input_dim)) - 1
        counter = 0

        def _block(in_channels, out_channels, counter=0, pool=False):
            block = [
                (
                    f"conv{counter}",
                    nn.Conv2d(
                        in_channels=in_channels,
                        out_channels=out_channels,
                        kernel_size=3,
                        stride=1,
                        padding=1,
                    ),
                ),
                (f"relu{counter}", nn.ReLU()),
            ]
            if pool:
                block += [(f"pool{counter}", O2MaxPool2d(kernel_size=2, stride=2))]
            return block

        layers = _block(in_channels, out_channels)
        for i in range(block_cnt):
            counter += 1
            layers += _block(out_channels, out_channels, counter)
        for i in range(end_cnt):
            counter += 1
            layers += _block(out_channels, out_channels, counter, pool=True)
        model = nn.Sequential(OrderedDict(layers))
        x = torch.rand(batch_size, in_channels, input_dim, input_dim)
        x.requires_grad = False
        t = model(x)
        t = torch.rand(t.shape)
        if wrapped:
            model = O2Model(model, nn.MSELoss())
        if device != "cpu":
            model = model.to(device)
            x = x.to(device)
            t = t.to(device)
        return model, x, t

    def save_results(self, df: pd.DataFrame, name: str):
        module_path = abspath(__file__)
        root_path = dirname(dirname(module_path))
        benchmark_path = os.path.join(root_path, "results/benchmark_test")
        if not os.path.exists(benchmark_path):
            os.makedirs(benchmark_path)
        file_path = os.path.join(benchmark_path, f"{name}.csv")
        df.to_csv(file_path)

    def run_results(self, grid, reps=3):
        df = pd.DataFrame(
            grid,
            columns=[
                "input_dim",
                "in_channels",
                "out_channels",
                "block_cnt",
                "batch_size",
                "o2grad",
                "device",
            ],
        )
        print(df)
        time_list = []
        mem_list = []
        err_list = []
        cached = [None, None]
        for element in tqdm.tqdm(grid):
            (
                input_dim,
                in_channels,
                out_channels,
                block_cnt,
                batch_size,
                wrapped,
                device,
            ) = element
            model, input, target = self.setup_conv2d(
                input_dim,
                in_channels,
                out_channels,
                block_cnt,
                batch_size,
                wrapped=wrapped,
                device=device,
            )
            print(model)
            try:
                err_tmp = []
                if wrapped:

                    def _get_hessian_o2():
                        model.zero_grad()
                        model.get_hessian_from_input(input, target)
                        model.clear_cache()
                        del hessian

                    timer = timeit.Timer(_get_hessian_o2)
                    t = timer.timeit(number=reps) / reps
                    mem_usg = memory_usage(
                        (_get_hessian_o2, [], {}), max_iterations=reps
                    )

                    def _get_hessian_o2err():
                        model.zero_grad()
                        hessian = model.get_hessian_from_input(input, target)
                        if cached[0] != element[:5]:
                            cached[0] = element[:5]
                            cached[1] = hessian
                            err_tmp.append(0)
                        else:
                            err = (cached[1] - hessian).abs().max().item()
                            err_tmp.append(err)
                        model.clear_cache()
                        del hessian

                    _get_hessian_o2err()

                else:

                    def _get_hessian():
                        model.zero_grad()
                        hessian = get_hessian(
                            model,
                            input=input,
                            target=target,
                            criterion=nn.MSELoss(),
                            progress=True,
                        )
                        del hessian

                    timer = timeit.Timer(_get_hessian)
                    t = timer.timeit(number=reps) / reps
                    mem_usg = memory_usage((_get_hessian, [], {}), max_iterations=reps)

                    def _get_hessian_err():
                        model.zero_grad()
                        hessian = get_hessian(
                            model,
                            input=input,
                            target=target,
                            criterion=nn.MSELoss(),
                            progress=True,
                        )
                        if cached[0] != element[:5]:
                            cached[0] = element[:5]
                            cached[1] = hessian
                            err_tmp.append(0)
                        else:
                            err = (cached[1] - hessian).abs().max().item()
                            err_tmp.append(err)
                        del hessian

                    _get_hessian_err()

                mem_usg = np.average(mem_usg)
                err = np.average(err_tmp)
                time_list.append(t)
                mem_list.append(mem_usg)
                err_list.append(err)
            except Exception as e:
                print(f"FAILED: {e}")
                time_list.append(None)
                mem_list.append(None)
                err_list.append(None)
            finally:
                del model, input, target
                torch.cuda.empty_cache()
        df["time"] = time_list
        df["mem"] = mem_list
        df["err"] = err_list
        print(df)
        return df

    def test_grid(self):
        input_dims = [8]
        input_dims = [4, 8, 16, 20]
        in_channels = [3]
        out_channels = [1, 5, 10, 15]
        block_cnts = [1, 5, 10, 20]
        batch_sizes = [1, 4, 8, 16]
        o2grad = [False, True]
        devices = ["cpu"]
        if torch.cuda.is_available():
            devices += ["cuda:2"]
        grid = [
            *itertools.product(
                input_dims,
                in_channels,
                out_channels,
                block_cnts,
                batch_sizes,
                o2grad,
                devices,
            )
        ]
        header_str = f"""
        \rConv2d Grid Benchmark
        \r------------------"""
        print(header_str)
        df = self.run_results(grid, reps=3)
        self.save_results(df, "conv2d_block")

    def test_mnist(self):
        input_dims = [28]
        in_channels = [1]
        out_channels = [10]
        block_cnts = [2]
        batch_sizes = [1, 2, 4]
        o2grad = [True, False]
        devices = ["cpu"]
        if torch.cuda.is_available():
            devices += ["cuda:2"]
        grid = [
            *itertools.product(
                input_dims,
                in_channels,
                out_channels,
                block_cnts,
                batch_sizes,
                o2grad,
                devices,
            )
        ]
        header_str = f"""
        \rConv2d Grid Benchmark
        \r------------------"""
        print(header_str)
        df = self.run_results(grid)
        self.save_results(df, "conv2d_mnist_")


if __name__ == "__main__":
    unittest.main()
