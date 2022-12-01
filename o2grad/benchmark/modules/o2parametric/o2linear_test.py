import os
from os.path import dirname, abspath, pardir
import unittest
import itertools
import timeit
import pandas as pd
import torch
import tqdm
from typing import List

from o2grad.utils import memory_usage
from o2grad.modules.o2parametric.o2linear import O2Linear


class O2LinearBenchmarkTest(unittest.TestCase):
    @staticmethod
    def _template(
        method_name: str,
        batch_sizes: List[int],
        input_dims: List[int],
        output_dims: List[int],
    ):
        devices = ["cpu"]
        if torch.cuda.is_available():
            devices += ["cuda:0"]
        parameter_table = [
            *itertools.product(batch_sizes, input_dims, output_dims, devices)
        ]
        measurements_df = pd.DataFrame(
            parameter_table, columns=["batch_size", "input_dim", "output_dim", "device"]
        )
        mem_sparse_list = []
        mem_dense_list = []
        t_sparse_list = []
        t_dense_list = []
        for (b, fan_in, fan_out, device) in tqdm.tqdm(parameter_table):
            x = torch.rand(b, fan_in, device=device)
            o2linear = O2Linear(fan_in, fan_out)
            method = getattr(o2linear, method_name)
            reps = 10

            def dense(x):
                return method(x, sparse=False)

            timer_dense = timeit.Timer(lambda: dense(x))
            t_dense = timer_dense.timeit(number=reps) / reps
            mem_dense = memory_usage(dense(x))
            mem_dense_list.append(mem_dense)
            t_dense_list.append(t_dense)

            def sparse(x):
                return method(x, sparse=True)

            timer_sparse = timeit.Timer(lambda: sparse(x))
            t_sparse = timer_sparse.timeit(number=reps) / reps
            mem_sparse = memory_usage(sparse(x))
            mem_sparse_list.append(mem_sparse)
            t_sparse_list.append(t_sparse)

        measurements_df["t_dense"] = t_dense_list
        measurements_df["t_sparse"] = t_sparse_list
        measurements_df["mem_dense"] = mem_dense_list
        measurements_df["mem_sparse"] = mem_sparse_list
        header_str = f"""
        \rO2Linear.{method_name}
        \r----------------------------------"""
        print(header_str)
        print(measurements_df)
        return measurements_df

    def save_results(self, df: pd.DataFrame, name: str):
        module_path = abspath(__file__)
        O2GRAD_HOME = os.path.join(dirname(module_path), pardir, pardir, pardir)
        benchmark_path = os.path.join(
            O2GRAD_HOME, "results/modules/o2parametric/o2linear"
        )
        if not os.path.exists(benchmark_path):
            os.makedirs(benchmark_path)
        file_path = os.path.join(benchmark_path, f"{name}.csv")
        df.to_csv(file_path)

    def test_get_output_input_jacobian(self):
        batch_sizes = [1, 2, 8, 16, 32]
        input_dims = [1, 2, 10, 20]
        output_dims = input_dims
        measurements_df = O2LinearBenchmarkTest._template(
            "get_output_input_jacobian", batch_sizes, input_dims, output_dims
        )
        self.save_results(measurements_df, "get_output_input_jacobian")

    def test_get_output_input_hessian(self):
        batch_sizes = [1, 2, 8, 16, 32]
        input_dims = [1, 2, 10, 20]
        output_dims = input_dims
        measurements_df = O2LinearBenchmarkTest._template(
            "get_output_input_hessian", batch_sizes, input_dims, output_dims
        )
        self.save_results(measurements_df, "get_output_input_hessian")

    def test_get_output_param_jacobian(self):
        batch_sizes = [1, 2, 8, 16, 32]
        input_dims = [1, 2, 10, 20]
        output_dims = input_dims
        measurements_df = O2LinearBenchmarkTest._template(
            "get_output_param_jacobian", batch_sizes, input_dims, output_dims
        )
        self.save_results(measurements_df, "get_output_param_jacobian")

    def test_get_output_param_hessian(self):
        batch_sizes = [1, 2, 8, 16, 32]
        input_dims = [1, 2, 10, 20]
        output_dims = input_dims
        measurements_df = O2LinearBenchmarkTest._template(
            "get_output_param_hessian", batch_sizes, input_dims, output_dims
        )
        self.save_results(measurements_df, "get_output_param_hessian")

    def test_get_mixed_output_param_hessian(self):
        batch_sizes = [1, 2, 8, 16, 32]
        input_dims = [1, 2, 10, 20]
        output_dims = input_dims
        measurements_df = O2LinearBenchmarkTest._template(
            "get_mixed_output_param_hessian", batch_sizes, input_dims, output_dims
        )
        self.save_results(measurements_df, "get_mixed_output_param_hessian")
