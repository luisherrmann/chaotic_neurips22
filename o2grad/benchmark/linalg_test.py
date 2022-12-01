import itertools
import unittest
import torch
from tqdm import tqdm
import pandas as pd
import timeit
from memory_profiler import LineProfiler as MemoryLineProfiler
from memory_profiler import memory_usage

from o2grad.linalg import eigs_from_dict2d
from o2grad.utils import matrix_from_dict2d

PRECISION = 1e-3


class LinalgBenchmarkTest(unittest.TestCase):
    def test_eigs_from_dict(self):
        grid = [(160, 10), (80, 20), (40, 40), (20, 80), (10, 160)]
        grid = [(sparse, *g) for sparse, g in itertools.product([False, True], grid)]
        df = pd.DataFrame(grid, columns=["sorted", "block_cnt", "size"])
        times_dense = []
        mem_dense = []
        times_sparse = []
        mem_sparse = []
        times = []
        mem = []
        reps = 1
        for sorted, size, block_cnt in tqdm(grid):
            dict2d = {s: {s: torch.rand(size, size)} for s in range(block_cnt)}

            def _get_eigs_vecs_dense():
                eigs, vecs = eigs_from_dict2d(
                    range(block_cnt), dict2d, layout=torch.strided, sorted=sorted
                )

            timer = timeit.Timer(_get_eigs_vecs_dense)
            t_dense = timer.timeit(number=reps) / reps
            mem_profiler = MemoryLineProfiler()
            mem_profiler.add_function(_get_eigs_vecs_dense)
            mem_profiler.enable()
            mem_usg = memory_usage(_get_eigs_vecs_dense)
            avg_mem_usg = sum(mem_usg) / len(mem_usg)
            mem_profiler.disable()
            del mem_profiler
            mem_dense.append(avg_mem_usg)

            times_dense.append(t_dense)

            def _get_eigs_vecs_sparse():
                eigs, vecs = eigs_from_dict2d(
                    range(block_cnt), dict2d, layout=torch.sparse_coo, sorted=sorted
                )

            timer = timeit.Timer(_get_eigs_vecs_sparse)
            t_sparse = timer.timeit(number=reps) / reps
            times_sparse.append(t_sparse)
            mem_profiler = MemoryLineProfiler()
            mem_profiler.add_function(_get_eigs_vecs_sparse)
            mem_profiler.enable()
            mem_usg = memory_usage(_get_eigs_vecs_sparse)
            avg_mem_usg = sum(mem_usg) / len(mem_usg)
            mem_profiler.disable()
            del mem_profiler
            mem_sparse.append(avg_mem_usg)

            def _get_eigs_vecs():
                matrix = matrix_from_dict2d(range(block_cnt), dict2d, as_type="dense")
                eigs, vecs = torch.linalg.eigh(matrix)

            timer = timeit.Timer(_get_eigs_vecs)
            t = timer.timeit(number=reps) / reps
            times.append(t)
            mem_profiler = MemoryLineProfiler()
            mem_profiler.add_function(_get_eigs_vecs)
            mem_profiler.enable()
            mem_usg = memory_usage(_get_eigs_vecs)
            avg_mem_usg = sum(mem_usg) / len(mem_usg)
            mem_profiler.disable()
            del mem_profiler
            mem.append(avg_mem_usg)

        df["t_dense"] = times_dense
        df["t_sparse"] = times_sparse
        df["t"] = times
        df["mem_dense"] = mem_dense
        df["mem_sparse"] = mem_sparse
        df["mem"] = mem
        print(df)


if __name__ == "__main__":
    unittest.main()
