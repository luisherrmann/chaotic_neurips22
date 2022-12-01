import unittest
import time
import numpy as np
import torch
from o2grad.multiidx import expand_multiindex
from o2grad.sparse import SparseSymmetricMatrix


class SparseSymmetricMatrixBenchmark(unittest.TestCase):
    def test_benchmark1(self):
        shape = [1000, 1000]
        numel = np.prod(shape)
        count = int(1e6)
        indices1 = torch.tensor(np.random.choice(numel, count))
        indices1 = expand_multiindex(indices1, shape)
        values1 = torch.rand(count)
        tensor1 = SparseSymmetricMatrix(indices1.T, values1, shape, variant=0)

        indices2 = torch.tensor(np.random.choice(numel, count))
        indices2 = expand_multiindex(indices2, shape)
        values2 = torch.rand(count)
        tensor2 = SparseSymmetricMatrix(indices2.T, values2, shape)

        start = time.time()
        tensor1 + tensor2
        end = time.time()
        print(f"Variant 1 Sum: {end - start}")

        tensor2_sparse = tensor2.to_sparse()
        start = time.time()
        tensor1.matmul(tensor2_sparse)
        end = time.time()
        print(f"Variant 2 Matmul: {end - start}")

    def test_benchmark2(self):
        shape = [10000, 10000]
        numel = np.prod(shape)
        count = int(1e6)
        indices1 = torch.tensor(np.random.choice(numel, count))
        indices1 = expand_multiindex(indices1, shape)
        values1 = torch.rand(count)
        tensor1 = SparseSymmetricMatrix(indices1.T, values1, shape, variant=1)

        indices2 = torch.tensor(np.random.choice(numel, count))
        indices2 = expand_multiindex(indices2, shape)
        values2 = torch.rand(count)
        tensor2 = SparseSymmetricMatrix(indices2.T, values2, shape)

        start = time.time()
        tensor1 + tensor2
        end = time.time()
        print(f"Variant 2 Sum: {end - start}")

        tensor2_sparse = tensor2.to_sparse()
        start = time.time()
        tensor1.matmul(tensor2_sparse)
        end = time.time()
        print(f"Variant 2 Matmul: {end - start}")


if __name__ == "__main__":
    unittest.main()
