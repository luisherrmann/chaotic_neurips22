import unittest
import os
import torch
from o2grad.sparse import SparseSymmetricMatrix, SparseStorageTensor


class SparseSymmetricMatrixTest(unittest.TestCase):
    def test_init(self):
        tensor = torch.tensor([[2, 3, 0], [3, 0, 1], [0, 1, 4]])
        sparse = tensor.to_sparse()
        hermitian = SparseSymmetricMatrix(
            sparse.indices(), sparse.values(), sparse.shape
        )
        tensor_ = hermitian.to_dense()
        self.assertTrue(torch.all(tensor == tensor_).item())

        # Assert that representation is correct for random matrix XX^T that is hermitian by construction.
        X = torch.rand(10, 10)
        tensor = torch.matmul(X, X.T)
        sparse = tensor.to_sparse()
        hermitian = SparseSymmetricMatrix(
            sparse.indices(), sparse.values(), sparse.shape
        )
        tensor_ = hermitian.to_dense()
        self.assertTrue(torch.allclose(tensor, tensor_))

        # Assert that non-hermitian quadratic matrix does not lead to correct reconstruction
        tensor = torch.tensor([[1, 1, 0], [1, 0, 0], [1, 0, 0]])
        sparse = tensor.to_sparse()
        hermitian = SparseSymmetricMatrix(
            sparse.indices(), sparse.values(), sparse.shape
        )
        tensor_ = hermitian.to_dense()
        self.assertFalse(torch.all(tensor == tensor_).item())

    def test_add(self):
        tensor1 = torch.tensor([[1, 0, 1], [0, 0, 0], [1, 0, 1]])
        sparse1 = tensor1.to_sparse()
        hermitian1 = SparseSymmetricMatrix(
            sparse1.indices(), sparse1.values(), sparse1.shape
        )
        tensor2 = torch.tensor([[1, 0, 0], [0, 0, 1], [0, 1, 1]])
        sparse2 = tensor2.to_sparse()
        hermitian2 = SparseSymmetricMatrix(
            sparse2.indices(), sparse2.values(), sparse2.shape
        )
        tensor3 = hermitian1 + hermitian2
        self.assertTrue(isinstance(tensor3, SparseSymmetricMatrix))

        tensor3_ = torch.tensor([[2, 0, 1], [0, 0, 1], [1, 1, 2]])
        self.assertTrue(all(tensor3.to_dense().reshape(-1) == tensor3_.reshape(-1)))

        tensor3 = hermitian1 + sparse2
        self.assertTrue(all(tensor3.to_dense().reshape(-1) == tensor3_.reshape(-1)))

    def test_sub(self):
        tensor1 = torch.tensor([[1, 0, 1], [0, 0, 0], [1, 0, 1]])
        sparse1 = tensor1.to_sparse()
        hermitian1 = SparseSymmetricMatrix(
            sparse1.indices(), sparse1.values(), sparse1.shape
        )
        tensor2 = torch.tensor([[1, 0, 0], [0, 0, 1], [0, 1, 1]])
        sparse2 = tensor2.to_sparse()
        hermitian2 = SparseSymmetricMatrix(
            sparse2.indices(), sparse2.values(), sparse2.shape
        )
        tensor3 = hermitian1 - hermitian2
        self.assertTrue(isinstance(tensor3, SparseSymmetricMatrix))

        tensor3_ = torch.tensor([[0, 0, 1], [0, 0, -1], [1, -1, 0]])
        self.assertTrue(all(tensor3.to_dense().reshape(-1) == tensor3_.reshape(-1)))

        tensor3 = hermitian1 - sparse2
        self.assertTrue(all(tensor3.to_dense().reshape(-1) == tensor3_.reshape(-1)))

    def test_matmul(self):
        tensor1 = torch.tensor([[1, 0, 1], [0, 0, 0], [1, 0, 1]])
        sparse1 = tensor1.to_sparse()
        hermitian1 = SparseSymmetricMatrix(
            sparse1.indices(), sparse1.values(), sparse1.shape
        )
        tensor2 = torch.tensor([[1, 0, 0], [0, 0, 1], [0, 1, 1]])
        sparse2 = tensor2.to_sparse()
        hermitian2 = SparseSymmetricMatrix(
            sparse2.indices(), sparse2.values(), sparse2.shape
        )
        result = torch.matmul(
            tensor1, tensor2
        )  # torch.tensor([[1, 1, 1], [0, 0, 0], [1, 1, 1]])
        # dense
        result_ = hermitian1.matmul(tensor2)
        self.assertTrue(
            isinstance(result_, torch.Tensor) and result_.layout == torch.strided
        )
        self.assertTrue(torch.allclose(result, result_))
        result_ = hermitian1.matmul(sparse2)
        # sparse
        self.assertTrue(
            isinstance(result_, torch.Tensor) and result_.layout == torch.sparse_coo
        )
        self.assertTrue(torch.allclose(result, result_.to_dense()))
        # SparseSymmetricMatrix
        tril1 = hermitian1.tril.clone()
        tril2 = hermitian2.tril.clone()
        result_ = hermitian1.matmul(hermitian2)
        self.assertTrue(
            isinstance(result_, torch.Tensor) and result_.layout == torch.sparse_coo
        )
        self.assertTrue(torch.allclose(result, result_.to_dense()))
        self.assertTrue(
            torch.all(tril1.to_dense() == hermitian1.tril.to_dense()).item()
        )
        self.assertTrue(
            torch.all(tril2.to_dense() == hermitian2.tril.to_dense()).item()
        )


class SparseStorageTensorTest(unittest.TestCase):
    def test_init(self):
        filename = "tmp/tensor"
        tensor = torch.tensor([[1, 0, 0], [1, 0, 1], [0, 0, 1]], dtype=torch.double)
        tensor_sparse = tensor.to_sparse()
        indices, values = tensor_sparse.indices(), tensor_sparse.values()
        sst = SparseStorageTensor(filename, indices, values, tensor.shape)
        ifile, vfile = sst.ifile, sst.vfile
        sst.persist()
        del sst
        # Make sure values have been stored
        file_exists = os.path.exists(filename)
        self.assertTrue(file_exists)
        istorage = torch.LongStorage.from_file(ifile, shared=True, size=indices.numel())
        vstorage = torch.DoubleStorage.from_file(
            vfile, shared=True, size=values.numel()
        )
        indices_, values_ = torch.LongTensor(istorage), torch.DoubleTensor(vstorage)
        self.assertEqual(indices.numel(), indices_.numel())
        self.assertEqual(values.numel(), values_.numel())
        tensor_ = torch.sparse_coo_tensor(
            indices_.reshape(indices.T.shape).T, values_, tensor.shape
        )
        all_match = torch.all(tensor.reshape(-1) == tensor_.to_dense().reshape(-1))
        self.assertTrue(all_match)

    def test_to_dense(self):
        filename = "tmp/tensor"
        tensor = torch.tensor([[1, 0, 0], [1, 0, 1], [0, 0, 1]], dtype=torch.double)
        tensor_sparse = tensor.to_sparse()
        indices, values = tensor_sparse.indices(), tensor_sparse.values()
        sst = SparseStorageTensor(filename, indices, values, tensor.shape)
        tensor_ = sst.to_dense()
        self.assertTrue(torch.all(tensor == tensor_).item())

    def test_to_sparse(self):
        filename = "tmp/tensor"
        tensor = torch.tensor([[1, 0, 0], [1, 0, 1], [0, 0, 1]], dtype=torch.double)
        tensor_sparse = tensor.to_sparse()
        indices, values = tensor_sparse.indices(), tensor_sparse.values()
        sst = SparseStorageTensor(filename, indices, values, tensor.shape)
        tensor_sparse_ = sst.to_sparse()
        self.assertTrue(torch.all(tensor == tensor_sparse_.to_dense()).item())

    def test_from_file(self):
        filename = "tmp/tensor"
        tensor = torch.tensor([[1, 0, 0], [1, 0, 1], [0, 0, 1]], dtype=torch.double)
        tensor_sparse = tensor.to_sparse()
        indices, values = tensor_sparse.indices(), tensor_sparse.values()
        sst = SparseStorageTensor(filename, indices, values, tensor.shape)
        sst.persist()
        del sst
        sst_ = SparseStorageTensor.from_file(filename)
        self.assertTrue(torch.all(tensor == sst_.to_dense()).item())

    def test_add(self):
        tensor1 = torch.tensor([[1, 0, 0], [0, 1, 0], [1, 0, 0]])
        tensor2 = torch.tensor([[1, 0, 0], [0, 0, 0], [0, 0, 1]])
        tensor1_sparse = tensor1.to_sparse()
        tensor2.to_sparse()
        # With coalescing
        sst1 = SparseStorageTensor(
            "tmp/tensor",
            tensor1_sparse.indices(),
            tensor1_sparse.values(),
            tensor1.shape,
        )
        sst1.add(tensor2)
        all_match = torch.all((tensor1 + tensor2) == sst1.to_dense()).item()
        self.assertTrue(all_match)

    def test__add__(self):
        tensor1 = torch.tensor([[1, 0, 0], [0, 1, 0], [1, 0, 0]])
        tensor2 = torch.tensor([[1, 0, 0], [0, 0, 0], [0, 0, 1]])
        tensor1_sparse = tensor1.to_sparse()
        tensor2.to_sparse()
        sst1 = SparseStorageTensor(
            "tmp/tensor",
            tensor1_sparse.indices(),
            tensor1_sparse.values(),
            tensor1.shape,
        )
        tensor3 = sst1 + tensor2
        all_match = torch.all(
            (tensor1 + tensor2).reshape(-1) == tensor3.to_dense().reshape(-1)
        )
        self.assertTrue(all_match)

    def test__sub__(self):
        tensor1 = torch.tensor([[1, 0, 0], [0, 1, 0], [1, 0, 0]])
        tensor2 = torch.tensor([[1, 0, 0], [0, 0, 0], [0, 0, 1]])
        tensor1_sparse = tensor1.to_sparse()
        tensor2.to_sparse()
        sst1 = SparseStorageTensor(
            "tmp/tensor",
            tensor1_sparse.indices(),
            tensor1_sparse.values(),
            tensor1.shape,
        )
        tensor3 = sst1 - tensor2
        all_match = torch.all(
            (tensor1 - tensor2).reshape(-1) == tensor3.to_dense().reshape(-1)
        )
        self.assertTrue(all_match)


if __name__ == "__main__":
    unittest.main()
