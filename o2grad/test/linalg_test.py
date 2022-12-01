import unittest
import torch

from o2grad.sparse import SparseSymmetricMatrix
from o2grad.linalg import (
    eigs_from_dict2d,
    matmul_mixed,
    sum_mixed,
    twin_matmul_mixed,
    matmul_1d_2d_mixed,
    matmul_1d_3d_mixed,
)
from o2grad.utils import matrix_from_dict2d

PRECISION = 1e-3


class LinalgTest(unittest.TestCase):
    def test_twin_matmul_mixed(self):
        tensor1 = torch.rand(10, 5)
        tensor1_sparse = tensor1.to_sparse()
        tensor2 = torch.rand(10, 10).tril()
        tensor2 = tensor2 + tensor2.T
        tensor2_sparse = tensor2.to_sparse()
        tensor2_sparse_sym = SparseSymmetricMatrix(
            tensor2_sparse.indices(), tensor2_sparse.values(), tensor2_sparse.shape
        )
        result12 = torch.einsum("ij,jk,kl -> il", tensor1.T, tensor2, tensor1)
        # dense - dense
        result = twin_matmul_mixed(tensor1, tensor2)
        self.assertTrue(result.layout == torch.strided)
        self.assertTrue(torch.allclose(result, result12))
        # dense - sparse
        result = twin_matmul_mixed(tensor1, tensor2_sparse)
        self.assertTrue(result.layout == torch.strided)
        self.assertTrue(torch.allclose(result, result12))
        # dense - SparseSymmetricMatrix
        result = twin_matmul_mixed(tensor1, tensor2_sparse_sym)
        self.assertTrue(result.layout == torch.strided)
        self.assertTrue(torch.allclose(result, result12))
        # sparse - dense
        result = twin_matmul_mixed(tensor1_sparse, tensor2)
        self.assertTrue(result.layout == torch.strided)
        self.assertTrue(torch.allclose(result, result12))
        # sparse - dense, second matrix symmetric
        tensor3 = tensor2 + tensor2.T
        result13 = torch.einsum("ij,jk,kl -> il", tensor1.T, tensor3, tensor1)
        result = twin_matmul_mixed(tensor1_sparse, tensor3, is_symmetric2=True)
        self.assertTrue(result.layout == torch.strided)
        self.assertTrue(torch.allclose(result13, result))
        # sparse - sparse
        result = twin_matmul_mixed(tensor1_sparse, tensor2_sparse)
        self.assertTrue(result.layout == torch.sparse_coo)
        self.assertTrue(torch.allclose(result.to_dense(), result12))
        # sparse - SparseSymmetricMatrix
        result = twin_matmul_mixed(tensor1_sparse, tensor2_sparse_sym)
        self.assertTrue(isinstance(result, SparseSymmetricMatrix))
        self.assertTrue(torch.allclose(result.to_dense(), result12))

    def test_matmul_1d_2d_mixed(self):
        tensor1 = torch.rand(10)
        tensor1_sparse = tensor1.to_sparse()
        tensor2 = torch.rand(10, 5)
        tensor2_sparse = tensor2.to_sparse()
        result12 = torch.einsum("i,ij->j", tensor1, tensor2)
        # dense - dense
        result = matmul_1d_2d_mixed(tensor1, tensor2)
        self.assertTrue(result.layout == torch.strided)
        self.assertTrue(torch.allclose(result, result12))
        # dense - sparse
        result = matmul_1d_2d_mixed(tensor1, tensor2_sparse)
        self.assertTrue(result.layout == torch.strided)
        self.assertTrue(torch.allclose(result, result12))
        # sparse - dense
        result = matmul_1d_2d_mixed(tensor1_sparse, tensor2)
        self.assertTrue(result.layout == torch.strided)
        self.assertTrue(torch.allclose(result, result12))
        # sparse - sparse
        result = matmul_1d_2d_mixed(tensor1_sparse, tensor2_sparse)
        self.assertTrue(result.layout == torch.sparse_coo)
        self.assertTrue(torch.allclose(result.to_dense(), result12))
        # Second tensor is zero everywhere
        tensor4 = torch.zeros(10, 5)
        tensor4_sparse = tensor4.to_sparse().coalesce()
        result14 = torch.einsum("i,ij->j", tensor1, tensor4)
        result = matmul_1d_2d_mixed(tensor1_sparse, tensor4_sparse)
        self.assertTrue(torch.allclose(result.to_dense(), result14))

    def test_matmul_1d_3d_mixed(self):
        tensor1 = torch.rand(10)
        tensor1_sparse = tensor1.to_sparse()
        tensor2 = torch.rand(10, 5, 7)
        tensor2_sparse = tensor2.to_sparse()
        result12 = torch.einsum("i,ijk->jk", tensor1, tensor2)
        # dense - dense
        result = matmul_1d_3d_mixed(tensor1, tensor2)
        self.assertTrue(result.layout == torch.strided)
        self.assertTrue(torch.allclose(result, result12))
        # dense - sparse
        result = matmul_1d_3d_mixed(tensor1, tensor2_sparse)
        self.assertTrue(result.layout == torch.strided)
        self.assertTrue(torch.allclose(result, result12))
        # sparse - dense
        result = matmul_1d_3d_mixed(tensor1_sparse, tensor2)
        self.assertTrue(result.layout == torch.strided)
        self.assertTrue(torch.allclose(result, result12))
        # sparse - sparse
        result = matmul_1d_3d_mixed(tensor1_sparse, tensor2_sparse)
        self.assertTrue(result.layout == torch.sparse_coo)
        self.assertTrue(torch.allclose(result.to_dense(), result12))
        # sparse - sparse, second matrix symmetric
        tensor3 = torch.rand(10, 5, 5)
        tensor3 = tensor3 + tensor3.transpose(2, 1)
        tensor3_sparse = tensor3.to_sparse()
        result13 = torch.einsum("i,ijk->jk", tensor1, tensor3)
        self.assertTrue(torch.allclose(result13, result13.T))
        result = matmul_1d_3d_mixed(tensor1_sparse, tensor3_sparse, is_symmetric=True)
        self.assertTrue(isinstance(result, SparseSymmetricMatrix))
        self.assertTrue(torch.allclose(result.to_dense(), result13))
        # Second tensor is zero everywhere
        tensor4 = torch.zeros(10, 5, 5)
        tensor4_sparse = tensor4.to_sparse().coalesce()
        result14 = torch.einsum("i,ijk->jk", tensor1, tensor4)
        result = matmul_1d_3d_mixed(tensor1_sparse, tensor4_sparse)
        self.assertTrue(torch.allclose(result.to_dense(), result14))

    def test_matmul_mixed(self):
        tensor1 = torch.rand(4, 5)
        tensor1_sparse = tensor1.to_sparse()
        tensor2 = torch.rand(5, 5).tril()
        tensor2 = tensor2 + tensor2.T
        tensor2_sparse = tensor2.to_sparse()
        tensor2_sym = SparseSymmetricMatrix(
            tensor2_sparse.indices(), tensor2_sparse.values(), tensor2.shape
        )
        result = torch.einsum("ij,jk->ik", tensor1, tensor2)
        # dense - dense
        result_ = matmul_mixed(tensor1, tensor2)
        self.assertTrue(result_.layout == torch.strided)
        self.assertTrue(torch.allclose(result, result_))
        result_ = matmul_mixed(tensor1, tensor2)
        self.assertTrue(result_.layout == torch.strided)
        self.assertTrue(torch.allclose(result, result_))
        # dense - sparse
        result_ = matmul_mixed(tensor1, tensor2_sparse)
        self.assertTrue(result_.layout == torch.strided)
        self.assertTrue(torch.allclose(result, result_))
        # dense - SparseSymmetricMatrix
        result_ = matmul_mixed(tensor1, tensor2_sym)
        self.assertTrue(result_.layout == torch.strided)
        self.assertTrue(torch.allclose(result, result_))
        # sparse - dense
        result_ = matmul_mixed(tensor1_sparse, tensor2)
        self.assertTrue(result_.layout == torch.strided)
        self.assertTrue(torch.allclose(result, result_))
        # sparse - sparse
        result_ = matmul_mixed(tensor1_sparse, tensor2_sparse)
        self.assertTrue(result_.layout == torch.sparse_coo)
        self.assertTrue(torch.allclose(result, result_.to_dense()))
        # sparse - SparseSymmetricMatrix
        result_ = matmul_mixed(tensor1_sparse, tensor2_sym)
        self.assertTrue(result_.layout == torch.sparse_coo)
        self.assertTrue(torch.allclose(result, result_.to_dense()))
        # SparseSymmetricMatrix - dense
        tensor1 = torch.rand(5, 5).tril()
        tensor1 = tensor1 + tensor1.T
        tensor1_sparse = tensor1.to_sparse()
        tensor1_sym = SparseSymmetricMatrix(
            tensor1_sparse.indices(), tensor1_sparse.values(), tensor1.shape
        )
        result = matmul_mixed(tensor1, tensor2)
        result_ = matmul_mixed(tensor1_sym, tensor2)
        self.assertTrue(result_.layout == torch.strided)
        self.assertTrue(torch.allclose(result, result_))
        result_ = matmul_mixed(tensor1_sym, tensor2, is_symmetric1=True)
        self.assertTrue(result_.layout == torch.strided)
        self.assertTrue(torch.allclose(result, result_))
        # SparseSymmetricMatrix - sparse
        result_ = matmul_mixed(tensor1_sym, tensor2_sparse)
        self.assertTrue(result_.layout == torch.sparse_coo)
        self.assertTrue(torch.allclose(result, result_.to_dense()))
        result_ = matmul_mixed(tensor1_sym, tensor2, is_symmetric1=True)
        self.assertTrue(result_.layout == torch.strided)
        self.assertTrue(torch.allclose(result, result_))
        # SparseSymmetricMatrix - SparseSymmetricMatrix
        result_ = matmul_mixed(tensor1_sym, tensor2_sym)
        self.assertTrue(
            isinstance(result_, torch.Tensor) and result_.layout == torch.sparse_coo
        )
        self.assertTrue(torch.allclose(result, result_.to_dense()))
        # dense - sparse
        result_ = matmul_mixed(
            tensor1, tensor2_sparse, is_symmetric1=True, is_symmetric2=True
        )
        self.assertTrue(result_.layout == torch.strided)
        self.assertTrue(torch.allclose(result, result_))

    def test_eigs_from_dict(self):
        block_cnt = 3
        size = 10
        dict2d = {s: {s: torch.rand(size, size)} for s in range(block_cnt)}
        eigs, vecs = eigs_from_dict2d(
            range(block_cnt), dict2d, layout=torch.strided, sorted=True
        )
        matrix = matrix_from_dict2d(range(block_cnt), dict2d, as_type="dense")
        eigs_, vecs_ = torch.linalg.eigh(matrix)
        self.assertTrue(torch.all(torch.abs(eigs - eigs_) < PRECISION))
        self.assertTrue(torch.all(torch.abs(vecs - vecs_) < PRECISION))
        eigs, vecs = eigs_from_dict2d(
            range(block_cnt), dict2d, layout=torch.sparse_coo, sorted=True
        )
        vecs = vecs.to_dense()
        self.assertTrue(torch.all(torch.abs(eigs - eigs_) < PRECISION))
        self.assertTrue(torch.all(torch.abs(vecs - vecs_) < PRECISION))

    def test_sum_mixed(self):
        tensor1 = torch.rand(5, 5).tril()
        tensor1 = tensor1 + tensor1.T
        tensor1_sparse = tensor1.to_sparse()
        tensor1_sym = SparseSymmetricMatrix(
            tensor1_sparse.indices(), tensor1_sparse.values(), tensor1.shape
        )
        tensor2 = torch.rand(5, 5).tril()
        tensor2 = tensor2 + tensor2.T
        tensor2_sparse = tensor2.to_sparse()
        tensor2_sym = SparseSymmetricMatrix(
            tensor2_sparse.indices(), tensor2_sparse.values(), tensor2.shape
        )
        result = tensor1 + tensor2
        # dense - dense
        result_ = sum_mixed(tensor1, tensor2)
        self.assertTrue(result_.layout == torch.strided)
        self.assertTrue(torch.allclose(result, result_))
        # dense - sparse
        result_ = sum_mixed(tensor1, tensor2_sparse)
        self.assertTrue(result_.layout == torch.strided)
        self.assertTrue(torch.allclose(result, result_))
        # dense - SparseSymmetricMatrix
        result_ = sum_mixed(tensor1, tensor2_sym)
        self.assertTrue(result_.layout == torch.strided)
        self.assertTrue(torch.allclose(result, result_))
        # sparse - dense
        result_ = sum_mixed(tensor1_sparse, tensor2)
        self.assertTrue(result_.layout == torch.strided)
        self.assertTrue(torch.allclose(result, result_))
        # sparse - sparse
        result_ = sum_mixed(tensor1_sparse, tensor2_sparse)
        self.assertTrue(result_.layout == torch.sparse_coo)
        self.assertTrue(torch.allclose(result, result_.to_dense()))
        # sparse - SparseSymmetricMatrix
        result_ = sum_mixed(tensor1_sparse, tensor2_sym)
        self.assertTrue(result_.layout == torch.sparse_coo)
        self.assertTrue(torch.allclose(result, result_.to_dense()))
        # SparseSymmetricMatrix - dense
        result_ = sum_mixed(tensor1_sym, tensor2)
        self.assertTrue(result_.layout == torch.strided)
        self.assertTrue(torch.allclose(result, result_))
        # SparseSymmetricMatrix - sparse
        result_ = sum_mixed(tensor1_sym, tensor2_sparse)
        self.assertTrue(result_.layout == torch.sparse_coo)
        self.assertTrue(torch.allclose(result, result_.to_dense()))
        # SparseSymmetricMatrix - SparseSymmetricMatrix
        result_ = sum_mixed(tensor1_sym, tensor2_sym)
        self.assertTrue(isinstance(result_, SparseSymmetricMatrix))
        self.assertTrue(torch.allclose(result, result_.to_dense()))
