import unittest
import torch
import torch.nn as nn

from o2grad.utils import (
    cartesian_prod_2d,
    get_conv_output_shape,
    get_tconv_output_shape,
    memory_usage,
    reshape,
    sparse_sum,
)


class UtilsTest(unittest.TestCase):
    def test_cartesian_prod_2d(self):
        # Test for two input tensors
        t1 = torch.cartesian_prod(torch.arange(2), torch.arange(2))
        t2 = torch.arange(3).reshape(-1, 1)
        t3 = torch.cartesian_prod(torch.arange(3), torch.arange(2))
        t12 = cartesian_prod_2d(t1, t2)
        t12_ = torch.cartesian_prod(torch.arange(2), torch.arange(2), torch.arange(3))
        self.assertTrue(torch.all(t12 == t12_).item())
        # Test for three input tensors
        t123 = cartesian_prod_2d(t1, t2, t3)
        t123_ = torch.cartesian_prod(
            torch.arange(2),
            torch.arange(2),
            torch.arange(3),
            torch.arange(3),
            torch.arange(2),
        )
        self.assertTrue(torch.all(t123 == t123_).item())
        # Test that passing no input tensors raises exception
        def _test_no_input():
            cartesian_prod_2d()

        self.assertRaises(ValueError, _test_no_input)
        # Test that passing single input tnesor raises exception
        def _test_single_input():
            t1 = torch.arange(2)
            cartesian_prod_2d(t1)

        self.assertRaises(ValueError, _test_single_input)
        # Test that objects other than tensors raise exception
        def _test_invalid():
            t1 = torch.arange(2)
            t2 = [1, 2, 3]
            cartesian_prod_2d(t1, t2)

        self.assertRaises(TypeError, _test_invalid)
        # Test that strided tensors raise exception
        def _test_strided():
            t1 = torch.arange(2).to_sparse()
            t2 = torch.arange(3)
            cartesian_prod_2d(t1, t2)

        self.assertRaises(ValueError, _test_strided)

    def _get_conv_output_shape(self, w, k, s, p, d):
        input = torch.rand(1, 1, w)
        output = nn.Conv1d(
            in_channels=1,
            out_channels=1,
            kernel_size=k,
            stride=s,
            padding=p,
            dilation=d,
        )(input)
        return output.numel()

    def _get_tconv_output_shape(self, w, k, s, p, d):
        input = torch.rand(1, 1, w)
        output = nn.ConvTranspose1d(
            in_channels=1,
            out_channels=1,
            kernel_size=k,
            stride=s,
            padding=p,
            dilation=d,
        )(input)
        return output.numel()

    def test_get_conv_output_shape(self):
        w, k, s, p, d = 5, 3, 1, 1, 1
        w_ = self._get_conv_output_shape(w, k, s, p, d)
        self.assertEqual(w_, get_conv_output_shape(w, k, s, p, d))

        w, k, s, p, d = 10, 3, 2, 1, 1
        w_ = self._get_conv_output_shape(w, k, s, p, d)
        self.assertEqual(w_, get_conv_output_shape(w, k, s, p, d))

        w, k, s, p, d = 10, 3, 1, 2, 1
        w_ = self._get_conv_output_shape(w, k, s, p, d)
        self.assertEqual(w_, get_conv_output_shape(w, k, s, p, d))

        w, k, s, p, d = 10, 3, 1, 1, 2
        w_ = self._get_conv_output_shape(w, k, s, p, d)
        self.assertEqual(w_, get_conv_output_shape(w, k, s, p, d))

        w, k, s, p, d = 20, 2, 1, 2, 3
        w_ = self._get_conv_output_shape(w, k, s, p, d)
        self.assertEqual(w_, get_conv_output_shape(w, k, s, p, d))

    def test_get_tconv_output_shape(self):
        w, k, s, p, d = 5, 3, 1, 1, 1
        w_ = self._get_tconv_output_shape(w, k, s, p, d)
        self.assertEqual(w_, get_tconv_output_shape(w, k, s, p, d))

        w, k, s, p, d = 10, 3, 2, 1, 1
        w_ = self._get_tconv_output_shape(w, k, s, p, d)
        self.assertEqual(w_, get_tconv_output_shape(w, k, s, p, d))

        w, k, s, p, d = 10, 3, 1, 2, 1
        w_ = self._get_tconv_output_shape(w, k, s, p, d)
        self.assertEqual(w_, get_tconv_output_shape(w, k, s, p, d))

        w, k, s, p, d = 10, 3, 1, 1, 2
        w_ = self._get_tconv_output_shape(w, k, s, p, d)
        self.assertEqual(w_, get_tconv_output_shape(w, k, s, p, d))

        w, k, s, p, d = 20, 2, 1, 2, 3
        w_ = self._get_tconv_output_shape(w, k, s, p, d)
        self.assertEqual(w_, get_tconv_output_shape(w, k, s, p, d))

        w = 10
        k = 2
        s = 2
        p = 2
        d = 2
        w_ = self._get_tconv_output_shape(w, k, s, p, d)
        self.assertEqual(w_, get_tconv_output_shape(w, k, s, p, d))

    def test_sparse_sum(self):
        indices1 = torch.LongTensor([[1, 3, 4]])
        indices2 = torch.LongTensor([[2, 3, 4, 5]])
        values1 = torch.FloatTensor([1.0, 2.0, 3.0])
        values2 = torch.FloatTensor([1.0, 2.0, 3.0, 4.0])
        shape = (6,)
        indices, values = sparse_sum(indices1, values1, indices2, values2, shape)
        tensor1 = torch.sparse_coo_tensor(indices1, values1, shape).to_dense()
        tensor2 = torch.sparse_coo_tensor(indices2, values2, shape).to_dense()
        sum_tensor = torch.sparse_coo_tensor(indices, values, shape).to_dense()
        self.assertTrue(torch.all(tensor1 + tensor2 == sum_tensor))

        tensor1 = torch.FloatTensor([[3, 0, 1], [3, 0, 0], [0, 1, 2]])
        tensor2 = torch.FloatTensor([[1, 0, 0], [1, 0, 1], [0, 0, 0]])
        shape = tensor1.shape
        tensor1_sparse = tensor1.to_sparse().coalesce()
        tensor2_sparse = tensor2.to_sparse().coalesce()
        indices1, values1 = tensor1_sparse.indices(), tensor1_sparse.values()
        indices2, values2 = tensor2_sparse.indices(), tensor2_sparse.values()

        indices, values = sparse_sum(indices1, values1, indices2, values2, shape)
        sum_tensor = torch.sparse_coo_tensor(indices, values, shape).to_dense()
        self.assertTrue(torch.all(tensor1 + tensor2 == sum_tensor))

    def test_reshape(self):
        tensor = torch.rand(4, 4)
        tensor_sparse = tensor.to_sparse()
        # Reshape to 2D tensor
        result = tensor.reshape(2, 8)
        result_ = reshape(tensor, (2, 8))
        self.assertTrue(torch.all(result == result_).item())
        result_ = reshape(tensor, [2, 8])
        self.assertTrue(torch.all(result == result_).item())
        result_ = reshape(tensor_sparse, (2, 8))
        self.assertTrue(torch.all(result == result_.to_dense()).item())
        # Reshape to 1D tensor
        result = tensor.reshape(16)
        result_ = reshape(tensor, (16,))
        self.assertTrue(torch.all(result == result_).item())
        result_ = reshape(tensor, [16])
        self.assertTrue(torch.all(result == result_).item())
        result_ = reshape(tensor_sparse, (16,))
        self.assertTrue(torch.all(result == result_.to_dense()).item())

        def _try_invalid():
            reshape(tensor, (2, 7))

        self.assertRaises(ValueError, _try_invalid)

    def test_memory_usage(self):
        x = torch.tensor([[1, 2, 3], [4, 5, 6]], dtype=torch.int32)
        self.assertTrue(memory_usage(x) == 4 * 6)


if __name__ == "__main__":
    unittest.main()
