import unittest
import numpy as np
import torch

from o2grad.modules.o2layer.activations import (
    O2PointwiseActivationFunction,
    O2Sigmoid,
    O2ReLU,
)


class O2SigmoidTest(unittest.TestCase):
    def test_get_output_input_jacobian(self):
        torch.manual_seed(0)
        x = torch.rand(2, 4)
        o2sigmoid = O2Sigmoid()

        dydx = o2sigmoid.get_output_input_jacobian(x).to_dense()
        dydx_ = super(
            O2PointwiseActivationFunction, o2sigmoid
        ).get_output_input_jacobian(x)

        self.assertEqual(np.prod(dydx.shape), np.prod(dydx_.shape))
        dydx = dydx.reshape(dydx_.shape)
        self.assertTrue(torch.allclose(dydx, dydx_))

    def test_get_output_input_hessian(self):
        torch.manual_seed(0)
        x = torch.rand(2, 4)
        o2sigmoid = O2Sigmoid()

        dydx2 = o2sigmoid.get_output_input_hessian(x).to_dense()
        dydx2_ = super(
            O2PointwiseActivationFunction, o2sigmoid
        ).get_output_input_hessian(x)

        self.assertEqual(np.prod(dydx2.shape), np.prod(dydx2_.shape))
        dydx2 = dydx2.reshape(dydx2_.shape)
        self.assertTrue(torch.allclose(dydx2, dydx2_))


class O2ReLUTest(unittest.TestCase):
    def test_get_output_input_jacobian(self):
        torch.manual_seed(0)
        x = torch.rand(2, 4)
        o2relu = O2ReLU()

        dydx = o2relu.get_output_input_jacobian(x).to_dense()
        dydx_ = super(O2PointwiseActivationFunction, o2relu).get_output_input_jacobian(
            x
        )

        self.assertEqual(np.prod(dydx.shape), np.prod(dydx_.shape))
        dydx = dydx.reshape(dydx_.shape)
        self.assertTrue(torch.allclose(dydx, dydx_))

    def test_get_output_input_hessian(self):
        torch.manual_seed(0)
        x = torch.rand(2, 4)
        o2sigmoid = O2Sigmoid()

        dydx2 = o2sigmoid.get_output_input_hessian(x).to_dense()
        dydx2_ = super(
            O2PointwiseActivationFunction, o2sigmoid
        ).get_output_input_hessian(x)

        self.assertEqual(np.prod(dydx2.shape), np.prod(dydx2_.shape))
        dydx2 = dydx2.reshape(dydx2_.shape)
        self.assertTrue(torch.allclose(dydx2, dydx2_))


if __name__ == "__main__":
    unittest.main()
