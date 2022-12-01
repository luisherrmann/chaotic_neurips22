import unittest
import torch
import torch.nn as nn
from o2grad.modules.o2loss import O2Loss


class O2LossTest(unittest.TestCase):
    def test_forward(self):
        epsilon = 0.01
        y = torch.rand(10, 5)
        target = y + torch.normal(0.0, epsilon, (10, 5))

        criterion = O2Loss(nn.MSELoss())
        criterion(y, target)
        dL2dx2 = criterion.dL2dx2
        if dL2dx2.layout == torch.sparse_coo:
            dL2dx2 = dL2dx2.to_dense()
        dL2dx2 = dL2dx2.reshape(50, 50)
        # Loss Hessian: 2 * 1/M * 1/n for M: number of samples and n: dimension
        self.assertTrue(torch.all(dL2dx2 == torch.eye(50) * 2 / 50))
