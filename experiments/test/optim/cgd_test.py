import unittest
import torch

from optim.cgd import calc_update


class CGDTest(unittest.TestCase):
    def test_calc_update(self):
        H = torch.tensor(
            [[0.5, 0, 0, 0], [0, 0.5, 0, 0], [0, 0, 2.1, 0], [0, 0, 0, -3.0]]
        )
        w, V = torch.linalg.eigh(H)
        lr = 1
        grads = torch.tensor([1.0, 1.0, 1.0, 1.0])
        grads_ = calc_update(grads, w, V, lr)
        result_ = torch.tensor([1.0, 1.0, 0.0, 0.0])
        self.assertTrue(torch.all(grads_ == result_).item())
        grads_ = calc_update(grads, w, V, lr, noise=True, noise_fac=0.5)
        correct = (
            torch.all(grads_[:2] == 1).item() and torch.all(grads_[2:] != 1).item()
        )
        self.assertTrue(correct)
        grads_ = calc_update(grads, w, V, lr, noise=True)
        correct = (
            torch.all(grads_[:2] == 1).item() and torch.all(grads_[2:] != 1).item()
        )
        self.assertTrue(correct)
        H = torch.tensor(
            [[0.5, 0, 0, 0], [0, 0.5, 0, 0], [0, 0, 6.0, 0], [0, 0, 0, -3.0]]
        )
        w, V = torch.linalg.eigh(H)
        grads_ = calc_update(grads, w, V, lr, k=1)
        result_ = torch.tensor([1.0, 1.0, 0.0, 1.0])
        self.assertTrue(torch.all(grads_ == result_).item())
        grads_ = calc_update(grads, w, V, lr, k=1, prune_k="rand")
        result1_ = torch.tensor([1.0, 1.0, 0.0, 1.0])
        result2_ = torch.tensor([1.0, 1.0, 1.0, 0.0])
        correct = (
            torch.all(grads_ == result1_).item() or torch.all(grads_ == result2_).item()
        )
        self.assertTrue(correct)
        H = torch.tensor(
            [[0.5, 0, 0, 0], [0, 0.5, 0, 0], [0, 0, 2.0, 0], [0, 0, 0, -3.0]]
        )
        w, V = torch.linalg.eigh(H)
        lr = 1
        grads = torch.tensor([1.0, 1.0, 1.0, 1.0])
        grads_ = calc_update(grads, w, V, lr, prune_set="neg")
        result_ = torch.tensor([1.0, 1.0, 1.0, 0.0])
        self.assertTrue(torch.all(grads_ == result_).item())
        H = torch.tensor(
            [[0.5, 0, 0, 0], [0, 0.5, 0, 0], [0, 0, -4.0, 0], [0, 0, 0, -3.0]]
        )
        w, V = torch.linalg.eigh(H)
        grads_ = calc_update(grads, w, V, lr, k=0, prune_set="neg")
        result_ = torch.tensor([1.0, 1.0, 1.0, 1.0])
        self.assertTrue(torch.all(grads_ == result_).item())
        grads_ = calc_update(grads, w, V, lr, k=1, prune_set="neg")
        result_ = torch.tensor([1.0, 1.0, 0.0, 1.0])
        self.assertTrue(torch.all(grads_ == result_).item())
        grads_ = calc_update(grads, w, V, lr, k=1, prune_set="neg", prune_k="rand")
        result1_ = torch.tensor([1.0, 1.0, 0.0, 1.0])
        result2_ = torch.tensor([1.0, 1.0, 1.0, 0.0])
        correct = (
            torch.all(grads_ == result1_).item() or torch.all(grads_ == result2_).item()
        )
        self.assertTrue(correct)
        H = torch.tensor(
            [[0.6, 0, 0, 0], [0, 0.5, 0, 0], [0, 0, -4.0, 0], [0, 0, 0, -3.0]]
        )
        w, V = torch.linalg.eigh(H)
        grads_ = calc_update(grads, w, V, lr, k=1, prune_set="pos")
        result_ = torch.tensor([0.0, 1.0, 1.0, 1.0])
        self.assertTrue(torch.all(grads_ == result_).item())
        grads_ = calc_update(grads, w, V, lr, k=1, prune_set="pos", prune_k="rand")
        result1_ = torch.tensor([0.0, 1.0, 1.0, 1.0])
        result2_ = torch.tensor([1.0, 0.0, 1.0, 1.0])
        correct = (
            torch.all(grads_ == result1_).item() or torch.all(grads_ == result2_).item()
        )
        self.assertTrue(correct)
