import unittest
import numpy as np
import torch
import math

from metrics.eigen import cosine_sim, subspace_sim


class MetricsTest(unittest.TestCase):
    def test_cosine_sim(self):
        a = torch.tensor([1, 0], dtype=torch.float32)
        b = torch.tensor([0, 1], dtype=torch.float32)
        self.assertEqual(cosine_sim(a, a).item(), 1.0)
        self.assertEqual(cosine_sim(a, b).item(), 0.0)
        vectors = torch.tensor([[1, 0, 1], [0, 1, 0], [0, 0, 1]], dtype=torch.float32)
        sims = cosine_sim(vectors, vectors)
        target = torch.tensor(
            [
                [1.0, 0.0, 1 / math.sqrt(2)],
                [0.0, 1.0, 0.0],
                [1 / math.sqrt(2), 0.0, 1.0],
            ],
            dtype=torch.float32,
        )
        self.assertTrue(torch.allclose(sims, target))

    def test_subspace_sim(self):
        d = 100
        m = np.random.randint(2, d)
        k = np.random.randint(1, m)

        V = torch.randn(d, k)
        V, _ = torch.linalg.qr(V)
        V_ = torch.cat([V, torch.randn(d, m - k)], dim=-1)
        W, _ = torch.linalg.qr(V_)

        self.assertAlmostEqual(subspace_sim(V, W).item(), 1.0, places=5)


if __name__ == "__main__":
    unittest.main()
