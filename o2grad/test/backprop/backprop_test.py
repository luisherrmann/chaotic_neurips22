import unittest
import itertools
import torch
import torch.nn as nn

from o2grad.backprop import get_hessian  # , get_hessian_fast


class BackpropTest(unittest.TestCase):
    def test_get_hessian(self):
        linear1 = nn.Linear(4, 2)
        linear2 = nn.Linear(2, 2)
        linear3 = nn.Linear(2, 2)
        model = nn.Sequential(
            *[linear1, nn.ReLU(), linear2, nn.ReLU(), linear3, nn.ReLU()]
        )
        criterion = nn.MSELoss()
        input = torch.rand(5, 4)
        target = torch.rand(5, 2)
        output = model(input)
        loss = criterion(output, target)
        loss.backward(create_graph=True)
        sz1 = sum([p.numel() for p in linear1.parameters()])
        sz2 = sz1 + sum([p.numel() for p in linear2.parameters()])
        sz3 = sz2 + sum([p.numel() for p in linear2.parameters()])
        full_hessian = get_hessian(model, progress=True)
        # full_hessian_fast = get_hessian_fast(model)
        # Check that results from both implementations match:
        # hessians_match = (full_hessian == full_hessian_fast).flatten()
        # self.assertTrue(all(hessians_match))
        # Check diagonal blocks match:
        block_hessian = get_hessian(model, progress=True, diagonal_blocks=True)
        block1_matches = (
            block_hessian[:sz1, :sz1] == full_hessian[:sz1, :sz1]
        ).flatten()
        self.assertTrue(all(block1_matches))
        block2_matches = (
            block_hessian[sz1:sz2, sz1:sz2] == full_hessian[sz1:sz2, sz1:sz2]
        ).flatten()
        self.assertTrue(all(block2_matches))
        block3_matches = (
            block_hessian[sz2:sz3, sz2:sz3] == full_hessian[sz2:sz3, sz2:sz3]
        ).flatten()
        self.assertTrue(all(block3_matches))
        # Check everything outside the diagonal blocks is zero:
        indices = block_hessian.to_sparse().coalesce().indices().T.tolist()
        block_indices = [
            *itertools.product(range(sz1), range(sz1)),
            *itertools.product(range(sz1, sz2), range(sz1, sz2)),
            *itertools.product(range(sz2, sz3), range(sz2, sz3)),
        ]
        # All indices should be in the set of valid indices:
        computed_set = set([tuple(x) for x in indices])
        valid_set = set([tuple(x) for x in block_indices])
        all_valid = len(computed_set - valid_set) == 0
        self.assertTrue(all_valid)
        # Check if results are also valid when passing input
        full_hessian_ = get_hessian(model, input=input, target=target, progress=True)
        self.assertTrue(torch.allclose(full_hessian, full_hessian_))


if __name__ == "__main__":
    unittest.main()
