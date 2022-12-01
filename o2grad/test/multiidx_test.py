import unittest
import torch
import itertools

from o2grad.multiidx import flatten_multiindex, reshape_multiindex, expand_multiindex


class MultiIdxTest(unittest.TestCase):
    def test_flatten_multiindex(self):
        shape = [4, 3, 2]
        idx = torch.LongTensor([*itertools.product(range(4), range(3), range(2))])
        idx_flat = flatten_multiindex(idx, shape)
        idx_flat_ = torch.LongTensor([*range(4 * 3 * 2)])
        self.assertEqual(idx_flat.shape, idx_flat_.shape)
        self.assertTrue(torch.all(idx_flat == idx_flat_))

    def test_expand_multiindex(self):
        idx = torch.LongTensor([1, 3, 5])
        # self.assertRaises(expand_multiindex(idx, [2, 2]))
        idx = torch.LongTensor(range(8))
        multi_index = torch.LongTensor(
            [*itertools.product(range(2), range(2), range(2))]
        )
        self.assertTrue(torch.all(expand_multiindex(idx, [2, 2, 2]) == multi_index))

        idx = torch.LongTensor(range(12))
        multi_index = torch.LongTensor(
            [*itertools.product(range(2), range(3), range(2))]
        )
        self.assertTrue(torch.all(expand_multiindex(idx, [2, 3, 2]) == multi_index))

    def test_reshape_multiindex(self):
        multi_index = torch.LongTensor([*itertools.product(range(2), range(2))])
        multi_index_re = torch.LongTensor([*itertools.product(range(4), range(1))])
        self.assertTrue(
            torch.all(reshape_multiindex(multi_index, (2, 2), (4, 1)) == multi_index_re)
        )
