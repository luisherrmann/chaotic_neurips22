import unittest
import torch.nn as nn
from o2grad.modules.mapping import is_module_supported


class FunctionTest(unittest.TestCase):
    def test_is_module_supported(self):
        class Identity(nn.Module):
            def __init__(self):
                super(Identity, self).__init__()

            def forward(self, x):
                return x

        id = Identity()
        self.assertFalse(is_module_supported(id))
        self.assertTrue(is_module_supported(id, custom_layers={Identity: Identity}))

        mock_layers = [
            nn.Linear(10, 5),
            nn.Conv2d(in_channels=3, out_channels=8, kernel_size=3),
        ]

        for layer in mock_layers:
            self.assertTrue(is_module_supported(layer))


if __name__ == "__main__":
    unittest.main()
