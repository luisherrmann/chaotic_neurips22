import torch
import torch.nn as nn
from torch.autograd.functional import jacobian, hessian
from addict import Dict as AttrDict

from o2grad.sparse import SparseSymmetricMatrix


class O2Loss(nn.Module):
    def __init__(self, criterion):
        super().__init__()
        self.TENSOR_NAMES = ["dLdx", "dL2dx2"]
        self.criterion = criterion
        self.settings = AttrDict(
            {"save_intermediate": [], "per_batch": False, "elem": 0}
        )
        self.input = None
        self.dLdx = None
        self.dL2dx2 = None
        self.nesting_level = 0
        self._shared_objects = None

    def forward(self, x: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if ("o2enabled" not in self.settings) or self.settings.o2enabled:
            dLdx = self.get_loss_input_jacobian(x, target)
            dL2dx2 = self.get_loss_input_hessian(x, target)
            self.input = x.clone().detach()
            self.dLdx, self.dL2dx2 = dLdx, dL2dx2
        loss = self.criterion(x, target)
        return loss

    def __repr__(self):
        return "O2" + repr(self.criterion)

    def get_loss_input_jacobian(
        self, x: torch.Tensor, target: torch.Tensor
    ) -> torch.Tensor:
        def _criterion(x):
            return self.criterion(x, target)

        dLdx = jacobian(_criterion, x)
        return dLdx

    def get_loss_input_hessian(
        self, x: torch.Tensor, target: torch.Tensor
    ) -> torch.Tensor:
        def _criterion(x):
            return self.criterion(x, target)

        dL2dx2 = hessian(_criterion, x)
        return dL2dx2

    def get_loss_input_hessian_cached(self):
        if "o2enabled" in self.settings and self.settings.o2enabled:
            self._shared_objects.callbacks.progress.init()
        x = self.input
        elem = None if not (self.settings.per_batch) else self.settings.elem
        if self.dL2dx2 is None:
            return self.dL2dx2
        elif elem is not None:
            b, *_ = x.shape
            dim = x.reshape(b, -1).shape[1]
            dL2dx2 = self.dL2dx2.reshape(b, dim, b, dim)[
                elem : elem + 1, :, elem : elem + 1, :
            ]
            dL2dx2 = dL2dx2.reshape(dim, dim).to_sparse()
        else:
            dL2dx2 = self.dL2dx2.reshape(x.numel(), x.numel()).to_sparse()
        dL2dx2 = SparseSymmetricMatrix(
            dL2dx2._indices(), dL2dx2._values(), dL2dx2.shape
        )
        return dL2dx2
