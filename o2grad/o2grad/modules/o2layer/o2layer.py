import torch
from torch.autograd.functional import jacobian
from abc import ABC, abstractmethod
from typing import Sequence, List
from addict import Dict as AttrDict

from ..o2module import O2Module
from o2grad.sparse import SparseSymmetricMatrix
from o2grad.linalg import matmul_1d_2d_mixed, twin_matmul_mixed, sum_mixed
from o2grad.utils import CallbackSequence, reshape


class O2Layer(O2Module, ABC):
    def __init__(self, module, save_intermediate: List[str] = []):
        super().__init__(module)
        self.TENSOR_NAMES = [
            "dLdx",
            "dL2dx2",
            "dLdy",
            "dL2dy2",
            "dydx",
            "dy2dx2",
            "dyydx",
        ]
        self.settings = AttrDict(
            {"save_intermediate": [], "per_batch": False, "elem": 0}
        )
        save_intermediate_f = [
            tname for tname in save_intermediate if tname in self.TENSOR_NAMES
        ]
        self.settings.save_intermediate.append(save_intermediate_f)
        self.input = None
        self.output = None
        self.output_dim = 0
        # Jacobian outputs w.r.t inputs
        self.dydx = None
        # Jacobian loss w.r.t inputs
        self.dLdx = None
        # Jacobian loss w.r.t outputs
        self.dLdy = None
        # Hessian outputs w.r.t inputs
        self.dy2dx2 = None
        # Hessian loss w.r.t inputs
        self.dL2dx2 = None
        # Hessian loss w.r.t outputs
        self.dL2dy2 = None
        # Pointer to next connected layer
        self.next_layer = None
        # Chained Jacobian outputs w.r.t inputs
        self.dyydx = None
        # Shared object
        self.V = dict()
        self.lid_to_name = None
        self.chain_dydx = False
        self.chain_end_dydx = False
        self.is_first_layer = False
        self._callbacks = AttrDict(
            {"on_capture": CallbackSequence(), "on_complete": CallbackSequence()}
        )
        self.nesting_level = 0

    def __repr__(self):
        return "O2" + repr(self.module)

    def set_chain_output_input_jacobian(self, value: bool) -> None:
        self.chain_dydx = value

    def set_chain_end_output_input_jacobian(self, value: bool) -> None:
        self.chain_end_dydx = value

    def get_chained_output_input_jacobian_cached(self) -> torch.Tensor:
        return self.dyydx

    def get_output(self, x: torch.Tensor):
        if (self.input is not None) and (self.output is not None) and (x is self.input):
            return self.output
        else:
            return self.module(x)

    @abstractmethod
    def get_output_input_jacobian(self, x: torch.Tensor) -> torch.Tensor:
        """Given input x, should return the Jacobian of the module output w.r.t. module input.

        For the sake of efficiency, the default implementation of this method should be overriden.

        Parameters:
        -----------
        x:  torch.Tensor
            Input activations.

        Returns:
        --------
        torch.Tensor with shape [m, n], where
        (1) m is the total dimension of the output shape, i.e. y.numel()
        (2) n is the total dimension of the input shape, i.e. x.numel()

        Example:
        --------
        Consider input activations x, where x.shape: [100, 2]
        Let module = nn.Linear(2, 3), so for a output y we have y.shape: [100, 3]

        The Jacobian returned should have the shape [100 * 3, 100 * 2].
        """

        def _jac(x):
            return jacobian(self.forward, x, create_graph=True)

        x = x.clone()
        nx = x.numel()
        dydx = _jac(x)
        dydx = dydx.reshape(-1, nx)
        return dydx

    @abstractmethod
    def get_output_input_hessian(self, x: torch.Tensor) -> torch.Tensor:
        """Given input x, should return the Hessian of the module output w.r.t. module input.

        For the sake of efficiency, the default implementation of this method should be overriden.

        Parameters:
        -----------
        x:  torch.Tensor
            Input activations.

        Returns:
        --------
        torch.Tensor with shape [m, n * n], where
        (1) m is the output shape, i.e. self(x).shape
        (2) n is the input shape, i.e. x.shape

        Example:
        --------
        Consider input activations x, where x.shape: [100, 2]
        Let module = nn.Linear(2, 3), so for a output y we have y.shape: [100, 3]

        The Jacobian returned should have the shape [100, 3, 100, 2, 100, 2].
        """

        def _jac(x):
            return jacobian(self.forward, x, create_graph=True)

        x = x.clone()
        nx = x.numel()
        dy2dx2 = jacobian(_jac, x)
        dy2dx2 = dy2dx2.reshape(-1, nx * nx)
        return dy2dx2

    def get_loss_input_hessian(
        self,
        dLdy: torch.Tensor,
        dL2dy2: torch.Tensor,
        dydx: torch.Tensor,
        dy2dx2: torch.Tensor,
        delete: Sequence[str] = [],
    ) -> torch.Tensor:
        """Given the gradient and Hessian of the outputs y, returns the Hessian of the loss w.r.t. module input.

        The default implementation of this method should NOT be overriden unless the new implementation leads to performance boost over the default implementation.

        Parameters:
        -----------
        dLdy: torch.Tensor
            The gradient w.r.t. the inputs. This tensor will always be a strided tensor.
        dLy2: torch.Tensor
            The Hessian of the loss w.r.t the inputs. Can be a strided or a sparse_coo tensor.

        Returns:
        --------
        torch.Tensor with shape [n, n], where n is the number of input parameters, i.e. x.numel().
        """
        fan_in = dydx.shape[-1]
        dL2dx2_term2 = twin_matmul_mixed(dydx, dL2dy2)
        if not self.try_cache("dydx", dydx) and "dydx" in delete:
            del dydx
        if not self.try_cache("dL2dy2", dL2dy2) and "dL2dy2" in delete:
            del dL2dy2
        # dL2dx2_term1 = matmul_1d_3d_mixed(dLdy, dy2dx2)
        dL2dx2_term1 = matmul_1d_2d_mixed(dLdy, dy2dx2)
        dL2dx2_term1 = reshape(dL2dx2_term1, (fan_in, fan_in))
        if not self.try_cache("dLdy", dLdy) and "dLdy" in delete:
            del dLdy
        if not self.try_cache("dy2dx2", dy2dx2) and "dy2dx2" in delete:
            del dy2dx2
        dL2dx2 = sum_mixed(dL2dx2_term1, dL2dx2_term2)
        del dL2dx2_term1, dL2dx2_term2
        if isinstance(dL2dx2, torch.Tensor) and dL2dx2.layout == torch.sparse_coo:
            dL2dx2 = SparseSymmetricMatrix(
                dL2dx2._indices(), dL2dx2._values(), (fan_in, fan_in)
            )
        return dL2dx2

    def get_loss_input_hessian_cached(self):
        return self.dL2dx2

    def add_default_callbacks(self):
        # self._callbacks.on_capture.add(lambda: backprop_step(self))
        self._callbacks.on_capture.add(self._callbacks.on_complete)

    @classmethod
    def from_module(cls, module):
        o2_pmodule = cls.__new__(cls)
        super(cls, o2_pmodule).__init__(module)
        return o2_pmodule
