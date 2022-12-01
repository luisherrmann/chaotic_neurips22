import torch
from abc import ABC, abstractmethod
from typing import Sequence, Union, Tuple

from ..o2layer.o2layer import O2Layer
from o2grad.sparse import SparseSymmetricMatrix
from o2grad.linalg import matmul_1d_2d_mixed, twin_matmul_mixed, matmul_mixed, sum_mixed
from o2grad.utils import reshape


class O2ParametricLayer(O2Layer, ABC):
    def __init__(self, module):
        super().__init__(module)
        self.SHARED_TENSOR_NAMES = ["dL2dw2" "V"]
        self.TENSOR_NAMES = self.TENSOR_NAMES + ["dydw", "dy2dw2", "dy2dxdw"]
        self.next_layer = None
        # Jacobian outputs w.r.t parameters
        self.dydw = None
        # Hessian outputs w.r.t. parameters
        self.dy2dw2 = None
        # Hessian outputs w.r.t. inputs and parameters
        self.dy2dxdw = None
        # Shared Objects:
        # Intermediate V
        self.V = None
        # Hessian loss w.r.t. parameters
        self.dL2dw2 = None
        self.lid_to_name = None

    @abstractmethod
    def get_output_param_jacobian(
        self, x: torch.Tensor, split: bool = False
    ) -> Union[torch.Tensor, Tuple[torch.Tensor]]:
        """Given activations x, should return the Jacobian of the module output w.r.t. module parameters (OPJ).

        Parameters:
        -----------
        x:  torch.Tensor
            Input activations.
        split: bool, optional
            If set to to False (default) returns a single tensor.
            Otherwise, returns a tensor for each parameter of the layer.

        Returns:
        --------
        torch.Tensor with shape [m, n], where
        (1) m is the total dimension of the output, i.e. x.numel()
        (2) n is the total number of parameters

        If module has separate parameters, n should correspond to:
        n = sum([p.numel() for p in self.parameters()]

        Example:
        --------
        Consider input activations x, where x.shape: [100, 2]
        Let module = nn.Linear(2, 3). There are two parameters layer.weight, layer.bias, where:

        module.weight.shape: [3, 2]
        module.bias.shape: [3]

        Thus n = 3 * 2 + 3 = 9

        The Jacobian returned should have the shape [200, 9].

        """

    @abstractmethod
    def get_output_param_hessian(self, x: torch.Tensor) -> torch.Tensor:
        """Given activations x, should return the Hessian dy2dw2 of the module output w.r.t. module parameters (OPH).

        Parameters:
        -----------
        x: torch.Tensor
            Input activations.

        Returns:
        --------
        A torch.Tensor with shape [m, n * n], where
        (1) m is the total output dimension, i.e. y.numel()
        (2) n is the total number of parameters

        If module has separate parameters, n should correspond to:
        n = sum([p.numel() for p in self.parameters()]

        Example:
        --------
        Consider input activations x, where x.shape: [100, 2]
        Let module = nn.Linear(2, 3). There are two parameters layer.weight, layer.bias, where:

        module.weight.shape: [3, 2]
        module.bias.shape: [3]

        Thus n = 3 * 2 + 3 = 9

        The Hessian returned should have the shape [100 * 3, 9 * 9].
        """

    @abstractmethod
    def get_mixed_output_param_hessian(self, x: torch.Tensor) -> torch.Tensor:
        """Given activations x, should return the mixed Hessian dy2dxdw of the module output w.r.t. inputs and module parameters (mOPH).

        Parameters:
        -----------
        x: torch.Tensor
            Input activations.

        Returns:
        --------
        A torch.Tensor with shape [m, o * n], where
        (1) m is the total output dimension, i.e. y.numel()
        (2) n is the total number of parameters
        (3) o is the total input dimension, i.e. x.numel()

        If module has separate parameters, n should correspond to:
        n = sum([p.numel() for p in self.parameters()]

        Example:
        --------
        Consider input activations x, where x.shape: [100, 2]
        Let module = nn.Linear(2, 3). There are two parameters layer.weight, layer.bias, where:

        module.weight.shape: [3, 2]
        module.bias.shape: [3]

        Thus n = 3 * 2 + 3 = 9

        The second Hessian returned should have the shape [100 * 3, 200 * 9].

        """

    def get_loss_param_hessian(
        self,
        dLdy: torch.Tensor,
        dL2dy2: Union[torch.Tensor, SparseSymmetricMatrix],
        dydw: torch.Tensor,
        dy2dw2: torch.Tensor,
        delete: Sequence[str] = [],
    ) -> torch.Tensor:
        """Given gradient, Hessian, OPJ and OPH, returns the Hessian of the loss w.r.t. module input.

            NOTE: The default implementation of this method should NOT be overriden
            unless the new implementation leads to performance boost over the default implementation.

        Parameters:
        -----------
        dLdy: torch.Tensor (strided|sparse_coo), (1D)
            The output gradient, or Loss-Output-Jacobian (LOJ).
        dL2dy2: torch.Tensor (strided|sparse_coo) | SparseSymmetricMatrix, (2D)
            The output Hessian, or Loss-Output-Hessian (LOH).
        dydw: torch.Tensor (strided|sparse_coo), (2D)
            The Output-Parameter-Jacobian (OPJ)
        dy2dw2: torch.Tensor (strided|sparse_coo), (2D)
            The Output-Parameter-Hessian (OPH)
        delete: Sequence[str]
            Names of tensors tagged for deletion

        Returns:
        --------
        torch.Tensor with shape [n, n], where n is the number of layer parameters, e.g. w.numel() + b.numel() for a linear layer.
        """
        pcount = dydw.shape[-1]
        dL2dw2_term2 = twin_matmul_mixed(dydw, dL2dy2)
        if not self.try_cache("dydw", dydw) and "dydw" in delete:
            del dydw
        if not self.try_cache("dL2dy2", dL2dy2) and "dL2dy2" in delete:
            del dL2dy2
        # dL2dw2_term1 = matmul_1d_3d_mixed(dLdy, dy2dw2)
        dL2dw2_term1 = matmul_1d_2d_mixed(dLdy, dy2dw2)
        if not self.try_cache("dLdy", dLdy) and "dLdy" in delete:
            del dLdy
        if not self.try_cache("dy2dw2", dy2dw2) and "dy2dw2" in delete:
            del dy2dw2
        dL2dw2_term1 = reshape(dL2dw2_term1, (pcount, pcount))
        dL2dw2 = sum_mixed(dL2dw2_term1, dL2dw2_term2)
        del dL2dw2_term1, dL2dw2_term2
        if isinstance(dL2dw2, torch.Tensor) and dL2dw2.layout == torch.sparse_coo:
            dL2dw2 = SparseSymmetricMatrix(
                dL2dw2._indices(), dL2dw2._values(), (pcount, pcount)
            )
        return dL2dw2

    def get_mixed_loss_param_hessian(
        self,
        dLdy: torch.Tensor,
        dL2dy2: Union[torch.Tensor, SparseSymmetricMatrix],
        dydx: torch.Tensor,
        dydw: torch.Tensor,
        dy2dxdw: torch.Tensor,
        delete: Sequence[str] = [],
    ) -> torch.Tensor:
        """Given OIJ, OIH, OPJ and mOPH, returns the Hessian of the loss w.r.t. module input.

        The default implementation of this method should NOT be overriden unless the new implementation leads to performance boost over the default implementation.

        Parameters:
        -----------
        dLdy: torch.Tensor (strided|sparse_coo), (1D)
            The output gradient, or Loss-Output-Jacobian (LOJ).
        dL2dy2: torch.Tensor (strided|sparse_coo) | SparseSymmetricMatrix, (2D)
            The output Hessian, or Loss-Output-Hessian (LOH).
        dydx: torch.Tensor (strided|sparse_coo), (2D)
            The Output-Input-Jacobian (OPJ)
        dydw: torch.Tensor (strided|sparse_coo), (2D)
            The Output-Parameter-Jacobian (OPJ)
        dy2dxdw: torch.Tensor (strided|sparse_coo), (2D)
            The mixed Output-Parameter-Hessian (mOPH)
        delete: Sequence[str]
            Names of tensors tagged for deletion

        Returns:
        --------
        torch.Tensor with shape [n, n], where n is the number of layer parameters, e.g. w.numel() + b.numel() for a linear layer.
        """
        fan_in = dydx.shape[-1]
        pcount = dydw.shape[-1]
        dL2dydw = matmul_mixed(dL2dy2, dydw, is_symmetric1=True)
        if not self.try_cache("dL2dy2", dL2dy2) and "dL2dy2" in delete:
            del dL2dy2
        if not self.try_cache("dydw", dydw) and "dydw" in delete:
            del dydw
        V_term1 = matmul_mixed(dydx, dL2dydw, is_transposed1=True)
        if not self.try_cache("dydx", dydx) and "dydx" in delete:
            del dydx
        del dL2dydw
        V_term2 = matmul_1d_2d_mixed(dLdy, dy2dxdw)
        V_term2 = reshape(V_term2, (fan_in, pcount))
        if not self.try_cache("dLdy", dLdy) and "dLdy" in delete:
            del dLdy
        if not self.try_cache("dy2dxdw", dy2dxdw) and "dy2dxdw" in delete:
            del dy2dxdw
        V = sum_mixed(V_term1, V_term2)
        del V_term1, V_term2
        return V
