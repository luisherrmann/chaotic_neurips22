from abc import ABC, abstractmethod
import torch
import torch.nn as nn

from .o2layer import O2Layer
from o2grad.utils import flatten_multiindex


class O2PointwiseActivationFunction(O2Layer, ABC):
    def __init__(self, module):
        super().__init__(module)

    @abstractmethod
    def get_first_derivative(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Given input x and output y of an activation function f, i.e. y = f(x),
        should return the first derivative f'(x).

        Parameter:
        ----------
        x: torch.Tensor
            The input to the activation function.
        y: torch.Tensor
            The output of the activation function.

        Returns:
        --------
        The first derivative of the activation function, i.e. f'(x), as torch.Tensor.
        """

    @abstractmethod
    def get_second_derivative(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Given input x and output y of an activation function f, i.e. y = f(x),
        should return the second derivative f''(x).

        Parameter:
        ----------
        x: torch.Tensor
            The input to the activation function.
        y: torch.Tensor
            The output of the activation function.

        Returns:
        --------
        The derivative of the activation function, i.e. f''(x), as torch.Tensor.
        """

    def build_input_jacobian(self, df: torch.Tensor, sparse=True) -> torch.Tensor:
        size = df.numel()
        shape = df.shape
        df = df.reshape(-1)

        idx = [[i] * 2 for i in range(size) if df[i] != 0]
        df = df[df != 0]
        idx = torch.tensor(idx, dtype=torch.int64, device=df.device).T

        dydx = torch.sparse_coo_tensor(idx, df, (size, size)).coalesce()
        if not sparse:
            dydx = dydx.to_dense()
            dydx = dydx.reshape([*shape, *shape])

        return dydx

    def build_input_hessian(self, df2: torch.Tensor, sparse=True) -> torch.Tensor:
        size = df2.numel()
        shape = df2.shape
        df2 = df2.reshape(-1)

        idx = [[i] * 3 for i in range(size) if df2[i] != 0]
        df2 = df2[df2 != 0]
        idx = torch.tensor(idx, dtype=torch.int64, device=df2.device).T
        idx = idx.reshape(3, -1)

        if not sparse:
            dy2dx2 = torch.sparse_coo_tensor(idx, df2, (size, size, size))
            dy2dx2 = dy2dx2.to_dense()
            dy2dx2 = dy2dx2.reshape([*shape, *shape, *shape])
        else:
            idx[1, :] = flatten_multiindex(idx[1:, :].T, (size, size))
            dy2dx2 = torch.sparse_coo_tensor(
                idx[:2, :], df2.reshape(-1), (size, size * size)
            ).coalesce()

        return dy2dx2

    def get_output_input_jacobian(self, x: torch.Tensor) -> torch.Tensor:
        y = self.get_output(x)
        df = self.get_first_derivative(x, y)
        dydx = self.build_input_jacobian(df)
        return dydx

    def get_output_input_hessian(self, x: torch.Tensor) -> torch.Tensor:
        y = self.get_output(x)
        df2 = self.get_second_derivative(x, y)
        dy2dx2 = self.build_input_hessian(df2)
        return dy2dx2


class O2Sigmoid(O2PointwiseActivationFunction):
    def __init__(self):
        super().__init__(nn.Sigmoid())

    def get_first_derivative(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return y * (1 - y)

    def get_second_derivative(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return y * (1 - y) * (1 - 2 * y)


class O2ReLU(O2PointwiseActivationFunction):
    def __init__(self):
        super().__init__(nn.ReLU())

    def get_first_derivative(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        # x = x.clone()
        # x[x <= 0] = 0
        # x[x > 0] = 1
        # return x
        pass

    def get_second_derivative(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        # x = x.clone()
        # x[:] = 0
        # return x
        pass

    def get_output_input_jacobian(self, x: torch.Tensor) -> torch.Tensor:
        size = x.numel()
        device = x.device
        x = x.reshape(-1)
        positions = torch.nonzero(x > 0).reshape(-1)
        nnz = len(positions)
        indices = torch.empty([2, nnz], dtype=torch.int64, device=device)
        indices[0, :] = positions
        indices[1, :] = positions
        values = torch.empty([nnz], device=device)
        values[:] = 1
        dydx = torch.sparse_coo_tensor(indices, values, (size, size)).coalesce()
        return dydx

    def get_output_input_hessian(self, x: torch.Tensor) -> torch.Tensor:
        size = x.numel()
        device = x.device
        indices, values = torch.tensor([], dtype=torch.int64, device=device).reshape(
            2, 0
        ), torch.tensor([], device=device)
        dy2dx2 = torch.sparse_coo_tensor(
            indices, values, (size, size * size)
        ).coalesce()
        return dy2dx2
