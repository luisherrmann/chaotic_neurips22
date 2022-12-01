import os
from os.path import dirname
from typing import Union, Sequence, Callable, List, Dict, overload
import numpy as np
import sortednp as snp
import torch
import torch.nn as nn
from .sparse import SparseSymmetricMatrix, SparseStorageTensor
from .multiidx import flatten_multiindex, reshape_multiindex


class CallbackSequence:
    def __init__(self, *args: Callable):
        self._check_inputs(*args)
        self.callbacks = [*args]

    @staticmethod
    def _check_inputs(*args) -> None:
        all_callables = all([isinstance(x, Callable) for x in args])
        if not all_callables:
            pos = [i for i, is_func in enumerate(all_callables) if not is_func]
            pos_str = ", ".join([str(i) for i in pos])
            raise TypeError(
                f"Expected Callables, but got wrong types at positions {pos_str}! "
            )

    def add(self, *args: Callable):
        self._check_inputs(*args)
        self.callbacks.extend([*args])

    def clear(self):
        self.callbacks = []

    def __call__(self):
        [cb() for cb in self.callbacks]


def module_has_parameters(module: nn.Module):
    """Checks if a torch layer has parameters.

    Parameters
    ----------
    module: torch.nn.Module
        The torch module to check.

    Returns
    -------
    True if module has parameters (even if require_grad=False), False otherwise.

    """
    if not isinstance(module, nn.Module):
        raise TypeError(f"Expected torch.nn.Module, but got {type(module)}!")
    return sum([p.numel() for p in module.parameters()]) > 0


def cartesian_prod_2d(*tensors: List[torch.Tensor]):
    """Given a set of 2d tensors, computes their cartesian product.
    NOTE: Other than torch.cartesian_prod(), the tensors do not have to be 1D here.

    Parameters:
    -----------
    *tensors: Any number of arbitrarily shaped 2D tensors

    Returns:
    --------
    The cartesian product of the respective tensors, to be interpreted as the cartesian product of two sets of tuples.

    Exceptions:
    -----------
    ValueError, TypeError
    """
    if len(tensors) < 2:
        raise ValueError(f"Expects at least two tensors")
    are_not_tensors = [not isinstance(x, torch.Tensor) for x in tensors]
    if any(are_not_tensors):
        pos = np.where(np.array(are_not_tensors))[0]
        pos_str = ", ".join([str(x) for x in pos])
        raise TypeError(f"Expected tensors, but arguments {pos_str} are not!")
    are_not_strided = [x.layout != torch.strided for x in tensors]
    if any(are_not_strided):
        pos = np.where(np.array(are_not_strided))[0]
        pos_str = ", ".join([str(x) for x in pos])
        raise ValueError(f"Expected strided tensors, but arguments {pos_str} are not!")
    A = tensors[0]
    for B in tensors[1:]:
        m1, n1 = A.shape
        m2, n2 = B.shape
        C = torch.empty([m1 * m2, n1 + n2], dtype=A.dtype, device=A.device)
        C[:, :n1] = A.repeat(1, m2).reshape(m1 * m2, n1)
        C[:, n1:] = B.repeat(m1, 1)
        del B
        A = C
    return A


def matrix_from_dict2d(
    order: Sequence[str],
    dict2d: Dict[str, Dict[str, Union[torch.Tensor, SparseSymmetricMatrix]]],
    as_type="dense",
    as_file=False,
    diagonal_blocks=False,
    device="cpu",
) -> Union[torch.Tensor, SparseSymmetricMatrix]:
    idx_low, idx_top = {}, {}
    i_top = 0
    size = 0
    for s in order:
        if s in dict2d:
            N_i = dict2d[s][s].shape[0]
            size += N_i
            i_low = i_top
            i_top = i_low + N_i
            idx_low[s] = i_low
            idx_top[s] = i_top
    if as_file:
        if as_type == "sparse":
            total_nnz = 0
            for s in dict2d:
                for t in dict2d[s]:
                    total_nnz += sparse_size(dict2d[s][t])
            indices = torch.tensor([], dtype=torch.int64).reshape(2, 0)
            values = torch.tensor([])
            matrix = SparseStorageTensor(as_file, indices, values, (size, size))
        elif as_type == "dense":
            dirpath = dirname(as_file)
            if not os.path.exists(dirpath):
                os.makedirs(dirpath)
            hstorage = torch.DoubleStorage.from_file(
                as_file, shared=True, size=size**2
            )
            matrix = torch.DoubleTensor(hstorage)
            matrix = matrix.reshape(size, size)
    else:
        if as_type == "dense":
            matrix = torch.empty((size, size), device=device)
            matrix[:, :] = 0
        else:
            indices = torch.tensor([], device=device, dtype=torch.int64).reshape(2, 0)
            values = torch.tensor([], device=device)
            matrix = torch.sparse_coo_tensor(indices, values, (size, size))
    for s in dict2d:
        for t in dict2d[s]:
            if not diagonal_blocks or (diagonal_blocks and s == t):
                i_low, i_top = idx_low[s], idx_top[s]
                j_low, j_top = idx_low[t], idx_top[t]
                if as_type == "dense":
                    if isinstance(dict2d[s][t], SparseSymmetricMatrix):
                        dict2d[s][t] = dict2d[s][t].to_dense()
                    if dict2d[s][t].layout == torch.sparse_coo:
                        dict2d[s][t] = to_dense(dict2d[s][t])
                    dict2d_st = dict2d[s][t]
                    if dict2d_st.device != device:
                        dict2d_st = dict2d_st.to(device=device)
                    matrix[i_low:i_top, j_low:j_top] = dict2d_st
                    if s != t:
                        matrix[j_low:j_top, i_low:i_top] = dict2d_st.T
                else:
                    if (
                        isinstance(dict2d[s][t], SparseSymmetricMatrix)
                        or dict2d[s][t].layout == torch.strided
                    ):
                        Hij_sparse = dict2d[s][t].to_sparse()
                    else:
                        Hij_sparse = dict2d[s][t]
                    if Hij_sparse.device != device:
                        Hij_sparse = Hij_sparse.to(device=device)
                    indices, values = Hij_sparse.indices(), Hij_sparse.values()
                    # Shift indices by offset determined through layer position
                    indices_shifted = indices.clone()
                    indices_shifted[0, :] += i_low
                    indices_shifted[1, :] += j_low
                    Hij = torch.sparse_coo_tensor(indices_shifted, values, (size, size))
                    matrix += Hij
                    if s != t and as_type != "symmetric":
                        matrix += Hij.t()
    if as_type == "sparse":
        matrix = matrix.coalesce()
    if as_type == "symmetric":
        matrix = SparseSymmetricMatrix(
            matrix._indices(), matrix._values(), matrix.shape, filter="triu"
        )
    return matrix


def replace_values(tensor: torch.Tensor, old_new_values: torch.Tensor):
    """Given 1d tensor and a 2d tensor with [old_value, new_value] pairs, replaces all occurrences of
    every old_value with the respective new_value.

    Example:
    --------
    tensor: torch.tensor([0, 1, 2, 1, 4, 2])
    old_new_vaues: torch.tensor([[1, 11], [2, 22]])

    => torch.tensor([0, 11, 22, 11, 4, 22])
    """
    assert len(tensor.shape) == 1
    assert len(old_new_values.shape) == 2 and old_new_values.shape[1] == 2
    mask = tensor == old_new_values[:, :1]
    return (1 - mask.sum(dim=0)) * tensor + (mask * old_new_values[:, 1:]).sum(dim=0)


def get_conv_output_shape(
    window: int, kernel: int, stride: int, padding: int, dilation: int
) -> int:
    """Given input window size, kernel size, stride, padding and dilation, computes the output window size of a convolution."""
    dilated_kernel = 1 + dilation * (kernel - 1)
    output_shape = int(np.floor((window + 2 * padding - dilated_kernel) / stride) + 1)
    return output_shape


def get_tconv_output_shape(
    window: int, kernel: int, stride: int, padding: int, dilation: int
) -> int:
    """Given input window size, kernel size, stride, padding and dilation, computes the output window size of a transposed convolution."""
    dilated_kernel = 1 + dilation * (kernel - 1)
    output_shape = (window - 1) * stride + dilated_kernel - 2 * padding
    return output_shape


def sparse_sum(indices1, values1, indices2, values2, shape):
    # Note: This function relies on the input matrices being coalesced.
    if indices1.shape[0] != indices2.shape[0]:
        raise ValueError(
            f"Size of indices1 and indices2 must be the same along dimension 0, but found {indices1.shape[0]} and {indices1.shape[0]}!"
        )

    if not indices1.shape[0] == 1:
        idx1_flat = flatten_multiindex(indices1.T, shape)
    else:
        idx1_flat = indices1.reshape(-1)
    idx1_flat = idx1_flat.type(torch.float64)

    if not indices2.shape[0] == 1:
        idx2_flat = flatten_multiindex(indices2.T, shape)
    else:
        idx2_flat = indices2.reshape(-1)
    idx2_flat = idx2_flat.type(torch.float64)

    device = idx1_flat.device
    if idx1_flat.device.type != "cpu":
        idx1_flat = idx1_flat.cpu()
    if idx2_flat.device.type != "cpu":
        idx2_flat = idx2_flat.cpu()

    idx1_flat = idx1_flat.numpy()
    idx2_flat = idx2_flat.numpy()
    idx, (union1, union2) = snp.merge(idx1_flat, idx2_flat, indices=True)
    n = len(idx)
    idx = torch.empty(indices1.shape[0], n, dtype=torch.int64, device=device)
    values = torch.empty(n, device=device)
    idx[:, union1] = indices1
    idx[:, union2] = indices2
    values[:] = 0
    values[union1] += values1
    values[union2] += values2
    return idx, values


def to_dense(sparse: torch.Tensor) -> torch.Tensor:
    assert sparse.layout == torch.sparse_coo
    indices, values = sparse._indices(), sparse._values()
    shape = sparse.shape
    idx_flat = flatten_multiindex(indices.T, shape)
    dense = torch.empty(sparse.numel(), dtype=sparse.dtype, device=sparse.device)
    dense[:] = 0
    dense[idx_flat] = values
    dense = dense.reshape(shape)
    return dense


def reshape(tensor: torch.Tensor, shape: Sequence[int]):
    """Reshapes the torch.Tensor or SparseSymmetricMatrix to the shape provided. Returns an object of the same type/layout."""
    shape = list(shape)
    shapes_match = np.prod(tensor.shape) == np.prod(shape)
    if not shapes_match:
        old_shape = ", ".join([str(x) for x in tensor.shape])
        new_shape = ", ".join([str(x) for x in shape])
        raise ValueError(
            f"New shape {new_shape} does not match old shape ({old_shape})!"
        )
    assert isinstance(tensor, torch.Tensor)
    if tensor.layout == torch.sparse_coo:
        idx, val = tensor.indices(), tensor.values()
        idx = reshape_multiindex(idx.T, tensor.shape, shape).T
        tensor = torch.sparse_coo_tensor(idx, val, shape).coalesce()
    else:
        tensor = tensor.reshape(shape)
    return tensor


def transpose(
    tensor: Union[torch.Tensor, SparseSymmetricMatrix],
    is_transposed=False,
    is_symmetric=False,
):
    if isinstance(tensor, torch.Tensor):
        tensor_T = tensor if (is_transposed or is_symmetric) else tensor.t()
        if tensor.layout == torch.sparse_coo:
            tensor_T = tensor_T.coalesce()
    elif isinstance(tensor, SparseSymmetricMatrix):
        tensor_T = tensor
    return tensor_T


def sparse_size(tensor: Union[torch.Tensor, SparseSymmetricMatrix]):
    if isinstance(tensor, torch.Tensor):
        if tensor.layout == torch.strided:
            return tensor.count_nonzero()
        elif tensor.layout == torch.sparse_coo:
            return len(tensor.values())
    elif isinstance(tensor, SparseSymmetricMatrix):
        return 2 * len(tensor.tril.values()) + len(tensor.diag.values())
    raise TypeError(f"Not an instance of torch.Tensor or SparseSymmetricMatrix")


def sizeof(dtype):
    return {
        torch.uint8: 1,
        torch.int8: 1,
        torch.int16: 2,
        torch.int32: 4,
        torch.int64: 8,
        torch.float16: 2,
        torch.float32: 4,
        torch.float64: 8,
    }[dtype]


@overload
def memory_usage(
    tensor: Union[torch.Tensor, SparseSymmetricMatrix],
    is_transposed=False,
    is_symmetric=False,
):
    ...


@overload
def memory_usage(
    tensor: Sequence[Union[torch.Tensor, SparseSymmetricMatrix]],
    is_transposed=False,
    is_symmetric=False,
):
    ...


def memory_usage(tensor, is_transposed=False, is_symmetric=False):
    if isinstance(tensor, torch.Tensor):
        if tensor.layout == torch.strided:
            return tensor.numel() * sizeof(tensor.dtype)
        elif tensor.layout == torch.sparse_coo:
            idx, val = tensor._indices(), tensor._values()
            return idx.numel() * sizeof(idx.dtype) + val.numel() * sizeof(val.dtype)
    elif isinstance(tensor, SparseSymmetricMatrix):
        return memory_usage(tensor.diag) + memory_usage(tensor.tril)
    elif isinstance(tensor, Sequence):
        return np.sum([memory_usage(t) for t in tensor])


def drop_zeros(tensor: torch.Tensor):
    """Removes zeros from a sparse tensor."""
    assert tensor.layout == torch.sparse_coo
    shape = tensor.shape
    indices, values = tensor._indices(), tensor._values()
    pos = values == 0
    indices = indices[:, pos]
    values = values[pos]
    return torch.sparse_coo_tensor(indices, values, shape).coalesce()
