from typing import Union
import warnings
import torch


def flatten_multiindex(
    indices: torch.Tensor, shape: Union[tuple, list]
) -> torch.Tensor:
    """Given a multi-index or a series of multi-indices and the shape of the corresponding tensor,
    returns a flattened series of multi-indices.

    Parameters:
    -----------
    indices: torch.Tensor
        A 1D or 2D tensor where each row corresponds to a multi-index.
    shape: tuple, list
        The shape of the reference tensor for the input multi-indices.
        The shape should have as many axes as the indices tensor has colums.

    Returns:
    --------
    Assuming the indices tensor has n rows, will return a 1D torch.Tensor with n entries.

    Exceptions:
    -----------
    ValueError
    """
    if not len(indices.shape) in [1, 2]:
        raise ValueError(
            f"Indices tensor must be 1D or 2D, but got a {len(indices.shape)}D tensor!"
        )

    if len(indices.shape) == 1:
        indices = indices.reshape([1, -1])

    if indices.shape[1] != len(shape):
        raise ValueError(
            f"Number of columns in indices must match length of shape, but got {indices.shape[1]} and {len(shape)}!"
        )

    factors = [1]
    for x in shape[:0:-1]:
        prev = factors[0]
        next = x * prev
        factors.insert(0, next)

    factors = torch.LongTensor(factors).reshape(1, -1)
    if indices.device != factors.device:
        factors = factors.to(indices.device)
    result = torch.sum(indices * factors, dim=1)
    return result


def expand_multiindex(indices: torch.Tensor, shape: Union[tuple, list]) -> torch.Tensor:
    """Given a series of flat indices, returns a multiindex version matching the shape provided.

    Parameters:
    -----------
    indices: torch.Tensor
        A 1D or 2D tensor where each row corresponds to a multi-index.
    shape: tuple, list
        The shape of the reference tensor.

    Returns:
    --------
    Assuming the indices tensor has n entries and len(shape)=m, will return a 2D torch.Tensor with n rows and m columns.

    Exceptions:
    -----------
    UserWarning, ValueError
    """
    if len(indices.shape) > 1:
        warnings.warn(
            f"You provided an indices tensor of shape {indices.shape}, proceeding with flattened tensor!",
            UserWarning,
        )
        indices = indices.flatten()

    indices_oor = indices > torch.prod(
        torch.tensor(shape, dtype=torch.int64, device=indices.device)
    )
    if torch.any(indices_oor):
        first_oor = indices[indices_oor][0]
        count_oor = torch.sum(indices_oor)
        raise IndexError(
            f"Index {first_oor} and {count_oor - 1} other(s) out of range!"
        )

    n, m = len(indices), len(shape)
    indices = indices.reshape(1, -1)
    result = torch.empty(n, m, dtype=torch.int64, device=indices.device)
    for i, dim in enumerate(reversed(shape)):
        i += 1
        result[:, -i] = indices % dim
        indices = torch.div(indices, dim, rounding_mode="trunc")

    return result


def reshape_multiindex(
    indices: torch.Tensor, shape1: Union[tuple, list], shape2: Union[tuple, list]
):
    """Given a series of flat indices and the shape of the corresponding tensor,
    returns a multiindex version matching the shape provided.

    Parameters:
    -----------
    indices: torch.Tensor
        A 1D tensor where each value corresponds to a flat index.
    shape1: tuple, list
        The shape of the reference tensor for the input multi-indices.
    shape2: tuple, list
        The shape of the reference tensor for the output multi-indices.

    Returns:
    --------
    Assuming the indices tensor has n entries and len(shape2)=m, will return a 2D torch.Tensor with n rows and m columns.

    Exceptions:
    -----------
    UserWarning, ValueError
    """
    flat_indices = flatten_multiindex(indices, shape1)
    reshape_indices = expand_multiindex(flat_indices, shape2)
    return reshape_indices
