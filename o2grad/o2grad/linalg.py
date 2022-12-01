from typing import Union, Sequence
import torch
from torch_sparse import spspmm

from .utils import matrix_from_dict2d, replace_values, reshape, transpose
from .multiidx import flatten_multiindex
from .sparse import SparseSymmetricMatrix


def twin_matmul_mixed(
    A: torch.Tensor,
    B: Union[torch.Tensor, SparseSymmetricMatrix],
    is_symmetric1=False,
    is_symmetric2=False,
) -> Union[torch.Tensor, SparseSymmetricMatrix]:
    """For tensors A, B as inputs, calculates the matrix product (AT)BA, where AT is the transposed matrix A.

    Parameters:
    -----------
    A: torch.Tensor
        A strided/sparse 2D tensor to be multiplied with tensor2.
    B: torch.Tensor, SparseSymmetricMatrix
        Either a trided/sparse 2D tensor or a SparseSymmetricMatrix.
    is_symmetric2: bool, optional
        If set, will assume that B is a symmetric matrix (which saves computation time).

    Returns:
    --------
    The product (AT)BA, which will be of the following type/layout:
    1. torch.Tensor, strided
        If any of the matrices A or B is dense
    2. torch.Tensor, sparse
        If A and B are both tensors with sparse layout
    3. SparseSymmetricMatrix
        If A is sparse and B is a SparseSymmetricMatrix
    """
    assert len(A.shape) == 2
    assert len(B.shape) == 2
    assert B.shape[1] == A.shape[0]
    if isinstance(B, SparseSymmetricMatrix):
        if A.layout == torch.sparse_coo:
            A = A.coalesce()
            AT = A if is_symmetric1 else A.t().coalesce()
            m, n = A.shape
            BA = B.matmul(A)
            ATBA_indices, ATBA_values = spspmm(
                AT.indices(), AT.values(), BA.indices(), BA.values(), n, m, n
            )
            del AT, BA
            ATBA = SparseSymmetricMatrix(ATBA_indices, ATBA_values, (n, n))
            del ATBA_indices, ATBA_values
        else:
            ATBA = torch.matmul(A.T, B.matmul(A))
    elif B.layout == torch.sparse_coo:
        if A.layout == torch.sparse_coo:
            AT = A if is_symmetric1 else A.t().coalesce()
            m, n = A.shape
            BA_indices, BA_values = spspmm(
                B.indices(), B.values(), A.indices(), A.values(), m, m, n
            )
            ATBA_indices, ATBA_values = spspmm(
                AT.indices(), AT.values(), BA_indices, BA_values, n, m, n
            )
            del AT, BA_indices, BA_values
            ATBA = torch.sparse_coo_tensor(ATBA_indices, ATBA_values, (n, n))
            del ATBA_indices, ATBA_values
        else:
            ATBA = torch.matmul(A.T, torch.matmul(B, A))
    elif B.layout == torch.strided:
        if A.layout == torch.sparse_coo:
            A_T = A if is_symmetric1 else A.t().coalesce()
            B_T = B if is_symmetric2 else B.T
            ATBA = torch.matmul(A_T, torch.matmul(A_T, B_T).T)
            del A_T, B_T
        else:
            ATBA = torch.einsum("ij,ik->jk", A, torch.einsum("ij,jk->ik", B, A))
    return ATBA


def matmul_1d_3d_mixed(A: torch.Tensor, B: torch.Tensor, is_symmetric=False):
    """For 1D tensor A and 3D tensor B, computes the dot product AB.
    Equivalent to (AB)_jk = A_i * B_ijk

    Parameters:
    -----------
    A: torch.Tensor
        A strided/sparse 1D tensor
    B: torch.Tensor
        A strided/sparse 2D or 3D tensor. If a 2D tensors is provided, will assume dims 1 and 2 are compressed into dim 1.
        To uncompress dimension 1 after matmul, provide respective shape.
    is_symmetric: torch.Tensor
        If True, implies symmetry of B, with symmetry meaning B_ijk = B_ikj

    Returns:
    --------
    The product AB, which will be of the following type/layout:
    1. torch.Tensor, strided
        If any of the tensors A or B are dense
    2. torch.Tensor, sparse
        If both A and B are sparse
    3. SparseSymmetricMatrix
        If both A and B are sparse and additionally, is_symmetric is set to True
    """
    assert len(A.shape) == 1
    assert len(B.shape) == 3
    assert A.shape[0] == B.shape[0]
    m, n, p = B.shape
    if is_symmetric:
        assert n == p
    if B.layout == torch.sparse_coo:
        idx, val = B.indices(), B.values()
        idx_ = torch.empty(2, len(val))
        idx_[0, :] = idx[0, :]
        idx_[1, :] = flatten_multiindex(idx[1:, :].T, (n, p)).T
        B = torch.sparse_coo_tensor(idx_, val, (m, n * p)).coalesce()
    else:
        B = B.reshape(m, -1)
    result = matmul_1d_2d_mixed(A, B)
    result = reshape(result, (n, p))
    if is_symmetric:
        result = SparseSymmetricMatrix(result.indices(), result.values(), result.shape)
    return result


def matmul_1d_2d_mixed(
    A: torch.Tensor, B: torch.Tensor
) -> Union[torch.Tensor, SparseSymmetricMatrix]:
    """For 1D tensor A and 3D tensor B, computes the dot product AB.
    Equivalent to (AB)_j = A_i * B_ij

    Parameters:
    -----------
    A: torch.Tensor
        A strided/sparse 1D tensor
    B: torch.Tensor
        A strided/sparse 2D tensor

    Returns:
    --------
    The product AB, which will be of the following type/layout:
    1. torch.Tensor, strided
        If any of the tensors A or B are dense
    2. torch.Tensor, sparse
        If both A and B are sparse
    """
    assert len(A.shape) == 1
    assert len(B.shape) == 2
    assert A.shape[0] == B.shape[0]
    m, n = B.shape
    if B.layout == torch.sparse_coo:
        if A.layout == torch.sparse_coo:
            A_idx, A_val = A.indices(), A.values()
            B_idx, B_val = B.indices(), B.values()
            if B_idx.numel() == 0:
                idx = torch.tensor([], dtype=torch.int64, device=A.device).reshape(1, 0)
                val = torch.tensor([], device=A.device)
            else:
                count = A_val.numel()
                A_idx = torch.empty([2, count], dtype=torch.int64, device=A.device)
                A_idx[0, :] = torch.arange(count)
                A_idx[1, :] = torch.arange(count)
                idx, val = spspmm(A_idx, A_val, B_idx, B_val, m, m, n)
                if not val.numel() == 0:
                    intermediate = torch.sparse_coo_tensor(idx, val, (m, n))
                    AB = torch.sparse.sum(intermediate, dim=[0])
                    del intermediate
                    idx, val = AB.indices(), AB.values()
            AB = torch.sparse_coo_tensor(idx, val, (n,)).coalesce()
        else:
            B_T = B.t().coalesce()
            AB = torch.matmul(B_T, A)
            # AB = AB.to_sparse() TODO: Include this optimization in the code
    elif B.layout == torch.strided:
        AB = torch.matmul(A, B)
    return AB


def matmul_mixed(
    A: Union[torch.Tensor, SparseSymmetricMatrix],
    B: Union[torch.Tensor, SparseSymmetricMatrix],
    is_symmetric1=False,
    is_symmetric2=False,
    is_transposed1=False,
) -> torch.Tensor:
    """Calculates the matrix dot product between A, B."""
    if isinstance(A, SparseSymmetricMatrix):
        AB = A.matmul(B)
    elif isinstance(B, SparseSymmetricMatrix):
        AT = transpose(A, is_transposed=is_transposed1, is_symmetric=is_symmetric1)
        AB = transpose(B.matmul(AT))
    elif isinstance(A, torch.Tensor) and isinstance(B, torch.Tensor):
        if A.layout == torch.sparse_coo:
            A = A.t().coalesce() if (is_transposed1 and not is_symmetric1) else A
            if B.layout == torch.sparse_coo:
                m, k = A.shape
                _, n = B.shape
                indices, values = spspmm(
                    A._indices(), A._values(), B._indices(), B._values(), m, k, n
                )
                AB = torch.sparse_coo_tensor(indices, values, (m, n))
            else:
                AB = torch.matmul(A, B)
        elif A.layout == torch.strided:
            if B.layout == torch.sparse_coo:
                AT = A if (is_transposed1 or is_symmetric1) else A.T
                BT = B if is_symmetric2 else B.t().coalesce()
                AB = torch.matmul(BT, AT).T
            else:
                A = A.T if (is_transposed1 and not is_symmetric1) else A
                AB = torch.matmul(A, B)
    else:
        raise ValueError(
            f"Expects inputs of types torch.Tensor or SparseSymmetricMatrix, but got {type(A)} and {type(B)}!"
        )
    if AB.layout == torch.sparse_coo:
        AB = AB.coalesce()
    return AB


def eigs_from_dict2d(
    order: Sequence[str], dict2d, layout=torch.strided, sorted=False, device="cpu"
):
    as_type = "dense" if layout == torch.strided else "sparse"
    size = sum([dict2d[s][s].shape[0] for s in dict2d.keys()])
    vecs = {s: {} for s in dict2d.keys()}
    eigs = torch.empty(size, device="cpu")
    eigs[:] = 0
    low = 0
    is_diagonal = all(
        [(len(dict2d[s]) == 1 and s in dict2d[s]) for s in order if s in dict2d]
    )
    if is_diagonal:
        for s in order:
            if s in dict2d:
                matrix = dict2d[s][s]
                if (
                    isinstance(matrix, torch.Tensor)
                    and matrix.layout == torch.sparse_coo
                ):
                    matrix = matrix.to_dense()
                elif isinstance(matrix, SparseSymmetricMatrix):
                    matrix = matrix.to_dense()
                W, Q = torch.linalg.eigh(matrix)
                vecs[s][s] = Q
                high = low + len(W)
                eigs[low:high] = W
                low = high
        vecs = matrix_from_dict2d(order, vecs, as_type=as_type, device=device)
        if sorted:
            eigs, idx = torch.sort(eigs)
            if layout == torch.sparse_coo:
                old_new = torch.empty(len(idx), 2, dtype=torch.int64)
                old_new[:, 0] = idx
                old_new[:, 1] = torch.arange(len(idx), dtype=torch.int64)
                vecs_idx = vecs.indices()
                vecs_idx[1, :] = replace_values(vecs_idx[1, :], old_new)
                vecs = torch.sparse_coo_tensor(
                    vecs_idx, vecs.values(), vecs.shape
                ).coalesce()
            else:
                vecs = vecs[:, idx]
    else:
        matrix = matrix_from_dict2d(order, dict2d, as_type=as_type, device=device)
        eigs, vecs = torch.linalg.eigh(matrix)
    return eigs, vecs


def sum_mixed(
    A: Union[torch.Tensor, SparseSymmetricMatrix],
    B: Union[torch.Tensor, SparseSymmetricMatrix],
):
    """For tensors/SparseSymmetricMatrix A and B, calculates their sum.

    Parameters:
    -----------
    A: torch.Tensor
        A strided/sparse 1D tensor
    B: torch.Tensor
        A strided/sparse 2D tensor

    Returns:
    --------
    The sum A+B, which will be of the following type/layout:
    1. torch.Tensor, strided
        If any of the tensors A or B are dense
    2. torch.Tensor, sparse
        If both A and B are sparse or one of them is a SparseSymmetricMatrix
    3. SparseSymmmetricMatrix
        If both A and B are instances of SparseSymmetricMatrix
    """
    if isinstance(A, SparseSymmetricMatrix):
        result = A + B
    elif isinstance(B, SparseSymmetricMatrix):
        result = B + A
    elif isinstance(A, torch.Tensor) and isinstance(B, torch.Tensor):
        if A.layout == torch.sparse_coo and B.layout == torch.sparse_coo:
            result = A + B
        elif A.layout == torch.sparse_coo:
            result = B + A
        elif B.layout == torch.sparse_coo:
            result = A + B
        else:
            result = A + B
    else:
        raise ValueError(
            f"Expects inputs of types torch.Tensor or SparseSymmetricMatrix, but got {type(A)} and {type(B)}!"
        )
    return result
