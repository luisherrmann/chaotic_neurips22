import itertools
from typing import Union, Iterable, Sequence
import os
import shutil
import json
import torch
from torch_sparse import spspmm


def sparse_eye(n, dtype=torch.float32, device="cpu"):
    idx = torch.empty([2, n], dtype=dtype, device=device)
    idx[0, :] = torch.arange(n, dtype=torch.int64, device=device)
    idx[1, :] = idx[0, :]
    val = torch.empty([n], dtype=dtype, device=device)
    val[:] = 1
    return torch.sparse_coo_tensor(idx, val, (n, n)).coalesce()


class SparseSymmetricMatrix:
    def __init__(
        self,
        indices: torch.LongTensor,
        values: torch.Tensor,
        shape: Iterable[int],
        filter="tril",
        variant=1,
    ):
        super(type(self), self).__init__()
        assert len(shape) == 2
        m, n = shape
        assert m == n
        assert indices.device == values.device
        self.device = indices.device
        diag_pos = indices[0, :] == indices[1, :]
        self.diag = torch.sparse_coo_tensor(
            indices[:, diag_pos], values[diag_pos], shape
        ).coalesce()
        if filter == "tril":
            lower_pos = indices[0, :] > indices[1, :]
            self.tril = torch.sparse_coo_tensor(
                indices[:, lower_pos], values[lower_pos], shape
            ).coalesce()
        elif filter == "triu":
            upper_pos = indices[0, :] < indices[1, :]
            self.tril = (
                torch.sparse_coo_tensor(indices[:, upper_pos], values[upper_pos], shape)
                .t()
                .coalesce()
            )
        self.shape = shape
        self.variant = variant

    @classmethod
    def from_diag_tril_shape(
        cls,
        diag: torch.sparse_coo_tensor,
        lower: torch.sparse_coo_tensor,
        shape: Sequence[int],
        variant=1,
    ):
        ssm = cls.__new__(cls)
        super(SparseSymmetricMatrix, ssm).__init__()
        assert diag.layout == torch.sparse_coo and diag.is_coalesced()
        assert lower.layout == torch.sparse_coo and lower.is_coalesced()
        ssm.diag = diag
        ssm.tril = lower
        ssm.shape = shape
        ssm.variant = variant
        return ssm

    def to(self, device) -> None:
        dummy_indices = torch.tensor([], dtype=torch.int64, device=device).reshape(
            (len(self.shape), 0)
        )
        dummy_values = torch.tensor([], device=device)
        sparse_matrix = SparseSymmetricMatrix(
            dummy_indices, dummy_values, self.shape, self.variant
        )
        sparse_matrix.diag = self.diag.to(device=device)
        sparse_matrix.tril = self.tril.to(device=device)
        return sparse_matrix

    def to_sparse(self) -> torch.Tensor:
        sparse = (self.tril + self.diag + self.tril.t()).coalesce()
        return sparse

    def to_dense(self) -> torch.Tensor:
        dense = (self.tril + self.diag + self.tril.t()).to_dense()
        return dense

    def clone(self) -> torch.Tensor:
        return SparseSymmetricMatrix.from_diag_tril_shape(
            self.diag, self.tril, self.shape, self.variant
        )

    def nnz(self) -> int:
        return len(self.diag.values()) + len(self.tril.values())

    def __repr__(self) -> str:
        tril_lines = repr(self.tril).splitlines()
        diag_lines = repr(self.diag).splitlines()
        str = f"""\rSparseSymmetricMatrix(
        \r\ttril={tril_lines[0]}
        \r\t     {tril_lines[1]}
        \r\t     {tril_lines[2]}
        \r\t     {tril_lines[3]},
        \r\tdiag={diag_lines[0]}
        \r\t     {diag_lines[1]}
        \r\t     {diag_lines[2]}
        \r\t     {tril_lines[3]},
        \r\tsize={self.shape},
        \r\tnnz={self.nnz()}
        \r)"""
        return str

    def __del__(self) -> None:
        del self.diag, self.tril

    def __add__(
        self, other: Union[torch.Tensor, "SparseSymmetricMatrix"]
    ) -> torch.Tensor:
        if isinstance(other, SparseSymmetricMatrix):
            if self.shape != other.shape:
                raise ValueError(
                    f"Matrices must have matching dimensions, but got {self.shape} and {other.shape}!"
                )
            result = self.diag + other.diag
            result += self.tril + other.tril
            result = result.coalesce()
            result = SparseSymmetricMatrix(
                result.indices(), result.values(), result.shape
            )
        elif isinstance(other, torch.Tensor):
            if other.layout == torch.sparse_coo:
                result = (other + self.diag) + self.tril
                if self.variant == 0:
                    result += self.tril.t().coalesce()
                elif self.variant == 1:
                    self.tril = self.tril.t().coalesce()
                    result += self.tril
                    self.tril = self.tril.t().coalesce()
                result = result.coalesce()
            else:
                result = other + self.tril + self.diag + self.tril.t()

        else:
            raise TypeError(
                f"Other thensor must be of type {SparseSymmetricMatrix} or {torch.Tensor}!"
            )

        return result

    def __sub__(
        self, other: Union[torch.Tensor, "SparseSymmetricMatrix"]
    ) -> torch.Tensor:
        if isinstance(other, SparseSymmetricMatrix):
            other.diag *= -1
            other.tril *= -1
            result = self.__add__(other)
            other.diag *= -1
            other.tril *= -1
        elif isinstance(other, torch.Tensor):
            sub_other = -other
            result = self.__add__(sub_other)
        return result

    def matmul(
        self, other: Union[torch.Tensor, "SparseSymmetricMatrix"]
    ) -> torch.Tensor:
        m, n = self.shape
        n_, p = other.shape
        result_shape = [m, p]
        assert n == n_
        if isinstance(other, SparseSymmetricMatrix):
            # Notation AB   = (L + D + LT) (R + E + RT) = LR + LE + L(RT) + DR + DE + D(RT) + (LT)R + (LT)E + (LT)(RT)
            result = None
            is_L_transposed = False
            is_R_transposed = False
            for LL, RR in itertools.product(
                [("tril", False), ("diag", False), ("tril", True)],
                [("tril", False), ("diag", False), ("tril", True)],
            ):
                L_name, L_transpose = LL
                R_name, R_transpose = RR
                L = getattr(self, L_name)
                R = getattr(other, R_name)
                if self.variant == 0:
                    L = L.t().coalesce() if L_transpose else L
                    R = R.t().coalesce() if R_transpose else R
                elif self.variant == 1:
                    if L_transpose != is_L_transposed:
                        L = L.t().coalesce()
                        setattr(self, L_name, L)
                        is_L_transposed = not is_L_transposed
                    if R_transpose != is_R_transposed:
                        R = R.t().coalesce()
                        setattr(other, R_name, R)
                        is_R_transposed = not is_R_transposed
                LR_indices, LR_values = spspmm(
                    L.indices(), L.values(), R.indices(), R.values(), m, n, p
                )
                if result is None:
                    result = torch.sparse_coo_tensor(
                        LR_indices, LR_values, result_shape
                    )
                else:
                    result += torch.sparse_coo_tensor(
                        LR_indices, LR_values, result_shape
                    )
            if self.variant == 1:
                self.tril = self.tril.t().coalesce()
                other.tril = other.tril.t().coalesce()
        elif isinstance(other, torch.Tensor):
            if other.layout == torch.sparse_coo:
                # Notation = AB = (L + D + LT)B = LB + DB + LTB
                LB_indices, LB_values = spspmm(
                    self.tril.indices(),
                    self.tril.values(),
                    other.indices(),
                    other.values(),
                    m,
                    n,
                    p,
                )
                result = torch.sparse_coo_tensor(LB_indices, LB_values, result_shape)
                LD_indices, LD_values = spspmm(
                    self.diag.indices(),
                    self.diag.values(),
                    other.indices(),
                    other.values(),
                    m,
                    n,
                    p,
                )
                result += torch.sparse_coo_tensor(LD_indices, LD_values, result_shape)

                if self.variant == 0:
                    LT = self.tril.t().coalesce()
                    LTB_indices, LTB_values = spspmm(
                        LT.indices(),
                        LT.values(),
                        other.indices(),
                        other.values(),
                        m,
                        n,
                        p,
                    )
                elif self.variant == 1:
                    self.tril = (
                        self.tril.t().coalesce()
                    )  # TODO: Make a test that this actually reduces memory usage
                    LTB_indices, LTB_values = spspmm(
                        self.tril.indices(),
                        self.tril.values(),
                        other.indices(),
                        other.values(),
                        m,
                        n,
                        p,
                    )
                    self.tril = self.tril.t().coalesce()
                result += torch.sparse_coo_tensor(LTB_indices, LTB_values, result_shape)
                result = result.coalesce()
            else:
                LT = self.tril.t().coalesce()
                LB = torch.matmul(self.tril, other)
                result = LB
                LD = torch.matmul(self.diag, other)
                result += LD
                LTB = torch.matmul(LT, other)
                result += LTB

        return result


DTYPE2STYPE = {
    str(torch.int): torch.IntStorage,
    str(torch.long): torch.LongStorage,
    str(torch.float32): torch.FloatStorage,
    str(torch.float64): torch.DoubleStorage,
}
STYPE2TTYPE = {
    torch.IntStorage: torch.IntTensor,
    torch.LongStorage: torch.LongTensor,
    torch.FloatStorage: torch.FloatTensor,
    torch.DoubleStorage: torch.DoubleTensor,
}


class SparseStorageTensor:
    def __init__(
        self,
        filename: str,
        indices: torch.LongTensor,
        values: torch.Tensor,
        shape: Iterable[int],
        shared: bool = True,
        dtype=torch.float32,
    ):
        self.filename = filename
        self.ifile = None
        self.vfile = None
        self.istorage = None
        self.vstorage = None
        self.indices = None
        self.values = None
        self.shared = None
        self.should_persist = False
        self.shape = shape
        self.layout = torch.sparse_coo
        # Create folder to store indices, values
        if not os.path.exists(filename):
            os.makedirs(filename)
        # Open file and create store objects
        tensor = torch.sparse_coo_tensor(indices, values, size=shape)
        if not tensor.is_coalesced():
            tensor = tensor.coalesce()
            indices, values = tensor.indices(), tensor.values()
        metadata = dict(
            inumel=indices.numel(),
            vnumel=values.numel(),
            idtype=str(indices.dtype),
            vdtype=str(values.dtype),
            shared=shared,
            shape=shape,
        )
        self._setup_storage(filename, **metadata)
        # Indices stored transposed to allow appending of new indices and values
        self.indices[:] = indices.T.reshape(-1)
        self.values[:] = values.reshape(-1)
        metafile = os.path.join(filename, ".meta.json")
        with open(metafile, "w+") as f:
            json.dump(metadata, f)

    def _setup_storage(
        self, filename, inumel, vnumel, idtype, vdtype, shared, *args, **kwargs
    ):
        self.ifile = os.path.join(filename, "indices.pth")
        self.vfile = os.path.join(filename, "values.pth")
        self.istorage = DTYPE2STYPE[str(idtype)].from_file(
            self.ifile, shared=shared, size=inumel
        )
        self.vstorage = DTYPE2STYPE[str(vdtype)].from_file(
            self.vfile, shared=shared, size=vnumel
        )
        self.indices = STYPE2TTYPE[type(self.istorage)](self.istorage)
        self.values = STYPE2TTYPE[type(self.vstorage)](self.vstorage)
        self.shared = shared
        self.istorage.share_memory_()
        self.vstorage.share_memory_()

    @classmethod
    def from_file(cls, filename: str) -> "SparseStorageTensor":
        sst = cls.__new__(cls)
        metafile = os.path.join(filename, ".meta.json")
        with open(metafile, "r") as f:
            metadata = json.load(f)
        sst.ifile = os.path.join(filename, "indices.pth")
        sst.vfile = os.path.join(filename, "values.pth")
        sst._setup_storage(filename, **metadata)
        sst.shape = metadata["shape"]
        sst.layout = torch.sparse_coo
        return sst

    def to_dense(self) -> torch.Tensor:
        return self.to_sparse().to_dense()

    def to_sparse(self) -> torch.Tensor:
        N, ndims = len(self.values), len(self.shape)
        return torch.sparse_coo_tensor(
            self.indices.reshape(N, ndims).T, self.values, self.shape
        )

    def coalesce(self) -> "SparseStorageTensor":
        tensor = self.to_sparse()
        tensor = tensor.coalesce()
        return SparseStorageTensor(
            self.filename, tensor.indices(), tensor.values(), self.shape
        )

    def coalesce_(self) -> None:
        """Coalesces tensor in-place"""
        tensor = self.to_sparse()
        tensor = tensor.coalesce()
        self.indices, self.values = tensor.indices().T, tensor.values()

    def add(self, other: torch.Tensor, coalesce: bool = False) -> None:
        """Adds tensor in-place"""
        if isinstance(other, torch.Tensor):
            if other.layout == torch.strided:
                other = other.to_sparse()
            tensor = self.to_sparse()
            result = tensor + other
            result = result.coalesce()
            indices, values = result.indices(), result.values()
            isize, vsize = indices.numel(), values.numel()
            self.istorage = DTYPE2STYPE[str(self.indices.dtype)].from_file(
                self.ifile, shared=self.shared, size=isize
            )
            self.vstorage = DTYPE2STYPE[str(self.values.dtype)].from_file(
                self.vfile, shared=self.shared, size=vsize
            )
            self.indices = STYPE2TTYPE[type(self.istorage)](self.istorage)
            self.values = STYPE2TTYPE[type(self.vstorage)](self.vstorage)
            # Entire array needs to be rewritten to file.
            self.indices = indices.T.reshape(-1)
            self.values = values.reshape(-1)

    def __add__(self, other: torch.Tensor) -> "SparseStorageTensor":
        if isinstance(other, torch.Tensor):
            if other.layout == torch.strided:
                other = other.to_sparse()
            tensor = self.to_sparse()
            result = tensor + other
            result = result.coalesce()
            indices, values = result.indices(), result.values()
            result = SparseStorageTensor(
                self.filename, indices, values, shape=self.shape
            )
        else:
            raise ValueError(
                f"Expected operand of type torch.Tensor, but got {type(other)}!"
            )
        return result

    def sub(self, other: torch.Tensor, coalesce: bool = False) -> None:
        """Subtracts tensor in-place"""
        self.add(-other, coalesce=coalesce)

    def __sub__(self, other: torch.Tensor) -> "SparseStorageTensor":
        return self.__add__(-other)

    def persist(self) -> None:
        self.should_persist = True

    def unpersist(self) -> None:
        self.should_persist = False

    def __del__(self) -> None:
        del self.istorage, self.vstorage, self.indices, self.values
        if not self.should_persist:
            shutil.rmtree(self.filename)
