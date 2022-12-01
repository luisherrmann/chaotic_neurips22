from typing import Callable, List, Union
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.autograd as autograd

from o2grad.modules.o2layer.o2layer import O2Layer
from o2grad.sparse import SparseStorageTensor


def get_hessian(
    module: Union[nn.Module, List[nn.Parameter]],
    input: torch.Tensor = None,
    target: torch.Tensor = None,
    criterion: Callable[[torch.Tensor, torch.Tensor], torch.Tensor] = None,
    start: int = None,
    stop: int = None,
    as_file: str = None,
    progress: bool = False,
    diagonal: bool = False,
    diagonal_blocks: bool = False,
    sparse: bool = False,
):
    """Returns the Hessian of the torch module provided.
    If start and stop indices are specified, returns hessian[start:stop, start:stop] assuming hessian to be the full hessian.

    Parameters
    ---------
    input: torch.Tensor, optional
        The input to the module for which to compute the Hessian. Requires passing `target` and `criterion`.
    target: torch.Tensor, optional
        The target to the module for which to compute the Hessian. Requires passing `input` and `criterion`.
    criterion: torch.Tensor, optional
        The loss criterion from which to compute the Hessian using parameters `input` and `target`.
    module: Union[torch.nn.Module, List[nn.Parameter]]
        Either a list of parameters or a torch module for which to compute the Hessian.
    start: int, optional
        Start index
    stop: int, optional
        Stop index
    as_file: str, optional
        If specified, will store the returned matrix in a binary file of the given name on disk.
    progress: bool, optional
        If specified, will display the progress of calculating the Hessian using a tqdm progress bar.
    diagonal: bool, optional
        If specified, will only calculate the diagonal elements of the Hessian.
    diagonal_blocks: bool, optional
        If specified, will only calculate the diagonal blocks of the Hessian,
        i.e. only terms dL2dpdq where p, q are parameters from the same layer.
    sparse: bool, optional
        If specified, will save the Hessian as a sparse tensor in CSR representation.
    """

    is_nn_module = False
    if isinstance(module, list) and all([isinstance(p, nn.Parameter) for p in module]):
        parameters = module
    elif isinstance(module, nn.Module):
        is_nn_module = True
        parameters = module.parameters()
    else:
        raise TypeError(f"Expected torch.nn.Module, but got {type(module)}!")

    if input is not None and target is not None and criterion is not None:
        output = module(input)
        loss = criterion(output, target)
        loss.backward(create_graph=True)

    grad_parameters = [p for p in parameters if p.requires_grad]
    n = sum([p.numel() for p in grad_parameters])
    grads = torch.cat([p.grad.reshape(-1) for p in grad_parameters])

    diagonal_blocks = is_nn_module and diagonal_blocks
    if diagonal:
        grad_parameters = torch.cat([p.reshape(-1) for p in grad_parameters])
    if diagonal_blocks:
        grad_parameters = []
        block_limits = [0]
        for m in module.children():
            if isinstance(m, O2Layer):
                m = m.module
            has_parameters = len([*m.parameters()]) > 0
            if has_parameters:
                block_params = [p for p in m.parameters() if p.requires_grad]
                grad_parameters.append(block_params)
                mnumel = [p.numel() for p in m.parameters() if p.requires_grad]
                limit = block_limits[-1] + sum(mnumel)
                block_limits.append(limit)

    start = start or 0
    if bool(stop):
        if stop <= start:
            raise ValueError(
                f"Expects parameters start > stop, but got start: {start} and stop: {stop}!"
            )
    else:
        stop = n

    num_rows = stop - start
    if as_file:
        if sparse:
            indices = torch.LongTensor([]).reshape(2, 0)
            values = torch.FloatTensor([])
            hessian = SparseStorageTensor(as_file, indices, values, (num_rows, n))
        else:
            hstorage = torch.DoubleStorage.from_file(
                as_file, shared=True, size=num_rows * n
            )
            hessian = torch.DoubleTensor(hstorage)
            hessian = hessian.reshape(num_rows, n)
    else:
        if sparse:
            indices = torch.LongTensor([]).reshape(2, 0)
            values = torch.FloatTensor([])
            hessian = torch.sparse_coo_tensor(indices, values, (num_rows, n))
        else:
            hessian = torch.empty(num_rows, n, device="cpu")
    if diagonal_blocks and not sparse:
        hessian[:, :] = 0

    row_iter = range(num_rows)
    block = 0
    if progress:
        row_iter = tqdm(row_iter)
    for i in row_iter:
        if diagonal:
            grads_2nd = autograd.grad(
                grads[start + i],
                grad_parameters[start + i],
                create_graph=True,
                allow_unused=True,
            )
            dL2dpi2 = grads_2nd.cpu().detach()
            if sparse:
                indices = torch.tensor(
                    [start + i, start + i], dtype=torch.int64
                ).reshape(2, 1)
                dL2dpi2 = torch.sparse_coo_tensor(indices, values, hessian.shape)
                hessian = hessian + dL2dpi2
            else:
                hessian[i, i] = dL2dpi2
        elif diagonal_blocks:
            block = block if i < block_limits[block + 1] else block + 1
            low, high = block_limits[block], block_limits[block + 1]
            grads_2nd = autograd.grad(
                grads[start + i],
                grad_parameters[block],
                create_graph=True,
                allow_unused=True,
            )
            grads_2nd = torch.cat([g.reshape(-1) for g in grads_2nd]).reshape([1, -1])
            dL2dpidp = grads_2nd.cpu().detach()
            if sparse:
                dL2dpidp = dL2dpidp.to_sparse()
                indices, values = dL2dpidp.indices(), dL2dpidp.values()
                indices_shifted = indices.clone()
                indices_shifted[0, :] += i
                indices_shifted[1, :] += low
                dL2dpidp = torch.sparse_coo_tensor(
                    indices_shifted, values, hessian.shape
                )
                hessian = hessian + dL2dpidp
            else:
                hessian[i, low:high] = dL2dpidp
        else:
            grads_2nd = autograd.grad(
                grads[start + i], grad_parameters, create_graph=True, allow_unused=True
            )
            grads_2nd = torch.cat([g.reshape(-1) for g in grads_2nd]).reshape([1, -1])
            dL2dpidp = grads_2nd.cpu().detach()
            if sparse:
                dL2dpidp = dL2dpidp.to_sparse()
                indices, values = dL2dpidp.indices(), dL2dpidp.values()
                indices_shifted = indices.clone()
                indices_shifted[0, :] += i
                dL2dpidp = torch.sparse_coo_tensor(
                    indices_shifted, values, hessian.shape
                )
                hessian = hessian + dL2dpidp
            else:
                hessian[i, :] = dL2dpidp
    if sparse:
        hessian = hessian.coalesce()
    return hessian
