import warnings
from typing import Callable, Sequence, Union, List, Dict
from tqdm import tqdm
from collections import OrderedDict
from addict import Dict as AttrDict
import torch
import torch.nn as nn

from o2grad.linalg import eigs_from_dict2d
from o2grad.modules.o2module import O2Module
from o2grad.modules.o2loss import O2Loss
from o2grad.modules.o2parametric.o2parametric import O2ParametricLayer
from o2grad.backprop.o2backprop import backprop_step
from o2grad.sparse import SparseSymmetricMatrix
from o2grad.recursive import (
    add_callbacks,
    add_default_callbacks,
    get_o2modules,
    replace_with_o2modules,
    set_nesting_level,
    set_next_layer,
    try_clear_cache,
)
from o2grad.utils import matrix_from_dict2d


class Wrapper(nn.Module):
    def __init__(self, module):
        super().__init__()
        self.module = module

    def forward(self, input):
        return self.module(input)


PROGRESS_DEFAULT = True


class O2Model(nn.Module):
    def __init__(
        self,
        model: nn.Module,
        criterion: nn.Module,
        custom_layers: dict = {},
        memopt: bool = False,
        save_intermediate: Union[str, List[str]] = [],
    ):
        # super().__init__(model)
        super().__init__()
        self.module = model

        if not isinstance(criterion, O2Loss):
            self.criterion = O2Loss(criterion)

        model_ = replace_with_o2modules(model, custom_objects=custom_layers)
        name_to_layer = get_o2modules(model_)
        set_next_layer(model_, self.criterion)
        set_nesting_level(model_)
        self.module = model_

        INTERMEDIATE_TENSORS = [
            "dLdy",
            "dL2dx2",
            "dydx",
            "dy2dx2",
            "dydw",
            "dy2dw2",
            "dy2dxdw",
        ]
        save_intermediate_filter = []
        if isinstance(save_intermediate, str):
            if save_intermediate == "all":
                save_intermediate = INTERMEDIATE_TENSORS
            else:
                save_intermediate = [save_intermediate]
        for tname in save_intermediate:
            if tname not in INTERMEDIATE_TENSORS:
                warnings.warn(
                    f'{tname} is not valid intermediate tensor name.\nChoose one of: {", ".join(INTERMEDIATE_TENSORS)}'
                )
            else:
                save_intermediate_filter.append(tname)

        player_cnt = len(
            [l for l in name_to_layer.values() if isinstance(l, O2ParametricLayer)]
        )
        self.name_to_layer = name_to_layer
        self.layer_to_name = OrderedDict(
            [(id(layer), name) for name, layer in name_to_layer.items()]
        )
        self.forward_handles = OrderedDict()
        self.backward_handles = OrderedDict()
        self.settings = AttrDict(
            {
                "memopt": memopt,
                "save_intermediate": set(save_intermediate_filter),
                "o2enabled": False,
                "diagonal_blocks": False,
                "progress": PROGRESS_DEFAULT,
            }
        )
        self.progress = None

        def _init_callback():
            if self.settings.progress:
                self.progress = tqdm(total=player_cnt * (player_cnt + 1) // 2)
                # self.progress = tqdm(total=player_cnt, position=0, leave=True)

        def _update_callback():
            if self.settings.progress:
                self.progress.update()

        def _close_callback():
            if self.settings.progress:
                self.progress.close()

        self.shared_objects = AttrDict(
            {
                "V": dict(),
                "dL2dw2": dict(),
                "lid_to_name": self.layer_to_name,
                "callbacks": {
                    "progress": {
                        "init": _init_callback,
                        "step": _update_callback,
                        "complete": _close_callback,
                    }
                },
            }
        )
        self.distribute_settings(self.settings)
        self.sync_shared_objects()
        self.enable_o2backprop()

        def _backprop_step(layer):
            diagonal_blocks = self.settings.diagonal_blocks
            backprop_step(layer, diagonal_blocks, cache_cpu=True)

        # add_callbacks(model_, dict(on_children_complete = lambda layer: backprop_step(layer)))
        # add_callbacks(model_, dict(on_capture = lambda layer: backprop_step(layer)))
        add_callbacks(model_, dict(on_children_complete=_backprop_step))
        add_callbacks(model_, dict(on_capture=_backprop_step))
        add_default_callbacks(model_)

    def forward(self, input):
        if self.settings.o2enabled:
            self.clear_cache()
        return self.module(input)

    def distribute_settings(
        self, settings=None, condition: Callable[[nn.Module], bool] = lambda m: True
    ) -> None:
        for layer in self.name_to_layer.values():
            if condition(layer):
                if settings is None:
                    layer.settings.update(self.settings)
                else:
                    layer.settings.update(settings)
        if settings is None:
            self.criterion.settings.update(self.settings)
        else:
            self.criterion.settings.update(settings)

    def sync_shared_objects(self) -> None:
        for layer in self.name_to_layer.values():
            for so_name, so in self.shared_objects.items():
                setattr(layer, so_name, so)
        self.criterion._shared_objects = self.shared_objects

    def enable_o2backprop(self) -> None:
        """Adds hooks to model to compute and store the values required for second order backprop."""
        for layer_name, layer in self.name_to_layer.items():
            if layer_name not in self.forward_handles:
                self.forward_handles[layer_name] = layer.register_forward_hook(
                    _capture_activations
                )
            if layer_name not in self.backward_handles:
                self.backward_handles[layer_name] = layer.register_full_backward_hook(
                    _capture_backprops
                )
        self.settings.o2enabled = True
        self.distribute_settings()

    def disable_o2backprop(self) -> None:
        """Remove hooks added by add_hooks"""
        if self.forward_handles:
            for handle in self.forward_handles.values():
                handle.remove()
            self.forward_handles = OrderedDict()
        if self.backward_handles:
            for handle in self.backward_handles.values():
                handle.remove()
            self.backward_handles = OrderedDict()
        self.settings.o2enabled = False

    def enable_progressbar(self) -> None:
        self.settings.progress = True

    def disable_progressbar(self) -> None:
        self.settings.progress = False

    def clear_cache(self) -> None:
        V = self.shared_objects.V
        for key in list(V.keys()):
            del V[key]
        dL2dw2 = self.shared_objects.dL2dw2
        for key in list(dL2dw2.keys()):
            del dL2dw2[key]
        try_clear_cache(self.module, force=True)

    def get_hessian_as_dict(
        self, as_type: str = "dense", diagonal_blocks=False, device="cpu"
    ) -> Dict[str, Union[torch.Tensor, SparseSymmetricMatrix]]:
        """
        Returns the Hessian of the torch module provided.

        Parameters
        ---------
        as_type: str, optional
            One of 'dense' (torch.Tensor, strided), 'sparse' (torch.Tensor, sparse_coo), or 'symmetric' (SparseSymmetricMatrix).
            Specifies the common type of the tensors H[s][t] in the returned dictionary.
            If None, the entries H[s][t] may have different types and will be returned as computed in the backpropagation step.
        diagonal_blocks: bool, optional
            If specified, will only calculate the diagonal blocks of the Hessian,
            i.e. only terms dL2dpdq where p, q are parameters from the same layer.
        device: str, optional
            Puts all tensors on the device specified.
        """
        return self._get_hessian_as_dict(
            self.shared_objects.dL2dw2,
            as_type=as_type,
            diagonal_blocks=diagonal_blocks,
            device=device,
        )

    def get_hessian_from_input_as_dict(
        self,
        input: torch.Tensor,
        target: torch.Tensor,
        as_type: str = "dense",
        diagonal_blocks=False,
        per_batch=False,
    ) -> Dict[str, Union[torch.Tensor, SparseSymmetricMatrix]]:
        """
        Returns the Hessian of the torch module provided.

        Parameters
        ---------
        input: torch:Tensor, optional
            If provided, returns the Hessian given this input.
            NOTE: This must be used in combination with a target.
        target: torch.Tensor, optional
            If provided, returns the Hessian given this target.
            NOTE: This must be used in combination with an output.
        %(get_hessian_as_dict.parameters)s
        per_batch: bool, optional
            If provided, calculates the Hessian for each batch sample individually and then aggregates.
        """
        kwargs = dict(as_type=as_type, diagonal_blocks=diagonal_blocks)
        return self._get_hessian_from_input(
            input, target, per_batch, as_dict=True, **kwargs
        )

    def get_hessian(
        self, as_type: str = "dense", as_file=False, diagonal_blocks=False
    ) -> Union[torch.Tensor, SparseSymmetricMatrix]:
        """Returns the Hessian of the torch module provided.

        Parameters
        ---------
        as_type: str, optional
            One of 'dense' (torch.Tensor, strided), 'sparse' (torch.Tensor, sparse_coo), or 'symmetric' (SparseSymmetricMatrix).
            Specifies the type of the returned object.
        as_file: str, optional
            If specified, will store the returned matrix in a binary file of the given name on disk.
        diagonal_blocks: bool, optional
            If specified, will only calculate the diagonal blocks of the Hessian,
            i.e. only terms dL2dpdq where p, q are parameters from the same layer.
        """
        return self._get_hessian(
            self.name_to_layer.keys(),
            self.shared_objects.dL2dw2,
            as_type=as_type,
            as_file=as_file,
            diagonal_blocks=diagonal_blocks,
        )

    def get_hessian_from_input(
        self,
        input: torch.Tensor,
        target: torch.Tensor,
        as_type: str = "dense",
        as_file=False,
        diagonal_blocks=False,
        per_batch=False,
    ) -> Union[torch.Tensor, SparseSymmetricMatrix]:
        """Returns the Hessian of the torch module provided.

        Parameters
        ---------
        input: torch:Tensor, optional
            If provided, returns the Hessian given this input.
            NOTE: This must be used in combination with a target.
        target: torch.Tensor, optional
            If provided, returns the Hessian given this target.
            NOTE: This must be used in combination with an output.
        %(get_hessian.parameters)s
        per_batch: bool, optional
            If provided, calculates the Hessian for each batch sample individually and then aggregates.
        """
        kwargs = dict(as_type=as_type, as_file=as_file, diagonal_blocks=diagonal_blocks)
        return self._get_hessian_from_input(
            input, target, per_batch, as_dict=False, **kwargs
        )

    def _get_hessian_from_input(
        self, input, target, per_batch, as_dict=False, *args, **kwargs
    ):
        output = self.module(input)
        if per_batch:
            # Replace old criterion so loss calculated per batch element in single forward propagation step

            # should_replace_crit = (self.criterion.criterion.reduction != 'none')
            # if should_replace_crit:
            #     criterion_old = self.criterion
            #     self.criterion.criterion = type(self.criterion.criterion)(reduction='none')
            loss = self.criterion(output, target)
            hessian = None
            for i in range(input.shape[0]):
                self.settings.per_batch = True
                self.settings.elem = i
                self.distribute_settings()
                self.module.zero_grad()
                loss.backward(retain_graph=True)
                if as_dict:
                    elem_hessian = self._get_hessian_as_dict(
                        self.shared_objects.dL2dw2, *args, **kwargs
                    )
                    if hessian:
                        for s in elem_hessian:
                            for t in elem_hessian[t]:
                                hessian[s][t] = hessian[s][t] + elem_hessian[s][t]
                    else:
                        hessian = elem_hessian
                else:
                    elem_hessian = self._get_hessian(
                        self.name_to_layer.keys(),
                        self.shared_objects.dL2dw2,
                        *args,
                        **kwargs,
                    )
                    hessian = (
                        elem_hessian if hessian is None else hessian + elem_hessian
                    )
            del self.settings.per_batch, self.settings.elem
            self.distribute_settings()
            # Restore old criterion
            # if should_replace_crit:
            #     self.criterion = criterion_old
        else:
            self.clear_cache()
            self.module.zero_grad()
            loss = self.criterion(output, target)
            loss.backward()
            if as_dict:
                hessian = self._get_hessian_as_dict(
                    self.shared_objects.dL2dw2, *args, **kwargs
                )
            else:
                hessian = self._get_hessian(
                    self.name_to_layer.keys(),
                    self.shared_objects.dL2dw2,
                    *args,
                    **kwargs,
                )
        return hessian

    @staticmethod
    def _get_hessian(
        layers: Sequence[str],
        dL2dw2: Dict[str, Union[torch.Tensor, SparseSymmetricMatrix]],
        as_type="dense",
        as_file=False,
        diagonal_blocks=False,
        *args,
        **kwargs,
    ) -> Union[torch.Tensor, SparseSymmetricMatrix]:
        return matrix_from_dict2d(
            layers,
            dL2dw2,
            as_type=as_type,
            as_file=as_file,
            diagonal_blocks=diagonal_blocks,
        )

    @staticmethod
    def _get_hessian_as_dict(
        dL2dw2: Dict[str, Union[torch.Tensor, SparseSymmetricMatrix]],
        as_type="dense",
        diagonal_blocks=False,
        inplace=False,
        device="cpu",
        *args,
        **kwargs,
    ) -> Dict[str, Dict[str, torch.Tensor]]:
        dL2dw2_ = dL2dw2 if inplace else {s: {} for s in dL2dw2.keys()}
        for s in dL2dw2:
            for t in dL2dw2[s]:
                if (not diagonal_blocks) or (s == t):
                    dL2dwsdwt = dL2dw2[s][t]
                    is_dense = (
                        isinstance(dL2dwsdwt, torch.Tensor)
                        and dL2dwsdwt.layout == torch.strided
                    )
                    is_sparse = (
                        isinstance(dL2dwsdwt, torch.Tensor)
                        and dL2dwsdwt.layout == torch.sparse_coo
                    )
                    if as_type == "dense" and not is_dense:
                        dL2dw2_[s][t] = dL2dw2[s][t].to_dense()
                    elif as_type == "sparse" and not is_sparse:
                        dL2dw2_[s][t] = dL2dw2[s][t].to_sparse()
                    if not inplace and t not in dL2dw2_[s]:
                        dL2dw2_[s][t] = dL2dw2[s][t]
                    if device != "cpu":
                        dL2dw2_[s][t] = dL2dw2_[s][t].to(device=device)
        return dL2dw2_

    def get_hessian_eigs_from_input(
        self, input: torch.Tensor, target: torch.Tensor, diagonal_blocks=False
    ):
        hessian = self.get_hessian_from_input_as_dict(
            input, target, diagonal_blocks=diagonal_blocks
        )
        eigs, vecs = eigs_from_dict2d(self.name_to_layer.keys(), hessian)
        return eigs, vecs

    def _add_connections(self) -> None:
        last_layer = self.criterion
        for layer in list(self.name_to_layer.values())[::-1]:
            layer.next_layer = last_layer
            last_layer = layer
        last_layer.is_first_layer = True


def _capture_activations(layer: O2Module, input, output) -> None:
    pass


def _capture_backprops(layer: O2Module, grad_input, grad_output):
    # Get loss output Jacobian dLdy
    if isinstance(grad_output, tuple) and len(grad_output) == 1:
        grad_output = grad_output[0].clone()
    layer.dLdy = grad_output
    layer._callbacks.on_capture()
