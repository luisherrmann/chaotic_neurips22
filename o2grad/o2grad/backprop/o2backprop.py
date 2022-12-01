import torch

from o2grad.linalg import matmul_mixed
from o2grad.modules.o2layer.o2layer import O2Layer
from o2grad.modules.o2container.o2container import O2Container
from o2grad.modules.o2parametric.o2parametric import O2ParametricLayer


def backprop_step(layer, diagonal_blocks=False, cache_cpu=True):
    """Append backprop to layer.backprops_list in backward pass.
    NOTE: If diagonal_blocks is set to True, only diagonal blocks of the Hessian will be calculated.
    """
    # ic(id(layer), layer.__class__.__name__)
    # ic(id(layer), type(layer), id(layer.next_layer), type(layer.next_layer), layer.next_layer.__class__.__name__)
    with torch.no_grad():
        # Get input and reshape
        x = layer.input
        elem = None if not (layer.settings.per_batch) else layer.settings.elem
        if elem is not None:
            shape = x.shape
            N = shape[0]
            x = x.reshape(N, -1)[elem : elem + 1, :].reshape(1, *shape[1:])
        # Get Loss-Output-Jacobian dLdy
        if isinstance(layer, O2Layer):
            dLdy = layer.dLdy
            if elem is not None:
                N, *_ = layer.output.shape
                dLdy = dLdy.reshape(N, -1)[elem : elem + 1, :]
            dLdy = dLdy.reshape(-1)
            dLdy_sparse = dLdy.to_sparse().coalesce()
        # Get Loss-Output-Hessian dL2dy2
        dL2dy2 = layer.next_layer.get_loss_input_hessian_cached()
        dydx = layer.get_output_input_jacobian(x)
        if layer.chain_dydx:
            if layer.chain_end_dydx:
                layer.dyydx = dydx
            else:
                dzdy = layer.next_layer.dyydx
                layer.dyydx = matmul_mixed(dzdy, dydx)
        if isinstance(layer.next_layer, O2Layer) or isinstance(
            layer.next_layer, O2Container
        ):
            if layer.nesting_level == layer.next_layer.nesting_level:
                layer.next_layer.try_clear_cache("dyydx")
        if isinstance(layer, O2Layer):
            if isinstance(layer, O2ParametricLayer):
                s = layer.lid_to_name[id(layer)]
                dydw = layer.get_output_param_jacobian(x)
                dy2dw2 = layer.get_output_param_hessian(x)
                dy2dxdw = layer.get_mixed_output_param_hessian(x)
                dL2dw2 = layer.get_loss_param_hessian(
                    dLdy_sparse, dL2dy2, dydw, dy2dw2, delete=["dy2dxdw"]
                ).to(
                    "cpu"
                )  # , non_blocking=True)
                layer.callbacks.progress.step()
                layer.dL2dw2[s] = {}
                layer.dL2dw2[s][s] = dL2dw2
                if not diagonal_blocks:
                    for t in layer.V:
                        Vt = layer.V[t]
                        if cache_cpu and x.device != "cpu":
                            Vt = Vt.to(x.device)
                        layer.dL2dw2[s][t] = matmul_mixed(
                            dydw, Vt, is_transposed1=True
                        ).to(
                            "cpu"
                        )  # , non_blocking=True)
                        del Vt
                        layer.callbacks.progress.step()
                    if not layer.is_first_layer:
                        layer.V[s] = layer.get_mixed_loss_param_hessian(
                            dLdy,
                            dL2dy2,
                            dydx,
                            dydw,
                            dy2dxdw,
                            delete=["dydw", "dy2dxdw"],
                        )
            # Update mixed Hessians V[s] from V[s+1]
            if not (diagonal_blocks or layer.is_first_layer):
                for t in [*layer.V.keys()]:
                    s = layer.lid_to_name[id(layer)]
                    if t != s:
                        Vt = layer.V[t]
                        if cache_cpu and x.device != "cpu":
                            Vt = Vt.to(x.device)
                        Vt_ = matmul_mixed(dydx, Vt, is_transposed1=True)
                        del Vt
                        if cache_cpu and Vt_.device != "cpu":
                            layer.V[t] = Vt_.to(x.device)
                            del Vt_
                        else:
                            layer.V[t] = Vt_
            else:
                # layer.callbacks.progress.complete()
                for t in [*layer.V.keys()]:
                    del layer.V[t]
        # Get input Hessian for backpropagation to previous layer
        if layer.nesting_level != 0 or "dL2dx2" in layer.settings.save_intermediate:
            if isinstance(layer, O2Layer):
                dy2dx2 = layer.get_output_input_hessian(x)
                dL2dx2 = layer.get_loss_input_hessian(
                    dLdy_sparse, dL2dy2, dydx, dy2dx2, delete=["dLdy", "dydx", "dy2dx2"]
                )
            elif isinstance(layer, O2Container):
                dL2dy2 = layer.next_layer.get_loss_input_hessian_cached()
                dL2dx2 = layer.get_loss_input_hessian(dL2dy2)
            if isinstance(layer.next_layer, O2Layer) or isinstance(
                layer.next_layer, O2Container
            ):
                if layer.nesting_level == layer.next_layer.nesting_level:
                    layer.next_layer.try_clear_cache("dL2dx2")
            del dL2dy2
            layer.dL2dx2 = dL2dx2
