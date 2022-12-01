from typing import Optional, overload
import torch
from torch.optim._functional import sgd
from torch.optim import Optimizer


def project_grads(grads, V, keep_idx, noise=False, noise_fac=None):
    """Given gradients and the eigenvector matrix of the Hessian,
    projects the gradient onto the eigenvector components specified by the indices provided.

    Args:
        grads (torch.Tensor): The network gradient to be modified.
        V (torch.Tensor): The eigenvector matrix of the network Hessian.
        keep_idx (torch.Tensor): The indices of the components of V to preserve in the gradient.
        noise (bool, optional): If True, replaces the non-preserved (pruned) components
            of the eigenvectors with noise sampled from a normal distribution (default: False).
        noise_fac (float, optional): If noise is set, determines the scaling factor of the noise.
            By default, the noise level of the mean absolute value of the gradient is used.
    """
    device = grads.device
    V_ = V[:, keep_idx]
    V_T = V_.T
    grads_ = torch.einsum("ij,j->i", V_T, grads)
    grads_ = torch.einsum("ij,j->i", V_, grads_)
    if noise:
        discard_idx = torch.logical_not(keep_idx)
        discard_cnt = discard_idx.sum()
        V__ = V[:, discard_idx]
        if noise_fac is None:
            noise_fac = grads.abs().mean()
        noise = noise_fac * torch.randn(
            discard_cnt, device=device
        )  # TODO: Replace with mu * sigma * eps
        # TODO: Set noise level for each dimension through running mean
        noise_ = torch.einsum("ij,j->i", V__, noise)
        grads_ += noise_
    return grads_


def calc_update(
    grads,
    w,
    V,
    lr,
    k=None,
    prune_set="chaotic",
    prune_k="top_abs",
    noise=False,
    noise_fac=None,
    ret_idx=False,
):
    """Given gradients, eigenvalues and eigenvectors of the Hessian, and the learning rate,
    calculates and updated gradient for the optimization step according to the respective
    policy parameters.

    Args:
        grads (torch.Tensor): A 1D gradient tensor.
        w (torch.Tensor): A 1D tensor containing all eigenvalues of the Hessian.
        V (torch.Tensor): A 2D tensor containig all eigenvectors of the Hessian.
        lr (float): The learning rate of the GD algorithm
        k (int, optional): If specified, will only prune (at most) k eigenvectors
        prune_set (str, optional): One of 'pos', 'neg', 'chaotic'.
            Determines subset from which to prune eigenpairs.
            (Positive, negative or eigenvalues indicating chaotic behaviour)
        prune_k (str, optional): One of 'top_abs', 'rand'.
            Determines the strategy for pruning eigenvalues from the pruning set.
            (Largest by absolute value or random selection)
        noise (bool, optional): If set, will not prune the eigenvalues
            but add random Gaussian noise on respective directions.
        noise_fac (float, optional): If noise is set, will scale Gaussian noise with the noise_level specified.
        ret_idx (bool, optional): If ret_idx is set, will return indices of pruned eigenpairs.
    """
    assert len(w.shape) == 1 and len(V.shape) == 2
    assert V.shape[0] == V.shape[1]
    assert len(w) == V.shape[0]
    device = grads.device
    grads = grads.clone().reshape(-1)
    if prune_set == "neg":
        if k is not None:
            keep_idx = torch.empty(len(w), dtype=torch.bool, device=device)
            keep_idx[:] = 1
            if prune_k == "top_abs":
                discard_idx = torch.topk(w, k, largest=False)[1]
            elif prune_k == "rand":
                neg_idx = torch.where(w <= 0)[0]
                k_ = min(k, len(neg_idx))
                discard_idx = torch.randint(0, len(neg_idx), (k_,), device=device)
                discard_idx = neg_idx[discard_idx]
            keep_idx[discard_idx] = 0
            keep_idx[w > 0] = 1
        else:
            keep_idx = w > 0
    elif prune_set == "pos":
        if k is not None:
            keep_idx = torch.empty(len(w), dtype=torch.bool, device=device)
            keep_idx[:] = 1
            if prune_k == "top_abs":
                discard_idx = torch.topk(w, k)[1]
            elif prune_k == "rand":
                pos_idx = torch.where(w >= 0)[0]
                k_ = min(k, len(pos_idx))
                discard_idx = torch.randint(0, len(pos_idx), (k_,), device=device)
                discard_idx = pos_idx[discard_idx]
            keep_idx[discard_idx] = 0
            keep_idx[w < 0] = 1
        else:
            keep_idx = w < 0
    elif prune_set == "chaotic":
        w_ = (1 - lr * w) ** 2
        if k is not None:
            keep_idx = torch.empty(len(w_), dtype=torch.bool, device=device)
            keep_idx[:] = 1
            if prune_k == "top_abs":
                discard_idx = torch.topk(w_, k)[1]
            elif prune_k == "rand":
                chaos_idx = torch.where(w_ > 1)[0]
                k_ = min(k, len(chaos_idx))
                discard_idx = torch.randint(0, len(chaos_idx), (k_,), device=device)
                discard_idx = chaos_idx[discard_idx]
            keep_idx[discard_idx] = 0
            keep_idx[w_ <= 1] = 1
        else:
            keep_idx = w_ <= 1
    grads_ = project_grads(grads, V, keep_idx, noise=noise, noise_fac=noise_fac)
    if ret_idx:
        return grads_, keep_idx
    else:
        return grads_


class CGD(Optimizer):
    r"""Implements chaotic gradient descent (optionally with momentum).

    Nesterov momentum is based on the formula from
    `On the importance of initialization and momentum in deep learning`__.

    Args:
        params (iterable): Iterable of parameters to optimize or dicts defining
            parameter groups
        lr (float): Learning rate
        momentum (float, optional): Momentum factor (default: 0)
        weight_decay (float, optional): Weight decay (L2 penalty) (default: 0)
        dampening (float, optional): Dampening for momentum (default: 0)
        nesterov (bool, optional): Enables Nesterov momentum (default: False)
        gamma (float, optional): If smaller than 1.0, computes a mixture between
            a regular gradient update and a CGD gradient update following:
            .. math::
            \begin{aligned}
                \Delta p &\leftarrow (1 - \gamma) * g + \gamma * \Delta p
            \end{aligned}
            (default: 1.0)
        k (int, optional): If specified, will only prune (at most) k eigenvectors.
        prune_global (bool, optional): If set, will prune with the same eigenpairs wg, Vg throughout the entire
            training. These (wg, Vg) need to be provided separately.
            (default: False)
        prune_start: The step at which to start pruning. (default: 0)
        prune_set (str, optional): One of 'pos', 'neg', 'chaotic'.
            Determines subset from which to prune eigenpairs.
            (Positive, negative or eigenvalues indicating chaotic behaviour)
        prune_k (str, optional): One of 'top_abs', 'rand'.
            Determines the strategy for pruning eigenvalues from the pruning set.
            (Largest by absolute value or random selection)
        noise (bool, optional): If set, will not prune the eigenvalues
            but add random Gaussian noise on respective directions.
        noise_fac (float, optional): If noise is set, will scale Gaussian noise with the noise_level specified.
        wg (torch.Tensor, optional):
        Vg (torch.Tensor, optional): O
        ret_idx (bool, optional): If ret_idx is set, will return indices of pruned eigenpairs.

    Example:
        >>> optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
        >>> optimizer.zero_grad()
        >>> loss_fn(model(input), target).backward()
        >>> optimizer.step()

    __ http://www.cs.toronto.edu/%7Ehinton/absps/momentum.pdf

    .. note::
        The implementation of SGD with Momentum/Nesterov subtly differs from
        Sutskever et. al. and implementations in some other frameworks.

        Considering the specific case of Momentum, the update can be written as

        .. math::
            \begin{aligned}
                v_{t+1} & = \mu * v_{t} + g_{t+1}, \\
                p_{t+1} & = p_{t} - \text{lr} * v_{t+1},
            \end{aligned}

        where :math:`p`, :math:`g`, :math:`v` and :math:`\mu` denote the
        parameters, gradient, velocity, and momentum respectively.

        This is in contrast to Sutskever et. al. and
        other frameworks which employ an update of the form

        .. math::
            \begin{aligned}
                v_{t+1} & = \mu * v_{t} + \text{lr} * g_{t+1}, \\
                p_{t+1} & = p_{t} - v_{t+1}.
            \end{aligned}

        The Nesterov version is analogously modified.
    """

    def __init__(
        self,
        params,
        lr=1e-3,
        momentum=0,
        dampening=0,
        weight_decay=0,
        nesterov=False,
        maximize=False,
        gamma=1.0,
        k=None,
        prune_global=False,
        prune_start=0,
        prune_set="chaotic",
        prune_k="top_abs",
        noise=False,
        noise_fac=None,
        wg=None,
        Vg=None,
        ret_idx=False,
    ):
        if lr < 0.0:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if momentum < 0.0:
            raise ValueError("Invalid momentum value: {}".format(momentum))
        if weight_decay < 0.0:
            raise ValueError("Invalid weight_decay value: {}".format(weight_decay))

        defaults = dict(
            lr=lr,
            momentum=momentum,
            dampening=dampening,
            weight_decay=weight_decay,
            nesterov=nesterov,
            maximize=maximize,
        )
        if nesterov and (momentum <= 0 or dampening != 0):
            raise ValueError("Nesterov momentum requires a momentum and zero dampening")
        is_wg, is_Vg = isinstance(wg, torch.Tensor), isinstance(Vg, torch.Tensor)
        if prune_global and not (is_wg and is_Vg):
            raise TypeError(
                "If prune_global is set, both wg and Vg need to be provided!"
            )
        super().__init__(params, defaults)
        self.gamma = gamma
        self.k = k
        self.prune_global = prune_global
        self.prune_set = prune_set
        self.prune_start = prune_start
        self.prune_k = prune_k
        self.noise = noise
        self.noise_fac = noise_fac
        self.wg = wg
        self.Vg = Vg
        self.ret_idx = ret_idx
        self.state["step"] = 0

    def __setstate__(self, state):
        super().__setstate__(state)
        for group in self.param_groups:
            group.setdefault("nesterov", False)
            group.setdefault("maximize", False)

    @overload
    def step(self, closure=None):
        """Performs a single optimization step.
        Only use when CGD was initialized with `prune_global=True`.

        Args:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        ...

    @overload
    def step(self, hessian: torch.Tensor, closure=None):
        """Performs a single optimization step.

        Args:
            hessian (torch.Tensor): The Hessian of the network to use in the optimization step.
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        ...

    @overload
    def step(self, w: torch.Tensor, V: torch.Tensor, closure=None):
        ...

    @torch.no_grad()
    def step(
        self,
        hessian: torch.Tensor = None,
        w: torch.Tensor = None,
        V: torch.Tensor = None,
        closure=None,
    ):
        """Performs a single optimization step.

        Args:
            w (torch.Tensor): A 1D tensor of the eigevalues of the network Hessian
                to use for the optimization step.
            V (torch.Tensor): A 2D tensor of the eigevectors of the network Hessian
                to use for the optimization step.
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        if self.prune_global:
            w, V = self.wg, self.Vg
        if hessian is not None:
            assert isinstance(hessian, torch.Tensor)
            assert w is None and V is None
            w, V = torch.linalg.eigh(hessian)
        else:
            assert isinstance(w, torch.Tensor)
            assert isinstance(V, torch.Tensor)
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            params_with_grad = []
            d_p_list = []
            momentum_buffer_list = []
            weight_decay = group["weight_decay"]
            momentum = group["momentum"]
            dampening = group["dampening"]
            nesterov = group["nesterov"]
            maximize = group["maximize"]
            lr = group["lr"]

            grads = torch.cat([p.grad.reshape(-1) for p in group["params"]])
            if self.is_pruning():
                if self.prune_global:
                    w, V = self.wg, self.Vg
                if self.ret_idx:
                    projection, keep_idx = calc_update(
                        grads,
                        w,
                        V,
                        lr,
                        k=self.k,
                        prune_set=self.prune_set,
                        prune_k=self.prune_k,
                        noise=self.noise,
                        noise_fac=self.noise_fac,
                        ret_idx=True,
                    )
                    self.state["last_kept"] = keep_idx
                else:
                    projection = calc_update(
                        grads,
                        w,
                        V,
                        lr,
                        k=self.k,
                        prune_set=self.prune_set,
                        prune_k=self.prune_k,
                        noise=self.noise,
                        noise_fac=self.noise_fac,
                    )
            high = 0
            for p in group["params"]:
                if p.grad is not None:
                    params_with_grad.append(p)
                    # d_p_list.append(p.grad)
                    low = high
                    high = low + p.numel()
                    if self.is_pruning():
                        d_p = projection[low:high].reshape(p.shape)
                        d_p = (1 - self.gamma) * p.grad + self.gamma * d_p
                    else:
                        d_p = p.grad
                    d_p_list.append(d_p)

                    state = self.state[p]
                    if "momentum_buffer" not in state:
                        momentum_buffer_list.append(None)
                    else:
                        momentum_buffer_list.append(state["momentum_buffer"])

            sgd(
                params_with_grad,
                d_p_list,
                momentum_buffer_list,
                weight_decay=weight_decay,
                momentum=momentum,
                lr=lr,
                dampening=dampening,
                nesterov=nesterov,
                maximize=maximize,
            )

            # update momentum_buffers in state
            for p, momentum_buffer in zip(params_with_grad, momentum_buffer_list):
                state = self.state[p]
                state["momentum_buffer"] = momentum_buffer
            self.state["step"] += 1

        return loss

    def is_pruning(self) -> bool:
        """Returns a bool stating if the optimizer is pruning in the current step."""
        return self.state["step"] >= self.prune_start

    def get_last_idx_kept(self) -> Optional[torch.Tensor]:
        """Returns the indices of the last eigenvector components preserved in the modified gradient."""
        return self.state["last_kept"]
