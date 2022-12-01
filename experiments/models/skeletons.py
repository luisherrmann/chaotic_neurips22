import os
import os.path as path
from os.path import dirname, abspath
import datetime
from typing import Sequence, Union, Tuple, Optional
from collections import defaultdict
import numpy as np
import torch
import torch.optim as optim
import torch.nn.functional as F
import pytorch_lightning as pl
import matplotlib.pyplot as plt
from neptune.new.types import File
from neptune.new.integrations.pytorch_lightning import NeptuneLogger
from pytorch_lightning.loggers import TensorBoardLogger

from o2grad.sparse import sparse_eye
from o2grad.linalg import matmul_mixed, sum_mixed
from optim.cgd import CGD
from metrics.eigen import eigen_dissim, cosine_sim, subspace_sim, random_baseline_subsim


HOME = abspath(path.join(dirname(__file__), "../.."))


class MinimalFFNSkeleton(pl.LightningModule):
    def __init__(
        self,
        model: Union[torch.nn.Module, Sequence[torch.nn.Module]],
        optimizer: torch.optim.Optimizer,
        log_freq: int = 1,
        **kwargs,
    ):
        """Base class that includes everything for a simple training run of multiple models.
        Logs accuracy and loss.

        Args:
            model (Union[torch.nn.Module, Sequence[torch.nn.Module]]): A model or a list of models that will be trained.
            optimizer (torch.optim.Optimizer): An optimizer, e.g. SGD or CGD.
            log_freq (int, optional): The frequency with which metrics will be logged.
                This can slow down training, in that case we recommend
                setting it to 10. Defaults to 1.
        """
        super().__init__()
        self.logged_metrics = defaultdict(list)
        self.step = 0
        self.log_freq = log_freq
        if isinstance(model, torch.nn.Module):
            self.model = [model]
        elif isinstance(model, Sequence):
            self.model = [*model]
        else:
            raise TypeError(f"Invalid argument model of type {type(model)}")
        # Add modules so they are visible to PL
        self.parallel = len(self.model) > 1
        for i, m in enumerate(self.model):
            self.add_module(f"model[{self.maybe_num(i)}]", m)
        self.automatic_optimization = False
        if isinstance(optimizer, torch.optim.Optimizer):
            self.optim = [optim]
        elif isinstance(optimizer, Sequence):
            self.optim = [*optimizer]
        else:
            raise TypeError(f"Invalid argument optimizer of type {type(optimizer)}")
        self.model_name = ""
        self.model_path = ""

    def maybe_num(self, i: int):
        """Formats i if self.parallel = True

        Args:
            i (int): Model id.

        Returns:
            str: Formated id.
        """
        return f"[{i}]" if self.parallel else ""

    def log_metric(self, key: str, value: float, local_only: bool = False):
        """Log metric to logger at current step.

        Args:
            key (str): Name of the metric.
            value (float): Value to log.
            local_only (bool, optional): If True, saves the metric only locally in the class. Defaults to False.
        """
        if not local_only:
            if self.step % self.log_freq == 0:
                self.logger.experiment[key].log(value)
        self.logged_metrics[key].append((self.step, value))

    def log_dist_metrics(
        self,
        grad: torch.Tensor,
        w: torch.Tensor,
        V: torch.Tensor,
        name: str = "hessian",
        chaos: bool = False,
        i1=0,
        i2=0,
    ):
        """Logs the distances between models that where perturbed with different strategies.

        Args:
            grad (torch.Tensor): The current gradient.
            w (torch.Tensor): The current eigenvalues.
            V (torch.Tensor): The current eigenvectors of shape (model_dim, model_dim)
            name (str, optional): Name of the matrix. Defaults to "hessian".
            chaos (bool, optional): If True, computes metrics regarding the local Lyapunov matrix. Defaults to False.
            i1 (int, optional): Id of 1st model. Defaults to 0.
            i2 (int, optional): Id of 2nd model. Defaults to 0.
        """
        dist_metrics = {}
        m1, m2 = self.model[i1], self.model[i2]
        p1 = torch.cat([p.reshape(-1) for p in m1.parameters()])
        p2 = torch.cat([p.reshape(-1) for p in m2.parameters()])
        dist = torch.dist(p1, p2)
        i1, i2 = self.maybe_num(i1), self.maybe_num(i2)
        dist_to_origin = m2.perturb_dist
        perturb_strat = m2.perturb_strat
        # Calculate distances along axes of interest
        if w is None and V is None and grad is None:
            dist_metrics.update(
                {f"distances/dist{i1}{i2}[{dist_to_origin}]": dist.item()}
            )
        else:
            vmin, vmax = V[:, 0], V[:, -1]
            dist_along_max = torch.dist(torch.dot(p1, vmax), torch.dot(p2, vmax))
            dist_along_min = torch.dist(torch.dot(p1, vmin), torch.dot(p2, vmin))
            dist_along_grad = torch.dist(torch.dot(p1, grad), torch.dot(p2, grad))
            dist_metrics.update(
                {
                    f"distances/dist{i1}{i2}[{dist_to_origin}]": dist.item(),
                    f"distances/dist{i1}{i2}[{dist_to_origin}]:strat[{perturb_strat}]_along:{name}{i2}_max": dist_along_max,
                    f"distances/dist{i1}{i2}[{dist_to_origin}]:strat[{perturb_strat}]_along:{name}{i2}_min": dist_along_min,
                    f"distances/dist{i1}{i2}[{dist_to_origin}]:strat[{perturb_strat}]_along:grad": dist_along_grad,
                }
            )
            if chaos:
                if torch.any(w < 1):
                    wmin_chaotic_idx = torch.argmin(w[w < 1])
                    vmin_chaotic = V[:, wmin_chaotic_idx]
                    dist_along_min_chaotic = torch.dist(
                        torch.dot(p1, vmin_chaotic), torch.dot(p2, vmin_chaotic)
                    )
                    dist_metrics.update(
                        {
                            f"distances/dist{i1}{i2}[{dist_to_origin}]:strat[{perturb_strat}]_along:{name}{i2}_chaotic_min": dist_along_min_chaotic
                        }
                    )
                if torch.any(w > 1):
                    wmax_nonchaotic_idx = torch.argmax(w[w > 1])
                    vmax_nonchaotic = V[:, wmax_nonchaotic_idx]
                    dist_along_max_nonchaotic = torch.dist(
                        torch.dot(p1, vmax_nonchaotic), torch.dot(p2, vmax_nonchaotic)
                    )
                    dist_metrics.update(
                        {
                            f"distances/dist{i1}{i2}[{dist_to_origin}]:strat[{perturb_strat}]_along:{name}{i2}_nonchaotic_max": dist_along_max_nonchaotic
                        }
                    )
            else:
                if torch.any(w > 0):
                    wposmin_idx = torch.argmin(w[w > 0])
                    vposmin = V[:, wposmin_idx]
                    dist_along_posmin = torch.dist(
                        torch.dot(p1, vposmin), torch.dot(p2, vposmin)
                    )
                    dist_metrics.update(
                        {
                            f"distances/dist{i1}{i2}[{dist_to_origin}]:strat[{perturb_strat}]_along:{name}{i2}_pos_min": dist_along_posmin
                        }
                    )
                if torch.any(w < 0):
                    wnegmax_idx = torch.argmax(w[w < 0])
                    vnegmax = V[:, wnegmax_idx]
                    dist_along_negmax = torch.dist(
                        torch.dot(p1, vnegmax), torch.dot(p2, vnegmax)
                    )
                    dist_metrics.update(
                        {
                            f"distances/dist{i1}{i2}[{dist_to_origin}]:strat[{perturb_strat}]_along:{name}{i2}_neg_max": dist_along_negmax
                        }
                    )
        for k, v in dist_metrics.items():
            self.log_metric(k, v)

    def forward(self, x: torch.Tensor):
        """Forward call for all models.

        Args:
            x (torch.Tensor): Input.

        Returns:
            list: List of outputs of each model.
        """
        return [m(x) for m in self.model]

    def on_train_start(self):
        """Initializes Training."""
        if self.logger:
            if isinstance(self.logger, NeptuneLogger):
                self.model_name = self.logger.experiment["sys/id"].fetch()
        else:
            time_now = datetime.now()
            self.model_name = time_now.strftime("%Y-%m-%d-%H-%M-%S")
        logging_dirs = path.join(HOME, "logs", self.model_name)
        if not path.exists(logging_dirs):
            os.makedirs(logging_dirs)
        print("Logging dirs:", logging_dirs)
        self.logging_dirs = logging_dirs
        storage_dirs = path.join(HOME, "storage", self.model_name)
        if not path.exists(storage_dirs):
            os.makedirs(storage_dirs)
        print("Storage dirs:", storage_dirs)
        self.storage_dirs = storage_dirs

    def training_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx):
        """Computes the training step.

        Args:
            batch (Tuple[torch.Tensor, torch.Tensor]): Input and labels.

        Returns:
            dict: Dictionary with current loss and accuracy.
        """
        x, y = batch
        y_hat = self.forward(x)
        optims = self.optimizers()
        return_dict = {}
        if not isinstance(optims, Sequence):
            optims = [optims]
        for i, (y_, opt) in enumerate(zip(y_hat, optims)):
            opt.zero_grad()
            loss = F.cross_entropy(y_, y)
            accuracy = torch.mean((torch.argmax(y_, dim=1) == y).to(torch.float64))
            self.manual_backward(loss, create_graph=True)
            opt.step()
            loss_name = f"loss{i}" if len(y_hat) > 0 else "loss"
            acc_name = f"acc{i}" if len(y_hat) > 0 else "acc"
            if self.step % self.log_freq == 0:
                self.logger.run[loss_name].log(loss)
                self.logger.run[acc_name].log(accuracy)
                self.log(acc_name, accuracy, prog_bar=True)
            return_dict[loss_name] = loss
            return_dict[acc_name] = accuracy

        # Distance metric without eigenthings
        with torch.no_grad():
            for i, m in enumerate(self.model):
                grads = torch.cat([p.grad.reshape(-1) for p in m.parameters()])
                self.log_dist_metrics(
                    grads, vmin=None, vmax=None, name="hessian", i1=0, i2=i
                )
        self.step += 1
        return return_dict

    def validation_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx):
        """Validation step.

        Args:
            batch (Tuple[torch.Tensor, torch.Tensor]): Input and labes.
        """
        x, y = batch
        y_hat = self.forward(x)
        val_metrics = {}
        for i, y_ in enumerate(y_hat):
            loss = F.cross_entropy(y_, y)
            accuracy = torch.mean((torch.argmax(y_, dim=1) == y).to(torch.float64))
            loss_name = f"val_loss{i}" if len(y_hat) > 0 else "val_loss"
            acc_name = f"val_acc{i}" if len(y_hat) > 0 else "val_acc"
            val_metrics.update({loss_name: loss, acc_name: accuracy})
            self.log(loss_name, loss, prog_bar=True)
            self.log(acc_name, accuracy, prog_bar=True)
        self.logger.log_metrics(val_metrics)

    def test_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx):
        """Testing step.

        Args:
            batch (Tuple[torch.Tensor, torch.Tensor]): Input and labels
        """
        x, y = batch
        y_hat = self.forward(x)
        test_metrics = {}
        for i, y_ in enumerate(y_hat):
            loss = F.cross_entropy(y_, y)
            accuracy = torch.mean((torch.argmax(y_hat, dim=1) == y).to(torch.float64))
            loss_name = f"test_loss{i}" if len(y_hat) > 0 else "test_loss"
            acc_name = f"test_acc{i}" if len(y_hat) > 0 else "test_acc"
            test_metrics.update({loss_name: loss, acc_name: accuracy})
            self.log(loss_name, loss, prog_bar=True)
            self.log(acc_name, accuracy, prog_bar=True)
        self.logger.log_metrics(test_metrics)

    def configure_optimizers(self):
        """Returns the current optimizer. Used for changing the optimizer.

        Returns:
            torch.optim.Optimizer: Current optimizer.
        """
        return self.optim


class FFNSkeleton(MinimalFFNSkeleton):
    def __init__(
        self,
        model: Union[torch.nn.Module, Sequence[torch.nn.Module]],
        optimizer: torch.optim.Optimizer,
        eps_eigh: float = 0.0,
        plot_eigs_every: int = 100,
        plot_mats_every: int = 200,
        log_eigenthings: bool = True,
        log_subsim: bool = True,
        log_lyapunov: bool = True,
        log_lyapunov_subsim: bool = False,
        log_freq: int = 1,
        save_lyapunov_at: Optional[int] = None,
        t_lya_subspace_sim: list = None,
        t_hessian_subspace_sim: list = None,
        save_hessian_dist_every: int = 0,
    ):
        """The main class for logging the experiments. Has every metric that we show in the paper.

        Args:
            model (Union[torch.nn.Module, Sequence[torch.nn.Module]]): A model or a list of models for training
            optimizer (torch.optim.Optimizer): An optimizer, e.g. SGD or CGD.
            eps_eigh (float, optional): A value that can be added for stability of the eigenvalue decomposition. Defaults to 0.0.
            plot_eigs_every (int, optional): Plots the eigenvalue densities every given steps. Defaults to 100.
            plot_mats_every (int, optional): Plots the Hessian as image every given steps. Defaults to 200.
            log_eigenthings (bool, optional): If True, logs the eigenvalue decomposition of the Hessian. Defaults to True.
            log_subsim (bool, optional): If True, logs the subspace similarity between Hessians. Defaults to True.
            log_lyapunov (bool, optional): If True, logs the metrics regarding the Lyapunov matrix. Defaults to True.
            log_lyapunov_subsim (bool, optional): If True, logs the Lyapunov subspace similarities. Defaults to False.
            log_freq (int, optional): The frequency with which metrics will be logged.
                This can slow down training, in that case we recommend
                setting it to 10. Defaults to 1.
            save_lyapunov_at (Optional[int], optional): Saves the Lyapunov eigenspectrum and eigenvalues at given step. Defaults to None.
            t_lya_subspace_sim (list, optional): Defines the timesteps at which to compare the Lyapunov eigenspaces to later eigenspaces. Defaults to None.
            t_hessian_subspace_sim (list, optional): Defines the timesteps at which to compare the Hessian eigenspaces to later eigenspaces.. Defaults to None.
            save_hessian_dist_every (int, optional): Saves Hessian eigenvalue distribution every given steps. Defaults to 0.
        """
        super().__init__(model, optimizer, log_freq)
        self.step = 0
        self.epoch = 0
        self.logging_dirs = None
        self.logged_metrics = defaultdict(list)
        self.eps_eigh = eps_eigh
        self.log_eigenthings = log_eigenthings
        self.log_subsim = log_subsim
        self.log_freq = log_freq
        self.log_lyapunov = log_lyapunov
        self.save_lyapunov_at = save_lyapunov_at
        self.log_lyapunov_subsim = log_lyapunov_subsim
        self.t_lya_subspace_sim = t_lya_subspace_sim

        if self.t_lya_subspace_sim is not None:
            n_params = sum([p.numel() for p in self.model[0].parameters()])
            self.lya_subspace_random_comp = []
            for _ in self.t_lya_subspace_sim:
                self.lya_subspace_random_comp += [torch.randperm(n_params)]
        self.t_hessian_subspace_sim = t_hessian_subspace_sim
        if self.t_hessian_subspace_sim is not None:
            n_params = sum([p.numel() for p in self.model[0].parameters()])
            self.hessian_subspace_random_comp = []
            for _ in self.t_hessian_subspace_sim:
                self.hessian_subspace_random_comp += [torch.randperm(n_params)]
        self.save_hessian_dist_every = save_hessian_dist_every
        self.plot_eigs_every = plot_eigs_every
        self.plot_mats_every = plot_mats_every
        self.grads_old = dict()
        self.Y = dict()
        self.Y_block = dict()
        self.Y_diag = dict()
        self.G = dict()
        self.G_block = dict()
        self.G_diag = dict()

    def on_train_epoch_end(self) -> None:
        """Called at the end of every epoch.
        Saves eigenvalues and eigenvectors if self.save_lyapunov_at == self.epoch.
        """
        if self.log_lyapunov and self.save_lyapunov_at == self.epoch:
            for i, Y in enumerate(self.Y.values()):
                lyapunov = matmul_mixed(Y.T, Y)
                w_lya, V_lya = torch.linalg.eigh(lyapunov)
                w_lya = w_lya.to(dtype=torch.float64)
                w_lya = w_lya ** (1 / self.step)
                self.save_eigenthings(w_lya, V_lya, name="lyapunov", i=0)
        self.epoch += 1

    def on_train_end(self) -> None:
        """Called at the end of training.
        Saves eigenvalues and eigenvectors.
        """
        if self.log_lyapunov and self.save_lyapunov_at is None:
            for i, Y in enumerate(self.Y.values()):
                lyapunov = matmul_mixed(Y.T, Y)
                w_lya, V_lya = torch.linalg.eigh(lyapunov)
                w_lya = w_lya.to(dtype=torch.float64)
                # w_lya = w_lya**(1 / self.step)
                self.save_eigenthings(w_lya, V_lya, name="lyapunov", i=0)

    def log_metric(self, key, value, local_only=False):
        """Log metric to logger at current step.

        Args:
            key (str): Name of the metric.
            value (float): Value to log.
            local_only (bool, optional): If True, saves the metric only locally in the class. Defaults to False.
        """
        if not local_only:
            if isinstance(self.logger, NeptuneLogger):
                self.logger.experiment[key].log(value)
            elif isinstance(self.logger, TensorBoardLogger):
                self.logger.log_metrics({key: value})
        self.logged_metrics[key].append((self.step, value))

    def forward(self, x: torch.Tensor):
        """Forward call for all models.

        Args:
            x (torch.Tensor): Input.

        Returns:
            list: List of outputs of each model.
        """
        return [m(x) for m in self.model]

    def log_eigval_metrics(
        self,
        w: torch.Tensor,
        lr: float,
        name: str = "hessian",
        chaos: bool = False,
        i: int = 0,
    ):
        """Logs the eigenvalue metrics.

        Args:
            w (torch.Tensor): The eigenvalues.
            lr (float): The current learning rate.
            name (str, optional): The name of the matrix of which the eigenvalues are computed. Defaults to "hessian".
            chaos (bool, optional): If True, calculates metrics regarding the local Lyapunov matrix. Defaults to False.
            i (int, optional): Id of the model. Defaults to 0.
        """
        eigval_metrics = {}
        name = f"{name}{self.maybe_num(i)}"
        idx_nan_or_inf = torch.logical_or(w.isnan(), w.isinf())
        w = w[~idx_nan_or_inf]
        max_eig = torch.max(w)
        min_eig = torch.min(w)
        eigval_metrics.update(
            {
                f"eigvals/{name}:max_eig": max_eig,
                f"eigvals/{name}:min_eig": min_eig,
                f"eigvals/{name}:eigs_std": torch.std(w),
                f"eigvals/{name}:eigs_mean": torch.mean(w),
            }
        )
        if chaos:
            if max_eig.dtype != torch.float64:
                max_eig = max_eig.to(dtype=torch.float64)
            max_exp_eig = 1 / 2 * torch.log(max_eig)
            if min_eig.dtype != torch.float64:
                min_eig = min_eig.to(dtype=torch.float64)
            min_exp_eig = 1 / 2 * torch.log(min_eig)
            chaotic_eigs_cnt = torch.sum(w > 1)
            nonchaotic_eigs_cnt = torch.sum(w < 1)
            edge_eigs_cnt = torch.sum(w == 1)
            eigval_metrics.update(
                {
                    f"eigvals/{name}:max_exp_eig": max_exp_eig,
                    f"eigvals/{name}:min_exp_eig": min_exp_eig,
                    f"eigvals/{name}:chaotic_cnt": chaotic_eigs_cnt,
                    f"eigvals/{name}:non_chaotic_cnt": nonchaotic_eigs_cnt,
                    f"eigvals/{name}:edge_chaotic_cnt": edge_eigs_cnt,
                }
            )
            if nonchaotic_eigs_cnt > 0:
                max_nonchaotic_eig = torch.max(w[w < 1])
                if max_nonchaotic_eig.dtype != torch.float64:
                    max_nonchaotic_eig = max_nonchaotic_eig.to(dtype=torch.float64)
                max_nonchaotic_exp_eig = 1 / 2 * torch.log(max_nonchaotic_eig)
                eigval_metrics.update(
                    {
                        f"eigvals/{name}:max_non_chaotic_eig": max_nonchaotic_eig,
                        f"eigvals/{name}:max_non_chaotic_exp_eig": max_nonchaotic_exp_eig,
                    }
                )
        else:
            nnz_eigs = torch.sum(w != 0) / w.numel()
            pos_eigs_cnt = torch.sum(w > 0)
            pos_chaotic_eigs_cnt = torch.sum(w[w > 0] >= 2 / lr)
            neg_eigs_cnt = torch.sum(w < 0)
            zero_eigs_cnt = torch.sum(w == 0)
            eigval_metrics.update(
                {
                    f"eigvals/{name}:nnz_eigs": nnz_eigs,
                    f"eigvals/{name}:pos_eigs_cnt": pos_eigs_cnt,
                    f"eigvals/{name}:pos_chaotic_eigs_cnt": pos_chaotic_eigs_cnt,
                    f"eigvals/{name}:neg_eigs_cnt": neg_eigs_cnt,
                    f"eigvals/{name}:zero_eigs_cnt": zero_eigs_cnt,
                    f"eigvals/{name}:eigs_log10_std": torch.log10(torch.std(w)),
                }
            )
        self.logger.log_metrics(eigval_metrics)

    def log_eigvec_metrics(
        self,
        V1: torch.Tensor,
        V2: torch.Tensor,
        name1: str = "hessian",
        name2: str = "hessian",
        i1: int = 0,
        i2: int = 0,
    ):
        """Logs the eigenvector metrics such as overlaps between minimal and maximal eigenvalues
        and average overlap.

        Args:
            V1 (torch.Tensor): Eigenvectors of 1st matrix of shape (model_dim, model_dim).
            V2 (torch.Tensor): Eigenvectors of 2nd matrix of shape (model_dim, model_dim).
            name1 (str, optional): Name of 1st matrix. Defaults to "hessian".
            name2 (str, optional): Name of 2nd matrix. Defaults to "hessian".
            i1 (int, optional): Model id of 1st model. Defaults to 0.
            i2 (int, optional): Model id of 2nd model. Defaults to 0.
        """
        eigvec_metrics = {}
        # Max eigvalue
        name1, name2 = f"{name1}{self.maybe_num(i1)}", f"{name2}{self.maybe_num(i2)}"
        max_abs_cossim = cosine_sim(V1[:, -1], V2[:, -1]).abs().item()
        avg_abs_cossim = cosine_sim(V1[:, -1], V2).abs().mean().item()
        # Min eigvalue
        min_abs_cossim = cosine_sim(V1[:, 0], V2[:, 0]).abs().item()
        min_abs_cossim = cosine_sim(V1[:, 0], V2[:, -1]).abs().item()
        min_abs_cossim = cosine_sim(V1[:, -1], V2[:, 0]).abs().item()
        avg_abs_cossim = cosine_sim(V1[:, 0], V2).abs().mean().item()
        self_dissim_l2 = eigen_dissim(V1, V2, metric="l2")
        eigvec_metrics.update(
            {
                f"eigvecs/{name1}-{name2}:max-max:abs_cossim": max_abs_cossim,
                f"eigvecs/{name1}-{name2}:max-all:avg_abs_cossim": avg_abs_cossim,
                f"eigvecs/{name1}-{name2}:min-min:abs_cossim": min_abs_cossim,
                f"eigvecs/{name1}-{name2}:min-max:abs_cossim": min_abs_cossim,
                f"eigvecs/{name1}-{name2}:max-min:abs_cossim": min_abs_cossim,
                f"eigvecs/{name1}-{name2}:min-all:avg_abs_cossim": avg_abs_cossim,
                f"eigvecs/{name1}-{name2}:dissim_l2": self_dissim_l2,
            }
        )
        self.logger.log_metrics(eigvec_metrics)

    def log_eigvec_grad_metrics(
        self,
        w: torch.Tensor,
        grads: torch.Tensor,
        V: torch.Tensor,
        H: torch.Tensor = None,
        name: str = "hessian",
        chaos: bool = False,
        i: int = 0,
    ):
        """Logs metrics that involve eigenvectors and the gradient
        such as eigenvalue-gradient overlaps and curvature in the direction of the gradient.

        Args:
            w (torch.Tensor): Eigenvalues of a matrix.
            grads (torch.Tensor): Gradient.
            V (torch.Tensor): Eigenvectors of shape (model_dim, model_dim) of a matrix
            H (torch.Tensor, optional): The Hessian. Defaults to None.
            name (str, optional): _description_. Defaults to "hessian".
            chaos (bool, optional): _description_. Defaults to False.
            i (int, optional): _description_. Defaults to 0.
        """
        # Gradient
        eigvec_grad_metrics = {}
        name = f"{name}{self.maybe_num(i)}"
        if H is not None:
            grad_norm = grads / grads.norm()
            curvature_along_gradient = grad_norm.T @ H @ grad_norm
            eigvec_grad_metrics.update(
                {f"hessian:curvature_along_gradient": curvature_along_gradient}
            )
        max_cossim_grad = cosine_sim(grads, V[:, -1]).abs().item()
        min_cossim_grad = cosine_sim(grads, V[:, 0]).abs().item()
        avg_cossim_grad = cosine_sim(grads, V).abs().mean().item()
        eigvec_grad_metrics.update(
            {
                f"eigvecs/{name}:max-grad:abs_cossim": max_cossim_grad,
                f"eigvecs/{name}:min-grad:abs_cossim": min_cossim_grad,
            }
        )
        if chaos:
            is_wnonchaotic = w < 1
            if torch.any(is_wnonchaotic):
                wmin_chaotic_idx = torch.argmin(w[is_wnonchaotic])
                vmin_chaotic = V[:, wmin_chaotic_idx]
                min_chaotic_cossim_grad = cosine_sim(grads, vmin_chaotic)
                eigvec_grad_metrics.update(
                    {
                        f"eigvecs/{name}:chaotic_min-grad:abs_cossim": min_chaotic_cossim_grad
                    }
                )
            is_wchaotic = w > 1
            if torch.any(is_wchaotic):
                wmax_nonchaotic_idx = torch.argmax(w[is_wchaotic])
                vmax_nonchaotic = V[:, wmax_nonchaotic_idx]
                max_nonchaotic_cossim_grad = cosine_sim(grads, vmax_nonchaotic)
                eigvec_grad_metrics.update(
                    {
                        f"eigvecs/{name}:chaotic_min-grad:abs_cossim": max_nonchaotic_cossim_grad
                    }
                )
        else:
            is_wpos = w > 0
            if torch.any(is_wpos):
                wposmin_idx = torch.argmin(w[is_wpos])
                vposmin = V[:, wposmin_idx]
                posmin_cossim_grad = cosine_sim(grads, vposmin)
                eigvec_grad_metrics.update(
                    {f"eigvecs/{name}:pos_min-grad:abs_cossim": posmin_cossim_grad}
                )
                avg_abs_cossim_pos_grad = (
                    cosine_sim(grads, V[:, is_wpos]).abs().mean().item()
                )
                eigvec_grad_metrics.update(
                    {
                        f"eigvecs/{name}:all_pos-grad:avg_abs_cossim": avg_abs_cossim_pos_grad
                    }
                )
            is_wneg = w < 0
            if torch.any(is_wneg):
                wnegmax_idx = torch.argmax(w[is_wneg])
                vnegmax = V[:, wnegmax_idx]
                negmax_cossim_grad = cosine_sim(grads, vnegmax)
                avg_abs_cossim_neg_grad = (
                    cosine_sim(grads, V[:, is_wneg]).abs().mean().item()
                )
                eigvec_grad_metrics.update(
                    {
                        f"eigvecs/{name}:neg_max-grad:abs_cossim": negmax_cossim_grad,
                        f"eigvecs/{name}:all_neg-grad:avg_abs_cossim": avg_abs_cossim_neg_grad,
                    }
                )
        eigvec_grad_metrics.update(
            {f"eigvecs/{name}:all-grad:avg_abs_cossim": avg_cossim_grad}
        )
        self.logger.log_metrics(eigvec_grad_metrics)

    def log_subspace_metric(
        self,
        w_V: torch.Tensor,
        w_W: torch.Tensor,
        V: torch.Tensor,
        W: torch.Tensor,
        id: int = None,
        t: int = None,
        lr: float = None,
        name: str = "lyapunov",
    ):
        """Calculates subspace similarities spanned by eigenvalues from different timesteps.
        Eigenspectrum can come from the Lyapunov matrix or the Hessian.

        Args:
            w_V (torch.Tensor): Eigenvalues corresponding to V
            w_W (torch.Tensor): Eigenvalues corresponding to W
            V (torch.Tensor): Eigenvectors of shape (model_dim, model_dim)
            W (torch.Tensor): Eigenvectors of shape (model_dim, model_dim)
            id (int, optional): Id corresponding to the timestep that is compared to.
                Used for random comparison. Defaults to None.
            t (int, optional): The timestep that will be compared to. Defaults to None.
            lr (float, optional): The current learning rate. Defaults to None.
            name (str, optional): "hessian" or "lyapunov" depending on the origin of the eigenspectrum. Defaults to "lyapunov".
        """
        assert (
            id is not None
        ), "Define id for comparison to a particular random subspace"
        subspace_metrics = {}
        if name == "lyapunov":
            rand_idx_V = self.lya_subspace_random_comp[id][: (w_V > 1).sum()]
            rand_idx_W = self.lya_subspace_random_comp[id][: (w_W > 1).sum()]
            metrics = {
                "chaotic-chaotic": [V[:, w_V > 1], W[:, w_W > 1]],
                "edge_w_chaotic-edge_w_chaotic": [V[:, w_V >= 1], W[:, w_W >= 1]],
                "non_chaotic-non_chaotic": [V[:, w_V < 1], W[:, w_W < 1]],
                "chaotic-non_chaotic": [V[:, w_V > 1], W[:, w_W < 1]],
                "non_chaotic-chaotic": [V[:, w_V < 1], W[:, w_W > 1]],
                "random-random": [V[:, rand_idx_V], W[:, rand_idx_W]],
            }
            for metric_name, spaces in metrics.items():
                subsim = subspace_sim(*spaces).cpu().item()
                randbase_sim = random_baseline_subsim(*spaces).cpu().item()
                rand_comp_sim = subspace_sim(*metrics["random-random"]).cpu().item()
                subspace_metrics.update(
                    {
                        f"subsim/{name}-t={t}:{metric_name}": subsim,
                        f"subsim/{name}-t={t}:baseline:{metric_name}": randbase_sim,
                        f"subsim/{name}-t={t}:{name}-baseline:{metric_name}": subsim
                        - randbase_sim,
                        f"subsim/{name}-t={t}:{name}-random:{metric_name}": subsim
                        - rand_comp_sim,
                    }
                )
                del subsim

        else:
            rand_idx_V = self.hessian_subspace_random_comp[id][: (w_V > 1).sum()]
            rand_idx_W = self.hessian_subspace_random_comp[id][: (w_W > 1).sum()]
            metrics = {
                "neg-neg": [V[:, w_V < 0], W[:, w_W < 0]],
                "edge-edge": [V[:, w_V == 0], W[:, w_W == 0]],
                "pos-pos": [V[:, w_V > 0], W[:, w_W > 0]],
                "pos-neg": [V[:, w_V > 0], W[:, w_W < 0]],
                "neg-pos": [V[:, w_V < 0], W[:, w_W > 0]],
                "chaotic-chaotic": [
                    V[:, (w_V < 0) + (w_V >= 2 / lr)],
                    W[:, (w_V < 0) + (w_V >= 2 / lr)],
                ],
                "random-random": [V[:, rand_idx_V], W[:, rand_idx_W]],
            }
            for metric_name, spaces in metrics.items():
                subsim = subspace_sim(*spaces)
                randbase_sim = random_baseline_subsim(*spaces)
                rand_comp_sim = subspace_sim(*metrics["random-random"])
                subspace_metrics.update(
                    {
                        f"subsim/{name}-t={t}:{metric_name}": subsim,
                        f"subsim/{name}-t={t}:baseline:{metric_name}": randbase_sim,
                        f"subsim/{name}-t={t}:{name}-baseline:{metric_name}": subsim
                        - randbase_sim,
                        f"subsim/{name}-t={t}:{name}-random:{metric_name}": subsim
                        - rand_comp_sim,
                    }
                )
        self.logger.log_metrics(subspace_metrics)

    def create_and_log_subsim(
        self, name: str, w: torch.Tensor, V: torch.Tensor, lr: float = None
    ):
        """Initializes subspace comparison and logs subspace similarites.

        Args:
            name (str): Name of the matrix.
            w (torch.Tensor): Eigenvalues of the current step.
            V (torch.Tensor): Eigenvector of the current step with shape (model_dim, model_dim).
            lr (float, optional): Current learning rate. Defaults to None.
        """
        if name == "lyapunov":
            t_subspace_sim = "t_lya_subspace_sim"
            comp_basis = "lyapunov_comp_basis"
            comp_eigvals = "lyapunov_comp_eigvals"
        elif name == "hessian":
            t_subspace_sim = "t_hessian_subspace_sim"
            comp_basis = "hessian_comp_basis"
            comp_eigvals = "hessian_comp_eigvals"
        else:
            raise NotImplementedError(f"Subsim is not implemented for {name}")

        ts = getattr(self, t_subspace_sim)
        if ts is not None:
            if not hasattr(self, comp_basis):
                setattr(self, comp_basis, [None] * len(ts))
                setattr(self, comp_eigvals, [None] * len(ts))

            for i, t in enumerate(ts):
                if t == self.step:
                    comp_basis_ = getattr(self, comp_basis)
                    comp_basis_[i] = V
                    setattr(self, comp_basis, comp_basis_)
                    comp_eigvals_ = getattr(self, comp_eigvals)
                    comp_eigvals_[i] = w
                    setattr(self, comp_eigvals, comp_eigvals_)
                elif self.step > t:
                    self.log_subspace_metric(
                        getattr(self, comp_eigvals)[i],
                        w,
                        getattr(self, comp_basis)[i],
                        V,
                        id=i,
                        t=t,
                        lr=lr,
                        name=name,
                    )

    def plot_eigs(self, wdict, eigs_min=None, eigs_max=None, name="hessian"):
        """ """
        fig = plt.figure()
        for key, w in wdict.items():
            if (eigs_min is None) and (eigs_max is None):
                bins = 200
            else:
                bins = np.linspace(eigs_min, eigs_max, 200)
            idx_nan_or_inf = torch.logical_or(w.isnan(), w.isinf())
            w = w[~idx_nan_or_inf]
            if isinstance(w, torch.Tensor) and w.device != "cpu":
                w = w.to("cpu")
            counts, bins = np.histogram(w, bins=bins)
            plt.hist(bins[:-1], bins, weights=counts, alpha=0.5, label=key)
        plt.xlim(eigs_min, eigs_max)
        plt.yscale("log")
        plt.grid(axis="y")
        plt.title(f"Step {self.step}")
        plt.legend()
        plot_path = path.join(self.logging_dirs, "hist_tmp.png")
        plt.savefig(plot_path, dpi=500)
        plt.close()
        if isinstance(self.logger, NeptuneLogger):
            self.logger.experiment[f"eigvals/{name}:eigs"].log(File(plot_path))
        elif isinstance(self.logger, TensorBoardLogger):
            self.logger.experiment[f"eigvals/{name}:eigs"].add_figure(fig)

    def save_eigenthings(
        self, w: torch.Tensor, V: torch.Tensor, name: str = "lyapunov", i: int = 0
    ):
        """Saves eigenvalues and eigenvectors in self.storage_dirs. This is important for
        running the distance experiments

        Args:
            w (torch.Tensor): Eigenvalues.
            V (_type_): Eigenvectors of shape (model_dim, model_dim)
            name (str, optional): Name of the matrix from which the eigenspectrum is from. Defaults to "lyapunov".
            i (int, optional): Model id. Defaults to 0.
        """
        name = f"{name}{self.maybe_num(i)}"
        if not path.exists(self.storage_dirs):
            os.makedirs(self.storage_dirs)
        eigenthings = (w.to(dtype=torch.float32), V.to(dtype=torch.float32))
        torch.save(eigenthings, f"{self.storage_dirs}/eigenthings-{name}.pt")

    def load_eigenthings(
        self, name="lyapunov", i=0
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Loads saved eigenvalues and eigenvectors from self.storage_dirs for model with id=i.

        Args:
            name (str, optional): Name of the matrix from which to load the eigenspectrum. Defaults to "lyapunov".
            i (int, optional): Model id. Defaults to 0.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Eigenvalues, eigenvectors.
        """
        name = f"{name}{self.maybe_num(i)}"
        w, V = torch.load(f"{self.storage_dirs}/eigenthings-{name}.pt")
        return w, V

    def load_eigenthings_all(self, name: str = "lyapunov") -> Tuple[list, list]:
        """Loads eigenvalues and eigenvectors for all models.

        Args:
            name (str, optional): Name of the matrix from which to load the eigenspectrum. Defaults to "lyapunov".

        Returns:
            Tuple[list, list]: List of eigenvalues, list of eigenvectors.
        """
        w_list, V_list = [], []
        for i in range(len(self.model)):
            w, V = self.load_eigenthings(name=name, i=i)
            w_list.append(w)
            V_list.append(V)
        return w_list, V_list

    def save_hessian_dist(self, w: torch.Tensor):
        """Saves distribution of the hessian to self.storage_dirs.

        Args:
            w (torch.Tensor): Eigenvalues.
        """
        directory = f"{self.storage_dirs}/hessian_eig_dist"
        if not os.path.exists(directory):
            os.makedirs(directory)
        torch.save(w, f"{directory}/hessian_eigvals_step:{self.step}.pt")

    def training_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx
    ) -> dict:
        """Performs a training step and calculates all the metrics that are enabled
        as defined in ./config.

        Args:
            batch (Tuple[torch.Tensor, torch.Tensor]): Input and labels.

        Returns:
            dict: Loss and accuracy.
        """
        x, y = batch
        optims = self.optimizers()
        mbn = self.maybe_num
        if not isinstance(optims, Sequence):
            optims = [optims]
        for i, (m, opt) in enumerate(zip(self.model, optims)):
            # NOTE: For sgd/cgd only
            opt.zero_grad()
            y_hat = m.forward(x)
            loss = m.criterion(y_hat, y)
            self.manual_backward(loss)
            accuracy = torch.mean((torch.argmax(y_hat, dim=1) == y).to(torch.float64))
            self.step += 1

            with torch.no_grad():
                if any(
                    [
                        isinstance(opt, CGD) and not opt.prune_global,
                        self.log_subsim,
                        self.log_eigenthings,
                        self.log_lyapunov,
                        self.log_lyapunov_subsim,
                    ]
                ):
                    lr = opt.param_groups[0]["lr"]
                    momentum = opt.param_groups[0]["momentum"]
                    grads = torch.cat([p.grad.reshape(-1) for p in m.parameters()])
                    hessian = m.get_hessian().to(x.device)
                    n = hessian.shape[0]
                    if self.eps_eigh > 0:
                        # I = sparse_eye(n, device=hessian.device)
                        hessian = hessian + self.eps_eigh
                    w, V = torch.linalg.eigh(hessian)

            if isinstance(opt, CGD) and not opt.prune_global:
                opt.step(w, V)
            else:
                opt.step()

            with torch.no_grad():
                if any(
                    [
                        self.log_subsim,
                        self.log_eigenthings,
                        self.log_lyapunov,
                        self.log_lyapunov_subsim,
                    ]
                ):
                    wc = (1 - lr * w) ** 2
                    wc, sorted_idx = torch.sort(wc)
                    Vc = V[:, sorted_idx]
                if self.log_subsim:
                    self.create_and_log_subsim("hessian", w, V, lr=lr)
                if self.log_eigenthings:
                    if i not in self.grads_old:
                        self.grads_old[i] = grads
                    grads_overlap = cosine_sim(self.grads_old[i], grads)
                    self.grads_old[i] = grads.clone().detach()
                    sparsity = torch.sum(hessian.reshape(-1) == 0) / hessian.numel()
                    eigenthings_metrics = {
                        f"grad_old{mbn(i)}-grad{mbn(i)}:overlap": grads_overlap,
                        f"hessian{mbn(i)}:sparsity": sparsity,
                    }
                    self.logger.log_metrics(eigenthings_metrics)
                    self.log_eigval_metrics(w, lr, name=f"hessian", i=i)
                    self.log_eigvec_grad_metrics(w, grads, V, name="hessian", i=i)
                    self.log_eigvec_metrics(
                        V, V, name1="hessian", name2="hessian", i1=i, i2=i
                    )
                    vmin, vmax = V[:, 0], V[:, -1]
                    self.log_dist_metrics(grads, w, V, name="hessian", i1=0, i2=i)
                    # Chaos:
                    self.log_eigval_metrics(wc, lr, name="chaos", chaos=True, i=i)
                    self.log_eigvec_grad_metrics(
                        wc, grads, Vc, name="chaos", chaos=True, i=i
                    )
                    self.log_eigvec_metrics(
                        Vc, Vc, name1="chaos", name2="chaos", i1=i, i2=i
                    )
                    vmin, vmax = Vc[:, 0], Vc[:, -1]
                    self.log_dist_metrics(
                        grads, wc, Vc, name="chaos", chaos=True, i1=0, i2=i
                    )
                    if self.save_hessian_dist_every != 0:
                        if self.step % self.save_hessian_dist_every == 0:
                            self.save_hessian_dist(w)
                if self.parallel > 0 and self.model[0].perturb_strat is not None:
                    for i, m in enumerate(self.model):
                        self.log_dist_metrics(
                            None, None, None, name="hessian", i1=0, i2=i
                        )

                if self.log_lyapunov_subsim or self.log_lyapunov:
                    I = sparse_eye(n, device=x.device)
                    hessian_ = hessian
                    if isinstance(opt, CGD) and opt.is_pruning():
                        idx_kept = opt.get_last_idx_kept()
                        if isinstance(idx_kept, torch.Tensor):
                            if opt.prune_global:
                                Vg = opt.Vg[:, idx_kept]
                                Pg = torch.matmul(Vg, Vg.T)
                                hessian_ = Pg.matmul(hessian.matmul(Pg))
                            else:
                                w_ = w[idx_kept]
                                V_ = V[:, idx_kept]
                                n_ = len(w_)
                                idx = torch.empty((2, n_), device=w.device)
                                idx[0, :] = torch.arange(
                                    n_, dtype=torch.int64, device=w.device
                                )
                                idx[1, :] = idx[0, :]
                                W = torch.sparse_coo_tensor(idx, w_, (n_, n_))
                                hessian_ = V_.matmul(W.matmul(V_.T))
                    if i not in self.Y:
                        self.Y[i] = I
                    if i not in self.G:
                        self.G[i] = hessian_
                    HY = matmul_mixed(hessian_, self.Y[i])
                    self.Y[i] = sum_mixed(self.Y[i], -lr * HY)
                    if momentum > 0:
                        self.Y[i] = sum_mixed(self.Y[i], -lr * self.G[i])
                        self.G[i] = sum_mixed(momentum * self.G[i], HY)
                    del HY
                    lyapunov = matmul_mixed(self.Y[i].T, self.Y[i])
                    if self.eps_eigh > 0:
                        # I = sparse_eye(n, device=x.device)
                        lyapunov = lyapunov + self.eps_eigh
                    w_lya, V_lya = torch.linalg.eigh(lyapunov)
                    w_lya = w_lya.to(dtype=torch.float64)
                    w_lya = w_lya ** (1 / self.step)
                if self.log_lyapunov_subsim:
                    self.create_and_log_subsim("lyapunov", w_lya, V_lya)
                if self.log_lyapunov:
                    # Lyapunov exponents:
                    self.log_eigval_metrics(w_lya, lr, name="lyapunov", chaos=True, i=i)
                    self.log_eigvec_grad_metrics(
                        w_lya, grads, V_lya, name="lyapunov", chaos=True, i=i
                    )
                    if self.log_eigenthings:
                        self.log_eigvec_metrics(
                            V_lya, V, name1="lyapunov", name2="hessian", i1=i, i2=i
                        )
                        self.log_eigvec_metrics(
                            V_lya, Vc, name1="lyapunov", name2="chaos", i1=i, i2=i
                        )
                    self.log_dist_metrics(
                        grads, w_lya, V_lya, name="lyapunov", chaos=True, i1=0, i2=i
                    )
                try:
                    del V
                except NameError:
                    pass
                try:
                    del V_lya
                except NameError:
                    pass
                # Plot eigenvalue distributions of matrices
                if (self.plot_mats_every > 0) and (
                    self.step % self.plot_mats_every == 0
                ):
                    hessian = hessian.cpu()
                    plt.figure()
                    eps = 1e-8
                    plt.imshow(torch.log10(eps + hessian.abs()))
                    plt.clim(-8, 1)
                    plt.colorbar()
                    plt.title(f"Step {self.step}")
                    plt.savefig("matrices_tmp.png", dpi=500)
                    plt.close()
                if (self.plot_eigs_every > 0) and (
                    self.step % self.plot_eigs_every == 0
                ):
                    hessian_eigs = dict()
                    chaos_eigs = dict()
                    lyapunov_eigs = dict()
                    if self.log_eigenthings:
                        w, wc = w.cpu(), wc.cpu()
                        hessian_eigs["hessian"] = w
                        chaos_eigs["chaos"] = wc
                    if self.log_lyapunov:
                        w_lya = w_lya.cpu()
                    if isinstance(self.logger, NeptuneLogger):
                        self.logger.experiment["matrices/hessian"].log(
                            File("matrices_tmp.png")
                        )
                    self.plot_eigs(
                        hessian_eigs, eigs_min=-5, eigs_max=+5, name="hessian_all"
                    )
                    self.plot_eigs(
                        chaos_eigs,
                        eigs_min=1 - 10 * lr,
                        eigs_max=1 + 10 * lr,
                        name="chaos_all",
                    )
                    self.plot_eigs(lyapunov_eigs, name="lyapunov_all")
                    for key, eigs in hessian_eigs.items():
                        self.plot_eigs({key: eigs}, eigs_min=-5, eigs_max=+5, name=key)
                    for key, eigs in chaos_eigs.items():
                        self.plot_eigs(
                            {key: eigs},
                            eigs_min=1 - 10 * lr,
                            eigs_max=1 + 10 * lr,
                            name=key,
                        )
                    for key, eigs in lyapunov_eigs:
                        self.plot_eigs(
                            {key: eigs},
                            eigs_min=1 - 10 * lr,
                            eigs_max=1 + 10 * lr,
                            name=key,
                        )

            loss_name = (
                f"loss{mbn(i)}:dist[{self.model[i].perturb_dist}]:strat[{self.model[i].perturb_strat}]"
                if self.parallel
                else "loss"
            )
            acc_name = (
                f"acc{mbn(i)}:dist[{self.model[i].perturb_dist}]:strat[{self.model[i].perturb_strat}]"
                if self.parallel
                else "acc"
            )
            if self.step % self.log_freq == 0:
                if isinstance(self.logger, NeptuneLogger):
                    self.logger.experiment[loss_name].log(loss)
                    self.logger.experiment[acc_name].log(accuracy)
                self.log(acc_name, accuracy, prog_bar=True)
            try:
                del hessian
            except NameError:
                pass
            m.clear_cache()
            self.log("acc", accuracy, prog_bar=True)

        return {"loss": loss, "acc": accuracy}
