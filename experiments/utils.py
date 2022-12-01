from typing import Union, Mapping, Any
from omegaconf import DictConfig
import torch
from scipy import optimize
import numpy as np


def perturb_model(
    model: torch.nn.Module,
    dist: float = 1e-1,
    w: torch.Tensor = None,
    V: torch.Tensor = None,
    perturb_strat: str = "max_chaos",
    model_id: int = None,
):
    """Perturbes a model given the Lyapunov eigenvalues and eigenvectors in directions defined by perturb_strat.

    Args:
        model (torch.nn.Module): The model to be perturbed.
        dist (float, optional): The perturbation distance. Defaults to 1e-1.
        w (torch.Tensor, optional): The eigenvalues. Defaults to None.
        V (torch.Tensor, optional): The eigenvectors, if None this will result in a random direction. Defaults to None.
        perturb_strat (str, optional): Defines the perturbation strategy, e.g. 'all' for all strategies. Defaults to "max_chaos".
        model_id (int, optional): In case of perturb_strat='all', this will select the right strategy for the model. Defaults to None.

    Returns:
        torch.Tensor: Distance for logging.
        str: Perturbation strategy for logging.
    """
    with torch.no_grad():
        if V is not None:
            perturb_dict = {
                "max_chaos": V[:, -1],
                "randn_chaos": V[:, w > 1]
                @ torch.randn(int((w > 1).sum()), device=V.device),
                "max_convergence": V[:, 0],
                "randn_convergence": V[:, w < 1]
                @ torch.randn(int((w < 1).sum()), device=V.device),
                "randn": torch.randn_like(V[:, 0]),
            }
            if perturb_strat != "all":
                direction = perturb_dict[perturb_strat]
                direction /= direction.norm()
            else:
                strat = list(perturb_dict.keys())[model_id % 5]
                direction = perturb_dict[strat]
                direction /= direction.norm()
                perturb_strat = strat
        param_id = 0
        for p in model.parameters():
            if V is None:
                eps_ = torch.randn(p.shape, device=p.device)
                eps = dist * eps_ / eps.norm()
            else:
                size = p.shape.numel()
                eps = dist * direction[param_id : param_id + size].reshape(p.shape)
                eps = eps.to(p.device)
                param_id += size
            p.data = p.data + eps

    return dist, perturb_strat


def _rec_log_dictconf(run, conf, key="", sep="/"):
    if isinstance(conf, dict) or isinstance(conf, DictConfig):
        for k in conf.keys():
            _rec_log_dictconf(run, conf[k], key=f"{key}{sep}{k}", sep=sep)
    else:
        run[key] = conf


def log_dictconf(run, conf: Union[DictConfig, Mapping[str, Any]], path="parameters"):
    _rec_log_dictconf(run, conf, key=path)


def lstrip_multiline(x):
    lines = x.split("\n")
    lines_stripped = [l.lstrip() for l in lines]
    return "\n".join(lines_stripped)


def fit_theoretical_bound(
    y_data: torch.Tensor,
    log_interval_scaling: float = 1,
    initial_distance: float = 1e-1,
):
    """Fits the curve (a*t + b*t^2 + distance^2)^(1/2) to the
    distance data. Use this function to plot the theoretical bounds.

    Args:
        y_data (torch.Tensor): The distance data that should be fit to the curve.
        log_interval_scaling (float, optional): This defines the timesteps of the x data.
        Usually the logger logs the data every 10 steps which would result in timesteps (0,10,20,...).
        Defaults to 1.
        initial_distance (float, optional): Defines the previously defined perturbation distance before the training.
        Defaults to 1e-1.

    Returns:
        tuple: timesteps and y data.
    """
    x_data = np.arange(len(y_data)) / 1

    def f(data, a, b):
        return a * data + b * data ** 2 + initial_distance ** 2

    a, b = optimize.curve_fit(f, x_data, y_data ** 2)[0]

    return x_data * log_interval_scaling, np.sqrt(f(x_data, a, b))
