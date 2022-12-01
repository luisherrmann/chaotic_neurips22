import re
import os
import os.path as path
from os.path import abspath
import glob
import hydra
from omegaconf import OmegaConf, DictConfig, open_dict
from typing import Union, Sequence
import torch
import torch.nn as nn
import torch.optim as optim
from pytorch_lightning import Trainer, LightningModule
from pytorch_lightning.utilities.seed import seed_everything
from pytorch_lightning.loggers import TensorBoardLogger
from neptune.new.integrations.pytorch_lightning import NeptuneLogger

from o2grad.backprop.o2model import O2Model
from models.skeletons import FFNSkeleton
from optim.cgd import CGD
from models.torch import get_model
from dataset import prepare_ds
from utils import log_dictconf, lstrip_multiline, perturb_model


SCRIPT_PATH = abspath(__file__)
low, high = re.search("experiments", SCRIPT_PATH).span()
HOME = SCRIPT_PATH[:high]


@hydra.main(config_path="config", config_name="train", version_base=None)
def main(args: DictConfig):
    MSG = f"""
    Run hyperparamters
    ------------------
    """
    print(lstrip_multiline(MSG))
    print(OmegaConf.to_yaml(args))
    single_or_multirun(args)


def single_or_multirun(args: DictConfig):
    """Starts a single or multiple runs as defined in args.

    Args:
        args (DictConfig): Configurations that get loaded when calling main()
            as defined in the .yml files in the config folder.
    """
    if args.optim.name == "cgd" and args.optim.prune_global:
        if args.optim.prune_global_checkpoint:
            checkpoint_dir = args.optim.prune_global_checkpoint
            eigenthings_path = os.path.join(checkpoint_dir, "eigenthings-lyapunov.pt")
            if os.path.exists(eigenthings_path):
                wg, Vg = torch.load(eigenthings_path)
            else:
                pattern = os.path.join(checkpoint_dir, "eigenthings-lyapunov[*].pt")
                wg, Vg = [], []
                for eigenthings_path in glob.glob(pattern):
                    wg_, Vg_ = torch.load(eigenthings_path)
                    wg.append(wg_)
                    Vg.append(Vg_)
        else:
            if not args.logging.log_lyapunov:
                raise Exception(
                    "Global pruning only works with lyapunov matrix calculation enabled!"
                )
            args_sgd = OmegaConf.create(args)
            default_args_sgd = hydra.compose(config_name="optim/sgd")
            # Copy parameters from sgd
            with open_dict(default_args_sgd):
                for k in default_args_sgd.optim.keys():
                    if k != "name":
                        default_args_sgd.optim[k] = args.optim[k]
            args_sgd.optim = default_args_sgd.optim
            logging_cfg = hydra.compose(config_name="logging/preglobal")
            args_sgd.logging = logging_cfg.logging
            print("SGD:")
            print(OmegaConf.to_yaml(args_sgd))
            model = process_and_train(args_sgd)
            wg, Vg = model.load_eigenthings_all(name="lyapunov")
        process_and_train(args, wg, Vg)
    else:
        process_and_train(args)


def process_and_train(
    args: DictConfig,
    wg: Union[torch.Tensor, Sequence[torch.Tensor]] = None,
    Vg: Union[torch.Tensor, Sequence[torch.Tensor]] = None,
) -> LightningModule:
    if isinstance(wg, Sequence):
        wg = [*wg]
    if isinstance(Vg, Sequence):
        Vg = [*Vg]

    trainer_kwargs = {}
    all_tags = [args.dataset, args.model.name, args.optim.name]
    if args.debug:
        all_tags += ["debug"]
    if args.optim.name == "cgd":
        all_tags += [args.optim.prune_set]
        if args.optim.noise:
            all_tags += ["noise"]
        if args.optim.prune_global:
            all_tags += ["global"]
    if args.parallel > 1:
        all_tags += ["distance"]
    if args.logging.t_lya_subsim is not None:
        all_tags += ["subsim"]
    if args.optim.momentum != 0:
        all_tags += ["momentum"]
    all_tags += list(args.logging.tags)
    # replace the part below to log your own experiments on Neptune

    if args.logging.logger_type == "neptune":
        neptune_logger = NeptuneLogger(
            # api_key="XXXX",
            # project="username/project",
            project="luisherrmann/ChaosPaper",
            capture_stdout=False,
            capture_stderr=True,
            log_model_checkpoints=False,
            tags=all_tags,
        )
        trainer_kwargs["logger"] = neptune_logger
        args_dict = OmegaConf.to_container(args, resolve=True)
        log_dictconf(neptune_logger.experiment, args_dict, path="parameters")
    elif args.logging.logger_type == "tensorboard":
        TB_PATH = os.path.join(HOME, "logs", "tensorboard")
        run_type = "-".join(
            [
                str(x)
                for x in (
                    args.dataset,
                    args.model.name,
                    args.model.activation,
                    args.optim.momentum,
                )
            ]
        )
        RUN_PATH = os.path.join(TB_PATH, run_type)
        if not os.path.exists(RUN_PATH):
            os.makedirs(RUN_PATH)
        tb_logger = TensorBoardLogger(RUN_PATH)
        trainer_kwargs["logger"] = tb_logger
        dict_args = OmegaConf.to_container(args)
        tb_logger.log_hyperparams(dict_args)
    DATA_PATH = path.join(HOME, "data")
    train_loader, val_loader = prepare_ds(
        DATA_PATH, args.dataset, args.batch_size, debug=args.debug
    )

    models = []
    optimizers = []
    for i in range(args.parallel):
        args.seed = int(seed_everything(args.seed))
        model = get_model(args.model)

        if i > 0:
            if args.logging.lyapunov_perturb_eigs is not None:
                lya_eigvals, lya_eigvecs = torch.load(
                    args.logging.lyapunov_perturb_eigs
                )

            else:
                lya_eigvecs = None
                lya_eigvals = None
            if args.perturb_strat != "all":
                dist, strat = perturb_model(
                    model,
                    dist=args.perturb_eps[i],
                    w=lya_eigvals,
                    V=lya_eigvecs,
                    perturb_strat=args.perturb_strat,
                )
            else:
                dist, strat = perturb_model(
                    model,
                    dist=args.perturb_eps[(i - 1) // 5],
                    w=lya_eigvals,
                    V=lya_eigvecs,
                    perturb_strat=args.perturb_strat,
                    model_id=i - 1,
                )
        criterion = nn.CrossEntropyLoss()
        o2model = O2Model(model, criterion)
        o2model.disable_progressbar()
        if i > 0:
            o2model.perturb_dist = dist
            o2model.perturb_strat = strat
        else:
            o2model.perturb_dist = 0
            o2model.perturb_strat = "origin"
        models.append(o2model)
        if args.optim.name == "cgd":
            wg_i = wg[i] if isinstance(wg, list) else wg
            Vg_i = Vg[i] if isinstance(Vg, list) else Vg
            o2model.enable_o2backprop()
            opt = CGD(
                o2model.parameters(),
                lr=args.optim.lr,
                momentum=args.optim.momentum,
                k=args.optim.k,
                prune_global=args.optim.prune_global,
                prune_set=args.optim.prune_set,
                prune_start=args.optim.prune_start,
                prune_k=args.optim.prune_k,
                noise=args.optim.noise,
                noise_fac=args.optim.noise_fac,
                wg=wg_i,
                Vg=Vg_i,
                ret_idx=True,
            )
        elif args.optim.name == "sgd":
            opt = optim.SGD(
                o2model.parameters(), momentum=args.optim.momentum, lr=args.optim.lr
            )
        optimizers.append(opt)
        pl_model = FFNSkeleton(
            models,
            optimizer=optimizers,
            eps_eigh=args.eps_eigh,
            plot_mats_every=args.logging.plot_mats_every,
            plot_eigs_every=args.logging.plot_eigs_every,
            log_eigenthings=args.logging.log_eigenthings,
            log_subsim=args.logging.log_subsim,
            log_lyapunov=args.logging.log_lyapunov,
            log_lyapunov_subsim=args.logging.log_lyapunov_subsim,
            log_freq=args.logging.log_freq,
            save_lyapunov_at=args.logging.save_lyapunov_at,
            save_hessian_dist_every=args.logging.save_hessian_dist_every,
            t_lya_subspace_sim=args.logging.t_lya_subsim,
        )

    if args.gpus:
        gpus = [int(s.split(":")[1]) for s in args.gpus]
        trainer_kwargs["accelerator"] = "gpu"
        trainer_kwargs["devices"] = gpus
        trainer_kwargs["auto_select_gpus"] = False
    else:
        trainer_kwargs["accelerator"] = "cpu"
    trainer = Trainer(max_epochs=args.max_epochs, **trainer_kwargs)
    trainer.fit(pl_model, train_loader, val_loader)
    return pl_model


if __name__ == "__main__":
    main()
