import os
import glob
import unittest
import hydra
from hydra.core.global_hydra import GlobalHydra
import torch
import neptune.new as neptune

from train import process_and_train, single_or_multirun
from o2grad.linalg import matmul_mixed


class TrainTest(unittest.TestCase):
    def test_process_and_train(self):
        hydra.initialize(config_path="../config", version_base=None)
        cfg = hydra.compose(
            "train",
            overrides=[
                "max_epochs=1",
                "seed=42",
                "debug=true",
                "logging.log_lyapunov=true",
            ],
        )
        pl_model = process_and_train(cfg)
        print(pl_model.model_name)
        print(pl_model.storage_dirs)

    def test_tb_logger(self):
        hydra.initialize(config_path="../config", version_base=None)
        cfg = hydra.compose(
            "train",
            overrides=[
                "max_epochs=1",
                "seed=42",
                "debug=true",
                "logging.logger_type=tensorboard",
            ],
        )
        pl_model = process_and_train(cfg)
        log_path = os.path.join(
            pl_model.logger.save_dir,
            "lightning_logs",
            f"version_{pl_model.logger.version}",
        )
        tblogs = glob.glob(f"{log_path}/events.out.tfevents.*")
        self.assertTrue(len(tblogs) > 0)

    def test_neptune_logger(self):
        hydra.initialize(config_path="../config", version_base=None)
        cfg = hydra.compose(
            "train",
            overrides=[
                "max_epochs=1",
                "seed=42",
                "debug=true",
                "logging.logger_type=neptune",
            ],
        )
        pl_model = process_and_train(cfg)
        run_id = pl_model.model_name
        project_name = pl_model.logger._project_name
        run = neptune.init(project=project_name, run=run_id, mode="read-only")
        for key in ("acc", "loss"):
            vals = run[key].fetch_values()
            self.assertTrue(len(vals) > 0)
        run.stop()

    def test_prune_global(self):
        hydra.initialize(config_path="../config", version_base=None)
        cfg = hydra.compose(
            "train",
            overrides=[
                "max_epochs=1",
                "seed=42",
                "debug=true",
                "optim=cgd",
                "optim.prune_global=true",
            ],
        )
        single_or_multirun(cfg)

    def test_prune_global_from_checkpoint(self):
        hydra.initialize(config_path="../config", version_base=None)
        cfg = hydra.compose(
            "train",
            overrides=[
                "max_epochs=1",
                "seed=42",
                "debug=true",
            ],
        )
        pl_model = process_and_train(cfg)
        model_path = pl_model.storage_dirs
        del pl_model
        cfg = hydra.compose(
            "train",
            overrides=[
                "max_epochs=1",
                "seed=42",
                "debug=true",
                "optim=cgd",
                "optim.prune_global=true",
                f"optim.prune_global_checkpoint={model_path}",
            ],
        )
        single_or_multirun(cfg)

    def test_save_lyapunov_at(self):
        hydra.initialize(config_path="../config", version_base=None)
        cfg = hydra.compose(
            "train",
            overrides=[
                "max_epochs=2",
                "seed=42",
                "debug=true",
                "logging.log_lyapunov=True",
                "logging.save_lyapunov_at=0",
            ],
        )
        pl_model = process_and_train(cfg)
        pattern = os.path.join(pl_model.storage_dirs, "eigenthings-lyapunov*.pt")
        for i, lya_path in enumerate(glob.glob(pattern)):
            w_lya, V_lya = torch.load(lya_path)
            Y = pl_model.Y[i]
            lyapunov = matmul_mixed(Y.T, Y)
            w_lya_, V_lya_ = torch.linalg.eigh(lyapunov)
            self.assertTrue(not torch.all(w_lya == w_lya_))
            self.assertTrue(not torch.all(V_lya == V_lya_))

    def test_reproducibility(self):
        hydra.initialize(config_path="../config", version_base=None)
        cfg = hydra.compose(
            "train", overrides=["max_epochs=1", "seed=42", "debug=true"]
        )
        model1 = process_and_train(cfg)
        model2 = process_and_train(cfg)
        self.assertFalse(model1 is model2)
        for (k1, p1), (k2, p2) in zip(
            model1.named_parameters(), model2.named_parameters()
        ):
            self.assertTrue(k1 == k2)
            self.assertTrue(torch.allclose(p1, p2))

    def setUp(self) -> None:
        if GlobalHydra().is_initialized():
            GlobalHydra().instance().clear()


if __name__ == "__main__":
    unittest.main()
