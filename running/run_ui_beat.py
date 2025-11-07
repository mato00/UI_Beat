from __future__ import annotations

import argparse
import os
import sys
script_dir = os.path.dirname(os.path.realpath(__file__))
parent_dir = os.path.dirname(os.path.dirname(script_dir))
sys.path.append(parent_dir)
from pathlib import Path
from typing import Dict

import torch
import wandb
from torch.utils.data import DataLoader

from ui_beat.config.BeatConfig import BeatConfig, load_config
from ui_beat.dataset.dataset import CPSC2019Dataset, cpsc2019_collate_fn
from ui_beat.models.multi_head import encoder4qrs, decoder4qrs, phi_qrs
from ui_beat.models.qrs_model import QRSModel
from ui_beat.training.qrs_trainer import BeatTrainer


def main(args) -> None:

    config = load_config(args.config)
    config.sweep = bool(args.sweep or config.sweep)

    if config.offline:
        os.environ["WANDB_MODE"] = "offline"

    run = wandb.init(
        project=config.project_name,
        name=config.run_name,
        config=None if config.sweep else config.to_wandb_config(),
        reinit=True,
    )

    if config.sweep:
        sweep_updates = wandb.config.as_dict() if hasattr(wandb.config, "as_dict") else dict(wandb.config)
        config.update_from_mapping(sweep_updates)
        config.log_dir = Path(f"{config.training_type}_{config.model_type}_log")
        config.model_save_dir = config.log_dir / "models"
    else:
        config.log_dir = Path(config.log_dir)
        config.model_save_dir = config.log_dir / run.id / "models"

    config.log_dir.mkdir(parents=True, exist_ok=True)
    if config.model_save_dir is not None:
        config.model_save_dir.mkdir(parents=True, exist_ok=True)

    dataset_root = Path(config.dataset_root)
    train_root = dataset_root / "cpsc2019_train"
    test_root = dataset_root / "cpsc2019_test"

    train_dataset = CPSC2019Dataset(root=train_root)
    test_dataset = CPSC2019Dataset(root=test_root)

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        collate_fn=cpsc2019_collate_fn,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        collate_fn=cpsc2019_collate_fn,
    )

    trainer = BeatTrainer(
        batch_size=config.batch_size,
        encoder_qrs=encoder4qrs(),
        decoder_qrs=decoder4qrs(),
        phi_qrs=phi_qrs(),
        alpha_lr=config.alpha_lr,
        theta_lr=config.theta_lr,
        early_stop_patience=config.early_stop_patience,
        model_save_path=config.model_save_dir,
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    )

    def log_callback(epoch: int, step: int, total_steps: int, metrics: Dict[str, float], phase: str) -> None:
        prefix = "train" if phase in {"batch", "epoch"} else phase
        log_dict = {f"{prefix}/{key}": value for key, value in metrics.items()}
        log_dict["epoch"] = epoch + 1
        if phase == "batch":
            wandb.log(log_dict, commit=False)
        else:
            wandb.log(log_dict)

    best_model_path = config.model_save_dir / "best_model.pt" if config.model_save_dir else None

    trainer.train(
        dataloader=train_loader,
        epochs=config.epochs,
        log_interval=config.log_interval,
        log_callback=log_callback,
        val_loader=test_loader,
        best_model_path=best_model_path,
    )

    if best_model_path and best_model_path.exists():
        checkpoint = torch.load(best_model_path, map_location=trainer.device)
        full_model = QRSModel(encoder4qrs(), decoder4qrs(), phi_qrs()).to(trainer.device)
        full_model.load_state_dict(checkpoint["model_state"])
        full_model.eval()

        test_metrics = trainer.evaluate_full_model(test_loader, full_model)
        wandb.log({f"test_{k}": v for k, v in test_metrics.items()})

    run.finish()


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Train QRS/PB models with wandb logging.")
    parser.add_argument("--config", type=str, default="BeatConfig", choices=["BeatConfig"])
    parser.add_argument("--sweep", default=False, action="store_true")
    args = parser.parse_args()

    main(args)
