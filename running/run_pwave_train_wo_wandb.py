from __future__ import annotations

import argparse
import datetime as dt
import os
import sys
script_dir = os.path.dirname(os.path.realpath(__file__))
parent_dir = os.path.dirname(os.path.dirname(script_dir))
sys.path.append(parent_dir)
from pathlib import Path
from typing import Dict

import torch
from torch.utils.data import DataLoader

from config.BeatConfig import BeatConfig, load_config
from dataset.pwave_dataset import PWaveSegmentDataset, pwave_collate_fn
from models.multi_head import decoder4qrs, encoder4qrs, phi_qrs
from models.qrs_model import QRSModel
from training.beat_trainer import BeatTrainer


def _ensure_dirs(config: BeatConfig) -> None:
    """
    Ensure logging/model directories exist, filling defaults if needed.
    """
    if not config.run_name:
        timestamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
        config.run_name = f"pwave_{timestamp}"

    config.log_dir = Path(config.log_dir)
    if config.model_save_dir is None:
        config.model_save_dir = config.log_dir / config.run_name / "models"
    else:
        config.model_save_dir = Path(config.model_save_dir)

    config.log_dir.mkdir(parents=True, exist_ok=True)
    config.model_save_dir.mkdir(parents=True, exist_ok=True)


def main(args: argparse.Namespace) -> None:
    config = load_config(args.config)
    config.sweep = False
    _ensure_dirs(config)

    dataset_root = Path(config.dataset_root) / "p_wave_dataset"
    train_dataset = PWaveSegmentDataset(root=dataset_root, split="train")
    val_dataset = PWaveSegmentDataset(root=dataset_root, split="val")
    test_dataset = PWaveSegmentDataset(root=dataset_root, split="test")

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        collate_fn=pwave_collate_fn,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        collate_fn=pwave_collate_fn,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        collate_fn=pwave_collate_fn,
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
        metric_str = ", ".join(f"{key}: {value:.4f}" for key, value in metrics.items())
        print(f"[{prefix.upper()}] epoch {epoch + 1}/{config.epochs} step {step}/{total_steps} - {metric_str}")

    best_model_path = config.model_save_dir / "best_model.pt"

    trainer.train(
        dataloader=train_loader,
        epochs=config.epochs,
        log_interval=config.log_interval,
        log_callback=log_callback,
        val_loader=val_loader,
        best_model_path=best_model_path,
    )

    if best_model_path.exists():
        checkpoint = torch.load(best_model_path, map_location=trainer.device)
        full_model = QRSModel(encoder4qrs(), decoder4qrs(), phi_qrs()).to(trainer.device)
        full_model.load_state_dict(checkpoint["model_state"])
        full_model.eval()

        test_metrics = trainer.evaluate_full_model(test_loader, full_model)
        metrics_str = ", ".join(f"{k}: {v:.4f}" for k, v in test_metrics.items())
        print(f"[TEST] {metrics_str}")
        print(f"Best model stored at: {best_model_path}")
    else:
        print("No best model found. Check training logs for issues.")


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Train the P-wave detector without wandb logging.")
    parser.add_argument("--config", type=str, default="BeatConfig", choices=["BeatConfig"])
    args = parser.parse_args()

    main(args)
