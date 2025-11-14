from __future__ import annotations

import os
import sys
script_dir = os.path.dirname(os.path.realpath(__file__))
parent_dir = os.path.dirname(os.path.dirname(script_dir))
sys.path.append(parent_dir)
import itertools
import random
from pathlib import Path
from typing import Callable, Dict, Optional

import torch
from torch import Tensor
from torch.utils.data import DataLoader

from UI_Beat.models.qrs_model import QRSModel
from UI_Beat.utils.loss import bce_loss_func, cce_loss_func, sim_loss_func


torch.backends.cudnn.benchmark = True


class BeatTrainer:
    """
    PyTorch reimplementation of the original TensorFlow trainer.
    """

    def __init__(
        self,
        batch_size: int,
        encoder_qrs: torch.nn.Module,
        decoder_qrs: torch.nn.Module,
        phi_qrs: torch.nn.Module,
        alpha_lr: float,
        theta_lr: float,
        early_stop_patience: int = 20,
        model_save_path: Optional[Path] = None,
        device: Optional[torch.device] = None,
    ) -> None:
        self.batch_size = batch_size
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model_save_path = Path(model_save_path) if model_save_path else None
        self.early_stop_patience = early_stop_patience

        self.en1 = encoder_qrs.to(self.device)
        self.de1 = decoder_qrs.to(self.device)
        self.phi1 = phi_qrs.to(self.device)

        self.bin_loss = bce_loss_func
        self.cat_loss = cce_loss_func
        self.sim_loss = sim_loss_func

        self.alpha_optimizer = torch.optim.Adam(
            itertools.chain(self.en1.parameters(), self.de1.parameters()),
            lr=alpha_lr,
            betas=(0.9, 0.9),
            eps=1e-8,
            amsgrad=True,
        )

        self.theta_optimizer = torch.optim.Adam(
            self.phi1.parameters(),
            lr=theta_lr,
            betas=(0.9, 0.9),
            eps=1e-8,
            amsgrad=True,
        )

        self.start_epoch = 0
        self.restore()

    def train(
        self,
        dataloader: DataLoader,
        epochs: int,
        log_interval: int = 10,
        log_callback: Optional[
            Callable[[int, int, int, Dict[str, float], str], None]
        ] = None,
        val_loader: Optional[DataLoader] = None,
        best_model_path: Optional[Path] = None,
    ) -> None:
        """
        Main training loop consuming a PyTorch DataLoader.
        """
        total_epochs = self.start_epoch + epochs
        best_metric = float("-inf")
        best_path = (
            Path(best_model_path)
            if best_model_path is not None
            else (self.model_save_path / "best_model.pt" if self.model_save_path else None)
        )
        epochs_without_improvement = 0

        for epoch in range(self.start_epoch, total_epochs):
            epoch_loss = 0.0
            epoch_bin_acc = 0.0

            for step, batch in enumerate(dataloader, start=1):
                loss_value, logits_qrs, target_qrs = self.aeu_train_step(batch)

                epoch_loss += loss_value
                epoch_bin_acc += self.binary_accuracy(logits_qrs, target_qrs)

                if step % log_interval == 0 or step == len(dataloader):
                    avg_loss = epoch_loss / step
                    avg_bin_acc = epoch_bin_acc / step
                    self._log_progress(epoch, step, len(dataloader), avg_loss, avg_bin_acc)
                    if log_callback is not None:
                        metrics = {
                            "loss": avg_loss,
                            "qrs_acc": avg_bin_acc,
                        }
                        log_callback(epoch, step, len(dataloader), metrics, "batch")

            num_batches = max(len(dataloader), 1)
            print(
                f"Epoch {epoch + 1}/{total_epochs} - "
                f"loss: {epoch_loss / num_batches:.4f} - "
                f"QRS acc: {epoch_bin_acc / num_batches:.4f}"
            )
        if log_callback is not None:
            metrics = {
                "loss": epoch_loss / num_batches,
                "qrs_acc": epoch_bin_acc / num_batches,
            }
            log_callback(epoch, num_batches, num_batches, metrics, "epoch")

        if val_loader is not None:
            val_metrics = self.evaluate(val_loader)
            print(
                f"Validation - loss: {val_metrics['loss']:.4f} - "
                f"QRS acc: {val_metrics['qrs_acc']:.4f}"
            )
            if log_callback is not None:
                log_callback(epoch, num_batches, num_batches, val_metrics, "val")
                log_callback(epoch, num_batches, num_batches, val_metrics, "test")

            if val_metrics["qrs_acc"] > best_metric:
                best_metric = val_metrics["qrs_acc"]
                self.save_best_model(best_path)
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1
                if epochs_without_improvement >= self.early_stop_patience:
                    print(
                        f"No validation improvement for {self.early_stop_patience} consecutive epochs. "
                        "Early stopping triggered."
                    )
                    self.save_checkpoint(epoch + 1)
                    return

            self.save_checkpoint(epoch + 1)

    def evaluate(self, dataloader: DataLoader) -> Dict[str, float]:
        self.en1.eval()
        self.de1.eval()

        total_loss = 0.0
        total_acc = 0.0
        total_batches = 0

        with torch.no_grad():
            for batch in dataloader:
                qrs_x, qrs_y = self._prepare_batch(batch)
                logits = self.de1(self.en1(qrs_x))
                loss = self.bin_loss(logits, qrs_y)
                total_loss += loss.item()
                total_acc += self.binary_accuracy(logits, qrs_y)
                total_batches += 1

        self.en1.train()
        self.de1.train()

        if total_batches == 0:
            return {"loss": 0.0, "qrs_acc": 0.0}

        return {
            "loss": total_loss / total_batches,
            "qrs_acc": total_acc / total_batches,
        }

    def evaluate_full_model(
        self,
        dataloader: DataLoader,
        model: Optional[torch.nn.Module] = None,
    ) -> Dict[str, float]:
        model = model or QRSModel(self.en1, self.de1, self.phi1)
        model = model.to(self.device)
        model.eval()

        total_loss = 0.0
        total_acc = 0.0
        total_batches = 0

        with torch.no_grad():
            for batch in dataloader:
                qrs_x, qrs_y = self._prepare_batch(batch)
                logits = model(qrs_x, return_projection=False)
                loss = self.bin_loss(logits, qrs_y)
                total_loss += loss.item()
                total_acc += self.binary_accuracy(logits, qrs_y)
                total_batches += 1

        if total_batches == 0:
            return {"loss": 0.0, "qrs_acc": 0.0}

        return {
            "loss": total_loss / total_batches,
            "qrs_acc": total_acc / total_batches,
        }

    def _extract_batch(self, batch: object) -> tuple[Tensor, Tensor]:
        if isinstance(batch, dict):
            qrs_x = batch["signal"]
            qrs_y = batch["mask"]
        elif isinstance(batch, (list, tuple)):
            if len(batch) < 2:
                raise ValueError("Batch tuple must contain at least signal and mask tensors.")
            qrs_x = batch[0]
            qrs_y = batch[1]
        else:
            raise TypeError(f"Unsupported batch type: {type(batch)}")
        return qrs_x, qrs_y

    def _prepare_batch(self, batch: object) -> tuple[Tensor, Tensor]:
        qrs_x, qrs_y = self._extract_batch(batch)
        qrs_x = qrs_x.to(self.device)
        qrs_y = qrs_y.to(self.device)
        if qrs_x.dim() == 3 and qrs_x.shape[1] > qrs_x.shape[2]:
            qrs_x = qrs_x.transpose(1, 2).contiguous()
        if qrs_y.dim() == 2:
            qrs_y = qrs_y.unsqueeze(-1)
        qrs_y = qrs_y.to(torch.float32)
        return qrs_x, qrs_y

    def aeu_train_step(self, batch: object) -> tuple[float, Tensor, Tensor]:
        """
        Performs one optimisation step across all sub-networks.
        """

        qrs_x, qrs_y = self._prepare_batch(batch)

        # Encoder/decoder branch for QRS
        self.en1.train()
        self.de1.train()
        self.alpha_optimizer.zero_grad()
        z1 = self.en1(qrs_x)
        lgt1 = self.de1(z1)
        alpha_loss = self.bin_loss(lgt1, qrs_y)
        alpha_loss.backward()
        self.alpha_optimizer.step()
        z1_detached = z1.detach()
        lgt1_detached = lgt1.detach()

        # Label-biased transformations
        self.theta_optimizer.zero_grad()
        self.phi1.train()
        mask_q = qrs_y
        mask_nq = 1.0 - mask_q
        z1_p, z1_n = self.phi1(z1_detached)
        r_p = self.estimate_r(z1_detached, mask_q)
        r_n = self.estimate_r(z1_detached, mask_nq)
        sim_loss_p = self.sim_loss(z1_detached, z1_p, 1.0 - mask_q) + self.sim_loss(r_p, z1_p, mask_q)
        sim_loss_n = self.sim_loss(z1_detached, z1_n, 1.0 - mask_nq) + self.sim_loss(r_n, z1_n, mask_nq)
        theta_loss = sim_loss_p + sim_loss_n
        theta_loss.backward()
        self.theta_optimizer.step()

        total_loss = alpha_loss.item() + theta_loss.item()

        return total_loss, lgt1_detached, qrs_y.detach()

    def estimate_r(self, z: Tensor, mask: Tensor) -> Tensor:
        if mask.dim() == 2:
            mask = mask.unsqueeze(-1)
        mask = mask.to(z.dtype)
        denominator = mask.sum()
        if denominator.item() == 0:
            base = torch.zeros(z.size(-1), device=z.device, dtype=z.dtype)
        else:
            weighted = (z * mask).sum(dim=(0, 1))
            base = weighted / denominator
        return base.unsqueeze(0).unsqueeze(0).expand(z.size(0), z.size(1), -1)

    @staticmethod
    def binary_accuracy(pred: Tensor, target: Tensor) -> float:
        pred_label = (pred >= 0.5).float()
        target = target.float()
        return (pred_label == target).float().mean().item()

    def save_best_model(self, path: Optional[Path]) -> None:
        if path is None:
            return
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        full_model = QRSModel(self.en1, self.de1, self.phi1)
        bundle = {
            "model_state": full_model.state_dict(),
            "model_class": QRSModel.__name__,
        }
        torch.save(bundle, path)

    def save_checkpoint(self, epoch: int) -> None:
        if not self.model_save_path:
            return
        self.model_save_path.mkdir(parents=True, exist_ok=True)
        state = {
            "epoch": epoch,
            "en1": self.en1.state_dict(),
            "de1": self.de1.state_dict(),
            "phi1": self.phi1.state_dict(),
            "alpha_optimizer": self.alpha_optimizer.state_dict(),
            "theta_optimizer": self.theta_optimizer.state_dict(),
        }
        torch.save(state, self.model_save_path / "trainer_state.pt")

    def restore(self) -> None:
        if not self.model_save_path:
            return
        state_path = self.model_save_path / "trainer_state.pt"
        if not state_path.exists():
            return

        state = torch.load(state_path, map_location=self.device)
        self.en1.load_state_dict(state["en1"])
        self.de1.load_state_dict(state["de1"])
        self.phi1.load_state_dict(state["phi1"])
        self.alpha_optimizer.load_state_dict(state["alpha_optimizer"])
        self.theta_optimizer.load_state_dict(state["theta_optimizer"])
        self.start_epoch = state.get("epoch", 0)

    @staticmethod
    def _log_progress(epoch: int, step: int, total_steps: int, loss: float, bin_acc: float) -> None:
        print(
            f"[Epoch {epoch + 1:03d} | Step {step:04d}/{total_steps:04d}] "
            f"loss: {loss:.4f} | QRS acc: {bin_acc:.4f}"
        )
