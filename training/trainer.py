from __future__ import annotations

import os
import sys
script_dir = os.path.dirname(os.path.realpath(__file__))
parent_dir = os.path.dirname(os.path.dirname(script_dir))
sys.path.append(parent_dir)
import itertools
import random
from pathlib import Path
from typing import Callable, Dict, Optional, Tuple

import torch
from torch import Tensor
from torch.utils.data import DataLoader

from ui_beat.dataset.dataset import HBDataset
from ui_beat.models.multi_head import (
    decoder4pb,
    decoder4qrs,
    encoder4pb,
    encoder4qrs,
    phi_pb,
    phi_qrs,
)
from ui_beat.utils.loss import bce_loss_func, cce_loss_func, sim_loss_func


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
        encoder_pb: torch.nn.Module,
        decoder_pb: torch.nn.Module,
        phi_qrs: torch.nn.Module,
        phi_pb: torch.nn.Module,
        alpha_lr: float,
        beta_lr: float,
        theta_lr: float,
        delta_lr: float,
        model_save_path: Optional[Path] = None,
        device: Optional[torch.device] = None,
    ) -> None:
        self.batch_size = batch_size
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model_save_path = Path(model_save_path) if model_save_path else None

        self.en1 = encoder_qrs.to(self.device)
        self.de1 = decoder_qrs.to(self.device)
        self.en2 = encoder_pb.to(self.device)
        self.de2 = decoder_pb.to(self.device)
        self.phi1 = phi_qrs.to(self.device)
        self.phi2 = phi_pb.to(self.device)

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
        self.beta_optimizer = torch.optim.Adam(
            itertools.chain(self.en2.parameters(), self.de2.parameters()),
            lr=beta_lr,
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
        self.delta_optimizer = torch.optim.Adam(
            self.phi2.parameters(),
            lr=delta_lr,
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
    ) -> None:
        """
        Main training loop consuming a PyTorch DataLoader.
        """
        total_epochs = self.start_epoch + epochs
        for epoch in range(self.start_epoch, total_epochs):
            epoch_loss = 0.0
            epoch_bin_acc = 0.0
            epoch_cat_acc = 0.0

            for step, batch in enumerate(dataloader, start=1):
                loss_value, logits_qrs, logits_pb, target_qrs, target_pb = self.aeu_train_step(batch)

                epoch_loss += loss_value
                epoch_bin_acc += self.binary_accuracy(logits_qrs, target_qrs)
                epoch_cat_acc += self.categorical_accuracy(logits_pb, target_pb)

                if step % log_interval == 0 or step == len(dataloader):
                    avg_loss = epoch_loss / step
                    avg_bin_acc = epoch_bin_acc / step
                    avg_cat_acc = epoch_cat_acc / step
                    self._log_progress(epoch, step, len(dataloader), avg_loss, avg_bin_acc, avg_cat_acc)
                    if log_callback is not None:
                        metrics = {
                            "loss": avg_loss,
                            "qrs_acc": avg_bin_acc,
                            "pb_acc": avg_cat_acc,
                        }
                        log_callback(epoch, step, len(dataloader), metrics, "batch")

            num_batches = max(len(dataloader), 1)
            print(
                f"Epoch {epoch + 1}/{total_epochs} - "
                f"loss: {epoch_loss / num_batches:.4f} - "
                f"QRS acc: {epoch_bin_acc / num_batches:.4f} - "
                f"PB acc: {epoch_cat_acc / num_batches:.4f}"
            )
            if log_callback is not None:
                metrics = {
                    "loss": epoch_loss / num_batches,
                    "qrs_acc": epoch_bin_acc / num_batches,
                    "pb_acc": epoch_cat_acc / num_batches,
                }
                log_callback(epoch, num_batches, num_batches, metrics, "epoch")

            self.save_checkpoint(epoch + 1)

    def aeu_train_step(
        self, batch: Tuple[Tensor, Tensor, Tensor, Tensor, Tensor]
    ) -> Tuple[float, Tensor, Tensor, Tensor, Tensor]:
        """
        Performs one optimisation step across all sub-networks.
        """

        qrs_x, qrs_y, pb_x, pb_y, noise = batch
        qrs_x = qrs_x.to(self.device)
        qrs_y = qrs_y.to(self.device)
        pb_x = pb_x.to(self.device)
        pb_y = pb_y.to(self.device)
        noise = noise.to(self.device)

        qrs_x = qrs_x.permute(0, 2, 1).contiguous()
        pb_x = pb_x.permute(0, 2, 1).contiguous()
        noise = noise.permute(0, 2, 1).contiguous()

        if qrs_y.dim() == 2:
            qrs_y = qrs_y.unsqueeze(-1)
        qrs_y = qrs_y.to(torch.float32)
        pb_y = pb_y.to(torch.float32)

        c1 = random.uniform(0.65, 0.95)
        x1 = c1 * qrs_x + (1.0 - c1) * noise
        c2 = random.choice([-1, 1])
        x2 = c2 * pb_x

        # Encoder/decoder branch for QRS
        self.en1.train()
        self.de1.train()
        self.alpha_optimizer.zero_grad()
        z1 = self.en1(x1)
        lgt1 = self.de1(z1)
        alpha_loss = self.bin_loss(lgt1, qrs_y)
        alpha_loss.backward()
        self.alpha_optimizer.step()
        z1_detached = z1.detach()
        lgt1_detached = lgt1.detach()

        # Encoder/decoder branch for PB
        self.beta_optimizer.zero_grad()
        self.en1.eval()
        with torch.no_grad():
            z21 = self.en1(x2)
        self.en1.train()
        self.en2.train()
        self.de2.train()
        z22 = self.en2(x2)
        z2 = z21 + z22
        lgt2 = self.de2(z2)
        beta_loss = self.cat_loss(lgt2, pb_y)
        beta_loss.backward()
        self.beta_optimizer.step()
        z2_detached = z2.detach()
        lgt2_detached = lgt2.detach()

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

        self.delta_optimizer.zero_grad()
        self.phi2.train()
        mask_nr = pb_y[..., 0].unsqueeze(-1)
        mask_n = pb_y[..., 1].unsqueeze(-1)
        mask_s = pb_y[..., 2].unsqueeze(-1)
        mask_v = pb_y[..., 3].unsqueeze(-1)
        r_nr = self.estimate_r(z2_detached, mask_nr)
        r_n_pb = self.estimate_r(z2_detached, mask_n)
        r_s = self.estimate_r(z2_detached, mask_s)
        r_v = self.estimate_r(z2_detached, mask_v)

        z2_nr, z2_n, z2_s, z2_v = self.phi2(z2_detached)
        sim_loss_nr = self.sim_loss(z2_detached, z2_nr, 1.0 - mask_nr) + self.sim_loss(r_nr, z2_nr, mask_nr)
        sim_loss_n_pb = self.sim_loss(z2_detached, z2_n, 1.0 - mask_n) + self.sim_loss(r_n_pb, z2_n, mask_n)
        sim_loss_s = self.sim_loss(z2_detached, z2_s, 1.0 - mask_s) + self.sim_loss(r_s, z2_s, mask_s)
        sim_loss_v = self.sim_loss(z2_detached, z2_v, 1.0 - mask_v) + self.sim_loss(r_v, z2_v, mask_v)
        delta_loss = sim_loss_nr + sim_loss_n_pb + sim_loss_s + sim_loss_v
        delta_loss.backward()
        self.delta_optimizer.step()

        total_loss = alpha_loss.item() + beta_loss.item()
        return total_loss, lgt1_detached, lgt2_detached, qrs_y.detach(), pb_y.detach()

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

    @staticmethod
    def categorical_accuracy(pred: Tensor, target: Tensor) -> float:
        pred_classes = pred.argmax(dim=-1)
        target_classes = target.argmax(dim=-1)
        return (pred_classes == target_classes).float().mean().item()

    def save_checkpoint(self, epoch: int) -> None:
        if not self.model_save_path:
            return
        self.model_save_path.mkdir(parents=True, exist_ok=True)
        state = {
            "epoch": epoch,
            "en1": self.en1.state_dict(),
            "de1": self.de1.state_dict(),
            "en2": self.en2.state_dict(),
            "de2": self.de2.state_dict(),
            "phi1": self.phi1.state_dict(),
            "phi2": self.phi2.state_dict(),
            "alpha_optimizer": self.alpha_optimizer.state_dict(),
            "beta_optimizer": self.beta_optimizer.state_dict(),
            "theta_optimizer": self.theta_optimizer.state_dict(),
            "delta_optimizer": self.delta_optimizer.state_dict(),
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
        self.en2.load_state_dict(state["en2"])
        self.de2.load_state_dict(state["de2"])
        self.phi1.load_state_dict(state["phi1"])
        self.phi2.load_state_dict(state["phi2"])
        self.alpha_optimizer.load_state_dict(state["alpha_optimizer"])
        self.beta_optimizer.load_state_dict(state["beta_optimizer"])
        self.theta_optimizer.load_state_dict(state["theta_optimizer"])
        self.delta_optimizer.load_state_dict(state["delta_optimizer"])
        self.start_epoch = state.get("epoch", 0)

    @staticmethod
    def _log_progress(epoch: int, step: int, total_steps: int, loss: float, bin_acc: float, cat_acc: float) -> None:
        print(
            f"[Epoch {epoch + 1:03d} | Step {step:04d}/{total_steps:04d}] "
            f"loss: {loss:.4f} | QRS acc: {bin_acc:.4f} | PB acc: {cat_acc:.4f}"
        )


if __name__ == "__main__":
    BASE_DIR = Path(__file__).resolve().parents[1]
    DATA_ROOT = BASE_DIR / "data"
    MODEL_SAVE_PATH = BASE_DIR / "experiments" / "hb_analysis_v2" / "saved_model"

    QRS_DATA_PATH = DATA_ROOT / "qrs_training_set" / "data.npy"
    QRS_REF_PATH = DATA_ROOT / "qrs_training_set" / "refs.npy"
    PB_DATA_PATH = DATA_ROOT / "pb_training_set" / "con_data_v2.npy"
    PB_REF_PATH = DATA_ROOT / "pb_training_set" / "con_refs_v2.npy"
    NOISE_PATH = DATA_ROOT / "pb_training_set" / "noises.npy"

    BATCH_SIZE = 400
    EPOCHS = 20
    ALPHA_LR = 1e-3
    BETA_LR = 1e-3
    THETA_LR = 1e-3
    DELTA_LR = 1e-3

    dataset = HBDataset(
        qrs_data_path=QRS_DATA_PATH,
        qrs_ref_path=QRS_REF_PATH,
        pb_data_path=PB_DATA_PATH,
        pb_ref_path=PB_REF_PATH,
        noise_path=NOISE_PATH,
        n_class=4,
    )

    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)

    trainer = BeatTrainer(
        batch_size=BATCH_SIZE,
        encoder_qrs=encoder4qrs(),
        decoder_qrs=decoder4qrs(),
        encoder_pb=encoder4pb(),
        decoder_pb=decoder4pb(),
        phi_qrs=phi_qrs(),
        phi_pb=phi_pb(),
        alpha_lr=ALPHA_LR,
        beta_lr=BETA_LR,
        theta_lr=THETA_LR,
        delta_lr=DELTA_LR,
        model_save_path=MODEL_SAVE_PATH,
    )

    trainer.train(dataloader, epochs=EPOCHS)
