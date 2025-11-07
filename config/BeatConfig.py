from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional


BASE_DIR = Path(__file__).resolve().parents[1]


@dataclass
class BeatConfig:
    project_name: str = "qrs_detection"
    run_name: Optional[str] = None
    offline: bool = False

    training_type: str = "qrs"
    model_type: str = "multi_head"

    batch_size: int = 400
    epochs: int = 20
    log_interval: int = 10
    early_stop_patience: int = 20

    alpha_lr: float = 1e-3
    beta_lr: float = 1e-3
    theta_lr: float = 1e-3
    delta_lr: float = 1e-3

    dataset_root: Path = field(default_factory=lambda: BASE_DIR / "data")
    log_dir: Path = field(default_factory=lambda: BASE_DIR / "experiments" / "qrs_detection" / "logs")
    model_save_dir: Optional[Path] = None

    sweep: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.dataset_root, Path):
            self.dataset_root = Path(self.dataset_root)
        if not isinstance(self.log_dir, Path):
            self.log_dir = Path(self.log_dir)
        if self.model_save_dir is not None and not isinstance(self.model_save_dir, Path):
            self.model_save_dir = Path(self.model_save_dir)

    def to_wandb_config(self) -> Dict[str, Any]:
        payload = asdict(self)
        payload["dataset_root"] = str(self.dataset_root)
        payload["log_dir"] = str(self.log_dir)
        payload["model_save_dir"] = str(self.model_save_dir) if self.model_save_dir else None
        return payload

    def update_from_mapping(self, updates: Dict[str, Any]) -> None:
        path_fields = {"dataset_root", "log_dir", "model_save_dir"}
        for key, value in updates.items():
            if not hasattr(self, key):
                continue
            if key in path_fields and value is not None:
                setattr(self, key, Path(value))
            else:
                setattr(self, key, value)


CONFIG_REGISTRY: Dict[str, BeatConfig] = {
    "BeatConfig": BeatConfig(),
}


def load_config(name: str) -> BeatConfig:
    if name not in CONFIG_REGISTRY:
        raise ValueError(f"Unknown config '{name}'. Available options: {list(CONFIG_REGISTRY.keys())}")
    config = CONFIG_REGISTRY[name]
    return BeatConfig(**asdict(config))


__all__ = ["BeatConfig", "load_config"]
