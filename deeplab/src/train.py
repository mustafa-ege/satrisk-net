from __future__ import annotations

import argparse
import random
from contextlib import nullcontext
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from torch import nn, optim
from torch.amp import GradScaler

from src.data.datamodule import DataConfig, build_dataloaders
from src.models import create_model
from src.training.metrics import BinarySegmentationMeter


def log_step(message: str) -> None:
    """Standardized stdout logger for long-running steps."""
    print(f"[STEP] {message}", flush=True)


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments (currently only the config path)."""
    parser = argparse.ArgumentParser(description="Train segmentation model (U-Net/DeepLabV3) on CEMS Wildfire dataset.")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/unet_baseline.yaml",
        help="Path to YAML config.",
    )
    return parser.parse_args()


def load_config(path: str) -> Dict:
    """Load YAML configuration into a dictionary."""
    log_step(f"Loading config from {path}")
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def set_seed(seed: int) -> None:
    """Ensure reproducibility across Python, NumPy, and PyTorch."""
    log_step(f"Setting random seed to {seed}")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def prepare_dataloaders(config: Dict) -> Dict[str, torch.utils.data.DataLoader]:
    """Construct PyTorch dataloaders based on config settings."""
    log_step("Building dataloaders")
    data_cfg = DataConfig(
        dataset_root=config["data"]["dataset_root"],
        split_dirs=config["data"]["split_dirs"],
        image_suffix=config["data"]["image_suffix"],
        mask_suffix=config["data"]["mask_suffix"],
        batch_size=config["data"]["batch_size"],
        num_workers=config["data"]["num_workers"],
        band_norm=config["data"].get("band_norm", "reflectance"),
        target_size=config["data"].get("target_size"),
        augmentations=config["data"].get("augmentations"),
    )
    loaders = build_dataloaders(data_cfg)
    for split, loader in loaders.items():
        log_step(f"{split.capitalize()} set: {len(loader.dataset)} samples")
    return loaders


def get_autocast(device: torch.device, amp: bool):
    """Return appropriate autocast context based on device and AMP flag."""
    if amp and device.type == "cuda":
        return torch.amp.autocast(device_type="cuda")
    return nullcontext()


def get_grad_scaler(device: torch.device, amp: bool):
    """Instantiate the appropriate GradScaler for the current device."""
    device_type = "cuda" if device.type == "cuda" else "cpu"
    return torch.amp.GradScaler(device_type, enabled=amp and device.type == "cuda")


def build_optimizer(
    model: nn.Module,
    model_cfg: Dict,
    training_cfg: Dict,
) -> optim.Optimizer:
    """
    Construct the optimizer for the given model and config.

    - Supports differential learning rates for DeepLabV3 backbone vs. head.
    - Falls back to a standard AdamW optimizer for other models.
    """
    learning_rate = float(training_cfg["learning_rate"])
    weight_decay = float(training_cfg.get("weight_decay", 0.0))
    model_type = model_cfg.get("type", "unet")

    # Differential LR for pretrained DeepLabV3 backbones
    if model_type == "deeplabv3" and model_cfg.get("pretrained", False):
        backbone_lr = training_cfg.get("backbone_lr", learning_rate * 0.1)
        head_lr = training_cfg.get("head_lr", learning_rate)

        backbone_params = []
        head_params = []
        for name, param in model.named_parameters():
            if "backbone" in name:
                backbone_params.append(param)
            else:
                head_params.append(param)

        optimizer = optim.AdamW(
            [
                {"params": backbone_params, "lr": backbone_lr, "weight_decay": weight_decay},
                {"params": head_params, "lr": head_lr, "weight_decay": weight_decay},
            ]
        )
        log_step(f"Using differential LR: backbone={backbone_lr}, head={head_lr}")
        return optimizer

    # Standard optimizer for all parameters
    return optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
    )


def train_model(
    config: Dict,
    model: nn.Module,
    dataloaders: Dict[str, torch.utils.data.DataLoader],
    device: torch.device,
) -> Dict:
    """
    High-level training routine shared by CLI and notebooks.

    Returns a dictionary containing:
    - history: per-epoch loss and metrics
    - best_state: best checkpoint (epoch, model_state, optimizer_state, metrics)
    - best_epoch, best_metrics
    - checkpoint_path (if a checkpoint was saved)
    """
    training_cfg = config["training"]
    model_cfg = config["model"]

    criterion = nn.BCEWithLogitsLoss()
    grad_clip_value = training_cfg.get("grad_clip_norm")
    grad_clip = float(grad_clip_value) if grad_clip_value is not None else None

    optimizer = build_optimizer(model, model_cfg, training_cfg)
    scaler = get_grad_scaler(device, training_cfg.get("amp", True))

    best_iou = 0.0
    best_state = None
    total_epochs = training_cfg["epochs"]

    history = {
        "epoch": [],
        "train_loss": [],
        "val_loss": [],
        "iou": [],
        "precision": [],
        "recall": [],
    }

    log_step(f"Training for {total_epochs} epochs")

    for epoch in range(1, total_epochs + 1):
        log_step(f"Epoch {epoch}/{total_epochs}: training phase")
        train_loss = train_one_epoch(
            model,
            dataloaders["train"],
            optimizer,
            criterion,
            device,
            scaler,
            training_cfg.get("amp", True),
            grad_clip,
        )

        log_step(f"Epoch {epoch}/{total_epochs}: validation phase")
        val_loss, metrics = evaluate(
            model,
            dataloaders["val"],
            criterion,
            device,
            training_cfg.get("amp", True),
            config["metrics"]["threshold"],
        )

        print(
            f"Epoch {epoch:03d} | "
            f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | "
            f"IoU: {metrics['iou']:.4f} | "
            f"Precision: {metrics['precision']:.4f} | "
            f"Recall: {metrics['recall']:.4f}"
        )

        history["epoch"].append(epoch)
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["iou"].append(metrics["iou"])
        history["precision"].append(metrics["precision"])
        history["recall"].append(metrics["recall"])

        if metrics["iou"] > best_iou:
            best_iou = metrics["iou"]
            best_state = {
                "epoch": epoch,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "metrics": metrics,
            }

    checkpoint_path = None
    if best_state:
        output_path = Path("artifacts")
        output_path.mkdir(exist_ok=True)
        checkpoint_path = output_path / f"{config['run_name']}_best.pth"
        torch.save(best_state, checkpoint_path)

    return {
        "history": history,
        "best_state": best_state,
        "best_epoch": best_state["epoch"] if best_state else None,
        "best_metrics": best_state["metrics"] if best_state else None,
        "checkpoint_path": checkpoint_path,
    }


def train_one_epoch(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    optimizer: optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    scaler: GradScaler,
    amp: bool,
    grad_clip: float | None,
) -> float:
    """Train the model for one epoch and return average loss."""
    model.train()
    running_loss = 0.0
    total = 0

    for batch in loader:
        images = batch["image"].to(device)
        masks = batch["mask"].to(device)

        optimizer.zero_grad(set_to_none=True)

        with get_autocast(device, amp):
            logits = model(images)
            loss = criterion(logits, masks)

        scaler.scale(loss).backward()

        if grad_clip is not None:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

        scaler.step(optimizer)
        scaler.update()

        running_loss += loss.item() * images.size(0)
        total += images.size(0)

    return running_loss / total


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    device: torch.device,
    amp: bool,
    threshold: float,
) -> Tuple[float, Dict[str, float]]:
    """Run validation and compute loss plus IoU/precision/recall."""
    model.eval()
    running_loss = 0.0
    total = 0
    meter = BinarySegmentationMeter(threshold=threshold)

    for batch in loader:
        images = batch["image"].to(device)
        masks = batch["mask"].to(device)

        with get_autocast(device, amp):
            logits = model(images)
            loss = criterion(logits, masks)
            probs = torch.sigmoid(logits)

        meter.update(probs, masks)
        running_loss += loss.item() * images.size(0)
        total += images.size(0)

    metrics = meter.compute()
    return running_loss / total, {
        "iou": metrics.iou,
        "precision": metrics.precision,
        "recall": metrics.recall,
    }


def main() -> None:
    """Main training orchestration function."""
    args = parse_args()
    config = load_config(args.config)
    set_seed(config.get("seed", 42))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log_step(f"Using device: {device}")

    dataloaders = prepare_dataloaders(config)

    model_cfg = config["model"]
    model_type = model_cfg.get("type", "unet")
    log_step(f"Initializing {model_type.upper()} model")
    model = create_model(model_cfg).to(device)

    result = train_model(config, model, dataloaders, device)

    if result["checkpoint_path"] is not None:
        print(f"Saved best model to {result['checkpoint_path']}")

    log_step("Training run complete")


if __name__ == "__main__":
    main()

