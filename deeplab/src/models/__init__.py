from __future__ import annotations

from typing import Dict

from .deeplabv3 import DeepLabV3
from .unet import UNet


def create_model(cfg: Dict):
    """
    Factory function used by training / evaluation code to build a model
    based on a simple config dictionary.

    Expected keys in `cfg`:
      - type: "unet" or "deeplabv3"
      - in_channels: int
      - num_classes: int
    Plus optional backbone / pretrained / aux_loss for DeepLabV3,
    and features / dropout for UNet.
    """
    model_type = cfg.get("type", "unet").lower()

    if model_type == "unet":
        return UNet(
            in_channels=cfg.get("in_channels", 12),
            num_classes=cfg.get("num_classes", 1),
            features=cfg.get("features", [64, 128, 256, 512]),
            dropout=cfg.get("dropout", 0.0),
        )

    if model_type == "deeplabv3":
        return DeepLabV3(
            in_channels=cfg.get("in_channels", 12),
            num_classes=cfg.get("num_classes", 1),
            backbone=cfg.get("backbone", "resnet50"),
            pretrained=cfg.get("pretrained", True),
            aux_loss=cfg.get("aux_loss", False),
        )

    raise ValueError(f"Unsupported model type '{model_type}'. Use 'unet' or 'deeplabv3'.")


