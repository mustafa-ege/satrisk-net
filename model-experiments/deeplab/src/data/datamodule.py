from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple

from torch.utils.data import DataLoader

from .cems_dataset import (
    AugmentationConfig,
    CEMSWildfireDataset,
    build_transforms,
)


@dataclass
class DataConfig:
    """Structured settings for building datasets and loaders."""

    dataset_root: str
    split_dirs: Dict[str, str]
    image_suffix: str
    mask_suffix: str
    batch_size: int
    num_workers: int
    band_norm: str = "reflectance"
    target_size: Optional[Tuple[int, int]] = None
    augmentations: Dict[str, bool] | None = None


def create_dataloader(
    dataset: CEMSWildfireDataset,
    batch_size: int,
    num_workers: int,
    shuffle: bool,
    drop_last: bool = False,
) -> DataLoader:
    """Helper that standardizes DataLoader construction across splits."""
    return DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=shuffle,
        pin_memory=True,
        drop_last=drop_last,
    )


def build_datasets(cfg: DataConfig) -> Dict[str, CEMSWildfireDataset]:
    """Instantiate train/val/test datasets with correct augmentations."""
    target_size = tuple(cfg.target_size) if cfg.target_size else None
    aug_cfg = AugmentationConfig(**cfg.augmentations) if cfg.augmentations else AugmentationConfig()
    train_tfms = build_transforms(aug_cfg, target_size)
    eval_tfms = build_transforms(AugmentationConfig(False, False, False), target_size)

    datasets: Dict[str, CEMSWildfireDataset] = {}
    for split, dir_name in cfg.split_dirs.items():
        transform = train_tfms if split == "train" else eval_tfms
        datasets[split] = CEMSWildfireDataset(
            root=Path(cfg.dataset_root),
            split_dir=dir_name,
            image_suffix=cfg.image_suffix,
            mask_suffix=cfg.mask_suffix,
            transform=transform,
            band_norm=cfg.band_norm,
        )
    return datasets


def build_dataloaders(cfg: DataConfig) -> Dict[str, DataLoader]:
    """Create dataloaders for every split, shuffling only the training data."""
    datasets = build_datasets(cfg)
    loaders: Dict[str, DataLoader] = {}
    for split, dataset in datasets.items():
        # Drop last batch during training to avoid BatchNorm issues with batch_size=1
        # This is especially important for models like DeepLabV3 with aux_loss
        drop_last = split == "train"
        loaders[split] = create_dataloader(
            dataset,
            batch_size=cfg.batch_size,
            num_workers=cfg.num_workers,
            shuffle=split == "train",
            drop_last=drop_last,
        )
    return loaders

