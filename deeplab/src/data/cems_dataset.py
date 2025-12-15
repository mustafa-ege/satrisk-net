from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import albumentations as A
import numpy as np
import rasterio
import torch
from torch.utils.data import Dataset


@dataclass
class AugmentationConfig:
    """Flags controlling simple geometric augmentations."""

    horizontal_flip: bool = True
    vertical_flip: bool = True
    random_rotate: bool = True


def build_transforms(
    config: AugmentationConfig,
    target_size: Optional[Tuple[int, int]] = None,
) -> Optional[A.Compose]:
    """Create albumentations pipeline derived from configuration toggles."""
    transforms: List[A.BasicTransform] = []
    if config.horizontal_flip:
        transforms.append(A.HorizontalFlip(p=0.5))
    if config.vertical_flip:
        transforms.append(A.VerticalFlip(p=0.5))
    if config.random_rotate:
        transforms.append(A.Rotate(limit=90, p=0.5))

    if target_size:
        height, width = target_size
        transforms.append(A.Resize(height=height, width=width))

    if not transforms:
        return None

    return A.Compose(
        transforms,
        additional_targets={"mask": "mask"},
    )


class CEMSWildfireDataset(Dataset):
    """Torch dataset that pairs Sentinel-2 tiles with burned-area masks."""

    def __init__(
        self,
        root: Path | str,
        split_dir: str,
        image_suffix: str = "_S2L2A.tif",
        mask_suffix: str = "_DEL.tif",
        transform: Optional[Callable] = None,
        band_norm: str = "reflectance",
    ) -> None:
        self.root = Path(root)
        self.split_dir = split_dir
        self.image_suffix = image_suffix
        self.mask_suffix = mask_suffix
        self.transform = transform
        self.band_norm = band_norm

        # Cache all Sentinel-2 tile paths once so __getitem__ stays lightweight
        self.image_paths = self._gather_images()
        if not self.image_paths:
            raise ValueError(
                f"No Sentinel tiles found in {self.root/self.split_dir}. "
                "Ensure the dataset path is correct."
            )

    def _gather_images(self) -> List[Path]:
        """Collect every Sentinel-2 tile for the requested split."""
        split_path = self.root / self.split_dir
        return sorted(split_path.rglob(f"*{self.image_suffix}"))

    def __len__(self) -> int:
        return len(self.image_paths)

    def _load_tiff(self, path: Path) -> np.ndarray:
        """Read a GeoTIFF as a numpy array with shape (C, H, W)."""
        with rasterio.open(path) as src:
            arr = src.read().astype(np.float32)
        return arr

    def _normalize(self, image: np.ndarray) -> np.ndarray:
        """Apply either reflectance scaling or per-tile standardization."""
        if self.band_norm == "reflectance":
            image /= 10000.0
        elif self.band_norm == "standard":
            mean = image.mean(axis=(1, 2), keepdims=True)
            std = image.std(axis=(1, 2), keepdims=True) + 1e-6
            image = (image - mean) / std
        return image

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        image_path = self.image_paths[idx]
        # Replace the spectral suffix with the delineation suffix to get label
        mask_path = Path(str(image_path).replace(self.image_suffix, self.mask_suffix))

        if not mask_path.exists():
            raise FileNotFoundError(f"Mask not found for {image_path}")

        # Load raw GeoTIFF arrays
        image = self._load_tiff(image_path)
        mask = self._load_tiff(mask_path)

        image = self._normalize(image)
        mask = (mask > 0).astype(np.float32)

        # rasterio loads as (C, H, W); albumentations expects (H, W, C)
        # rasterio -> (C, H, W); albumentations wants channels last
        image = np.transpose(image, (1, 2, 0))
        mask = mask.squeeze()

        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            image = augmented["image"]
            mask = augmented["mask"]

        # Return PyTorch tensors in NCHW format
        image_tensor = torch.from_numpy(image).permute(2, 0, 1).contiguous()
        mask_tensor = torch.from_numpy(mask).unsqueeze(0)

        return {
            "image": image_tensor,
            "mask": mask_tensor,
            "image_path": str(image_path),
        }

