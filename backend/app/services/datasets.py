from __future__ import annotations

import json
import os
from functools import lru_cache
from pathlib import Path

from pydantic import BaseModel, Field
from pydantic import field_validator

from app.services.dataset_scanner import scan_wildfire_dataset


class DatasetRegion(BaseModel):
    id: str
    bbox: tuple[float, float, float, float] = Field(
        ...,
        description="(min_lon, min_lat, max_lon, max_lat)",
        min_length=4,
        max_length=4,
    )
    image_path: str
    mask_path: str

    @field_validator("image_path", "mask_path", mode="before")
    @classmethod
    def _strip_data_prefix(cls, v: str) -> str:
        if isinstance(v, str) and (v.startswith("data/") or v.startswith("data\\")):
            return v.split("/", 1)[1] if "/" in v else v.split("\\", 1)[1]
        return v


class DatasetIndex(BaseModel):
    regions: list[DatasetRegion]


def _default_index_path() -> Path:
    return Path(__file__).resolve().parents[1] / "data"


def _dataset_root_from_env() -> Path | None:
    raw = os.environ.get("SATRISK_DATASET_ROOT", "").strip()
    if not raw:
        return None
    root = Path(raw)
    return root if root.is_dir() else None


def _regions_from_filesystem(root: Path) -> list[DatasetRegion]:
    root_resolved = root.resolve()
    samples = scan_wildfire_dataset(root_resolved)
    regions: list[DatasetRegion] = []
    for s in samples:
        img = Path(s.image_path).resolve()
        msk = Path(s.mask_path).resolve()
        try:
            image_rel = img.relative_to(root_resolved).as_posix()
            mask_rel = msk.relative_to(root_resolved).as_posix()
        except ValueError:
            image_rel = img.as_posix()
            mask_rel = msk.as_posix()
        regions.append(
            DatasetRegion(
                id=s.id,
                bbox=s.bbox,
                image_path=image_rel,
                mask_path=mask_rel,
            )
        )
    return regions


@lru_cache(maxsize=16)
def _load_dataset_index_cached(dataset_root_resolved: str) -> DatasetIndex:
    if dataset_root_resolved:
        root = Path(dataset_root_resolved)
        if root.is_dir():
            return DatasetIndex(regions=_regions_from_filesystem(root))
    return DatasetIndex(regions=[])


def load_dataset_index() -> DatasetIndex:
    """
    Load regions from ``SATRISK_DATASET_ROOT`` when that directory exists.
    """
    root = _dataset_root_from_env()
    dataset_root_key = str(root.resolve()) if root is not None else ""
    return _load_dataset_index_cached(dataset_root_key)


def list_regions() -> list[DatasetRegion]:
    return load_dataset_index().regions


def get_region(region_id: str) -> DatasetRegion | None:
    regions = load_dataset_index().regions
    for region in regions:
        if region.id == region_id:
            return region
    return None


