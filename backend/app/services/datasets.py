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
    # backend/app/services/datasets.py -> backend/app/data/datasets.mock.json
    return Path(__file__).resolve().parents[1] / "data" / "datasets.mock.json"


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
def _load_dataset_index_cached(mock_index_resolved: str, dataset_root_resolved: str) -> DatasetIndex:
    if dataset_root_resolved:
        root = Path(dataset_root_resolved)
        if root.is_dir():
            return DatasetIndex(regions=_regions_from_filesystem(root))

    index_path = Path(mock_index_resolved)
    raw = json.loads(index_path.read_text(encoding="utf-8"))
    regions = [DatasetRegion.model_validate(item) for item in raw]
    return DatasetIndex(regions=regions)


def load_dataset_index(path: str | Path | None = None) -> DatasetIndex:
    """
    Load regions from ``SATRISK_DATASET_ROOT`` when that directory exists; otherwise
    load the JSON mock index (or the explicit ``path`` for the mock file).
    """
    mock_path = Path(path) if path is not None else _default_index_path()
    root = _dataset_root_from_env()
    dataset_root_key = str(root.resolve()) if root is not None else ""
    return _load_dataset_index_cached(str(mock_path.resolve()), dataset_root_key)


def list_regions(*, index_path: str | Path | None = None) -> list[DatasetRegion]:
    return load_dataset_index(index_path).regions


def get_region(region_id: str, *, index_path: str | Path | None = None) -> DatasetRegion | None:
    regions = load_dataset_index(index_path).regions
    for region in regions:
        if region.id == region_id:
            return region
    return None

