from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

from pydantic import BaseModel, Field
from pydantic import field_validator


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


@lru_cache(maxsize=1)
def load_dataset_index(path: str | Path | None = None) -> DatasetIndex:
    index_path = Path(path) if path is not None else _default_index_path()
    raw = json.loads(index_path.read_text(encoding="utf-8"))
    regions = [DatasetRegion.model_validate(item) for item in raw]
    return DatasetIndex(regions=regions)


def list_regions(*, index_path: str | Path | None = None) -> list[DatasetRegion]:
    return load_dataset_index(index_path).regions


def get_region(region_id: str, *, index_path: str | Path | None = None) -> DatasetRegion | None:
    regions = load_dataset_index(index_path).regions
    for region in regions:
        if region.id == region_id:
            return region
    return None

