from __future__ import annotations

import logging
from pathlib import Path

import rasterio
from pydantic import BaseModel, Field
from rasterio.warp import transform_bounds

logger = logging.getLogger(__name__)


class ScannedDatasetSample(BaseModel):
    """One on-disk wildfire sample: Sentinel-2 stack + damage label mask."""

    id: str
    image_path: str = Field(..., description="Absolute path to *_S2L2A.tif")
    mask_path: str = Field(..., description="Absolute path to *_DEL.tif")
    bbox: tuple[float, float, float, float] = Field(
        ...,
        description="[minLon, minLat, maxLon, maxLat] in EPSG:4326 from the S2L2A GeoTIFF",
    )


def _bounds_wgs84(image_path: Path) -> tuple[float, float, float, float]:
    """Read raster bounds and return ``(min_lon, min_lat, max_lon, max_lat)`` (no full-array read)."""
    with rasterio.open(image_path) as src:
        if src.crs is None:
            raise ValueError("GeoTIFF has no CRS; cannot derive lon/lat bounds")
        left, bottom, right, top = transform_bounds(
            src.crs,
            "EPSG:4326",
            src.bounds.left,
            src.bounds.bottom,
            src.bounds.right,
            src.bounds.top,
        )
    return (left, bottom, right, top)


def _pick_tif_pair(sample_dir: Path) -> tuple[Path, Path] | None:
    images = sorted(sample_dir.glob("*_S2L2A.tif"))
    masks = sorted(sample_dir.glob("*_DEL.tif"))
    if not images or not masks:
        return None
    return images[0], masks[0]


def scan_wildfire_dataset(root: Path | str) -> list[ScannedDatasetSample]:
    """
    Walk ``root`` and collect every directory that contains at least one
    ``*_S2L2A.tif`` and one ``*_DEL.tif``. Other assets (CM, GRA, …) are ignored.

    Sample ``id`` is the directory name that holds the pair (e.g. ``EMSR230_AOI01_01``).
    """
    base = Path(root)
    if not base.is_dir():
        return []

    seen_dirs: set[Path] = set()
    samples: list[ScannedDatasetSample] = []

    for path in base.rglob("*"):
        if not path.is_file():
            continue
        if not (path.name.endswith("_S2L2A.tif") or path.name.endswith("_DEL.tif")):
            continue

        sample_dir = path.parent
        resolved = sample_dir.resolve()
        if resolved in seen_dirs:
            continue

        pair = _pick_tif_pair(sample_dir)
        if pair is None:
            continue

        image_path, mask_path = pair
        try:
            bbox = _bounds_wgs84(image_path)
        except Exception as exc:  # noqa: BLE001 — skip unreadable / non-georeferenced stacks
            logger.warning(
                "Skipping sample %s (%s): %s",
                sample_dir.name,
                image_path,
                exc,
            )
            continue

        seen_dirs.add(resolved)
        samples.append(
            ScannedDatasetSample(
                id=sample_dir.name,
                image_path=str(image_path.resolve().as_posix()),
                mask_path=str(mask_path.resolve().as_posix()),
                bbox=bbox,
            )
        )

    samples.sort(key=lambda s: s.id)
    return samples
