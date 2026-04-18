from __future__ import annotations

import hashlib
from pathlib import PurePosixPath

from app.services.datasets import DatasetRegion


AllowedModel = str  # "unet" | "deeplab" | "segformer" (validated at API layer)


def prediction_relpath(*, model: AllowedModel, region: DatasetRegion) -> str:
    return str(PurePosixPath("predictions") / model / f"{region.id}.txt")


def deterministic_iou(*, model: AllowedModel, region_id: str) -> float:
    h = hashlib.sha256(f"{model}:{region_id}".encode("utf-8")).digest()
    # Map first 2 bytes -> [0.40, 0.95] to look realistic while clearly "mock".
    x = int.from_bytes(h[:2], "big") / 65535.0
    return round(0.40 + x * (0.95 - 0.40), 4)

