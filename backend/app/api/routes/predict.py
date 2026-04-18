from __future__ import annotations

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from app.services.datasets import get_region
from app.services.predictions import deterministic_iou, prediction_relpath

router = APIRouter(tags=["predict"])


class PredictRequest(BaseModel):
    region_id: str = Field(..., min_length=1)
    model: str = Field(..., description="unet | deeplab | segformer")


class Metrics(BaseModel):
    iou: float


class PredictResponse(BaseModel):
    region_id: str
    mask_url: str
    ground_truth_url: str
    metrics: Metrics


@router.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest) -> PredictResponse:
    model = req.model.lower().strip()
    if model not in {"unet", "deeplab", "segformer"}:
        raise HTTPException(status_code=400, detail="Invalid model. Use: unet | deeplab | segformer")

    region = get_region(req.region_id)
    if region is None:
        raise HTTPException(status_code=404, detail="Unknown region_id")

    pred_path = prediction_relpath(model=model, region=region)

    return PredictResponse(
        region_id=region.id,
        mask_url=f"/static/{pred_path}",
        ground_truth_url=f"/static/{region.mask_path}",
        metrics=Metrics(iou=deterministic_iou(model=model, region_id=region.id)),
    )

