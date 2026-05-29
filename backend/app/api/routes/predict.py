from __future__ import annotations

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from app.services.datasets import DatasetRegion, _dataset_root_from_env, get_region
from app.services.geotiff_predictions import build_geotiff_prediction_result, export_region_quicklook_png

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

    root = _dataset_root_from_env()
    if root is None:
        raise HTTPException(
            status_code=500,
            detail="SATRISK_DATASET_ROOT is not set; cannot resolve ground truth GeoTIFF paths.",
        )
    try:
        # Create a stable preview image path the frontend can load directly.
        export_region_quicklook_png(image_path=(root / region.image_path).resolve(), region_id=region.id)
        try:
            result = build_geotiff_prediction_result(
                region,
                model=model,
                dataset_root=root,
            )
            print(f"[predict] Served precomputed result for region={region.id} model={model}", flush=True)
        except FileNotFoundError as exc:
            print(f"[predict] Precomputed result not found for region={region.id} model={model}. Trying live inference...", flush=True)
            try:
                from app.services.inference import run_live_inference
                result = run_live_inference(region, model)
                print(f"[predict] Successfully ran live inference for region={region.id} model={model}", flush=True)
            except Exception as live_exc:
                print(f"[predict] Live inference failed or model not available for region={region.id} model={model}: {live_exc}", flush=True)
                raise HTTPException(
                    status_code=404,
                    detail=f"Precomputed prediction missing, and live inference failed: {str(live_exc)}",
                ) from live_exc
    except ValueError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    return PredictResponse(
        region_id=region.id,
        mask_url=f"/static/{result.mask_relpath}",
        ground_truth_url=f"/static/{result.ground_truth_relpath}",
        metrics=Metrics(iou=result.iou),
    )


