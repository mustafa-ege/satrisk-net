"""
Upload generated static PNG assets to Cloudinary and rewrite metrics.json with Cloudinary URLs.

Supports both:
1. Unsigned uploads using a Cloudinary Upload Preset (recommended & safest).
2. Signed uploads using Cloudinary API Key and API Secret.
"""

from __future__ import annotations

import argparse
import hashlib
import time
import json
from pathlib import Path
import requests

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Upload prediction masks, ground truths, and quicklooks to Cloudinary.",
    )
    p.add_argument(
        "--cloud-name",
        type=str,
        required=True,
        help="Your Cloudinary Cloud Name.",
    )
    p.add_argument(
        "--upload-preset",
        type=str,
        help="Cloudinary Unsigned Upload Preset name (recommended).",
    )
    p.add_argument(
        "--api-key",
        type=str,
        help="Cloudinary API Key (for signed uploads).",
    )
    p.add_argument(
        "--api-secret",
        type=str,
        help="Cloudinary API Secret (for signed uploads).",
    )
    p.add_argument(
        "--delete-local",
        action="store_true",
        help="Delete local PNG files after successful upload and URL rewriting.",
    )
    p.add_argument(
        "--public-root",
        type=str,
        default=str(PROJECT_ROOT / "frontend" / "public"),
        help="Directory path to the frontend public folder.",
    )
    return p.parse_args()


def upload_to_cloudinary(
    file_path: Path,
    public_id: str,
    cloud_name: str,
    upload_preset: str | None = None,
    api_key: str | None = None,
    api_secret: str | None = None,
) -> str:
    """Upload a file to Cloudinary and return its secure URL."""
    url = f"https://api.cloudinary.com/v1_1/{cloud_name}/image/upload"
    
    data = {}
    
    if upload_preset:
        # Unsigned upload
        data["upload_preset"] = upload_preset
        data["public_id"] = public_id
    elif api_key and api_secret:
        # Signed upload
        timestamp = int(time.time())
        # Parameters to sign must be sorted alphabetically
        params_to_sign = f"public_id={public_id}&timestamp={timestamp}{api_secret}"
        signature = hashlib.sha1(params_to_sign.encode("utf-8")).hexdigest()
        
        data["api_key"] = api_key
        data["timestamp"] = str(timestamp)
        data["public_id"] = public_id
        data["signature"] = signature
    else:
        raise ValueError(
            "Either --upload-preset OR both --api-key and --api-secret must be provided."
        )

    with open(file_path, "rb") as f:
        files = {"file": f}
        res = requests.post(url, data=data, files=files)
        
    if res.status_code != 200:
        raise RuntimeError(
            f"Cloudinary upload failed for {file_path.name} ({res.status_code}): {res.text}"
        )
        
    response_data = res.json()
    return response_data["secure_url"]


def main() -> None:
    args = parse_args()
    public_root = Path(args.public_root).resolve()
    cloud_name = args.cloud_name.strip()
    
    if not args.upload_preset and not (args.api_key and args.api_secret):
        raise SystemExit(
            "Error: You must provide either --upload-preset (unsigned) OR both --api-key and --api-secret (signed)."
        )

    metrics_json_path = public_root / "data" / "metrics.json"
    if not metrics_json_path.is_file():
        raise SystemExit(f"metrics.json not found at {metrics_json_path}. Run export_predictions.py first!")

    with open(metrics_json_path, "r", encoding="utf-8") as f:
        metrics = json.load(f)

    print(f"[cloudinary] Loaded {len(metrics)} metric records from metrics.json.", flush=True)

    # We will upload prediction masks, ground truths, and region quicklooks
    # We will maintain a map of local public path -> uploaded Cloudinary URL to avoid redundant uploads
    uploaded_urls: dict[str, str] = {}
    
    # Folders mapping
    # Predictions: frontend/public/predictions/{model_name}/{region_id}.png -> satrisk/predictions/{model_name}/{region_id}
    # Ground Truths: frontend/public/ground_truth/{region_id}.png -> satrisk/ground_truth/{region_id}
    # Region Quicklooks: frontend/public/region_images/{region_id}.png -> satrisk/region_images/{region_id}

    n_uploaded = 0
    updated_metrics = []

    for item in metrics:
        region_id = item["region_id"]
        model_name = item["model_name"]
        
        # Resolve paths
        # Relative paths look like '/predictions/deeplab/EMSR207_AOI01_01.png'
        pred_rel = item["prediction_path"].lstrip("/")
        gt_rel = item["ground_truth_path"].lstrip("/")
        image_rel = item["image_path"].lstrip("/")

        pred_local = public_root / pred_rel
        gt_local = public_root / gt_rel
        image_local = public_root / image_rel

        # Helper to upload if not already uploaded in this run
        def get_or_upload(local_path: Path, public_id: str) -> str:
            nonlocal n_uploaded
            if not local_path.is_file():
                print(f"[cloudinary] Warning: File not found locally: {local_path}", flush=True)
                return f"/{local_path.relative_to(public_root).as_posix()}"
                
            local_str = str(local_path)
            if local_str in uploaded_urls:
                return uploaded_urls[local_str]

            print(f"[cloudinary] Uploading {local_path.name} -> public_id: {public_id}...", flush=True)
            cloudinary_url = upload_to_cloudinary(
                file_path=local_path,
                public_id=public_id,
                cloud_name=cloud_name,
                upload_preset=args.upload_preset,
                api_key=args.api_key,
                api_secret=args.api_secret,
            )
            uploaded_urls[local_str] = cloudinary_url
            n_uploaded += 1
            return cloudinary_url

        try:
            # Upload prediction
            pred_public_id = f"satrisk/predictions/{model_name}/{region_id}"
            pred_cloud_url = get_or_upload(pred_local, pred_public_id)

            # Upload ground truth
            gt_public_id = f"satrisk/ground_truth/{region_id}"
            gt_cloud_url = get_or_upload(gt_local, gt_public_id)

            # Upload visual region image
            image_public_id = f"satrisk/region_images/{region_id}"
            image_cloud_url = get_or_upload(image_local, image_public_id)

            # Update metrics item
            updated_item = {
                "region_id": region_id,
                "model_name": model_name,
                "iou": item["iou"],
                "image_path": image_cloud_url,
                "image path": image_cloud_url,
                "prediction_path": pred_cloud_url,
                "prediction path": pred_cloud_url,
                "ground_truth_path": gt_cloud_url,
                "ground truth path": gt_cloud_url,
            }
            updated_metrics.append(updated_item)

        except Exception as err:
            print(f"[cloudinary] Error processing region {region_id} model {model_name}: {err}", flush=True)
            # Maintain original on error
            updated_metrics.append(item)

    # Save updated metrics.json back
    with open(metrics_json_path, "w", encoding="utf-8") as f:
        json.dump(updated_metrics, f, indent=2)
    print(f"[cloudinary] Successfully updated metrics.json with Cloudinary URLs.", flush=True)

    # Delete local PNG assets if requested
    if args.delete_local:
        print("[cloudinary] Cleaning up local PNG assets to keep git repository clean...", flush=True)
        n_deleted = 0
        for local_str in uploaded_urls.keys():
            p = Path(local_str)
            if p.is_file():
                try:
                    p.unlink()
                    n_deleted += 1
                except Exception as e:
                    print(f"[cloudinary] Warning: Could not delete local file {p}: {e}", flush=True)
        print(f"[cloudinary] Deleted {n_deleted} local PNG files.", flush=True)

    print(f"[cloudinary] Done! Uploaded {n_uploaded} assets in total.", flush=True)


if __name__ == "__main__":
    main()
