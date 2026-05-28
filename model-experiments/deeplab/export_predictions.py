"""
Batch-run a saved segmentation checkpoint and write binary prediction GeoTIFFs.

This script is intentionally explicit: it does not assume defaults for paths. You
must pass the dataset location, model config, checkpoint, and output directory.

Predictions are stored under a separate output tree that mirrors paths relative to
``--dataset-root`` (e.g. ``<output-root>/test/EMSR207/AOI01/.../stem_PRED.tif``).
"""

from __future__ import annotations

import argparse
import inspect
import sys
from pathlib import Path

import numpy as np
import rasterio
import torch
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.datamodule import DataConfig, build_datasets
from src.models import create_model


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Run segmentation checkpoint on a split and save *_PRED.tif rasters.",
    )
    p.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to a .pth/.pt file (either a dict with 'model_state' or a raw state_dict).",
    )
    p.add_argument(
        "--config",
        type=str,
        required=True,
        help="YAML config that describes the model and data preprocessing.",
    )
    p.add_argument(
        "--dataset-root",
        type=str,
        required=True,
        help="Root directory that contains the split folder (see --split-dir).",
    )
    p.add_argument(
        "--split-dir",
        type=str,
        required=True,
        help="Subdirectory under dataset-root to scan for *_S2L2A.tif (use '.' if root is already the AOI).",
    )
    p.add_argument(
        "--output-root",
        type=str,
        required=True,
        help="Directory under which mirrored relative paths and *_PRED.tif files are written.",
    )
    p.add_argument(
        "--split-key",
        type=str,
        default="infer",
        help="Internal split name for DataConfig (only affects logging; use any non-train string).",
    )
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"])
    p.add_argument(
        "--overwrite",
        action="store_true",
        help="Write even if the target *_PRED.tif already exists.",
    )
    return p.parse_args()


def load_config(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def extract_state_dict(ckpt_obj: object) -> dict:
    """
    Accept either:
    - training checkpoint dict: {'model_state': state_dict, ...}
    - raw state_dict: {'layer.weight': tensor, ...}
    """
    if (
        isinstance(ckpt_obj, dict)
        and "model_state" in ckpt_obj
        and isinstance(ckpt_obj.get("model_state"), dict)
    ):
        return ckpt_obj["model_state"]  # type: ignore[return-value]

    if isinstance(ckpt_obj, dict) and ckpt_obj and all(isinstance(k, str) for k in ckpt_obj.keys()):
        return ckpt_obj  # type: ignore[return-value]

    raise TypeError(
        "Unsupported checkpoint format. Expected a dict with key 'model_state' or a raw state_dict dict."
    )


def output_raster_path(
    image_path: Path,
    dataset_root: Path,
    output_root: Path,
    image_suffix: str,
) -> Path:
    root_res = dataset_root.resolve()
    img_res = image_path.resolve()
    rel = img_res.relative_to(root_res)
    new_name = rel.name.replace(image_suffix, "_PRED.tif")
    return (output_root / rel.parent / new_name).resolve()


def write_pred_geotiff(
    *,
    ref_image_path: Path,
    pred_hw: np.ndarray,
    dst_path: Path,
) -> None:
    """Write single-band uint8 mask using CRS/transform/size from ref_image_path."""
    pred_hw = np.asarray(pred_hw, dtype=np.float32)
    if pred_hw.ndim != 2:
        raise ValueError(f"expected HxW prediction, got shape {pred_hw.shape}")

    dst_path.parent.mkdir(parents=True, exist_ok=True)

    with rasterio.open(ref_image_path) as src:
        if src.height != pred_hw.shape[0] or src.width != pred_hw.shape[1]:
            raise ValueError(
                f"prediction shape {pred_hw.shape} does not match ref {src.height}x{src.width}"
            )
        profile = src.profile.copy()
        profile.update(
            count=1,
            dtype=np.uint8,
            nodata=None,
        )

    binary = (pred_hw > 0.5).astype(np.uint8)
    with rasterio.open(dst_path, "w", **profile) as dst:
        dst.write(binary, 1)


def main() -> None:
    args = parse_args()
    cfg_path = Path(args.config).expanduser().resolve()
    ckpt_path = Path(args.checkpoint).expanduser().resolve()
    dataset_root = Path(args.dataset_root).resolve()
    output_root = Path(args.output_root).resolve()

    if not cfg_path.is_file():
        raise SystemExit(f"config not found: {cfg_path}")
    if not ckpt_path.is_file():
        raise SystemExit(f"checkpoint not found: {ckpt_path}")
    if not dataset_root.is_dir():
        raise SystemExit(f"dataset-root is not a directory: {dataset_root}")

    config = load_config(cfg_path)
    threshold = float(config.get("metrics", {}).get("threshold", 0.5))

    device = torch.device(
        "cuda" if args.device == "cuda" and torch.cuda.is_available() else "cpu"
    )
    print(f"[export] device={device}", flush=True)

    model_cfg = config["model"]
    in_channels = int(model_cfg.get("in_channels", 12))

    data_cfg = DataConfig(
        dataset_root=str(dataset_root),
        split_dirs={args.split_key: args.split_dir},
        image_suffix=config["data"]["image_suffix"],
        mask_suffix=config["data"]["mask_suffix"],
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        band_norm=config["data"].get("band_norm", "reflectance"),
        target_size=config["data"].get("target_size"),
        augmentations=config["data"].get("augmentations"),
    )
    datasets = build_datasets(data_cfg)
    dataset = datasets[args.split_key]
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )

    print(f"[export] samples={len(dataset)} split_dir={args.split_dir!r}", flush=True)

    load_kw: dict = {"map_location": device}
    if "weights_only" in inspect.signature(torch.load).parameters:
        load_kw["weights_only"] = False
    ckpt_obj = torch.load(ckpt_path, **load_kw)
    state_dict = extract_state_dict(ckpt_obj)
    model = create_model(model_cfg).to(device)
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing:
        print(f"[export] warning: missing keys: {len(missing)}", flush=True)
    if unexpected:
        print(f"[export] warning: unexpected keys: {len(unexpected)}", flush=True)
    model.eval()

    n_written = 0
    for batch in loader:
        images = batch["image"].to(device)
        if images.shape[1] > in_channels:
            images = images[:, :in_channels]
        elif images.shape[1] < in_channels:
            raise RuntimeError(
                f"model expects {in_channels} bands but batch has {images.shape[1]}"
            )

        raw_paths = batch["image_path"]
        if isinstance(raw_paths, str):
            paths = [raw_paths]
        else:
            paths = list(raw_paths)

        with torch.no_grad():
            logits = model(images)
            probs = torch.sigmoid(logits)

        for i, image_path_str in enumerate(paths):
            image_path = Path(image_path_str)
            prob = probs[i, 0].detach().cpu()

            with rasterio.open(image_path) as src:
                h0, w0 = src.height, src.width

            prob_up = F.interpolate(
                prob.unsqueeze(0).unsqueeze(0),
                size=(h0, w0),
                mode="bilinear",
                align_corners=False,
            )[0, 0]

            pred_np = (prob_up.numpy() >= threshold).astype(np.float32)

            out_path = output_raster_path(
                image_path,
                dataset_root,
                output_root,
                config["data"]["image_suffix"],
            )
            if out_path.is_file() and not args.overwrite:
                print(f"[export] skip exists: {out_path}", flush=True)
                continue

            write_pred_geotiff(
                ref_image_path=image_path,
                pred_hw=pred_np,
                dst_path=out_path,
            )
            n_written += 1
            print(f"[export] wrote {out_path}", flush=True)

    print(f"[export] done. wrote {n_written} file(s) under {output_root}", flush=True)


if __name__ == "__main__":
    main()
