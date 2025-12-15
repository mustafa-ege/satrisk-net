from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.datamodule import DataConfig, build_datasets
from src.models import create_model
from train import evaluate


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate saved segmentation model checkpoint.")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="artifacts/unet_baseline_best.pth",
        help="Path to the checkpoint file.",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/unet_baseline.yaml",
        help="Path to the training config (for model/data settings).",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="test",
        choices=["train", "val", "test"],
        help="Dataset split to evaluate.",
    )
    return parser.parse_args()


def load_config(path: str) -> dict:
    import yaml

    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def main() -> None:
    args = parse_args()
    config = load_config(args.config)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[EVAL] Using device: {device}")

    ckpt = torch.load(args.checkpoint, map_location=device)

    model_cfg = config["model"]
    model = create_model(model_cfg).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    data_cfg = DataConfig(
        dataset_root=config["data"]["dataset_root"],
        split_dirs={args.split: config["data"]["split_dirs"][args.split]},
        image_suffix=config["data"]["image_suffix"],
        mask_suffix=config["data"]["mask_suffix"],
        batch_size=1,
        num_workers=2,
        band_norm=config["data"].get("band_norm", "reflectance"),
        target_size=config["data"].get("target_size"),
        augmentations=None,
    )
    datasets = build_datasets(data_cfg)
    loader = torch.utils.data.DataLoader(
        datasets[args.split],
        batch_size=1,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
    )

    criterion = torch.nn.BCEWithLogitsLoss()
    _, metrics = evaluate(
        model,
        loader,
        criterion,
        device,
        amp=False,
        threshold=config["metrics"]["threshold"],
    )

    print(f"[EVAL] Split: {args.split}")
    print(
        f"[EVAL] IoU: {metrics['iou']:.4f} | "
        f"Precision: {metrics['precision']:.4f} | "
        f"Recall: {metrics['recall']:.4f}"
    )


if __name__ == "__main__":
    main()

