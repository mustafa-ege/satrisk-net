from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class MetricResult:
    """Container for aggregated segmentation metrics."""

    iou: float
    precision: float
    recall: float


class BinarySegmentationMeter:
    """Accumulates TP/FP/FN to compute IoU, precision and recall."""

    def __init__(self, threshold: float = 0.5, eps: float = 1e-6) -> None:
        self.threshold = threshold
        self.eps = eps
        self.reset()

    def reset(self) -> None:
        self.tp = 0.0
        self.fp = 0.0
        self.fn = 0.0

    def update(self, probs: torch.Tensor, targets: torch.Tensor) -> None:
        """Update confusion counts using batched predictions."""
        preds = (probs >= self.threshold).float()

        tp = (preds * targets).sum().item()
        fp = (preds * (1 - targets)).sum().item()
        fn = ((1 - preds) * targets).sum().item()

        self.tp += tp
        self.fp += fp
        self.fn += fn

    def compute(self) -> MetricResult:
        """Return the final metric values after processing all batches."""
        precision = self.tp / (self.tp + self.fp + self.eps)
        recall = self.tp / (self.tp + self.fn + self.eps)
        iou = self.tp / (self.tp + self.fp + self.fn + self.eps)
        return MetricResult(iou=iou, precision=precision, recall=recall)

