from __future__ import annotations

import torch
import torch.nn as nn
from torchvision import models
from torchvision.models.segmentation import DeepLabV3_ResNet50_Weights, DeepLabV3_ResNet101_Weights


class DeepLabV3(nn.Module):
    """DeepLabV3 model adapted for multi-channel input and binary segmentation."""

    def __init__(
        self,
        in_channels: int = 3,
        num_classes: int = 1,
        backbone: str = "resnet50",
        pretrained: bool = True,
        aux_loss: bool = False,
    ) -> None:
        """
        Initialize DeepLabV3 model.

        Args:
            in_channels: Number of input channels (default: 3, but can be 12 for Sentinel-2)
            num_classes: Number of output classes (1 for binary segmentation)
            backbone: Backbone architecture ('resnet50' or 'resnet101')
            pretrained: Whether to use pretrained weights
            aux_loss: Whether to use auxiliary loss (not used in this implementation)
        """
        super().__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes

        # Use new weights API instead of deprecated pretrained parameter
        if pretrained:
            weights_50 = DeepLabV3_ResNet50_Weights.COCO_WITH_VOC_LABELS_V1
            weights_101 = DeepLabV3_ResNet101_Weights.COCO_WITH_VOC_LABELS_V1
        else:
            weights_50 = None
            weights_101 = None

        # Load torchvision DeepLabV3 with ResNet backbone
        # Note: When using pretrained weights, aux_loss must be True
        if backbone == "resnet50":
            deeplab = models.segmentation.deeplabv3_resnet50(
                weights=weights_50, num_classes=21, aux_loss=aux_loss or pretrained
            )
        elif backbone == "resnet101":
            deeplab = models.segmentation.deeplabv3_resnet101(
                weights=weights_101, num_classes=21, aux_loss=aux_loss or pretrained
            )
        else:
            raise ValueError(f"Unsupported backbone: {backbone}. Use 'resnet50' or 'resnet101'")

        # Adapt first layer for custom input channels
        if in_channels != 3:
            # Replace the first conv layer to accept in_channels instead of 3
            old_conv = deeplab.backbone.conv1
            new_conv = nn.Conv2d(
                in_channels,
                old_conv.out_channels,
                kernel_size=old_conv.kernel_size,
                stride=old_conv.stride,
                padding=old_conv.padding,
                bias=old_conv.bias is not None,
            )
            # Initialize new conv layer weights
            # For non-3 channels, initialize with repeated RGB weights or random
            if in_channels > 3 and pretrained:
                # Repeat RGB weights for additional channels
                with torch.no_grad():
                    new_conv.weight[:, :3] = old_conv.weight.data
                    # Initialize remaining channels with small random values
                    nn.init.kaiming_normal_(new_conv.weight[:, 3:], mode="fan_out", nonlinearity="relu")
            else:
                # Initialize all channels randomly
                nn.init.kaiming_normal_(new_conv.weight, mode="fan_out", nonlinearity="relu")
            deeplab.backbone.conv1 = new_conv

        # Replace classifier head for binary segmentation
        # The classifier is a Sequential: [ASPP, Conv, BN, ReLU, Dropout, Conv]
        # We need to replace the last Conv2d layer (index 5)
        # Find the last Conv2d layer in the classifier
        for i in range(len(deeplab.classifier) - 1, -1, -1):
            if isinstance(deeplab.classifier[i], nn.Conv2d):
                # Get the input channels from the existing layer
                in_channels = deeplab.classifier[i].in_channels
                deeplab.classifier[i] = nn.Conv2d(in_channels, num_classes, kernel_size=1)
                break

        self.model = deeplab

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input tensor of shape (B, C, H, W)

        Returns:
            Logits tensor of shape (B, num_classes, H, W)
        """
        output = self.model(x)
        # DeepLabV3 returns a dict with 'out' key, extract it
        if isinstance(output, dict):
            return output["out"]
        return output

