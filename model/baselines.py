"""
P1-05: Fair Comparison Baselines (Trained from Scratch under Identical Splits & Budget).

Provides:
  1. PlainCNNBaseline:        5-stage Plain Conv-BN-ReLU-MaxPool + GlobalAvgPool (1.82M params)
  2. ResidualGAPBaseline:     5-stage Strided Residual Conv + GlobalAvgPool (5.00M params)
  3. ResNet18Scratch:         Standard ResNet-18 architecture initialized from scratch (11.18M params)
  4. MobileNetV3SmallScratch: Standard MobileNetV3-Small initialized from scratch (1.52M params)
"""

import torch
import torch.nn as nn
from torchvision import models

from architecture import CarComponentCNN, ModelConfig


def build_plain_cnn_baseline(num_classes: int = 5) -> CarComponentCNN:
    cfg = ModelConfig(
        num_classes=num_classes,
        backbone="plain",
        use_coordconv=False,
        use_multiscale=False,
        use_cbam=False,
        pooling="global_avg",
        use_film=False,
    )
    return CarComponentCNN(config=cfg)


def build_residual_gap_baseline(num_classes: int = 5) -> CarComponentCNN:
    cfg = ModelConfig(
        num_classes=num_classes,
        backbone="residual",
        use_coordconv=False,
        use_multiscale=False,
        use_cbam=False,
        pooling="global_avg",
        use_film=False,
    )
    return CarComponentCNN(config=cfg)


class ResNet18Scratch(nn.Module):
    """Standard ResNet-18 trained from scratch (weights=None) for 5-component state classification."""

    def __init__(self, num_classes: int = 5):
        super().__init__()
        self.net = models.resnet18(weights=None)
        in_features = self.net.fc.in_features
        self.net.fc = nn.Linear(in_features, num_classes)

    def forward(self, x: torch.Tensor, return_pose: bool = False):
        logits = self.net(x)
        if return_pose:
            dummy_pose = torch.zeros(x.size(0), 8, device=x.device, dtype=x.dtype)
            return logits, dummy_pose
        return logits


class MobileNetV3SmallScratch(nn.Module):
    """Standard MobileNetV3-Small trained from scratch (weights=None) for 5-component state classification."""

    def __init__(self, num_classes: int = 5):
        super().__init__()
        self.net = models.mobilenet_v3_small(weights=None)
        in_features = self.net.classifier[-1].in_features
        self.net.classifier[-1] = nn.Linear(in_features, num_classes)

    def forward(self, x: torch.Tensor, return_pose: bool = False):
        logits = self.net(x)
        if return_pose:
            dummy_pose = torch.zeros(x.size(0), 8, device=x.device, dtype=x.dtype)
            return logits, dummy_pose
        return logits
