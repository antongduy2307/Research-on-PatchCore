from __future__ import annotations

import torch
from torch import nn
from torchvision.models import Wide_ResNet50_2_Weights, wide_resnet50_2


class FeatureExtractor(nn.Module):
    def __init__(self, backbone_name: str, layers: list[str]) -> None:
        super().__init__()
        if backbone_name != "wide_resnet50_2":
            raise ValueError(f"Unsupported backbone: {backbone_name}")

        valid_layers = {"layer2", "layer3"}
        if set(layers) != valid_layers:
            raise ValueError("This baseline expects layers ['layer2', 'layer3'].")

        self.layers = layers
        self.backbone = wide_resnet50_2(weights=Wide_ResNet50_2_Weights.DEFAULT)
        self.backbone.eval()

        for parameter in self.backbone.parameters():
            parameter.requires_grad = False

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        outputs: dict[str, torch.Tensor] = {}
        x = self.backbone.conv1(x)
        x = self.backbone.bn1(x)
        x = self.backbone.relu(x)
        x = self.backbone.maxpool(x)

        x = self.backbone.layer1(x)
        x = self.backbone.layer2(x)
        outputs["layer2"] = x

        x = self.backbone.layer3(x)
        outputs["layer3"] = x
        return outputs
