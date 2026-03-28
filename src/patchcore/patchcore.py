from __future__ import annotations

import math

import torch
import torch.nn.functional as F


class PatchCore:
    def __init__(self, local_agg: bool = True) -> None:
        self.local_agg = local_agg
        self.memory_bank: torch.Tensor | None = None
        self.feature_dim: int | None = None
        self.memory_bank_size_before: int = 0
        self.memory_bank_size_after: int = 0

    def _aggregate(self, feature_map: torch.Tensor) -> torch.Tensor:
        if not self.local_agg:
            return feature_map
        return F.avg_pool2d(feature_map, kernel_size=3, stride=1, padding=1)

    def extract_patch_embeddings(self, features: dict[str, torch.Tensor]) -> tuple[torch.Tensor, tuple[int, int]]:
        layer2 = self._aggregate(features["layer2"])
        layer3 = self._aggregate(features["layer3"])
        layer3 = F.interpolate(
            layer3,
            size=layer2.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )

        embedding_map = torch.cat([layer2, layer3], dim=1)
        batch_size, channels, height, width = embedding_map.shape
        patches = embedding_map.permute(0, 2, 3, 1).reshape(batch_size, height * width, channels)
        return patches, (height, width)

    def build_memory_bank(self, patch_embeddings: torch.Tensor, subsample_ratio: float, seed: int) -> None:
        if not 0 < subsample_ratio <= 1:
            raise ValueError("subsample_ratio must be in (0, 1].")

        self.memory_bank_size_before = patch_embeddings.shape[0]
        self.feature_dim = patch_embeddings.shape[1]

        num_samples = max(1, math.ceil(self.memory_bank_size_before * subsample_ratio))
        generator = torch.Generator(device=patch_embeddings.device)
        generator.manual_seed(seed)
        indices = torch.randperm(self.memory_bank_size_before, generator=generator, device=patch_embeddings.device)[
            :num_samples
        ]

        self.memory_bank = patch_embeddings[indices].contiguous()
        self.memory_bank_size_after = self.memory_bank.shape[0]

    @torch.no_grad()
    def score(self, patch_embeddings: torch.Tensor, patch_shape: tuple[int, int], output_size: int) -> tuple[torch.Tensor, torch.Tensor]:
        if self.memory_bank is None:
            raise RuntimeError("Memory bank has not been built.")

        batch_maps = []
        batch_scores = []
        height, width = patch_shape

        for sample_patches in patch_embeddings:
            distances = torch.cdist(sample_patches, self.memory_bank)
            patch_scores = distances.min(dim=1).values
            anomaly_map = patch_scores.view(1, 1, height, width)
            anomaly_map = F.interpolate(
                anomaly_map,
                size=(output_size, output_size),
                mode="bilinear",
                align_corners=False,
            )
            batch_maps.append(anomaly_map.squeeze(0))
            batch_scores.append(patch_scores.max())

        return torch.stack(batch_scores), torch.stack(batch_maps)
