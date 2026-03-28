from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from src.datasets.mvtec import IMAGENET_MEAN, IMAGENET_STD


def denormalize_image(image_tensor: torch.Tensor) -> np.ndarray:
    mean = torch.tensor(IMAGENET_MEAN, device=image_tensor.device).view(3, 1, 1)
    std = torch.tensor(IMAGENET_STD, device=image_tensor.device).view(3, 1, 1)
    image = image_tensor * std + mean
    image = image.clamp(0.0, 1.0)
    return image.permute(1, 2, 0).cpu().numpy()


def save_anomaly_visualizations(
    images: list[torch.Tensor],
    anomaly_maps: list[torch.Tensor],
    image_scores: list[float],
    image_paths: list[str],
    save_dir: str | Path,
    max_samples: int,
) -> Path:
    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)

    num_samples = min(max_samples, len(images))
    for index in range(num_samples):
        image_np = denormalize_image(images[index])
        anomaly_map = anomaly_maps[index].squeeze().cpu().numpy()

        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        axes[0].imshow(image_np)
        axes[0].set_title("Original")
        axes[0].axis("off")

        axes[1].imshow(image_np)
        axes[1].imshow(anomaly_map, cmap="jet", alpha=0.45)
        axes[1].set_title(f"Overlay | score={image_scores[index]:.4f}")
        axes[1].axis("off")

        output_file = save_path / f"{Path(image_paths[index]).stem}_heatmap.png"
        fig.tight_layout()
        fig.savefig(output_file, dpi=200, bbox_inches="tight")
        plt.close(fig)

    return save_path
