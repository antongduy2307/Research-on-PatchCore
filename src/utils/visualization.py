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
    labels: list[int],
    image_scores: list[float],
    image_paths: list[str],
    save_dir: str | Path,
    max_samples: int,
) -> Path:
    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)
    good_dir = save_path / "good"
    anomaly_dir = save_path / "anomaly"
    good_dir.mkdir(parents=True, exist_ok=True)
    anomaly_dir.mkdir(parents=True, exist_ok=True)

    if max_samples <= 0:
        num_samples = len(images)
    else:
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

        image_path = Path(image_paths[index])
        subtype = image_path.parent.name
        label_dir = good_dir if labels[index] == 0 else anomaly_dir
        score_prefix = f"{image_scores[index]:012.6f}"
        output_file = label_dir / f"{score_prefix}__{subtype}__{image_path.stem}_heatmap.png"
        fig.tight_layout()
        fig.savefig(output_file, dpi=200, bbox_inches="tight")
        plt.close(fig)

    return save_path


def save_score_distribution(
    labels: list[int],
    image_scores: list[float],
    save_dir: str | Path,
    filename: str = "distribution_score.png",
) -> Path:
    if len(labels) != len(image_scores):
        raise ValueError("Labels and image scores must have the same length.")

    good_scores = [score for label, score in zip(labels, image_scores) if label == 0]
    anomaly_scores = [score for label, score in zip(labels, image_scores) if label == 1]

    if not good_scores or not anomaly_scores:
        raise ValueError("Distribution plot requires both good and anomalous scores.")

    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)
    output_path = save_path / filename

    all_scores = np.asarray(good_scores + anomaly_scores, dtype=np.float32)
    score_min = float(all_scores.min())
    score_max = float(all_scores.max())
    if np.isclose(score_min, score_max):
        bins = 10
    else:
        bins = np.linspace(score_min, score_max, 21)

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))

    axes[0].hist(good_scores, bins=bins, color="#2E8B57", alpha=0.85, edgecolor="black")
    axes[0].set_title("Good Score Histogram")
    axes[0].set_xlabel("Score")
    axes[0].set_ylabel("Count")

    axes[1].hist(anomaly_scores, bins=bins, color="#C0392B", alpha=0.85, edgecolor="black")
    axes[1].set_title("Anomaly Score Histogram")
    axes[1].set_xlabel("Score")
    axes[1].set_ylabel("Count")

    box = axes[2].boxplot(
        [good_scores, anomaly_scores],
        labels=["good", "anomaly"],
        patch_artist=True,
    )
    box["boxes"][0].set(facecolor="#2E8B57", alpha=0.65)
    box["boxes"][1].set(facecolor="#C0392B", alpha=0.65)
    axes[2].set_title("Score Boxplot")
    axes[2].set_ylabel("Score")

    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return output_path
