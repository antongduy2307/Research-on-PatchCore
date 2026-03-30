from __future__ import annotations

from pathlib import Path

import torch

from src.datasets.mvtec import IMAGENET_MEAN, IMAGENET_STD


def build_memory_bank_metadata(config: dict) -> dict:
    return {
        "dataset_root": config["data"]["root"],
        "category": config["data"]["category"],
        "image_size": config["data"]["image_size"],
        "crop_size": config["data"]["crop_size"],
        "normalize_mean": list(IMAGENET_MEAN),
        "normalize_std": list(IMAGENET_STD),
        "backbone": config["model"]["backbone"],
        "layers": list(config["model"]["layers"]),
        "local_agg": bool(config["model"]["local_agg"]),
        "subsampling_method": config["memory"].get("subsampling_method", "greedy_coreset"),
        "subsample_ratio": float(config["memory"]["subsample_ratio"]),
        "random_seed": int(config["memory"]["random_seed"]),
    }


def find_matching_memory_bank(models_root: str | Path, metadata: dict) -> Path | None:
    models_root = Path(models_root)
    candidate = models_root / metadata["category"] / "memory_bank.pt"
    if not candidate.exists():
        return None

    checkpoint = torch.load(candidate, map_location="cpu")
    if checkpoint.get("metadata", {}) == metadata:
        return candidate
    return None


def save_memory_bank(
    models_root: str | Path,
    metadata: dict,
    memory_bank: torch.Tensor,
    memory_bank_size_before: int,
) -> Path:
    save_dir = Path(models_root) / metadata["category"]
    save_dir.mkdir(parents=True, exist_ok=True)

    save_path = save_dir / "memory_bank.pt"
    torch.save(
        {
            "metadata": metadata,
            "memory_bank": memory_bank.detach().cpu(),
            "memory_bank_size_before": int(memory_bank_size_before),
            "memory_bank_size_after": int(memory_bank.shape[0]),
            "feature_dim": int(memory_bank.shape[1]),
        },
        save_path,
    )
    return save_path


def load_memory_bank(memory_bank_path: str | Path, device: torch.device) -> dict:
    checkpoint = torch.load(memory_bank_path, map_location="cpu")
    checkpoint["memory_bank"] = checkpoint["memory_bank"].to(device)
    return checkpoint
