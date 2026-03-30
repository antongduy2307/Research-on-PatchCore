from __future__ import annotations

from datetime import datetime
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
        "subsample_ratio": float(config["memory"]["subsample_ratio"]),
        "random_seed": int(config["memory"]["random_seed"]),
    }


def find_matching_memory_bank(models_root: str | Path, metadata: dict) -> Path | None:
    models_root = Path(models_root)
    if not models_root.exists():
        return None

    candidates = sorted(models_root.glob(f"*/{metadata['category']}/memory_bank.pt"), reverse=True)
    for candidate in candidates:
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
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_dir = Path(models_root) / timestamp / metadata["category"]
    save_dir.mkdir(parents=True, exist_ok=True)

    save_path = save_dir / "memory_bank.pt"
    torch.save(
        {
            "metadata": metadata,
            "memory_bank": memory_bank.detach().cpu(),
            "memory_bank_size_before": int(memory_bank_size_before),
            "memory_bank_size_after": int(memory_bank.shape[0]),
            "feature_dim": int(memory_bank.shape[1]),
            "created_at": timestamp,
        },
        save_path,
    )
    return save_path


def load_memory_bank(memory_bank_path: str | Path, device: torch.device) -> dict:
    checkpoint = torch.load(memory_bank_path, map_location="cpu")
    checkpoint["memory_bank"] = checkpoint["memory_bank"].to(device)
    return checkpoint
