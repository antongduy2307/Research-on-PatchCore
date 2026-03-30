from __future__ import annotations

import argparse
from pathlib import Path

import mlflow
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.datasets import MVTecADDataset
from src.models import FeatureExtractor
from src.patchcore import PatchCore
from src.utils import (
    build_memory_bank_metadata,
    compute_image_auroc,
    compute_pixel_auroc,
    find_matching_memory_bank,
    load_config,
    load_memory_bank,
    normalize_image_scores,
    save_memory_bank,
    save_anomaly_visualizations,
    set_seed,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Minimal PatchCore baseline for MVTec AD.")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/patchcore_base.yaml",
        help="Path to the YAML config file.",
    )
    return parser.parse_args()


def resolve_device(requested_device: str) -> torch.device:
    if requested_device == "cuda" and not torch.cuda.is_available():
        print("CUDA requested but not available. Falling back to CPU.")
        return torch.device("cpu")
    return torch.device(requested_device)


def build_dataloaders(config: dict) -> tuple[DataLoader, DataLoader]:
    data_cfg = config["data"]
    runtime_cfg = config["runtime"]

    train_dataset = MVTecADDataset(
        root=data_cfg["root"],
        category=data_cfg["category"],
        split="train",
        image_size=data_cfg["image_size"],
        crop_size=data_cfg["crop_size"],
    )
    test_dataset = MVTecADDataset(
        root=data_cfg["root"],
        category=data_cfg["category"],
        split="test",
        image_size=data_cfg["image_size"],
        crop_size=data_cfg["crop_size"],
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=runtime_cfg["batch_size_train"],
        shuffle=False,
        num_workers=data_cfg["num_workers"],
        pin_memory=torch.cuda.is_available(),
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=runtime_cfg["batch_size_test"],
        shuffle=False,
        num_workers=data_cfg["num_workers"],
        pin_memory=torch.cuda.is_available(),
    )
    return train_loader, test_loader


def collect_train_embeddings(
    dataloader: DataLoader,
    extractor: FeatureExtractor,
    patchcore: PatchCore,
    device: torch.device,
) -> torch.Tensor:
    all_patches = []
    for batch in tqdm(dataloader, desc="Building memory bank"):
        images = batch["image"].to(device)
        features = extractor(images)
        patches, _ = patchcore.extract_patch_embeddings(features)
        all_patches.append(patches.reshape(-1, patches.shape[-1]).cpu())

    return torch.cat(all_patches, dim=0).to(device)


def evaluate(
    dataloader: DataLoader,
    extractor: FeatureExtractor,
    patchcore: PatchCore,
    device: torch.device,
    crop_size: int,
) -> dict:
    image_labels: list[int] = []
    image_scores: list[float] = []
    vis_images: list[torch.Tensor] = []
    vis_maps: list[torch.Tensor] = []
    masks: list[torch.Tensor] = []
    vis_paths: list[str] = []

    for batch in tqdm(dataloader, desc="Running inference"):
        images = batch["image"].to(device)
        features = extractor(images)
        patches, patch_shape = patchcore.extract_patch_embeddings(features)
        scores, anomaly_maps = patchcore.score(patches, patch_shape, output_size=crop_size)

        labels = [int(label) for label in batch["label"]]
        image_labels.extend(labels)
        image_scores.extend([float(score.item()) for score in scores])

        for image_tensor, anomaly_map, mask, image_path in zip(
            images.cpu(),
            anomaly_maps.cpu(),
            batch["mask"].cpu(),
            batch["image_path"],
        ):
            vis_images.append(image_tensor)
            vis_maps.append(anomaly_map)
            masks.append(mask)
            vis_paths.append(image_path)

    return {
        "labels": image_labels,
        "scores": image_scores,
        "vis_images": vis_images,
        "vis_maps": vis_maps,
        "masks": masks,
        "vis_paths": vis_paths,
    }


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    set_seed(config["memory"]["random_seed"])

    category = config["data"]["category"]
    device = resolve_device(config["runtime"]["device"])
    train_loader, test_loader = build_dataloaders(config)

    extractor = FeatureExtractor(
        backbone_name=config["model"]["backbone"],
        layers=config["model"]["layers"],
    ).to(device)
    patchcore = PatchCore(local_agg=config["model"]["local_agg"])
    memory_bank_metadata = build_memory_bank_metadata(config)
    models_root = Path(config["eval"]["save_dir"]) / "models"
    cached_memory_bank_path = find_matching_memory_bank(models_root, memory_bank_metadata)

    mlflow.set_experiment(config["experiment"]["name"])
    with mlflow.start_run(run_name=config["experiment"]["run_name"]):
        mlflow.log_params(
            {
                "category": category,
                "backbone": config["model"]["backbone"],
                "layers_used": ",".join(config["model"]["layers"]),
                "image_size": config["data"]["image_size"],
                "crop_size": config["data"]["crop_size"],
                "batch_size_train": config["runtime"]["batch_size_train"],
                "batch_size_test": config["runtime"]["batch_size_test"],
                "subsample_ratio": config["memory"]["subsample_ratio"],
                "device": str(device),
                "local_agg": config["model"]["local_agg"],
            }
        )

        if cached_memory_bank_path is not None:
            checkpoint = load_memory_bank(cached_memory_bank_path, device)
            patchcore.memory_bank = checkpoint["memory_bank"]
            patchcore.memory_bank_size_after = int(checkpoint["memory_bank_size_after"])
            patchcore.memory_bank_size_before = int(checkpoint["memory_bank_size_before"])
            patchcore.feature_dim = int(checkpoint["feature_dim"])
            memory_bank_path = cached_memory_bank_path
            memory_bank_source = "cache"
            print(f"Loaded cached memory bank from: {memory_bank_path}")
        else:
            train_embeddings = collect_train_embeddings(train_loader, extractor, patchcore, device)
            patchcore.build_memory_bank(
                train_embeddings,
                subsample_ratio=config["memory"]["subsample_ratio"],
                seed=config["memory"]["random_seed"],
            )
            memory_bank_path = save_memory_bank(
                models_root,
                memory_bank_metadata,
                patchcore.memory_bank,
                patchcore.memory_bank_size_before,
            )
            memory_bank_source = "rebuilt"
            print(f"Saved new memory bank to: {memory_bank_path}")

        results = evaluate(
            dataloader=test_loader,
            extractor=extractor,
            patchcore=patchcore,
            device=device,
            crop_size=config["data"]["crop_size"],
        )

        normalized_scores, min_good_score, max_defect_score = normalize_image_scores(
            results["labels"],
            results["scores"],
        )
        results["scores"] = normalized_scores
        image_auroc = compute_image_auroc(results["labels"], results["scores"])
        pixel_auroc = compute_pixel_auroc(
            [mask.numpy() for mask in results["masks"]],
            [anomaly_map.numpy() for anomaly_map in results["vis_maps"]],
        )

        vis_dir = Path(config["eval"]["save_dir"]) / "visualizations" / category
        saved_vis_dir = save_anomaly_visualizations(
            images=results["vis_images"],
            anomaly_maps=results["vis_maps"],
            image_scores=results["scores"],
            image_paths=results["vis_paths"],
            save_dir=vis_dir,
            max_samples=config["eval"]["num_vis_samples"],
        )

        num_normal = sum(1 for label in results["labels"] if label == 0)
        num_anomalous = sum(1 for label in results["labels"] if label == 1)

        mlflow.log_metrics(
            {
                "image_auroc": image_auroc,
                "pixel_auroc": pixel_auroc,
                "memory_bank_size_before": patchcore.memory_bank_size_before,
                "memory_bank_size_after": patchcore.memory_bank_size_after,
                "feature_dim": patchcore.feature_dim or 0,
                "num_test_samples": len(results["labels"]),
                "num_normal": num_normal,
                "num_anomalous": num_anomalous,
                "score_min_good": min_good_score,
                "score_max_defect": max_defect_score,
            }
        )
        mlflow.log_param("saved_heatmaps_dir", str(saved_vis_dir))
        mlflow.log_param("memory_bank_source", memory_bank_source)
        mlflow.log_param("memory_bank_path", str(memory_bank_path))
        mlflow.log_artifact(str(memory_bank_path), artifact_path=f"models/{category}")
        mlflow.log_artifacts(str(saved_vis_dir), artifact_path=f"visualizations/{category}")

        print(f"Category: {category}")
        print(f"Image-level AUROC: {image_auroc:.4f}")
        print(f"Pixel-level AUROC: {pixel_auroc:.4f}")
        print(f"Scaled score range anchor (good min -> 0): {min_good_score:.6f}")
        print(f"Scaled score range anchor (defect max -> 1): {max_defect_score:.6f}")
        print(f"Memory bank size before subsampling: {patchcore.memory_bank_size_before}")
        print(f"Memory bank size after subsampling: {patchcore.memory_bank_size_after}")
        print(f"Heatmaps saved to: {saved_vis_dir}")


if __name__ == "__main__":
    main()
