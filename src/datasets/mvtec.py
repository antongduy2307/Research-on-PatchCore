from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.transforms import InterpolationMode


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


@dataclass(frozen=True)
class SampleRecord:
    image_path: Path
    label: int
    mask_path: Path | None
    defect_type: str


class MVTecADDataset(Dataset):
    def __init__(
        self,
        root: str | Path,
        category: str,
        split: str,
        image_size: int = 256,
        crop_size: int = 224,
    ) -> None:
        if split not in {"train", "test"}:
            raise ValueError(f"Unsupported split: {split}")

        self.root = Path(root)
        self.category = category
        self.split = split
        self.category_dir = self.root / category

        if not self.category_dir.exists():
            raise FileNotFoundError(f"Category directory not found: {self.category_dir}")

        self.image_transform = transforms.Compose(
            [
                transforms.Resize(image_size, interpolation=InterpolationMode.BILINEAR),
                transforms.CenterCrop(crop_size),
                transforms.ToTensor(),
                transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ]
        )
        self.mask_transform = transforms.Compose(
            [
                transforms.Resize(image_size, interpolation=InterpolationMode.NEAREST),
                transforms.CenterCrop(crop_size),
                transforms.ToTensor(),
            ]
        )
        self.samples = self._collect_samples()
        self.crop_size = crop_size

    def _collect_samples(self) -> list[SampleRecord]:
        split_dir = self.category_dir / self.split
        if not split_dir.exists():
            raise FileNotFoundError(f"Split directory not found: {split_dir}")

        samples: list[SampleRecord] = []
        defect_dirs = sorted([path for path in split_dir.iterdir() if path.is_dir()])

        for defect_dir in defect_dirs:
            defect_type = defect_dir.name
            if self.split == "train" and defect_type != "good":
                continue

            image_paths = sorted(defect_dir.glob("*.png"))
            for image_path in image_paths:
                is_good = defect_type == "good"
                mask_path = None
                if self.split == "test" and not is_good:
                    mask_name = f"{image_path.stem}_mask.png"
                    candidate = self.category_dir / "ground_truth" / defect_type / mask_name
                    mask_path = candidate if candidate.exists() else None

                samples.append(
                    SampleRecord(
                        image_path=image_path,
                        label=0 if is_good else 1,
                        mask_path=mask_path,
                        defect_type=defect_type,
                    )
                )

        if not samples:
            raise RuntimeError(f"No samples found in {split_dir}")

        return samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor | int | str]:
        sample = self.samples[index]
        image = Image.open(sample.image_path).convert("RGB")
        image_tensor = self.image_transform(image)

        if sample.mask_path is not None:
            mask = Image.open(sample.mask_path).convert("L")
            mask_tensor = self.mask_transform(mask)
            mask_tensor = (mask_tensor > 0.5).float()
        else:
            mask_tensor = torch.zeros((1, self.crop_size, self.crop_size), dtype=torch.float32)

        return {
            "image": image_tensor,
            "label": sample.label,
            "mask": mask_tensor,
            "image_path": str(sample.image_path),
            "defect_type": sample.defect_type,
        }
