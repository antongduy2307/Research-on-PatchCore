# PatchCore Baseline Report

## 1. Overview
This document freezes the current PatchCore baseline as the reference point for future experiments. The baseline was executed across all 15 MVTec AD categories available under `data/mvtec` using the current non-parametric PatchCore pipeline with a frozen pretrained `wide_resnet50_2` feature extractor.

## 2. Code Version
- Branch: Main  
- Commit: a704dbb471537564022b91f7c996e772cdc79988
- Date: 28 Mar 2026
- Entry script: Run PatchCore baseline on dataset MVTec
  
## 3. Environment
- OS: Windows 11 (`Windows-11-10.0.26200-SP0`)
- Python: `3.13.12`
- PyTorch: `2.11.0+cu128`
- CUDA: PyTorch CUDA runtime `12.8` on NVIDIA driver reporting CUDA `13.2`
- GPU: `NVIDIA GeForce RTX 4060 Laptop GPU (8 GB VRAM)`
- Dependency snapshot: [outputs/dependency_snapshot.txt](/e:/patchcore_repro/outputs/dependency_snapshot.txt), [requirements.txt](/e:/patchcore_repro/requirements.txt)

## 4. Dataset and Preprocessing
- Dataset: MVTec AD
- Root: `data/mvtec`
- Classes tested: `bottle`, `cable`, `capsule`, `carpet`, `grid`, `hazelnut`, `leather`, `metal_nut`, `pill`, `screw`, `tile`, `toothbrush`, `transistor`, `wood`, `zipper`
- Train split: `train/good` only
- Test split: `test/good` plus all defect folders, with ground-truth masks loaded when available
- Resize: `256`
- Crop: `224` center crop
- Normalize: ImageNet mean/std, mean=`(0.485, 0.456, 0.406)`, std=`(0.229, 0.224, 0.225)`
- Custom preprocessing: None beyond resize, center crop, tensor conversion, and normalization

## 5. Baseline Model Definition
- Model type: PatchCore baseline for anomaly detection, non-parametric memory-bank method
- Backbone: pretrained `wide_resnet50_2`
- Extracted layers: `layer2`, `layer3`
- Feature aggregation: local average pooling `3x3`, stride `1`, padding `1`; `layer3` is bilinearly upsampled to match `layer2`
- Memory bank construction: all train-image patch embeddings are extracted and concatenated
- Subsampling method: random subsampling
- Subsampling ratio: `0.1`
- Distance metric: Euclidean distance via `torch.cdist`
- Patch score rule: minimum distance from each test patch to the memory bank
- Image score rule: maximum patch score within the image, then linearly normalized to `[0, 1]` using `min(good_scores)` and `max(defect_scores)`
- Anomaly map generation: reshape patch scores to the feature-grid spatial size and bilinearly upsample to `224x224`
- Notes: no training loop, no optimizer, no gradient updates, no FAISS, no attention, no coreset sampling

## 6. MLflow Tracking
- Experiment: `patchcore_baseline`
- Run names:
  - `mvtec_bottle_wrn50`
  - `mvtec_cable_wrn50`
  - `mvtec_capsule_wrn50`
  - `mvtec_carpet_wrn50`
  - `mvtec_grid_wrn50`
  - `mvtec_hazelnut_wrn50`
  - `mvtec_leather_wrn50`
  - `mvtec_metal_nut_wrn50`
  - `mvtec_pill_wrn50`
  - `mvtec_screw_wrn50`
  - `mvtec_tile_wrn50`
  - `mvtec_toothbrush_wrn50`
  - `mvtec_transistor_wrn50`
  - `mvtec_wood_wrn50`
  - `mvtec_zipper_wrn50`
- Run IDs:
  - `mvtec_bottle_wrn50`: `4c760ec2c90f4c3db5ed07c7aadd977c`
  - `mvtec_cable_wrn50`: `6b6dea44b75349a492e730bf73a417b2`
  - `mvtec_capsule_wrn50`: `24eee0f0954d4789884c7c2f90c7e7b4`
  - `mvtec_carpet_wrn50`: `5582e6e9df074f41b2f6f8caaee4aede`
  - `mvtec_grid_wrn50`: `ea4b9f7dcf8b4d799be160598cf9daa6`
  - `mvtec_hazelnut_wrn50`: `aba6144602c944f1b190d354edec487e`
  - `mvtec_leather_wrn50`: `ea48de8400f3444182a151e2654a56cd`
  - `mvtec_metal_nut_wrn50`: `f885534069c0406e9d17370bfbb613ba`
  - `mvtec_pill_wrn50`: `8c3f6a296d0f45ee8ceabaaf42f3ae68`
  - `mvtec_screw_wrn50`: `a6fa2b5d74164359a122d6f06fe3de92`
  - `mvtec_tile_wrn50`: `67a5e40faaa04a178e90c9b1687d8d0a`
  - `mvtec_toothbrush_wrn50`: `98f91ac6f0314fd384ea539a0a6b5d0d`
  - `mvtec_transistor_wrn50`: `3bed550bf1a14df09a958fcbf91dbc3a`
  - `mvtec_wood_wrn50`: `c984c57c1b6c4119bece907c32fde324`
  - `mvtec_zipper_wrn50`: `f1649b903cf54dd1b671ae0b7a2b1968`

## 7. Results

### Class: bottle
- Image-level AUROC: `1.0000`
- Pixel-level AUROC: `0.9837`
- Memory bank size before: `163856`
- Memory bank size after: `16386`
- Feature dimension: `1536`
- Heatmaps: `outputs/visualizations/bottle`
- Notes: perfect image-level separation in this run

### Class: screw
- Image-level AUROC: `0.9227`
- Pixel-level AUROC: `0.9862`
- Memory bank size before: `250880`
- Memory bank size after: `25088`
- Feature dimension: `1536`
- Heatmaps: `outputs/visualizations/screw`
- Notes: strongest localization among the weaker image-level categories

### Class: cable
- Image-level AUROC: `0.9799`
- Pixel-level AUROC: `0.9789`
- Memory bank size before: `175616`
- Memory bank size after: `17562`
- Feature dimension: `1536`
- Heatmaps: `outputs/visualizations/cable`
- Notes: strong overall baseline performance

### Class: capsule
- Image-level AUROC: `0.9438`
- Pixel-level AUROC: `0.9858`
- Memory bank size before: `171696`
- Memory bank size after: `17170`
- Feature dimension: `1536`
- Heatmaps: `outputs/visualizations/capsule`
- Notes: image-level ranking is weaker than pixel localization

### Class: carpet
- Image-level AUROC: `0.9888`
- Pixel-level AUROC: `0.9876`
- Memory bank size before: `219520`
- Memory bank size after: `21952`
- Feature dimension: `1536`
- Heatmaps: `outputs/visualizations/carpet`
- Notes: strong at both image and pixel levels

### Class: grid
- Image-level AUROC: `0.9620`
- Pixel-level AUROC: `0.9734`
- Memory bank size before: `206976`
- Memory bank size after: `20698`
- Feature dimension: `1536`
- Heatmaps: `outputs/visualizations/grid`
- Notes: good overall, but below the best categories

### Class: hazelnut
- Image-level AUROC: `1.0000`
- Pixel-level AUROC: `0.9847`
- Memory bank size before: `306544`
- Memory bank size after: `30655`
- Feature dimension: `1536`
- Heatmaps: `outputs/visualizations/hazelnut`
- Notes: perfect image-level separation in this run

### Class: leather
- Image-level AUROC: `1.0000`
- Pixel-level AUROC: `0.9901`
- Memory bank size before: `192080`
- Memory bank size after: `19208`
- Feature dimension: `1536`
- Heatmaps: `outputs/visualizations/leather`
- Notes: strongest pixel-level score in this sweep

### Class: metal_nut
- Image-level AUROC: `0.9956`
- Pixel-level AUROC: `0.9804`
- Memory bank size before: `172480`
- Memory bank size after: `17248`
- Feature dimension: `1536`
- Heatmaps: `outputs/visualizations/metal_nut`
- Notes: highly stable baseline result

### Class: pill
- Image-level AUROC: `0.9343`
- Pixel-level AUROC: `0.9680`
- Memory bank size before: `209328`
- Memory bank size after: `20933`
- Feature dimension: `1536`
- Heatmaps: `outputs/visualizations/pill`
- Notes: among the weakest image-level categories in this sweep

### Class: tile
- Image-level AUROC: `0.9924`
- Pixel-level AUROC: `0.9527`
- Memory bank size before: `180320`
- Memory bank size after: `18032`
- Feature dimension: `1536`
- Heatmaps: `outputs/visualizations/tile`
- Notes: image-level classification is strong, localization is weaker than most categories

### Class: toothbrush
- Image-level AUROC: `0.9861`
- Pixel-level AUROC: `0.9864`
- Memory bank size before: `47040`
- Memory bank size after: `4704`
- Feature dimension: `1536`
- Heatmaps: `outputs/visualizations/toothbrush`
- Notes: compact train set but strong result

### Class: transistor
- Image-level AUROC: `1.0000`
- Pixel-level AUROC: `0.9607`
- Memory bank size before: `166992`
- Memory bank size after: `16700`
- Feature dimension: `1536`
- Heatmaps: `outputs/visualizations/transistor`
- Notes: perfect image-level separation, moderate localization relative to the top pixel results

### Class: wood
- Image-level AUROC: `0.9886`
- Pixel-level AUROC: `0.9361`
- Memory bank size before: `193648`
- Memory bank size after: `19365`
- Feature dimension: `1536`
- Heatmaps: `outputs/visualizations/wood`
- Notes: weakest pixel-level score in this sweep

### Class: zipper
- Image-level AUROC: `0.9808`
- Pixel-level AUROC: `0.9779`
- Memory bank size before: `188160`
- Memory bank size after: `18816`
- Feature dimension: `1536`
- Heatmaps: `outputs/visualizations/zipper`
- Notes: strong overall baseline result

## 8. Observations
- Which class performs strongly? Image-level performance is strongest on `bottle`, `hazelnut`, `leather`, and `transistor` with `1.0000` AUROC. Pixel-level performance is strongest on `leather` (`0.9901`), `carpet` (`0.9876`), `toothbrush` (`0.9864`), and `screw` (`0.9862`).
- Which class performs weakly? The weakest image-level classes are `screw` (`0.9227`), `pill` (`0.9343`), and `capsule` (`0.9438`). The weakest pixel-level classes are `wood` (`0.9361`) and `tile` (`0.9527`).
- Any obvious false positives? Based on metrics rather than exhaustive manual review, categories with lower image-level AUROC such as `screw`, `pill`, and `capsule` are the most likely to contain false-positive image rankings.
- Any obvious false negatives? Also based on metrics rather than exhaustive manual review, weaker image-level categories likely contain some false-negative image rankings, while `wood` and `tile` are the most likely to have under-localized anomalous regions.
- Are heatmaps visually meaningful? Overall yes. Pixel-level AUROC is high for most classes, which suggests the anomaly maps are generally informative. The most caution is warranted for `wood` and `tile`, where localization quality is relatively weaker.

## 9. Current Limitations
- Image-score normalization uses test-label information by anchoring `0` to the minimum good score and `1` to the maximum defect score, so that normalization is evaluation-time only and not deployable as-is.
- Memory bank subsampling is random only; no coreset selection or FAISS acceleration is used in this baseline.
- The sweep report is metric-driven; heatmap observations were not manually audited image-by-image for every sample in every category.

## 10. Next Steps
1. Add an automated all-category evaluation script that writes a machine-readable summary file directly instead of relying on console logs.
2. Separate deployment-time scoring from evaluation-time normalized scoring so future experiments can compare both raw and normalized metrics cleanly.
