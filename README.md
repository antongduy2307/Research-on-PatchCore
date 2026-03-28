# PatchCore Baseline

Minimal PatchCore baseline for anomaly detection on MVTec AD using PyTorch and MLflow.

## Install

Create and activate a virtual environment first, then install the CUDA-enabled PyTorch wheels before the rest of the dependencies.

For NVIDIA GPU on Windows in this project:

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt
```

If you do not want GPU support, install the CPU wheels instead:

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
```

Verify PyTorch can see CUDA:

```bash
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.version.cuda); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu')"
```

## Run

```bash
python run_patchcore.py --config configs/patchcore_base.yaml
```

The default config uses:

- dataset root: `data/mvtec`
- category: `bottle`
- backbone: `wide_resnet50_2`
- layers: `layer2`, `layer3`
- runtime device: `cuda` when available

## MLflow

Start the local MLflow UI with:

```bash
mlflow ui
```

Runs are stored in the default local tracking directory:

```text
./mlruns
```

Experiment name:

```text
patchcore_baseline
```

## Outputs

- Visualizations: `outputs/visualizations/<category>/`
- MLflow artifacts: `mlruns/`

Each saved visualization contains:

- original image
- anomaly heatmap overlay
