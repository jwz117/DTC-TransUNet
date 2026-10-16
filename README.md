# DTC-TransUNet

PyTorch implementation for binary landslide segmentation in remote-sensing imagery.

## Repository structure

```text
DTC-TransUNet/
├── models/
│   ├── modules.py
│   └── dtc_transunet.py
├── datasets/
│   └── npy_dataset.py
├── utils/
│   ├── losses.py
│   ├── metrics.py
│   └── checkpoint.py
├── configs/
│   └── default.yaml
├── trainer.py
├── train.py
├── requirements.txt
├── LICENSE
└── README.md
```

## Model components

- **MPE**: morphological boundary and regional-structure prior extraction.
- **DGM**: dynamic gating modulation of encoder features using morphological priors.
- **DTC-GCN**: dynamic graph construction and tri-source graph feature interaction.
- **GCA**: graph cross-attention.
- **Transformer bottleneck and decoder**: global dependency modeling and pixel-wise binary prediction.

The CLI seeds Python, NumPy, and PyTorch through `utils/reproducibility.py`. The file `configs/default.yaml` records reference settings; the current CLI does not automatically load this YAML file, so command-line arguments and code defaults control execution. Check the final manuscript against the code before citing this repository as an exact reproduction.

## Environment

Recommended Python 3.10 or 3.11. Install a PyTorch build matching your operating system and CUDA version from the official selector:

https://pytorch.org/get-started/locally/

Then install the other dependencies:

```bash
pip install -r requirements.txt
```

Check the environment:

```bash
python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.version.cuda)"
```

## Data format

The loader expects NumPy `.npy` arrays. Images should be `[N, C, H, W]`; masks should be `[N, H, W]` or `[N, 1, H, W]`. Use binary masks with background `0` and landslide `1`. Ensure images and masks have matching sample counts and order.

Expected local filenames:

```text
data/
├── train_images.npy
├── train_labels.npy
├── val_images.npy
├── val_labels.npy
├── test_images.npy
└── test_labels.npy
```

These files are not included in the repository. Update the paths in the CLI arguments if your files are stored elsewhere.

## Training

Inspect available arguments:

```bash
python train.py train --help
```

Example (paths must exist locally):

```bash
python train.py train \
  --train_images data/train_images.npy \
  --train_labels data/train_labels.npy \
  --val_images data/val_images.npy \
  --val_labels data/val_labels.npy \
  --test_images data/test_images.npy \
  --test_labels data/test_labels.npy \
  --epochs 100 --batch_size 16 --gpu_id 0
```

Use `--allow_cpu` to permit CPU execution when CUDA is unavailable. Use `--skip_final_test` to skip the final held-out test evaluation; in that case, test NPY files are not required during training.

## Standalone testing

```bash
python train.py test \
  --test_images data/test_images.npy \
  --test_labels data/test_labels.npy \
  --ckpt_path results/<run>/best_model.pth \
  --norm_stats results/<run>/norm_stats.npz \
  --gpu_id 0
```

The checkpoint should be produced by the same architecture/configuration used for evaluation. Prefer the saved training normalization statistics; computing normalization statistics from the test set is not recommended.

## Metrics

The reported mIoU is the arithmetic mean of background IoU and landslide IoU, calculated from the confusion matrix accumulated over the full evaluation set. The script also reports Precision, Recall, F1, and overall accuracy.

## Reproducibility notes

- Confirm the model modules and ablation settings against the final paper.
- Keep train/validation/test splits fixed and do not use test labels for checkpoint selection.
- Record the input size, input channels, seed, PyTorch/CUDA versions, and FLOPs counting convention.
- Run a complete train-validation-test cycle on the target dataset before reporting results.
- Add the final paper citation and verify third-party code/data licenses before public release.
- The segmentation loss currently uses CE + Dice + Tversky weights (0.3, 0.4, 0.3); these are set in `trainer.py`, not configurable from the CLI.
