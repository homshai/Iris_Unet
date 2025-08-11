# Iris Segmentation - Improved Version

This improved project includes:
- Multi-dataset balanced training (WeightedRandomSampler) to handle domain gaps.
- Stronger augmentations with `albumentations`.
- CBAM attention and an ASPP-like module in the decoder to improve robustness.
- Combined BCEWithLogits + Dice loss.
- Cosine scheduler with linear warmup and optional mixed precision training (AMP).
- Optionally freeze backbone for initial epochs to stabilize training across datasets.

## Usage
Train:
```
python main.py train --data-roots /path/datasetA /path/datasetB /path/datasetC --batch-size 16 --epochs 50 --img-size 256 --amp
```

Infer:
```
python main.py infer --weights checkpoints/best.pth --input /path/to/images --output results --img-size 256
```

Notes:
- Each dataset root must contain `train/images` and `train/masks` (or `images` and `masks`).
- The scripts use WeightedRandomSampler to balance sampling among datasets.
- For further performance, consider:
  - tuning learning rate, batch size, using TTA, stronger post-processing, and experiments with backbones.
