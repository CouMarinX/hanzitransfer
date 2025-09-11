# HanziTransfer

HanziTransfer provides utilities for synthesizing Chinese character images via
an experimental "基字融合" pipeline built on PyTorch.

## Project structure

- `hanzitransfer/fusion` – experimental "基字融合" pipeline.
- `output` – default location for generated data and trained models.
- `tests` – unit tests.

<!-- Legacy data generation, model training, and GUI documentation has been
removed. -->

## Fusion (基字融合)

Build a tiny demo dataset:

```bash
python scripts/build_pairs.py --out output/fusion/demo_pairs
```

Train the conditional UNet:

```bash
hanzi-fuse-train --data_root output/fusion/demo_pairs --epochs 5 --img_size 128
```

Run inference:

```bash
hanzi-fuse --base 木 --layout ⿰ --num 4 --ckpt output/fusion/checkpoints/hanzi_fusion_unet.pt
```

The command writes individual samples and a `grid.png` to the output directory
and prints containment and Chamfer metrics.

## Testing

Install the package and run the test suite with:

```bash
pip install -e .[test]  # or ensure dependencies are installed
pytest
```
