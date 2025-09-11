# HanziTransfer

HanziTransfer provides utilities for synthesizing Chinese character images via
an experimental "基字融合" pipeline built on PyTorch.

## Installation

Set up a virtual environment and install the project in editable mode:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

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

## Motif-Constrained Fusion (Publishable Prototype)

This prototype introduces a motif-constrained generation task where a base
character motif must be preserved within each synthesis.  Generation proceeds
via a symbolic IDS layer and a vector geometry stage which are projected onto a
feasible set ``C`` using a proximal operator ``Proj_C``.

```
Raster -> Skeleton -> Paths -> Proj_C -> Raster
```

### Metrics

- **Containment@τ/AUC** – how well the output covers the base motif.
- **IDS Validity** – structural legality of the predicted IDS tree.
- **Vector Quality** – self‑intersection count and curvature statistics.
- **Novelty@k** – distance to a small reference glyph set.

### Ethics & safety

Generated glyphs are filtered by a novelty heuristic to avoid accidental
collisions with existing Unicode characters.  Ensure that fonts used for
training permit derivative works.

### Quickstart (CPU)

```bash
PYTHONPATH=. python scripts/build_pairs.py --out output/fusion/demo_pairs
python -m hanzitransfer.fusion.train_fusion --data_root output/fusion/demo_pairs \
    --epochs 1 --batch_size 4 --img_size 128 --save_dir output/fusion/checkpoints
python -m hanzitransfer.fusion.infer_fusion --base 木 --layout ⿰ --num 4 \
    --ckpt output/fusion/checkpoints/hanzi_fusion_unet.pt \
    --guide-lambda 3.0 --proj-every 1 --tau 0.5 --outdir output/fusion/samples
PYTHONPATH=. python bench/run_bench.py  # optional benchmark
```

## Testing

Install the package and run the test suite with:

```bash
pip install -e .[test]  # or ensure dependencies are installed
pytest
```
