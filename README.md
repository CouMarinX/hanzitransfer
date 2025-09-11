# HanziTransfer

HanziTransfer provides utilities for synthesizing Chinese character images,
training convolutional or variational autoencoder models, and running a small
Tkinter GUI for style transfer experiments.

## Project structure

- `hanzitransfer/data` – dataset generation tools.
- `hanzitransfer/models` – CNN and CVAE models and training scripts.
- `hanzitransfer/inference` – model loading helpers and stroke renderers.
- `hanzitransfer/ui` – graphical interface built with Tkinter.
- `hanzitransfer/fusion` – experimental "基字融合" pipeline.
- `output` – default location for generated data and trained models.
- `tests` – unit tests.

## Data generation

Create synthetic training samples:

```bash
python -m hanzitransfer.data.testmaking --output output --limit 100
```

This writes `synth_base.npz` and `synth_target.npz` in the `output` directory.
Convert the arrays to `inputs.npy` and `targets.npy` if you plan to use the
CNN training script.

## Model training

Train the convolutional model on prepared arrays:

```bash
python -m hanzitransfer.models.cnn --data_dir output --epochs 10 --batch_size 32 --model_path hanzi_style_model.keras
```

A conditional variational autoencoder training script is also available:

```bash
python -m hanzitransfer.models.modeltraining_cvae
```

## Graphical interface

Once a model is saved as `hanzi_style_model.keras` in the repository root,
launch the GUI:

```bash
hanzi-ui
```

or

```bash
python -m hanzitransfer.ui.front_cnnver2
```

This GUI now includes a **Fusion (基字融合)** tab for generating new
characters conditioned on a base glyph.

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
and prints containment and Chamfer metrics. The Tkinter UI's **Fusion** tab can
also be used to generate candidates from the trained checkpoint.

## Testing

Install the package and run the test suite with:

```bash
pip install -e .[test]  # or ensure dependencies are installed
pytest
```
