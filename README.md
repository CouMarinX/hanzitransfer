# HanziTransfer

HanziTransfer provides utilities for synthesizing Chinese character images,
training convolutional or variational autoencoder models, and running a small
Tkinter GUI for style transfer experiments.

## Project structure

- `hanzitransfer/data` – dataset generation tools.
- `hanzitransfer/models` – CNN and CVAE models and training scripts.
- `hanzitransfer/inference` – model loading helpers and stroke renderers.
- `hanzitransfer/ui` – graphical interface built with Tkinter.
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

## Testing

Install the package and run the test suite with:

```bash
pip install -e .[test]  # or ensure dependencies are installed
pytest
```
