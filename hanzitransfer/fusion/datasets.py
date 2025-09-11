"""Datasets for the fusion pipeline."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

LAYOUTS: List[str] = ["⿰", "⿱", "⿴"]
LAYOUT_TO_INDEX: Dict[str, int] = {v: i for i, v in enumerate(LAYOUTS)}


def _load_file(path: Path) -> Dict[str, np.ndarray]:
    if path.suffix == ".npz":
        data = np.load(path, allow_pickle=True)
        return {k: data[k] for k in data.files}
    if path.suffix == ".pt":
        return torch.load(path, map_location="cpu")
    raise ValueError(f"Unsupported file type: {path.suffix}")


class PairsDataset(Dataset):
    """Simple dataset yielding (input_tensor, target_image, meta)."""

    def __init__(self, root: str | Path):
        self.root = Path(root)
        self.files = sorted(
            [p for p in self.root.iterdir() if p.suffix in {".npz", ".pt"}]
        )
        if not self.files:
            raise FileNotFoundError(f"No pair files found in {self.root}")

    def __len__(self) -> int:  # pragma: no cover - trivial
        return len(self.files)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, str]]:
        path = self.files[idx]
        data = _load_file(path)
        dt_base = torch.from_numpy(data["dt_base"]).float()
        img_target = torch.from_numpy(data["img_target"]).float()
        layout = data.get("layout", "⿰")
        layout_idx = LAYOUT_TO_INDEX.get(layout, 0)
        layout_onehot = torch.zeros(len(LAYOUTS), *dt_base.shape[1:])
        layout_onehot[layout_idx] = 1.0
        x = torch.cat([dt_base, layout_onehot], dim=0)
        y = img_target
        meta = {"base_char": data.get("base_char", ""), "target_char": data.get("target_char", ""), "layout": layout}
        return x, y, meta
