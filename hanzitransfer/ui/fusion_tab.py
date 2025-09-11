"""Tkinter Fusion tab using the inference utilities."""

from __future__ import annotations

import tkinter as tk
from pathlib import Path
from typing import List

from PIL import ImageTk

from ..fusion import metrics
from ..fusion.infer_fusion import generate

DEFAULT_CKPT = "output/fusion/checkpoints/hanzi_fusion_unet.pt"


class FusionTab:
    """Simple UI tab for character fusion."""

    def __init__(self, master: tk.Misc) -> None:
        self.frame = tk.Frame(master)
        self.base_var = tk.StringVar()
        self.layout_var = tk.StringVar(value="⿰")

        entry = tk.Entry(self.frame, textvariable=self.base_var, width=5)
        entry.grid(row=0, column=0, padx=5, pady=5)
        layout_menu = tk.OptionMenu(
            self.frame, self.layout_var, "⿰", "⿱", "⿴", "auto"
        )
        layout_menu.grid(row=0, column=1, padx=5, pady=5)
        btn = tk.Button(self.frame, text="Generate 4", command=self.generate)
        btn.grid(row=0, column=2, padx=5, pady=5)
        self.labels: List[tk.Label] = []
        self.metric_labels: List[tk.Label] = []
        for i in range(4):
            lbl = tk.Label(self.frame)
            lbl.grid(row=1, column=i, padx=5, pady=5)
            self.labels.append(lbl)
            mlbl = tk.Label(self.frame, text="")
            mlbl.grid(row=2, column=i)
            self.metric_labels.append(mlbl)
        self.save_button = tk.Button(self.frame, text="Save Best", command=self.save_best)
        self.save_button.grid(row=3, column=0, columnspan=4, pady=5)
        self.images: List[ImageTk.PhotoImage] = []

    def generate(self) -> None:
        base = self.base_var.get().strip() or "木"
        layout = self.layout_var.get()
        if layout == "auto":
            layout = "⿰"
        try:
            outs, base_tensor = generate(base, layout, 4, DEFAULT_CKPT, noise=0.0)
        except Exception as e:  # pragma: no cover - UI feedback
            print(f"Generation failed: {e}")
            return
        self.images.clear()
        for i, img in enumerate(outs):
            from PIL import Image

            pil_img = Image.fromarray((img.squeeze().numpy() * 255).astype("uint8"))
            photo = ImageTk.PhotoImage(pil_img, master=self.frame)
            self.labels[i].config(image=photo)
            self.labels[i].image = photo
            self.images.append(photo)
            contain = metrics.containment_at(img, base_tensor)
            self.metric_labels[i].config(text=f"{contain:.2f}")
        self.selected = 0

    def save_best(self) -> None:
        if not self.images:
            return
        outdir = Path("output/fusion/ui")
        outdir.mkdir(parents=True, exist_ok=True)
        img = self.labels[getattr(self, "selected", 0)].image
        img._PhotoImage__photo.write(str(outdir / "best.png"))

