"""Inference utilities for fusion."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable, List, Tuple

import torch
from torchvision.utils import make_grid, save_image

from .datasets import LAYOUT_TO_INDEX, LAYOUTS
from .ids import render_char
from .metrics import chamfer_dt, containment_at
from .models.cond_unet import CondUNet
from .sampling import guided_projected_sampling
from .vectorize import save_svg, vectorize


def generate(
    base: str,
    layout: str,
    num: int,
    ckpt: str,
    noise: float,
    img_size: int = 128,
    device: str = "cpu",
    guide_lambda: float = 3.0,
    proj_every: int = 0,
    tau: float = 0.5,
) -> Tuple[List[torch.Tensor], torch.Tensor]:
    """Generate ``num`` samples and return them along with base tensor."""

    device_t = torch.device(device)
    model = CondUNet(in_channels=1 + len(LAYOUTS))
    state = torch.load(ckpt, map_location=device_t)
    model.load_state_dict(state, strict=False)
    model.to(device_t).eval()

    base_img = render_char(base, size=img_size)
    base_tensor = torch.from_numpy(1.0 - base_img).unsqueeze(0).unsqueeze(0).float()
    layout_idx = LAYOUT_TO_INDEX.get(layout, 0)
    layout_onehot = torch.zeros(1, len(LAYOUTS), img_size, img_size)
    layout_onehot[:, layout_idx] = 1.0
    x = torch.cat([base_tensor, layout_onehot], dim=1).to(device_t)

    results: List[torch.Tensor] = []
    for _ in range(num):
        if proj_every > 0:
            y, _ = guided_projected_sampling(
                model,
                x.to(device_t),
                base_tensor,
                layout,
                steps=1,
                guide_lambda=guide_lambda,
                proj_every=proj_every,
                tau=tau,
            )
            results.append(y.cpu())
        else:
            with torch.no_grad():
                y = model(x, noise=noise).cpu()
            results.append(y)
    return results, base_tensor


def main(argv: Iterable[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Fuse a base character into new forms")
    parser.add_argument("--base", required=True)
    parser.add_argument("--layout", default="⿰")
    parser.add_argument("--num", type=int, default=4)
    parser.add_argument("--noise", type=float, default=0.0)
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--outdir", default="output/fusion/samples")
    parser.add_argument("--export-svg", action="store_true")
    parser.add_argument("--guide-lambda", type=float, default=3.0)
    parser.add_argument("--proj-every", type=int, default=0)
    parser.add_argument("--tau", type=float, default=0.5)
    args = parser.parse_args(list(argv) if argv is not None else None)

    outs, base_tensor = generate(
        base=args.base,
        layout=args.layout,
        num=args.num,
        ckpt=args.ckpt,
        noise=args.noise,
        guide_lambda=args.guide_lambda,
        proj_every=args.proj_every,
        tau=args.tau,
    )

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    for i, img in enumerate(outs):
        path = outdir / f"sample_{i}.png"
        save_image(img, path)
        if args.export_svg:
            paths = vectorize(img.squeeze().numpy())
            save_svg(paths, (img.shape[-1], img.shape[-2]), outdir / f"sample_{i}.svg")
    grid = make_grid(torch.cat(outs, dim=0), nrow=args.num)
    save_image(grid, outdir / "grid.png")

    for i, img in enumerate(outs):
        contain = containment_at(img, base_tensor)
        chamfer = chamfer_dt(img, base_tensor)
        print(f"sample_{i}: contain@0.5={contain:.3f} chamfer={chamfer:.3f}")


if __name__ == "__main__":  # pragma: no cover
    main()
