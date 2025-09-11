"""Build demo (base -> target) pairs for fusion training."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Iterable, List

import numpy as np
from scipy.ndimage import distance_transform_edt

from hanzitransfer.fusion.ids import guess_layout, render_char

MAPPING: Dict[str, List[str]] = {
    "木": ["校", "林", "棋"],
    "口": ["固", "國"],
    "女": ["娘", "嫣"],
}


def main(argv: Iterable[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Build demo pair dataset")
    parser.add_argument("--out", default="output/fusion/demo_pairs")
    parser.add_argument("--img_size", type=int, default=128)
    parser.add_argument("--repeats", type=int, default=20, help="augmentations per pair")
    args = parser.parse_args(list(argv) if argv is not None else None)

    outdir = Path(args.out)
    outdir.mkdir(parents=True, exist_ok=True)

    for base, targets in MAPPING.items():
        img_base = render_char(base, size=args.img_size)
        dt_base = distance_transform_edt(img_base > 0.5)
        dt_base = dt_base / dt_base.max()
        for tgt in targets:
            img_target = render_char(tgt, size=args.img_size)
            dt_target = distance_transform_edt(img_target > 0.5)
            dt_target = dt_target / dt_target.max()
            layout = guess_layout(img_target)
            for r in range(args.repeats):
                np.savez_compressed(
                    outdir / f"pair_{base}_{tgt}_{r}.npz",
                    base_char=base,
                    target_char=tgt,
                    layout=layout,
                    img_base=img_base[np.newaxis, ...],
                    img_target=img_target[np.newaxis, ...],
                    dt_base=dt_base[np.newaxis, ...],
                    dt_target=dt_target[np.newaxis, ...],
                )


if __name__ == "__main__":  # pragma: no cover
    main()
