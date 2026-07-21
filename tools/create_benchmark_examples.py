#!/usr/bin/env python3
"""Build the manuscript benchmark-example panel from the frozen image set."""

from pathlib import Path

import matplotlib.pyplot as plt
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
IMAGE_DIR = ROOT / "synthetic_benchmark" / "images"
OUTPUT = (
    ROOT
    / "results/no_human/runs/run_20260713_181934_jvcir_corrected"
    / "synthetic_benchmark/figures/benchmark_examples.png"
)

ROWS = [
    ("visual_complexity", "Structural\nclutter"),
    ("layout_order", "Geometric\norder"),
    ("colour_harmony", "Palette\ncoherence"),
    ("visual_intensity", "Visual\nsalience"),
    ("layout_hierarchy", "Focal\nhierarchy"),
]


def image_path(construct: str, level: int) -> Path:
    return IMAGE_DIR / f"A_poster_{construct}_L{level}_S0_I0000.png"


def main() -> None:
    fig = plt.figure(figsize=(10, 16), facecolor="white")
    grid = fig.add_gridspec(
        len(ROWS),
        3,
        width_ratios=[0.95, 1.45, 1.45],
        hspace=0.04,
        wspace=0.08,
    )

    for row_index, (construct, label) in enumerate(ROWS):
        label_ax = fig.add_subplot(grid[row_index, 0])
        label_ax.axis("off")
        label_ax.text(
            0.08,
            0.5,
            label,
            ha="left",
            va="center",
            fontsize=14,
            linespacing=1.5,
        )

        for col_index, level in enumerate((0, 4), start=1):
            path = image_path(construct, level)
            if not path.exists():
                raise FileNotFoundError(path)
            ax = fig.add_subplot(grid[row_index, col_index])
            with Image.open(path) as image:
                ax.imshow(image.convert("RGB"))
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_color("#b8b8b8")
                spine.set_linewidth(0.8)
            if row_index == 0:
                ax.set_title(f"Level {level}", fontsize=16, weight="semibold", pad=10)

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT, dpi=240, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    print(f"Wrote {OUTPUT}")


if __name__ == "__main__":
    main()
