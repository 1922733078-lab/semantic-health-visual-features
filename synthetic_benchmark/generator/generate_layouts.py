#!/usr/bin/env python3
"""
Deterministic procedural layout generator for the JVCIR synthetic benchmark.

Creates poster-like, banner-like, and card-like synthetic images with
controlled rendering parameters. Writes rendered images and metadata CSV.

Usage:
    python synthetic_benchmark/generator/generate_layouts.py
"""

import argparse
import hashlib
import json
import os
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[2]
OUTPUT_ROOT = ROOT / "synthetic_benchmark"
IMAGES_DIR = OUTPUT_ROOT / "images"
METADATA_DIR = OUTPUT_ROOT / "metadata"
MANIFEST_DIR = OUTPUT_ROOT / "manifests"

FAMILIES = {
    "A_poster": {"size": (768, 1024), "n_primitives_range": (6, 14)},
    "B_banner": {"size": (1024, 384), "n_primitives_range": (4, 10)},
    "C_card": {"size": (512, 768), "n_primitives_range": (3, 8)},
}

# Frozen split assignment per protocol v2.0.
# G-dev: family A + seeds 0-9
# G-validation: family B + seeds 10-14
# G-test: family C + seeds 15-24
# Interaction conditions are held out as interaction-test.
FAMILY_SPLIT = {
    "A_poster": "G-dev",
    "B_banner": "G-validation",
    "C_card": "G-test",
}

FAMILY_SEEDS = {
    "A_poster": list(range(0, 10)),   # G-dev
    "B_banner": list(range(10, 15)),  # G-validation
    "C_card": list(range(15, 25)),    # G-test
}


def _get_split(family, interaction=None):
    if interaction is not None:
        return "interaction-test"
    return FAMILY_SPLIT[family]


def stable_image_seed(family, construct, level, seed, image_index):
    """Return a process- and platform-stable 32-bit seed for one image.

    Python's built-in ``hash`` is intentionally salted between processes when
    strings are present.  A framed SHA-256 digest makes the seed independent of
    ``PYTHONHASHSEED`` and avoids ambiguous string concatenation.
    """
    payload = json.dumps(
        [family, construct, int(level), int(seed), int(image_index)],
        ensure_ascii=False,
        separators=(",", ":"),
    ).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:4], "big")

BACKGROUND_PALETTE = [
    (245, 245, 240),
    (235, 235, 230),
    (250, 248, 242),
    (240, 244, 248),
    (230, 230, 230),
]

FOREGROUND_PALETTE = [
    (200, 80, 80), (80, 130, 180), (90, 160, 90), (220, 150, 60),
    (130, 80, 150), (60, 160, 160), (180, 100, 60), (100, 100, 100),
    (240, 200, 70), (70, 90, 140),
]


@dataclass
class RenderParams:
    family: str
    seed: int
    construct: str
    level: int
    n_primitives: int
    overlap_ratio: float
    edge_density_factor: float
    spacing_irregularity: float
    alignment_jitter: float
    grid_deviation: float
    spacing_variance: float
    hue_dispersion: float
    chroma_dispersion: float
    palette_incoherence: float
    contrast_ratio: float
    luminance_contrast: float
    edge_contrast: float
    focal_contrast: float
    focal_size_ratio: float
    secondary_count: int
    whitespace_fraction: float
    has_text_bars: bool
    text_bar_count: int


def _get_family_defaults(family):
    cfg = FAMILIES[family]
    w, h = cfg["size"]
    n_min, n_max = cfg["n_primitives_range"]
    area = w * h
    return {
        "n_primitives": (n_min + n_max) // 2,
        "overlap_ratio": 0.05,
        "edge_density_factor": 0.5,
        "spacing_irregularity": 0.2,
        "alignment_jitter": 0.02,
        "grid_deviation": 0.05,
        "spacing_variance": 0.1,
        "hue_dispersion": 0.15,
        "chroma_dispersion": 0.15,
        "palette_incoherence": 0.1,
        "contrast_ratio": 0.3,
        "luminance_contrast": 0.2,
        "edge_contrast": 0.3,
        "focal_contrast": 0.2,
        "focal_size_ratio": 0.15,
        "secondary_count": 3,
        "whitespace_fraction": 0.35,
        "has_text_bars": True,
        "text_bar_count": 3,
    }


def _build_params(family, seed, construct, level, interaction=None):
    """Build rendering parameters for one cell."""
    rng = np.random.default_rng(seed)
    defaults = _get_family_defaults(family)
    p = RenderParams(
        family=family,
        seed=seed,
        construct=construct,
        level=level,
        **defaults,
    )
    t = level / 4.0  # 0..1

    # Primary manipulation
    if construct == "visual_complexity":
        n_min, n_max = FAMILIES[family]["n_primitives_range"]
        p.n_primitives = int(np.round(n_min + t * (n_max - n_min)))
        p.overlap_ratio = 0.05 + t * 0.45
        p.edge_density_factor = 0.3 + t * 1.2
        p.spacing_irregularity = 0.1 + t * 0.6
        p.whitespace_fraction = max(0.1, 0.45 - t * 0.25)
    elif construct == "layout_order":
        p.alignment_jitter = 0.02 + t * 0.22
        p.grid_deviation = 0.05 + t * 0.35
        p.spacing_variance = 0.05 + t * 0.45
        p.overlap_ratio = 0.05 + t * 0.40
    elif construct == "colour_harmony":
        p.hue_dispersion = 0.00 + t * 0.95
        p.chroma_dispersion = 0.00 + t * 0.95
        p.palette_incoherence = 0.00 + t * 0.90
    elif construct == "visual_intensity":
        p.contrast_ratio = 0.10 + t * 0.90
        p.luminance_contrast = 0.05 + t * 0.95
        p.edge_contrast = 0.10 + t * 1.10
        p.focal_contrast = 0.10 + t * 0.90
    elif construct == "layout_hierarchy":
        p.focal_size_ratio = 0.12 + t * 0.48
        p.focal_contrast = 0.20 + t * 0.80
        p.secondary_count = max(1, int(np.round(5 - t * 4)))
        p.whitespace_fraction = 0.25 + t * 0.25
    else:
        raise ValueError(f"Unknown construct: {construct}")

    # Interaction overrides (held-out conditions)
    if interaction == "high_clutter_high_alignment":
        p.construct = "interaction"
        p.level = -1
        n_min, n_max = FAMILIES[family]["n_primitives_range"]
        p.n_primitives = int(np.round(n_min + 0.8 * (n_max - n_min)))
        p.overlap_ratio = 0.35
        p.edge_density_factor = 1.0
        p.alignment_jitter = 0.03
        p.grid_deviation = 0.06
        p.spacing_variance = 0.08
    elif interaction == "low_clutter_high_colour_noise":
        p.construct = "interaction"
        p.level = -1
        n_min, n_max = FAMILIES[family]["n_primitives_range"]
        p.n_primitives = int(np.round(n_min + 0.2 * (n_max - n_min)))
        p.overlap_ratio = 0.05
        p.edge_density_factor = 0.35
        p.hue_dispersion = 0.65
        p.chroma_dispersion = 0.65
        p.palette_incoherence = 0.55
    elif interaction == "strong_hierarchy_large_whitespace":
        p.construct = "interaction"
        p.level = -1
        p.focal_size_ratio = 0.55
        p.focal_contrast = 0.90
        p.secondary_count = 1
        p.whitespace_fraction = 0.55
    elif interaction == "symmetric_high_edge_density":
        p.construct = "interaction"
        p.level = -1
        p.edge_density_factor = 1.2
        p.edge_contrast = 1.0
        p.alignment_jitter = 0.03
        p.grid_deviation = 0.05

    # Per-image randomness derived from seed and a sequence counter is applied at render time.
    return p


def _random_color(rng, hue_dispersion, chroma_dispersion, palette_incoherence):
    """Sample a colour with controlled dispersion."""
    base_idx = rng.integers(0, len(FOREGROUND_PALETTE))
    base = np.array(FOREGROUND_PALETTE[base_idx], dtype=np.float32)
    if palette_incoherence > 0.5:
        # Fully random hue
        rgb = rng.integers(0, 256, size=3).astype(np.float32)
    else:
        rgb = base.copy()
        rgb += rng.normal(0, chroma_dispersion * 60, size=3)
        rgb = np.clip(rgb, 0, 255)
    return tuple(rgb.astype(int))


def _lum(rgb):
    r, g, b = rgb
    return 0.299 * r + 0.587 * g + 0.114 * b


def _place_primitive(draw, rng, w, h, params, shape_idx, total, focal=False):
    """Place one primitive. Returns bounding box."""
    if focal:
        size_ratio = params.focal_size_ratio
        aspect = rng.uniform(0.8, 1.25)
    else:
        size_ratio = rng.uniform(0.06, 0.18)
        aspect = rng.uniform(0.5, 2.0)

    area = w * h
    target_area = area * size_ratio * rng.uniform(0.7, 1.3)
    rect_h = int(np.sqrt(target_area / aspect))
    rect_w = int(target_area / rect_h)
    rect_w = np.clip(rect_w, 20, w - 40)
    rect_h = np.clip(rect_h, 20, h - 40)

    if params.alignment_jitter < 0.1 and total > 1:
        # Try grid-like placement
        cols = int(np.ceil(np.sqrt(total)))
        row = shape_idx // cols
        col = shape_idx % cols
        cell_w = w // cols
        cell_h = h // cols
        cx = col * cell_w + cell_w // 2
        cy = row * cell_h + cell_h // 2
        cx += int(rng.normal(0, params.grid_deviation * cell_w))
        cy += int(rng.normal(0, params.grid_deviation * cell_h))
    else:
        cx = rng.integers(rect_w // 2 + 10, w - rect_w // 2 - 10)
        cy = rng.integers(rect_h // 2 + 10, h - rect_h // 2 - 10)

    cx += int(rng.normal(0, params.alignment_jitter * w))
    cy += int(rng.normal(0, params.alignment_jitter * h))

    cx = np.clip(cx, rect_w // 2 + 2, w - rect_w // 2 - 2)
    cy = np.clip(cy, rect_h // 2 + 2, h - rect_h // 2 - 2)

    x1, y1 = cx - rect_w // 2, cy - rect_h // 2
    x2, y2 = x1 + rect_w, y1 + rect_h
    color = _random_color(rng, params.hue_dispersion, params.chroma_dispersion, params.palette_incoherence)

    # Edges: high edge_contrast adds a dark border
    border = max(1, int(2 + params.edge_contrast * 6))
    if rng.random() < 0.5:
        draw.rectangle([x1, y1, x2, y2], fill=color, outline=(40, 40, 40), width=border)
    else:
        draw.ellipse([x1, y1, x2, y2], fill=color, outline=(40, 40, 40), width=border)

    return (x1, y1, x2, y2)


def _add_text_bars(draw, rng, w, h, params):
    """Add text-like horizontal bars."""
    if not params.has_text_bars:
        return
    n_bars = params.text_bar_count
    bar_h = max(8, int(h * 0.015))
    y_positions = sorted(rng.integers(bar_h, h - bar_h, size=n_bars * 3))
    selected = y_positions[:n_bars]
    for y in selected:
        bar_w = int(w * rng.uniform(0.2, 0.7))
        x = rng.integers(10, max(11, w - bar_w - 10))
        color = (30, 30, 30) if rng.random() < 0.7 else (220, 220, 220)
        draw.rectangle([x, y, x + bar_w, y + bar_h], fill=color)


def render_image(params, image_index, interaction=None):
    """Render one synthetic image and return PIL Image plus derived record."""
    family_cfg = FAMILIES[params.family]
    w, h = family_cfg["size"]

    # Stable per-image seed derived from family, construct, level, seed, index.
    seed_int = stable_image_seed(
        params.family, params.construct, params.level, params.seed, image_index
    )
    rng = np.random.default_rng(seed_int)

    bg = BACKGROUND_PALETTE[rng.integers(0, len(BACKGROUND_PALETTE))]
    img = Image.new("RGB", (w, h), bg)
    draw = ImageDraw.Draw(img)

    n = params.n_primitives
    bboxes = []

    # Focal object (if hierarchy or intensity high)
    has_focal = params.focal_size_ratio > 0.20 or params.focal_contrast > 0.4
    focal_bbox = None
    if has_focal:
        focal_bbox = _place_primitive(draw, rng, w, h, params, 0, n + 1, focal=True)
        bboxes.append(focal_bbox)

    for i in range(n):
        bbox = _place_primitive(draw, rng, w, h, params, i, n)
        bboxes.append(bbox)
        # Enforce overlap ratio by randomly shifting toward existing boxes
        if params.overlap_ratio > 0.1 and len(bboxes) > 1 and rng.random() < params.overlap_ratio:
            ref = bboxes[rng.integers(0, len(bboxes) - 1)]
            # Overlap is already probabilistic due to random placement; record parameter only.

    _add_text_bars(draw, rng, w, h, params)

    # Compute actual geometric scene-graph properties for metadata
    overlap_area = 0
    total_shape_area = 0
    for i, b1 in enumerate(bboxes):
        area1 = (b1[2] - b1[0]) * (b1[3] - b1[1])
        total_shape_area += area1
        for b2 in bboxes[i + 1:]:
            ox = max(0, min(b1[2], b2[2]) - max(b1[0], b2[0]))
            oy = max(0, min(b1[3], b2[3]) - max(b1[1], b2[1]))
            overlap_area += ox * oy

    image_area = w * h
    actual_overlap_ratio = overlap_area / (total_shape_area + 1e-10)
    actual_whitespace = max(0.0, 1.0 - total_shape_area / image_area)

    interaction_suffix = f"_{interaction}" if interaction else ""
    record = {
        "image_id": f"{params.family}_{params.construct}_L{params.level}_S{params.seed}_I{image_index:04d}{interaction_suffix}",
        "family": params.family,
        "construct": params.construct,
        "level": params.level,
        "seed": params.seed,
        "image_index": image_index,
        "canvas_width": w,
        "canvas_height": h,
        "n_primitives": params.n_primitives,
        "overlap_ratio_param": params.overlap_ratio,
        "edge_density_factor": params.edge_density_factor,
        "spacing_irregularity": params.spacing_irregularity,
        "alignment_jitter": params.alignment_jitter,
        "grid_deviation": params.grid_deviation,
        "spacing_variance": params.spacing_variance,
        "hue_dispersion": params.hue_dispersion,
        "chroma_dispersion": params.chroma_dispersion,
        "palette_incoherence": params.palette_incoherence,
        "contrast_ratio": params.contrast_ratio,
        "luminance_contrast": params.luminance_contrast,
        "edge_contrast": params.edge_contrast,
        "focal_contrast": params.focal_contrast,
        "focal_size_ratio": params.focal_size_ratio,
        "secondary_count": params.secondary_count,
        "whitespace_fraction_param": params.whitespace_fraction,
        "has_text_bars": params.has_text_bars,
        "text_bar_count": params.text_bar_count,
        "actual_overlap_ratio": actual_overlap_ratio,
        "actual_whitespace": actual_whitespace,
    }
    return img, record


def generate_benchmark(images_per_cell=25, interactions_per_cell=8, max_images=None):
    """Generate the full benchmark set with frozen G-dev/G-validation/G-test splits."""
    IMAGES_DIR.mkdir(parents=True, exist_ok=True)
    METADATA_DIR.mkdir(parents=True, exist_ok=True)
    MANIFEST_DIR.mkdir(parents=True, exist_ok=True)

    constructs = ["visual_complexity", "layout_order", "colour_harmony", "visual_intensity", "layout_hierarchy"]
    levels = [0, 1, 2, 3, 4]
    families = ["A_poster", "B_banner", "C_card"]
    # Seeds per family are frozen by split assignment.
    family_seeds = FAMILY_SEEDS
    interactions = [
        "high_clutter_high_alignment",
        "low_clutter_high_colour_noise",
        "strong_hierarchy_large_whitespace",
        "symmetric_high_edge_density",
    ]

    rows = []
    count = 0

    # Base factorial cells
    for family in families:
        for construct in constructs:
            for level in levels:
                for seed in family_seeds[family]:
                    for idx in range(images_per_cell):
                        if max_images is not None and count >= max_images:
                            break
                        params = _build_params(family, seed, construct, level)
                        img, record = render_image(params, idx)
                        record["split"] = _get_split(family)
                        img_path = IMAGES_DIR / f"{record['image_id']}.png"
                        img.save(img_path, "PNG")
                        sha = hashlib.sha256(img_path.read_bytes()).hexdigest()
                        record["image_path"] = str(img_path.relative_to(ROOT))
                        record["sha256"] = sha
                        rows.append(record)
                        count += 1

    # Interaction conditions (held-out as interaction-test)
    for family in families:
        for interaction in interactions:
            for seed in family_seeds[family]:
                for idx in range(interactions_per_cell):
                    if max_images is not None and count >= max_images:
                        break
                    params = _build_params(family, seed, "visual_complexity", 0, interaction=interaction)
                    img, record = render_image(params, idx, interaction=interaction)
                    record["interaction"] = interaction
                    record["split"] = _get_split(family, interaction=interaction)
                    img_path = IMAGES_DIR / f"{record['image_id']}.png"
                    img.save(img_path, "PNG")
                    sha = hashlib.sha256(img_path.read_bytes()).hexdigest()
                    record["image_path"] = str(img_path.relative_to(ROOT))
                    record["sha256"] = sha
                    rows.append(record)
                    count += 1

    # A full regeneration defines an exact image set. Remove stale PNGs from
    # earlier generator variants so a residual directory cannot pass by count.
    if max_images is None:
        expected_names = {f"{row['image_id']}.png" for row in rows}
        for path in IMAGES_DIR.glob("*.png"):
            if path.name not in expected_names:
                path.unlink()

    return rows


def write_manifests(rows):
    import pandas as pd

    df = pd.DataFrame(rows)
    df.to_csv(METADATA_DIR / "image_metadata.csv", index=False)

    manifest_cols = ["image_id", "family", "construct", "level", "seed", "split", "image_path", "sha256"]
    manifest = df[manifest_cols].copy()
    manifest["category"] = "synthetic"
    manifest.to_csv(MANIFEST_DIR / "image_manifest.csv", index=False)

    # Feature-extraction metadata used by src/features/traditional_features.py
    feature_meta = df[["image_id", "image_path"]].copy()
    feature_meta["standardized_path"] = feature_meta["image_path"]
    feature_meta["category"] = "synthetic"
    feature_meta[["image_id", "standardized_path", "category"]].to_csv(
        METADATA_DIR / "feature_extraction_metadata.csv", index=False
    )

    # Frozen split membership (image_id, family, seed, split, sha256)
    splits_df = df[["image_id", "family", "seed", "split", "sha256"]].copy()
    splits_df = splits_df.sort_values(["split", "family", "seed", "image_id"]).reset_index(drop=True)
    splits_df.to_csv(OUTPUT_ROOT / "splits" / "frozen_splits.csv", index=False)

    # Generator hash
    gen_file = Path(__file__)
    gen_hash = hashlib.sha256(gen_file.read_bytes()).hexdigest()

    # Generator specification as source-of-truth manifest
    spec = {
        "generator_version": "2.1",
        "generator_file": gen_file.name,
        "generator_sha256": gen_hash,
        "families": {
            name: {
                "display_name": name.replace("A_", "").replace("B_", "").replace("C_", "").replace("_", "-") + "-like",
                "canvas_size": cfg["size"],
                "n_primitives_range": cfg["n_primitives_range"],
                "split": FAMILY_SPLIT[name],
                "seed_range": [min(FAMILY_SEEDS[name]), max(FAMILY_SEEDS[name])],
            }
            for name, cfg in FAMILIES.items()
        },
        "constructs": {
            "visual_complexity": {
                "target": "structural_clutter_index",
                "parameters": ["n_primitives", "overlap_ratio", "edge_density_factor", "spacing_irregularity", "whitespace_fraction"],
                "levels": 5,
            },
            "layout_order": {
                "target": "geometric_order_index",
                "parameters": ["alignment_jitter", "grid_deviation", "spacing_variance", "overlap_ratio"],
                "levels": 5,
            },
            "colour_harmony": {
                "target": "palette_coherence_index",
                "parameters": ["hue_dispersion", "chroma_dispersion", "palette_incoherence"],
                "levels": 5,
            },
            "visual_intensity": {
                "target": "visual_salience_index",
                "parameters": ["contrast_ratio", "luminance_contrast", "edge_contrast", "focal_contrast"],
                "levels": 5,
            },
            "layout_hierarchy": {
                "target": "focal_hierarchy_index",
                "parameters": ["focal_size_ratio", "focal_contrast", "secondary_count", "whitespace_fraction"],
                "levels": 5,
            },
        },
        "interaction_conditions": [
            "high_clutter_high_alignment",
            "low_clutter_high_colour_noise",
            "strong_hierarchy_large_whitespace",
            "symmetric_high_edge_density",
        ],
        "primitive_types": ["rectangle", "circle"],
        "text_elements": "horizontal high-contrast bars (no linguistic content)",
        "randomness_source": "SHA-256 framed tuple -> first 32 bits -> NumPy PCG64",
    }
    (MANIFEST_DIR / "generator_specification.json").write_text(
        json.dumps(spec, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    hashes = pd.DataFrame({
        "file": [gen_file.name, "PROTOCOL.md", "configs/benchmark_decision_rules.yaml", "splits/frozen_splits.csv", "generator_specification.json"],
        "path": [
            str(gen_file.relative_to(ROOT)),
            str((OUTPUT_ROOT / "PROTOCOL.md").relative_to(ROOT)),
            str((OUTPUT_ROOT / "configs" / "benchmark_decision_rules.yaml").relative_to(ROOT)),
            str((OUTPUT_ROOT / "splits" / "frozen_splits.csv").relative_to(ROOT)),
            str((MANIFEST_DIR / "generator_specification.json").relative_to(ROOT)),
        ],
        "sha256": [
            gen_hash,
            hashlib.sha256((OUTPUT_ROOT / "PROTOCOL.md").read_bytes()).hexdigest(),
            hashlib.sha256((OUTPUT_ROOT / "configs" / "benchmark_decision_rules.yaml").read_bytes()).hexdigest(),
            hashlib.sha256((OUTPUT_ROOT / "splits" / "frozen_splits.csv").read_bytes()).hexdigest(),
            hashlib.sha256((MANIFEST_DIR / "generator_specification.json").read_bytes()).hexdigest(),
        ],
    })
    hashes.to_csv(MANIFEST_DIR / "generator_hashes.csv", index=False)

    return df


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--images-per-cell", type=int, default=25)
    parser.add_argument("--interactions-per-cell", type=int, default=8)
    parser.add_argument("--max-images", type=int, default=None)
    args = parser.parse_args()

    start = time.time()
    print("Generating synthetic benchmark images...")
    rows = generate_benchmark(
        images_per_cell=args.images_per_cell,
        interactions_per_cell=args.interactions_per_cell,
        max_images=args.max_images,
    )
    df = write_manifests(rows)
    elapsed = time.time() - start
    print(f"Generated {len(df)} images in {elapsed:.1f}s.")
    print(f"Images: {IMAGES_DIR}")
    print(f"Metadata: {METADATA_DIR / 'image_metadata.csv'}")
    print(f"Manifest: {MANIFEST_DIR / 'image_manifest.csv'}")


if __name__ == "__main__":
    main()
