#!/usr/bin/env python3
"""
Step 9 — Benign-transformation stability analysis.

Generates transformed variants for the D-stability image set, computes proxy
scores for each variant using the frozen standardization from the main trace,
and reports stability statistics per transformation.

Outputs:
    results/no_human/data/stability_variants_manifest.csv
    results/no_human/tables/stability_results.csv
"""

import hashlib
import json
import os
import time
import warnings
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import yaml
from scipy.stats import spearmanr

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[2]
STABILITY_CONFIG_PATH = ROOT / "configs" / "no_human_stability_transforms.yaml"
CONFIG_PATH = Path(os.environ.get("NO_HUMAN_CONFIG_PATH", ROOT / "configs" / "no_human_design.yaml"))
OUTPUT_ROOT = Path(os.environ.get("NO_HUMAN_OUTPUT_ROOT", ROOT / "results" / "no_human"))
TRACE_PATH = OUTPUT_ROOT / "data" / "per_image_metric_trace.csv"
SPLITS_PATH = OUTPUT_ROOT / "quality_control" / "canonical_splits.csv"
VARIANT_DIR = OUTPUT_ROOT / "data" / "stability_variants"

SEED = 42
RNG = np.random.default_rng(SEED)


def load_config():
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def get_image_manifest_path():
    cfg = load_config()
    return ROOT / cfg.get("paths", {}).get("image_manifest", "data/processed/no_human_image_manifest.csv")

METRICS = [
    "visual_complexity_proxy",
    "layout_order_proxy",
    "colour_harmony_proxy",
    "visual_intensity_proxy",
    "layout_hierarchy_proxy",
]


def load_frozen_stats():
    trace = pd.read_csv(TRACE_PATH)
    stats_json = trace["standardization_stats"].iloc[0]
    stats = json.loads(stats_json)
    return stats, trace


def get_stability_images():
    splits = pd.read_csv(SPLITS_PATH)
    stability = splits[splits["role"] == "D_stability"][["image_id", "category"]].copy()
    manifest = pd.read_csv(get_image_manifest_path())
    for col in ["relative_path", "standardized_path", "image_path"]:
        if col in manifest.columns:
            manifest = manifest.rename(columns={col: "standardized_path"})
            break
    return stability.merge(manifest[["image_id", "standardized_path"]], on="image_id", how="left")


def apply_transform(img_bgr, transform_type, params):
    h, w = img_bgr.shape[:2]
    if transform_type == "jpeg_compression":
        quality = params["quality"]
        _, buf = cv2.imencode(".jpg", img_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
        return cv2.imdecode(buf, cv2.IMREAD_COLOR)

    if transform_type == "brightness_shift":
        factor = params["factor"]
        hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV).astype(np.float32)
        hsv[:, :, 2] = np.clip(hsv[:, :, 2] * factor, 0, 255)
        return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)

    if transform_type == "contrast_shift":
        factor = params["factor"]
        gray_mean = float(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY).mean())
        adjusted = (img_bgr.astype(np.float32) - gray_mean) * factor + gray_mean
        return np.clip(adjusted, 0, 255).astype(np.uint8)

    if transform_type == "resize":
        scale = params["scale"]
        new_w, new_h = int(w * scale), int(h * scale)
        resized = cv2.resize(img_bgr, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        return cv2.resize(resized, (w, h), interpolation=cv2.INTER_LINEAR)

    if transform_type == "gaussian_noise":
        std = params["std"] * 255.0
        noise = RNG.normal(0, std, img_bgr.shape).astype(np.float32)
        noisy = np.clip(img_bgr.astype(np.float32) + noise, 0, 255)
        return noisy.astype(np.uint8)

    raise ValueError(f"Unknown transform type: {transform_type}")


def extract_features_for_image(img_path, image_id, category):
    """Import and call the frozen traditional feature extractor for one image."""
    import sys
    sys.path.insert(0, str(ROOT / "src" / "features"))
    from traditional_features import process_single_image
    result = process_single_image((str(img_path), image_id, category))
    return result


def standardize_and_score(feat_row, stats, percentile_bounds):
    """Given a feature dict, return proxy scores using frozen stats."""
    z = {}
    for col, mean in stats["mean"].items():
        std = stats["std"][col]
        val = feat_row.get(col, np.nan)
        z[col] = (val - mean) / std if std > 0 else 0.0
        z[col] = np.clip(z[col], -4, 4)
        if np.isnan(z[col]):
            z[col] = stats["median"][col]

    scores = {}
    metric_specs = {
        "visual_complexity_proxy": {
            "weights": [0.20, 0.15, 0.25, 0.15, 0.10, 0.05, -0.05, -0.05],
            "features": ["num_dominant_colors", "color_entropy", "edge_density",
                         "edge_orientation_entropy", "gradient_energy",
                         "text_block_count", "whitespace_ratio", "symmetry"],
        },
        "layout_order_proxy": {
            "weights": [0.25, 0.20, 0.15, -0.25, -0.10, -0.05],
            "features": ["symmetry", "whitespace_ratio", "rule_of_thirds",
                         "edge_density", "text_block_count", "lightness_contrast"],
        },
        "colour_harmony_proxy": {
            "weights": [0.30, 0.25, 0.20, -0.15, -0.10],
            "features": ["color_harmony", "warm_color_ratio", "saturation_std",
                         "hue_contrast", "num_dominant_colors"],
        },
        "visual_intensity_proxy": {
            "weights": [0.35, 0.25, 0.20, 0.10, 0.10],
            "features": ["saturation_mean", "lightness_contrast", "gradient_energy",
                         "warm_color_ratio", "saturation_std"],
        },
        "layout_hierarchy_proxy": {
            "weights": [0.25, 0.20, 0.15, 0.15, 0.10, 0.10, -0.05],
            "features": ["edge_density", "saliency_std", "fg_bg_ratio",
                         "text_block_count", "has_text", "saliency_mean", "whitespace_ratio"],
        },
    }

    for metric, spec in metric_specs.items():
        raw = sum(z[f] * w for f, w in zip(spec["features"], spec["weights"]))
        bounds = percentile_bounds[metric]
        score = 100 * (raw - bounds["p0.5"]) / (bounds["p99.5"] - bounds["p0.5"] + 1e-10)
        scores[metric] = float(np.clip(score, 0, 100))
    return scores


def process_variant(args):
    image_id, category, img_path, transform_name, level_id, params, out_dir = args
    try:
        img = cv2.imread(str(img_path))
        if img is None:
            return None
        variant_img = apply_transform(img, transform_name, params)
        out_path = out_dir / f"{image_id}_{transform_name}_{level_id}.jpg"
        cv2.imwrite(str(out_path), variant_img)
        feat = extract_features_for_image(out_path, f"{image_id}_{transform_name}_{level_id}", category)
        if feat is None:
            return None
        feat["original_image_id"] = image_id
        feat["transform"] = transform_name
        feat["level_id"] = level_id
        feat["variant_path"] = str(out_path.relative_to(ROOT)) if out_path.is_absolute() else str(out_path)
        # Compute SHA-256 of variant
        with open(out_path, "rb") as f:
            feat["variant_sha256"] = hashlib.sha256(f.read()).hexdigest()
        return feat
    except Exception as e:
        print(f"Error processing {image_id} {transform_name} {level_id}: {e}")
        return None


def main():
    start = time.time()
    config = yaml.safe_load(STABILITY_CONFIG_PATH.read_text(encoding="utf-8"))
    stats, trace = load_frozen_stats()
    percentile_bounds = json.loads(trace["percentile_bounds"].iloc[0])
    originals = trace.set_index("image_id")[METRICS].to_dict("index")

    stability_df = get_stability_images()
    print(f"D-stability set: {len(stability_df)} images")

    VARIANT_DIR.mkdir(parents=True, exist_ok=True)

    tasks = []
    for _, row in stability_df.iterrows():
        img_path = ROOT / row["standardized_path"]
        for transform_name, tspec in config["transformations"].items():
            for level in tspec["params"]:
                tasks.append((
                    row["image_id"], row["category"], img_path,
                    transform_name, level["level_id"], level, VARIANT_DIR,
                ))

    print(f"Generating {len(tasks)} variants...")
    from multiprocessing import Pool
    results = []
    with Pool(processes=4) as pool:
        for res in pool.imap_unordered(process_variant, tasks):
            if res is not None:
                results.append(res)

    print(f"Extracted features for {len(results)} variants")

    # Compute scores
    manifest_rows = []
    score_rows = []
    for res in results:
        oid = res["original_image_id"]
        transform = res["transform"]
        level_id = res["level_id"]
        scores = standardize_and_score(res, stats, percentile_bounds)
        manifest_rows.append({
            "original_image_id": oid,
            "transform": transform,
            "level_id": level_id,
            "variant_image_id": res["image_id"],
            "variant_path": res["variant_path"],
            "variant_sha256": res["variant_sha256"],
        })
        row = {"original_image_id": oid, "transform": transform, "level_id": level_id}
        row.update(scores)
        # Add deltas
        orig = originals[oid]
        for m in METRICS:
            row[f"{m}_delta"] = scores[m] - orig[m]
        score_rows.append(row)

    manifest_df = pd.DataFrame(manifest_rows)
    scores_df = pd.DataFrame(score_rows)

    manifest_df.to_csv(OUTPUT_ROOT / "data" / "stability_variants_manifest.csv", index=False)
    scores_df.to_csv(OUTPUT_ROOT / "data" / "stability_variant_scores.csv", index=False)

    # Aggregate stability statistics per transform x metric
    agg_rows = []
    for (transform, level_id), group in scores_df.groupby(["transform", "level_id"]):
        n = len(group)
        for m in METRICS:
            deltas = group[f"{m}_delta"].values
            orig_scores = [originals[oid][m] for oid in group["original_image_id"]]
            variant_scores = group[m].values
            rho, _ = spearmanr(orig_scores, variant_scores)
            within5 = np.mean(np.abs(deltas) <= 5.0)
            within10 = np.mean(np.abs(deltas) <= 10.0)
            agg_rows.append({
                "transform": transform,
                "level_id": level_id,
                "metric": m,
                "n": n,
                "mean_absolute_difference": float(np.mean(np.abs(deltas))),
                "median_absolute_difference": float(np.median(np.abs(deltas))),
                "rank_correlation_spearman": float(rho) if not np.isnan(rho) else 0.0,
                "within_5_percent_rate": float(within5),
                "within_10_percent_rate": float(within10),
            })
    stability_results = pd.DataFrame(agg_rows)
    stability_results.to_csv(OUTPUT_ROOT / "tables" / "stability_results.csv", index=False)

    elapsed = time.time() - start
    print(f"Done in {elapsed:.1f}s.")
    print(stability_results.groupby(["transform", "level_id"])["mean_absolute_difference"].mean())


if __name__ == "__main__":
    main()
