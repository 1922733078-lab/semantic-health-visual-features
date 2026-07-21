#!/usr/bin/env python3
"""
Step 10 — Controlled directional-sensitivity analysis.

Applies predefined structural manipulations to the D-stability image set,
computes proxy scores using frozen standardization, and reports whether
scores move in the expected direction.

Outputs:
    results/no_human/data/sensitivity_variants_manifest.csv
    results/no_human/tables/directional_sensitivity_results.csv
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
CONFIG_PATH = Path(os.environ.get("NO_HUMAN_CONFIG_PATH", ROOT / "configs" / "no_human_design.yaml"))
OUTPUT_ROOT = Path(os.environ.get("NO_HUMAN_OUTPUT_ROOT", ROOT / "results" / "no_human"))
TRACE_PATH = OUTPUT_ROOT / "data" / "per_image_metric_trace.csv"
SPLITS_PATH = OUTPUT_ROOT / "quality_control" / "canonical_splits.csv"
VARIANT_DIR = OUTPUT_ROOT / "data" / "sensitivity_variants"

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

MANIPULATIONS = {
    "add_shapes": {
        "expected": {"visual_complexity_proxy": 1},
        "params": {
            "mild": {"n": 3, "size": 0.05},
            "moderate": {"n": 6, "size": 0.08},
            "strong": {"n": 10, "size": 0.12},
        },
    },
    "sharpen": {
        "expected": {"visual_complexity_proxy": 1, "visual_intensity_proxy": 1},
        "params": {
            "mild": {"amount": 0.5},
            "moderate": {"amount": 1.0},
            "strong": {"amount": 1.5},
        },
    },
    "desaturate": {
        "expected": {"visual_intensity_proxy": -1},
        "params": {
            "mild": {"factor": 0.90},
            "moderate": {"factor": 0.70},
            "strong": {"factor": 0.50},
        },
    },
    "saturate": {
        "expected": {"visual_intensity_proxy": 1},
        "params": {
            "mild": {"factor": 1.10},
            "moderate": {"factor": 1.30},
            "strong": {"factor": 1.50},
        },
    },
    "add_text_bars": {
        "expected": {"layout_hierarchy_proxy": 1, "visual_complexity_proxy": 1},
        "params": {
            "mild": {"n": 2, "thickness": 2},
            "moderate": {"n": 4, "thickness": 3},
            "strong": {"n": 6, "thickness": 4},
        },
    },
    "reduce_symmetry": {
        "expected": {"layout_order_proxy": -1},
        "params": {
            "mild": {"offset": 0.02},
            "moderate": {"offset": 0.05},
            "strong": {"offset": 0.10},
        },
    },
    "add_colour_noise": {
        "expected": {"colour_harmony_proxy": -1},
        "params": {
            "mild": {"std": 0.005},
            "moderate": {"std": 0.01},
            "strong": {"std": 0.02},
        },
    },
    "reduce_whitespace": {
        "expected": {"layout_order_proxy": -1, "visual_complexity_proxy": 1},
        "params": {
            "mild": {"fill_ratio": 0.10},
            "moderate": {"fill_ratio": 0.20},
            "strong": {"fill_ratio": 0.30},
        },
    },
}


def load_frozen_stats():
    trace = pd.read_csv(TRACE_PATH)
    stats_json = trace["standardization_stats"].iloc[0]
    stats = json.loads(stats_json)
    percentile_bounds = json.loads(trace["percentile_bounds"].iloc[0])
    return stats, percentile_bounds, trace


def get_sensitivity_images(n_sample=50):
    splits = pd.read_csv(SPLITS_PATH)
    dev = splits[splits["role"] == "D_development"][["image_id", "category"]].copy()
    manifest = pd.read_csv(get_image_manifest_path())
    for col in ["relative_path", "standardized_path", "image_path"]:
        if col in manifest.columns:
            manifest = manifest.rename(columns={col: "standardized_path"})
            break
    dev = dev.merge(manifest[["image_id", "standardized_path"]], on="image_id", how="left")
    # Fixed sample for D-sensitivity
    if len(dev) > n_sample:
        dev = dev.sample(n=n_sample, random_state=SEED).reset_index(drop=True)
    return dev


def apply_manipulation(img_bgr, manipulation, severity, params):
    h, w = img_bgr.shape[:2]
    out = img_bgr.copy()

    if manipulation == "add_shapes":
        n = params["n"]
        size = int(min(h, w) * params["size"])
        for _ in range(n):
            x = RNG.integers(0, max(1, w - size))
            y = RNG.integers(0, max(1, h - size))
            color = [int(c) for c in RNG.integers(0, 255, 3)]
            cv2.rectangle(out, (x, y), (x + size, y + size), color, -1)

    elif manipulation == "sharpen":
        amount = params["amount"]
        kernel = np.array([[0, -1, 0], [-1, 5 + amount, -1], [0, -1, 0]])
        out = cv2.filter2D(out, -1, kernel)
        out = np.clip(out, 0, 255).astype(np.uint8)

    elif manipulation == "desaturate":
        factor = params["factor"]
        hsv = cv2.cvtColor(out, cv2.COLOR_BGR2HSV).astype(np.float32)
        hsv[:, :, 1] *= factor
        out = cv2.cvtColor(np.clip(hsv, 0, 255).astype(np.uint8), cv2.COLOR_HSV2BGR)

    elif manipulation == "saturate":
        factor = params["factor"]
        hsv = cv2.cvtColor(out, cv2.COLOR_BGR2HSV).astype(np.float32)
        hsv[:, :, 1] *= factor
        out = cv2.cvtColor(np.clip(hsv, 0, 255).astype(np.uint8), cv2.COLOR_HSV2BGR)

    elif manipulation == "add_text_bars":
        n = params["n"]
        thickness = params["thickness"]
        gap = h // (n + 1)
        for i in range(1, n + 1):
            y = i * gap
            cv2.line(out, (0, y), (w, y), (0, 0, 0), thickness)

    elif manipulation == "reduce_symmetry":
        offset = int(w * params["offset"])
        left = out[:, : w // 2, :]
        right = cv2.flip(left, 1)
        # Shift right half vertically to break symmetry
        shift = max(1, abs(offset))
        if offset > 0:
            right = np.roll(right, shift, axis=0)
            right[:shift, :] = 128
        else:
            right = np.roll(right, -shift, axis=0)
            right[-shift:, :] = 128
        out = np.hstack([left, right])

    elif manipulation == "add_colour_noise":
        std = params["std"] * 255.0
        noise = RNG.normal(0, std, out.shape).astype(np.float32)
        out = np.clip(out.astype(np.float32) + noise, 0, 255).astype(np.uint8)

    elif manipulation == "reduce_whitespace":
        fill_ratio = params["fill_ratio"]
        gray = cv2.cvtColor(out, cv2.COLOR_BGR2GRAY)
        edge = cv2.Canny(gray, 30, 100).astype(np.float32)
        kernel = np.ones((32, 32), np.float32) / (32 * 32)
        edge_density = cv2.filter2D(edge, -1, kernel)
        low_edge_mask = edge_density < 0.02
        fill_mask = (RNG.random(out.shape[:2]) < fill_ratio) & low_edge_mask
        fill_colour = np.array([128, 128, 128], dtype=np.uint8)
        out[fill_mask] = fill_colour

    return out


def extract_features_for_image(img_path, image_id, category):
    import sys
    sys.path.insert(0, str(ROOT / "src" / "features"))
    from traditional_features import process_single_image
    return process_single_image((str(img_path), image_id, category))


def standardize_and_score(feat_row, stats, percentile_bounds):
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
    image_id, category, img_path, manipulation, severity, params, out_dir = args
    try:
        img = cv2.imread(str(img_path))
        if img is None:
            return None
        variant_img = apply_manipulation(img, manipulation, severity, params)
        out_path = out_dir / f"{image_id}_{manipulation}_{severity}.jpg"
        cv2.imwrite(str(out_path), variant_img)
        feat = extract_features_for_image(out_path, f"{image_id}_{manipulation}_{severity}", category)
        if feat is None:
            return None
        feat["original_image_id"] = image_id
        feat["manipulation"] = manipulation
        feat["severity"] = severity
        feat["variant_path"] = str(out_path.relative_to(ROOT)) if out_path.is_absolute() else str(out_path)
        with open(out_path, "rb") as f:
            feat["variant_sha256"] = hashlib.sha256(f.read()).hexdigest()
        return feat
    except Exception as e:
        print(f"Error {image_id} {manipulation} {severity}: {e}")
        return None


def main():
    start = time.time()
    stats, percentile_bounds, trace = load_frozen_stats()
    originals = trace.set_index("image_id")[METRICS].to_dict("index")

    sensitivity_df = get_sensitivity_images(n_sample=50)
    print(f"D-sensitivity set: {len(sensitivity_df)} images")
    VARIANT_DIR.mkdir(parents=True, exist_ok=True)

    tasks = []
    for _, row in sensitivity_df.iterrows():
        img_path = ROOT / row["standardized_path"]
        for manipulation, spec in MANIPULATIONS.items():
            for severity, params in spec["params"].items():
                tasks.append((
                    row["image_id"], row["category"], img_path,
                    manipulation, severity, params, VARIANT_DIR,
                ))

    print(f"Generating {len(tasks)} variants...")
    # Single-process to avoid import/multiprocessing deadlocks in this environment
    results = []
    for i, task in enumerate(tasks):
        res = process_variant(task)
        if res is not None:
            results.append(res)
        if (i + 1) % 100 == 0:
            print(f"  processed {i + 1}/{len(tasks)} variants")

    print(f"Extracted features for {len(results)} variants")

    manifest_rows = []
    score_rows = []
    for res in results:
        oid = res["original_image_id"]
        manipulation = res["manipulation"]
        severity = res["severity"]
        scores = standardize_and_score(res, stats, percentile_bounds)
        manifest_rows.append({
            "original_image_id": oid,
            "manipulation": manipulation,
            "severity": severity,
            "variant_image_id": res["image_id"],
            "variant_path": res["variant_path"],
            "variant_sha256": res["variant_sha256"],
        })
        row = {"original_image_id": oid, "manipulation": manipulation, "severity": severity}
        row.update(scores)
        orig = originals[oid]
        for m in METRICS:
            row[f"{m}_delta"] = scores[m] - orig[m]
        score_rows.append(row)

    manifest_df = pd.DataFrame(manifest_rows)
    scores_df = pd.DataFrame(score_rows)
    manifest_df.to_csv(OUTPUT_ROOT / "data" / "sensitivity_variants_manifest.csv", index=False)
    scores_df.to_csv(OUTPUT_ROOT / "data" / "sensitivity_variant_scores.csv", index=False)

    severity_order = {"mild": 1, "moderate": 2, "strong": 3}
    scores_df["severity_ord"] = scores_df["severity"].map(severity_order)

    agg_rows = []
    for manipulation, spec in MANIPULATIONS.items():
        sub = scores_df[scores_df["manipulation"] == manipulation]
        for metric, expected_sign in spec["expected"].items():
            deltas = sub[f"{metric}_delta"].values
            prop_correct = np.mean((deltas * expected_sign) > 0)
            mean_delta = float(np.mean(deltas * expected_sign))
            # Monotonicity across severity
            rho, _ = spearmanr(sub["severity_ord"], sub[f"{metric}_delta"])
            agg_rows.append({
                "manipulation": manipulation,
                "metric": metric,
                "expected_direction": "increase" if expected_sign > 0 else "decrease",
                "n": len(sub),
                "prop_correct_direction": float(prop_correct),
                "mean_signed_delta": float(mean_delta),
                "spearman_severity_vs_delta": float(rho) if not np.isnan(rho) else 0.0,
            })
    sensitivity_results = pd.DataFrame(agg_rows)
    sensitivity_results.to_csv(OUTPUT_ROOT / "tables" / "directional_sensitivity_results.csv", index=False)

    elapsed = time.time() - start
    print(f"Done in {elapsed:.1f}s.")
    print(sensitivity_results)


if __name__ == "__main__":
    main()
