#!/usr/bin/env python3
"""Protocol-v4 operational descriptors and internal OpenCV holdout."""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import sys
from multiprocessing import Pool
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
from scipy.ndimage import gaussian_filter
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.features.traditional_features import process_single_image
from synthetic_benchmark.run_post_review_protocol_v3 import proxy_scores, ridge_fit, ridge_predict


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/no_human/runs/run_20260718_protocol_v4/external_holdout_v4"
IMAGE_DIR = OUT / "images"
N_BOOT = 2000
N_NULL = 100
RNG_SEED = 26071804
SEEDS = list(range(100, 110))
TARGETS = [
    "structural_clutter_index", "geometric_order_index", "palette_coherence_index",
    "visual_salience_index", "focal_hierarchy_index",
]
TARGET_MAP = {
    "visual_complexity_proxy": "structural_clutter_index",
    "layout_order_proxy": "geometric_order_index",
    "colour_harmony_proxy": "palette_coherence_index",
    "visual_intensity_proxy": "visual_salience_index",
    "layout_hierarchy_proxy": "focal_hierarchy_index",
}
EXTERNAL_MAP = {
    "canny_edge_density": ("structural_clutter_index", 1),
    "multiscale_lab_residual_entropy": ("structural_clutter_index", 1),
    "jpeg_bytes_per_pixel_q90": ("structural_clutter_index", 1),
    "hasler_susstrunk_colourfulness": ("palette_coherence_index", -1),
    "global_luminance_cv": ("visual_salience_index", 1),
    "horizontal_mirror_similarity": ("geometric_order_index", 1),
}


def prepare_original_targets(frame):
    """Hook for later protocols that freeze target scaling on development."""
    return frame


def rho(x, y) -> float:
    value = spearmanr(np.asarray(x), np.asarray(y)).statistic
    return float(value) if np.isfinite(value) else 0.0


def file_sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def factor_rows():
    rows = []
    for a in range(5):
        for b in range(5):
            for c in range(5):
                rows.append((a, b, c, (a + b + c) % 5, (a + 2*b + 3*c) % 5))
    return rows


def hsv_colour(hue: float, saturation: float, value: float):
    pixel = np.uint8([[[hue % 180, np.clip(saturation, 0, 255), np.clip(value, 0, 255)]]])
    return tuple(int(v) for v in cv2.cvtColor(pixel, cv2.COLOR_HSV2BGR)[0, 0])


def render(seed: int, row_index: int, levels, output: Path):
    clutter, disorder, incoherence, salience, hierarchy = levels
    rng = np.random.default_rng(seed * 1000 + row_index)
    canvas = np.full((512, 512, 3), 232, dtype=np.uint8)

    # Clutter: low-contrast micro-marks only.
    for _ in range([0, 18, 42, 72, 110][clutter]):
        x, y = rng.integers(8, 504, size=2)
        angle = rng.uniform(0, 2*np.pi)
        length = rng.integers(4, 13)
        colour = int(rng.integers(185, 220))
        cv2.line(canvas, (int(x), int(y)),
                 (int(x + length*np.cos(angle)), int(y + length*np.sin(angle))),
                 (colour, colour, colour), 1, cv2.LINE_AA)

    base_hue = (seed * 17) % 180
    hue_spread = [0, 5, 16, 38, 75][incoherence]
    object_value = [215, 190, 155, 105, 45][salience]
    jitter = [0, 3, 9, 18, 32][disorder]
    positions = [(112 + col*96, 112 + row*96) for row in range(4) for col in range(4)]
    for idx, (cx, cy) in enumerate(positions):
        dx, dy = rng.normal(0, jitter, size=2)
        h = base_hue + rng.uniform(-hue_spread, hue_spread)
        colour = hsv_colour(h, 150, object_value)
        radius = 18 if idx else [18, 25, 34, 45, 58][hierarchy]
        if idx == 0 and hierarchy:
            colour = hsv_colour(h, min(255, 150 + 20*hierarchy), max(20, object_value - 12*hierarchy))
        if idx % 2:
            cv2.rectangle(canvas, (int(cx+dx-radius), int(cy+dy-radius)),
                          (int(cx+dx+radius), int(cy+dy+radius)), colour, -1, cv2.LINE_AA)
        else:
            cv2.circle(canvas, (int(cx+dx), int(cy+dy)), radius, colour, -1, cv2.LINE_AA)
    cv2.imwrite(str(output), canvas, [cv2.IMWRITE_PNG_COMPRESSION, 9])


def generate_external(force=False):
    IMAGE_DIR.mkdir(parents=True, exist_ok=True)
    target_rows, image_rows = [], []
    design = factor_rows()
    for seed in SEEDS:
        for idx, levels in enumerate(design):
            image_id = f"X_opencv_S{seed}_R{idx:03d}"
            path = IMAGE_DIR / f"{image_id}.png"
            if force or not path.exists():
                render(seed, idx, levels, path)
            clutter, disorder, incoherence, salience, hierarchy = levels
            target_rows.append({
                "image_id": image_id, "family": "X_opencv_orthogonal", "seed": seed,
                "row_index": idx, "clutter_level": clutter, "disorder_level": disorder,
                "incoherence_level": incoherence, "salience_level": salience,
                "hierarchy_level": hierarchy, "structural_clutter_index": 25*clutter,
                "geometric_order_index": 100-25*disorder,
                "palette_coherence_index": 100-25*incoherence,
                "visual_salience_index": 25*salience,
                "focal_hierarchy_index": 25*hierarchy,
                "image_path": path.relative_to(ROOT).as_posix(),
            })
            image_rows.append({"relative_path": path.relative_to(OUT).as_posix(),
                               "sha256": file_sha(path), "bytes": path.stat().st_size})
    targets = pd.DataFrame(target_rows)
    targets.to_csv(OUT / "external_target_metadata.csv", index=False)
    pd.DataFrame(image_rows).to_csv(OUT / "external_image_manifest.csv", index=False)
    return targets


def operational_descriptors_worker(item):
    image_id, path = item
    bgr = cv2.imread(str(path))
    bgr = cv2.resize(bgr, (512, 512))
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).astype(float)
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY).astype(float)
    edges = cv2.Canny(gray.astype(np.uint8), 100, 200)
    rg = rgb[:, :, 0] - rgb[:, :, 1]
    yb = 0.5*(rgb[:, :, 0] + rgb[:, :, 1]) - rgb[:, :, 2]
    colourfulness = np.hypot(rg.std(ddof=1), yb.std(ddof=1)) + 0.3*np.hypot(rg.mean(), yb.mean())
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB).astype(float)
    channel_entropies = []
    for channel in range(3):
        total = 0.0
        plane = lab[:, :, channel]
        for sigma in (1.0, 2.0, 4.0):
            band = plane - gaussian_filter(plane, sigma=sigma)
            hist, _ = np.histogram(band, bins=64, range=(-128, 128))
            prob = hist[hist > 0] / hist.sum()
            total += float(-(prob*np.log2(prob)).sum())
        channel_entropies.append(total)
    subband = 0.84*channel_entropies[0] + 0.08*channel_entropies[1] + 0.08*channel_entropies[2]
    ok, encoded = cv2.imencode(".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, 90])
    if not ok:
        raise RuntimeError(f"JPEG encoding failed: {path}")
    return {
        "image_id": image_id,
        "canny_edge_density": float((edges > 0).mean()),
        "multiscale_lab_residual_entropy": subband,
        "jpeg_bytes_per_pixel_q90": float(len(encoded) / gray.size),
        "hasler_susstrunk_colourfulness": float(colourfulness),
        "global_luminance_cv": float(gray.std(ddof=1) / max(gray.mean(), 1e-9)),
        "horizontal_mirror_similarity": float(1 - np.abs(gray - np.fliplr(gray)).mean()/255),
    }


def operational_descriptors(items):
    with Pool(processes=8) as pool:
        return pd.DataFrame(pool.map(operational_descriptors_worker, items, chunksize=20))


def cluster_interval(frame, score, target):
    observed = rho(frame[score], frame[target])
    seeds = np.array(sorted(frame.seed.unique()))
    by_seed = {seed: frame.index[frame.seed.eq(seed)].to_numpy() for seed in seeds}
    rng = np.random.default_rng(RNG_SEED)
    boot = []
    for _ in range(N_BOOT):
        selected = rng.choice(seeds, size=len(seeds), replace=True)
        idx = np.concatenate([by_seed[seed] for seed in selected])
        boot.append(rho(frame.loc[idx, score], frame.loc[idx, target]))
    lo, hi = np.quantile(boot, [0.025, 0.975])
    return observed, float(lo), float(hi)


def metric_results(frame, generator):
    rows, per_seed = [], []
    for metric, (target, direction) in EXTERNAL_MAP.items():
        observed, lo, hi = cluster_interval(frame, metric, target)
        rows.append({"generator": generator, "metric": metric, "target": target,
                     "expected_direction": direction, "n_images": len(frame),
                     "n_seed_clusters": frame.seed.nunique(), "rho": observed,
                     "cluster_ci_lower": lo, "cluster_ci_upper": hi,
                     "responsive_expected_sign": bool(lo > 0 if direction > 0 else hi < 0)})
        for seed, group in frame.groupby("seed"):
            per_seed.append({"generator": generator, "seed": seed, "metric": metric,
                             "target": target, "rho": rho(group[metric], group[target])})
    return rows, per_seed


def composite_results(frame, scores):
    data = pd.concat([frame.reset_index(drop=True), scores.reset_index(drop=True)], axis=1)
    rows, per_seed = [], []
    for proxy, target in TARGET_MAP.items():
        observed, lo, hi = cluster_interval(data, proxy, target)
        non = {other: abs(rho(data[proxy], data[other])) for other in TARGETS if other != target}
        max_name = max(non, key=non.get)
        rows.append({"proxy": proxy, "target": target, "n_images": len(data),
                     "n_seed_clusters": data.seed.nunique(), "rho": observed,
                     "cluster_ci_lower": lo, "cluster_ci_upper": hi,
                     "max_abs_non_target_rho": non[max_name], "max_abs_non_target_name": max_name,
                     "selectivity_margin": abs(observed)-non[max_name]})
        for seed, group in data.groupby("seed"):
            per_seed.append({"seed": seed, "proxy": proxy, "target": target,
                             "rho": rho(group[proxy], group[target])})
    return pd.DataFrame(rows), pd.DataFrame(per_seed)


def fit_controls(original, original_features, external, external_features):
    feature_columns = [c for c in original_features.select_dtypes(include=[np.number]).columns
                       if c in external_features.columns]
    base = original.merge(original_features[["image_id"] + feature_columns], on="image_id", validate="one_to_one")
    ext = external.merge(external_features[["image_id"] + feature_columns], on="image_id", validate="one_to_one")
    dev, val, test = base.split.eq("G-dev"), base.split.eq("G-validation"), base.split.eq("G-test")
    alphas = [0.1, 1.0, 10.0, 100.0]
    positive, null_rows = [], []
    rng = np.random.default_rng(RNG_SEED)
    for target in TARGETS:
        candidates = []
        for alpha in alphas:
            model = ridge_fit(base.loc[dev, feature_columns].to_numpy(float), base.loc[dev, target].to_numpy(float), alpha)
            value = rho(ridge_predict(model, base.loc[val, feature_columns].to_numpy(float)), base.loc[val, target])
            candidates.append((value, alpha))
        validation_rho, alpha = max(candidates)
        train = dev | val
        model = ridge_fit(base.loc[train, feature_columns].to_numpy(float), base.loc[train, target].to_numpy(float), alpha)
        gtest_rho = rho(ridge_predict(model, base.loc[test, feature_columns].to_numpy(float)), base.loc[test, target])
        external_rho = rho(ridge_predict(model, ext[feature_columns].to_numpy(float)), ext[target])
        null_g, null_x = [], []
        dev_frame = base.loc[dev].copy()
        for permutation in range(N_NULL):
            shuffled = dev_frame[target].to_numpy(float).copy()
            for _, block in dev_frame.groupby(["seed", "construct"]):
                levels = np.array(sorted(block["level"].unique()))
                source_levels = rng.permutation(levels)
                for destination_level, source_level in zip(levels, source_levels):
                    destination_index = block.index[block["level"].eq(destination_level)]
                    source_index = block.index[block["level"].eq(source_level)]
                    destination_positions = dev_frame.index.get_indexer(destination_index)
                    source_positions = dev_frame.index.get_indexer(source_index)
                    shuffled[destination_positions] = rng.permutation(dev_frame[target].to_numpy(float)[source_positions])
            null_model = ridge_fit(dev_frame[feature_columns].to_numpy(float), shuffled, alpha)
            null_g.append(rho(ridge_predict(null_model, base.loc[test, feature_columns].to_numpy(float)), base.loc[test, target]))
            null_x.append(rho(ridge_predict(null_model, ext[feature_columns].to_numpy(float)), ext[target]))
        g975, x975 = np.quantile(np.abs(null_g), .975), np.quantile(np.abs(null_x), .975)
        positive.append({
            "target": target,
            "n_features": len(feature_columns),
            "selected_alpha": alpha,
            "validation_rho": validation_rho,
            "gtest_rho": gtest_rho,
            "internal_opencv_rho": external_rho,
            "gtest_structure_reference_abs_q975": g975,
            "internal_opencv_structure_reference_abs_q975": x975,
            # These two flags are descriptive diagnostics only.  The
            # structure-preserving permutations are not a zero-effect null and
            # therefore are not used to define a formal pass/fail decision.
            "both_domains_rho_ge_060_descriptive": bool(
                gtest_rho >= .60 and external_rho >= .60
            ),
            "exceeds_structure_reference_both_descriptive": bool(
                abs(gtest_rho) > g975 and abs(external_rho) > x975
            ),
        })
        for domain, values in [("G-test", null_g), ("internal_opencv", null_x)]:
            null_rows.append({"target": target, "domain": domain, "n_permutations": N_NULL,
                              "analysis_role": "structure_sensitivity_only_not_a_decision_null",
                              "restriction": "permute_level_blocks_within_development_seed_and_construct",
                              "mean_rho": np.mean(values), "rho_q025": np.quantile(values, .025),
                              "rho_q975": np.quantile(values, .975), "abs_rho_q975": np.quantile(np.abs(values), .975),
                              "max_abs_rho": np.max(np.abs(values))})
    return pd.DataFrame(positive), pd.DataFrame(null_rows)


def task_fingerprint(tasks):
    """Bind a cache to the ordered image identifiers and current image bytes."""
    digest = hashlib.sha256()
    for image_id, path in sorted(tasks, key=lambda item: item[0]):
        digest.update(image_id.encode("utf-8"))
        digest.update(b"\0")
        digest.update(file_sha(path).encode("ascii"))
        digest.update(b"\n")
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--force-regenerate",
        action="store_true",
        help="rerender all holdout images and recompute every feature/descriptor cache",
    )
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    print(
        "Protocol v4.2 execution mode: "
        + ("forced rerender and cache recomputation" if args.force_regenerate else "hash-bound cache reuse when valid")
    )
    if args.force_regenerate:
        for stale_name in (
            "published_metrics_original_gtest.csv",
            "published_metrics_external_generator.csv",
        ):
            stale_path = OUT / stale_name
            if stale_path.exists():
                stale_path.unlink()
    external = generate_external(force=args.force_regenerate)
    tasks = [(row.image_id, ROOT / row.image_path) for row in external.itertuples()]
    external_fingerprint = task_fingerprint(tasks)
    print(f"Internal OpenCV image-set fingerprint: {external_fingerprint}")
    binding_path = OUT / "cache_bindings.json"
    old_bindings = {}
    if binding_path.exists() and not args.force_regenerate:
        old_bindings = json.loads(binding_path.read_text(encoding="utf-8"))
    feature_cache = OUT / "external_traditional_features.csv"
    external_cache_valid = old_bindings.get("external_image_fingerprint") == external_fingerprint
    if feature_cache.exists() and external_cache_valid:
        external_features = pd.read_csv(feature_cache).sort_values("image_id")
    else:
        extracted = []
        with Pool(processes=8) as pool:
            for result in pool.map(process_single_image, [(str(path), image_id, "internal_opencv") for image_id, path in tasks], chunksize=10):
                if result is not None:
                    extracted.append(result)
        external_features = pd.DataFrame(extracted).sort_values("image_id")
    if len(external_features) != 1250:
        raise RuntimeError(f"Expected 1,250 external feature rows, found {len(external_features)}")
    external_features.to_csv(OUT / "external_traditional_features.csv", index=False)

    original = prepare_original_targets(
        pd.read_csv(ROOT / "synthetic_benchmark/metadata/target_metadata.csv")
    )
    original_features = pd.read_csv(ROOT / "synthetic_benchmark/metadata/synthetic_traditional_features.csv")
    original_g = original.loc[original.split.eq("G-test")].copy()
    original_tasks = [(row.image_id, ROOT / row.image_path) for row in original_g.itertuples()]
    original_fingerprint = task_fingerprint(original_tasks)
    print(f"Original G-test image-set fingerprint: {original_fingerprint}")
    original_cache_valid = old_bindings.get("original_gtest_image_fingerprint") == original_fingerprint
    original_metric_cache = OUT / "operational_descriptors_original_gtest.csv"
    external_metric_cache = OUT / "operational_descriptors_external_generator.csv"
    if original_metric_cache.exists() and original_cache_valid:
        metric_original = pd.read_csv(original_metric_cache)
    else:
        metric_original = operational_descriptors(original_tasks)
        metric_original.to_csv(original_metric_cache, index=False)
    if external_metric_cache.exists() and external_cache_valid:
        metric_external = pd.read_csv(external_metric_cache)
    else:
        metric_external = operational_descriptors(tasks)
        metric_external.to_csv(external_metric_cache, index=False)
    original_eval = original_g.merge(metric_original, on="image_id", validate="one_to_one").reset_index(drop=True)
    external_eval = external.merge(metric_external, on="image_id", validate="one_to_one").reset_index(drop=True)
    metric_rows, per_seed_rows = [], []
    for frame, label in [(original_eval, "original_G-test"), (external_eval, "internal_opencv")]:
        a, b = metric_results(frame, label)
        metric_rows.extend(a); per_seed_rows.extend(b)
    pd.DataFrame(metric_rows).to_csv(OUT / "external_metric_correlations.csv", index=False)
    pd.DataFrame(per_seed_rows).to_csv(OUT / "per_seed_external_metrics.csv", index=False)

    scores = proxy_scores(external_features.reset_index(drop=True))
    composites, per_seed_composites = composite_results(external_eval, scores)
    composites.to_csv(OUT / "composite_external_holdout.csv", index=False)
    per_seed_composites.to_csv(OUT / "per_seed_composite_external.csv", index=False)

    external_eval[TARGETS].corr(method="spearman").to_csv(OUT / "target_correlation_external.csv")
    positive, nulls = fit_controls(original, original_features, external, external_features)
    positive.to_csv(OUT / "positive_control_cross_generator.csv", index=False)
    nulls.to_csv(OUT / "restricted_null_controls.csv", index=False)
    labels = ["Clutter", "Order", "Palette", "Salience", "Hierarchy"]
    x = np.arange(len(positive))
    width = 0.36
    fig, ax = plt.subplots(figsize=(9.0, 5.6))
    ax.bar(x - width / 2, positive["gtest_rho"], width, label="Pillow G-test")
    ax.bar(x + width / 2, positive["internal_opencv_rho"], width, label="Internal OpenCV")
    ax.axhline(0.60, color="black", ls="--", lw=1, label="Local rho = 0.60 reference")
    ax.set_xticks(x, labels, rotation=25, ha="right")
    ax.set_ylabel("Spearman rho")
    ax.set_ylim(0, 0.95)
    ax.set_title("Pixel-side ridge probes across two internal renderers")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(
        OUT / "ridge_probe_cross_renderer.png",
        dpi=300,
        metadata={"Software": "Matplotlib deterministic protocol-v4.2 renderer"},
    )
    plt.close(fig)
    pd.DataFrame([
        {"domain": "original_G-test", "n_images": len(original_eval), "n_seed_clusters": original_eval.seed.nunique()},
        {"domain": "internal_opencv", "n_images": len(external_eval), "n_seed_clusters": external_eval.seed.nunique()},
    ]).to_csv(OUT / "analysis_units.csv", index=False)

    bindings = {
        "external_image_fingerprint": external_fingerprint,
        "original_gtest_image_fingerprint": original_fingerprint,
        "binding_algorithm": "SHA-256 over sorted image_id, NUL, image_sha256 records",
    }
    binding_path.write_text(json.dumps(bindings, indent=2), encoding="utf-8")
    print("Cache bindings written after feature and operational-descriptor computation.")

    metadata = {"protocol": "4.2-failure-mode-case-study", "run_id": "run_20260718_protocol_v4",
                "rng_seed": RNG_SEED, "n_boot": N_BOOT, "n_restricted_null": N_NULL,
                "external_generator": "separately_implemented_internal_opencv_holdout_v1",
                "force_regenerate": bool(args.force_regenerate),
                "cache_binding_file": "cache_bindings.json",
                "restricted_null_role": "structure_sensitivity_only_no_formal_gate",
                "bootstrap_p_values": False, "bh_q_values": False}
    (OUT / "run_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    artifact_rows = []
    for path in sorted(OUT.iterdir()):
        if path.is_file() and path.name != "artifact_hashes.csv":
            artifact_rows.append({"file": path.name, "sha256": file_sha(path), "bytes": path.stat().st_size})
    pd.DataFrame(artifact_rows).to_csv(OUT / "artifact_hashes.csv", index=False)
    print(f"Protocol v4 complete: {OUT}")


if __name__ == "__main__":
    main()
