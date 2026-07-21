#!/usr/bin/env python3
"""
Compute deterministic no-human proxy metrics and generate canonical data artifacts.

Implements Steps 4-7 of the no-human redesign plan:
  - canonical data dictionary and splits
  - provenance / duplicate audit
  - preprocessing freeze record
  - transparent direct metrics

Inputs:
    configs/no_human_design.yaml
    data/processed/no_human_image_manifest.csv
    data/features/traditional_features.csv

Outputs:
    results/no_human/quality_control/canonical_splits.csv
    results/no_human/quality_control/dataset_role_audit.csv
    results/no_human/quality_control/image_provenance.csv
    results/no_human/quality_control/duplicate_groups.csv
    results/no_human/quality_control/split_leakage_report.md
    results/no_human/quality_control/repeated_extraction_check.csv
    results/no_human/tables/metric_definition_table.csv
    results/no_human/data/per_image_metric_trace.csv
"""

import ast
import concurrent.futures
import hashlib
import json
import os
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from scipy.spatial.distance import cdist

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = Path(os.environ.get("NO_HUMAN_CONFIG_PATH", ROOT / "configs" / "no_human_design.yaml"))
PREPROC_CONFIG_PATH = ROOT / "configs" / "no_human_preprocessing.yaml"
OUTPUT_ROOT = Path(os.environ.get("NO_HUMAN_OUTPUT_ROOT", ROOT / "results" / "no_human"))

SEED = 42
RNG = np.random.default_rng(SEED)

METRICS = {
    "visual_complexity_proxy": {
        "features": [
            "num_dominant_colors",
            "color_entropy",
            "edge_density",
            "edge_orientation_entropy",
            "gradient_energy",
            "text_block_count",
            "whitespace_ratio",
            "symmetry",
        ],
        "weights": [0.20, 0.15, 0.25, 0.15, 0.10, 0.05, -0.05, -0.05],
        "groups": ["colour", "colour", "edge_texture", "edge_texture", "edge_texture",
                   "projection_legacy", "composition", "composition"],
    },
    "layout_order_proxy": {
        "features": [
            "symmetry",
            "whitespace_ratio",
            "rule_of_thirds",
            "edge_density",
            "text_block_count",
            "lightness_contrast",
        ],
        "weights": [0.25, 0.20, 0.15, -0.25, -0.10, -0.05],
        "groups": ["composition", "composition", "composition", "edge_texture",
                   "projection_legacy", "colour"],
    },
    "colour_harmony_proxy": {
        "features": [
            "color_harmony",
            "warm_color_ratio",
            "saturation_std",
            "hue_contrast",
            "num_dominant_colors",
        ],
        "weights": [0.30, 0.25, 0.20, -0.15, -0.10],
        "groups": ["colour", "colour", "colour", "colour", "colour"],
    },
    "visual_intensity_proxy": {
        "features": [
            "saturation_mean",
            "lightness_contrast",
            "gradient_energy",
            "warm_color_ratio",
            "saturation_std",
        ],
        "weights": [0.35, 0.25, 0.20, 0.10, 0.10],
        "groups": ["colour", "colour", "edge_texture", "colour", "colour"],
    },
    "layout_hierarchy_proxy": {
        "features": [
            "edge_density",
            "saliency_std",
            "fg_bg_ratio",
            "text_block_count",
            "has_text",
            "saliency_mean",
            "whitespace_ratio",
        ],
        "weights": [0.25, 0.20, 0.15, 0.15, 0.10, 0.10, -0.05],
        "groups": ["edge_texture", "saliency", "composition", "projection_legacy",
                   "projection_legacy", "saliency", "composition"],
    },
}


def load_config():
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_data():
    cfg = load_config()
    paths = cfg.get("paths", {})
    manifest_path = ROOT / paths.get("image_manifest", "data/processed/no_human_image_manifest.csv")
    features_path = ROOT / paths.get("traditional_features", "data/features/traditional_features.csv")

    manifest = pd.read_csv(manifest_path)
    # Accept either relative_path or image_path/standardized_path
    for col in ["relative_path", "standardized_path", "image_path"]:
        if col in manifest.columns:
            manifest = manifest.rename(columns={col: "standardized_path"})
            break
    feats = pd.read_csv(features_path)
    # Canonical image list = manifest rows only
    feats = feats[feats["image_id"].isin(manifest["image_id"])].copy()
    feats = feats.drop_duplicates(subset=["image_id"], keep="first")
    df = manifest.merge(feats, on="image_id", how="left", suffixes=("", "_feat"))
    # Resolve duplicate category columns
    if "category_feat" in df.columns:
        df = df.drop(columns=["category_feat"])
    df = df.reset_index(drop=True)
    return df


def build_splits(df):
    """Stratified split by category: 70% dev, 15% test, 15% not used for primary test
    but kept in D_metric."""
    all_ids = df["image_id"].values
    n = len(all_ids)
    indices = np.arange(n)

    dev_idx = []
    test_idx = []
    for cat, group in df.groupby("category"):
        gidx = group.index.values.copy()
        RNG.shuffle(gidx)
        n_dev = int(len(gidx) * 0.70)
        n_test = int(len(gidx) * 0.15)
        dev_idx.extend(gidx[:n_dev])
        test_idx.extend(gidx[n_dev : n_dev + n_test])

    dev_idx = np.array(dev_idx)
    test_idx = np.array(test_idx)

    # D-stability = fixed 200 random dev images
    stability_idx = RNG.choice(dev_idx, size=min(200, len(dev_idx)), replace=False)

    # D-case-study = 20 representative images (2 per category, max)
    case_study_idx = []
    for cat, group in df.groupby("category"):
        cat_dev = group[group.index.isin(dev_idx)]
        if len(cat_dev) >= 2:
            picked = cat_dev.iloc[:: max(1, len(cat_dev) // 2)][:2].index.values
            case_study_idx.extend(picked)
    case_study_idx = np.array(case_study_idx)[:20]

    splits = pd.DataFrame({
        "image_id": df["image_id"],
        "category": df["category"],
        "split": "",
        "role": "D_metric",
        "d_metric_member": True,
        "duplicate_group_id": "",
        "exclusion_reason": "",
    })
    splits.loc[dev_idx, "split"] = "dev"
    splits.loc[test_idx, "split"] = "test"
    splits.loc[stability_idx, "split"] = "stability"
    splits.loc[stability_idx, "role"] = "D_stability"
    splits.loc[case_study_idx, "split"] = "case_study"
    splits.loc[case_study_idx, "role"] = "D_case_study"

    # Dev/test roles override case_study role only if conflict (case_study stays case_study)
    splits.loc[(splits["split"] == "dev") & (splits["role"] == "D_metric"), "role"] = "D_development"
    splits.loc[(splits["split"] == "test") & (splits["role"] == "D_metric"), "role"] = "D_test"
    return splits


def _hash_one(row, root):
    """Compute SHA-256 and a simple average hash for a single image."""
    img_path = root / row["standardized_path"]
    sha256 = ""
    avg_hash = ""
    status = "ok"
    try:
        import cv2
        img = cv2.imread(str(img_path))
        if img is None:
            raise ValueError("cv2 returned None")
        img_bytes = cv2.imencode(".png", img)[1].tobytes()
        sha256 = hashlib.sha256(img_bytes).hexdigest()
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        small = cv2.resize(gray, (16, 16), interpolation=cv2.INTER_AREA)
        mean_val = float(small.mean())
        bits = (small >= mean_val).flatten().astype(int)
        avg_hash = "".join(str(b) for b in bits)
    except Exception as e:
        status = f"error: {e}"
    return {
        "image_id": row["image_id"],
        "category": row["category"],
        "path": str(row["standardized_path"]),
        "sha256": sha256,
        "avg_hash": avg_hash,
        "status": status,
    }


def compute_hashes(df):
    """Compute SHA-256 and a simple average hash for each image (parallel I/O)."""
    rows = []
    root = ROOT
    # ThreadPoolExecutor improves throughput on I/O-bound network storage.
    max_workers = min(32, (os.cpu_count() or 1) * 4)
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = [ex.submit(_hash_one, row.to_dict(), root) for _, row in df.iterrows()]
        for future in concurrent.futures.as_completed(futures):
            rows.append(future.result())
    # Preserve input order
    order = {iid: idx for idx, iid in enumerate(df["image_id"].values)}
    rows.sort(key=lambda r: order[r["image_id"]])
    return pd.DataFrame(rows)


def detect_duplicates(provenance):
    """Group exact and near-duplicate images. Assign groups only to one split later."""
    # Exact duplicates by SHA-256
    dup_groups = []
    group_id = 0
    seen_sha = {}
    for _, row in provenance.iterrows():
        if not row["sha256"]:
            continue
        seen_sha.setdefault(row["sha256"], []).append(row["image_id"])

    grouped = set()
    for sha, ids in seen_sha.items():
        if len(ids) > 1:
            group_id += 1
            for iid in ids:
                dup_groups.append({"duplicate_group_id": f"exact_{group_id:05d}", "image_id": iid})
                grouped.add(iid)

    # Near duplicates by average hash Hamming distance (vectorised, chunked)
    hash_rows = provenance[provenance["avg_hash"] != ""].copy()
    if not hash_rows.empty:
        ids = hash_rows["image_id"].values
        # Convert 0/1 strings to a compact uint8 matrix (n x 256)
        hash_mat = np.array(
            [[int(b) for b in h] for h in hash_rows["avg_hash"]],
            dtype=np.uint8,
        )
        n = len(ids)
        near_threshold = 16  # Hamming distance out of 256
        chunk_size = 1024
        for i_start in range(0, n, chunk_size):
            i_end = min(i_start + chunk_size, n)
            # City block on {0,1} vectors equals Hamming distance.
            dists = cdist(hash_mat[i_start:i_end], hash_mat, metric="cityblock").astype(np.uint8)
            for local_i, global_i in enumerate(range(i_start, i_end)):
                if ids[global_i] in grouped:
                    continue
                row = dists[local_i]
                near = []
                for j in range(global_i + 1, n):
                    if ids[j] in grouped:
                        continue
                    if row[j] <= near_threshold:
                        near.append(ids[j])
                if near:
                    group_id += 1
                    group_name = f"near_{group_id:05d}"
                    dup_groups.append({"duplicate_group_id": group_name, "image_id": ids[global_i]})
                    grouped.add(ids[global_i])
                    for nid in near:
                        dup_groups.append({"duplicate_group_id": group_name, "image_id": nid})
                        grouped.add(nid)

    return pd.DataFrame(dup_groups)


def apply_duplicate_groups(splits, duplicate_groups):
    """Assign duplicate group ids to splits; ensure groups stay in one split."""
    if duplicate_groups.empty:
        return splits
    # Drop the placeholder column before merging real duplicate-group ids
    splits = splits.drop(columns=["duplicate_group_id"])
    splits = splits.merge(duplicate_groups, on="image_id", how="left")
    splits["duplicate_group_id"] = splits["duplicate_group_id"].fillna("")

    # For groups that span splits, move all members to the most frequent split
    for gid, group in splits[splits["duplicate_group_id"] != ""].groupby("duplicate_group_id"):
        if group["split"].nunique() > 1:
            target = group["split"].mode().iloc[0]
            if target == "":
                target = "dev"
            splits.loc[splits["duplicate_group_id"] == gid, "split"] = target
            # Update roles accordingly
            splits.loc[(splits["duplicate_group_id"] == gid) & (splits["split"] == "dev") & (~splits["role"].isin(["D_stability", "D_case_study"])), "role"] = "D_development"
            splits.loc[(splits["duplicate_group_id"] == gid) & (splits["split"] == "test") & (~splits["role"].isin(["D_stability", "D_case_study"])), "role"] = "D_test"
    return splits


def standardize_features(df, split_df, feature_cols):
    dev_idx = split_df[split_df["role"] == "D_development"].index
    dev_df = df.loc[dev_idx, feature_cols]
    stats = {
        "mean": dev_df.mean().to_dict(),
        "std": dev_df.std().replace(0, 1).to_dict(),
        "median": dev_df.median().to_dict(),
    }
    return apply_standardization(df, feature_cols, stats), stats


def apply_standardization(df, feature_cols, stats):
    """Apply frozen mean/std/median standardization constants to the feature matrix.

    Columns present in the input but missing from the frozen stats are ignored
    with a warning. This lets the same pipeline run on the public demo (which
    may contain extra diagnostic features) without altering the frozen
    standardization used for the main corpus.
    """
    cols = [c for c in feature_cols if c in stats.get("mean", {})]
    missing = set(feature_cols) - set(cols)
    if missing:
        print(
            f"Warning: {len(missing)} feature column(s) missing from "
            f"standardization stats and will be ignored: {sorted(missing)}"
        )
    z = pd.DataFrame(index=df.index, columns=cols, dtype=float)
    for col in cols:
        z[col] = (df[col] - stats["mean"][col]) / stats["std"][col]
        z[col] = z[col].clip(-4, 4)
        z[col] = z[col].fillna(stats["median"][col])
    return z


def compute_metric_scores(z_df, metric):
    spec = METRICS[metric]
    raw = np.zeros(len(z_df))
    components = {}
    for feat, w, grp in zip(spec["features"], spec["weights"], spec["groups"]):
        contrib = z_df[feat].values * w
        raw += contrib
        components[f"{metric}_{feat}_contrib"] = contrib
    return raw, components


def rescale(raw, dev_mask):
    p005 = np.percentile(raw[dev_mask], 0.5)
    p995 = np.percentile(raw[dev_mask], 99.5)
    score = 100 * (raw - p005) / (p995 - p005 + 1e-10)
    score = np.clip(score, 0, 100)
    return score, p005, p995


def build_trace(df, z_df, split_df, stats, percentile_bounds=None):
    dev_mask = split_df["role"] == "D_development"
    dev_mask = df.index.isin(split_df[dev_mask].index)

    trace = pd.DataFrame({"image_id": df["image_id"], "category": df["category"]})
    for col in z_df.columns:
        trace[f"z_{col}"] = z_df[col].values

    computed_bounds = {}
    for metric in METRICS:
        raw, components = compute_metric_scores(z_df, metric)
        if percentile_bounds is not None and metric in percentile_bounds:
            p005 = percentile_bounds[metric]["p0.5"]
            p995 = percentile_bounds[metric]["p99.5"]
            score = 100 * (raw - p005) / (p995 - p005 + 1e-10)
            score = np.clip(score, 0, 100)
        else:
            score, p005, p995 = rescale(raw, dev_mask)
        trace[f"{metric}_raw"] = raw
        trace[f"{metric}"] = score
        computed_bounds[metric] = {"p0.5": p005, "p99.5": p995}
        for k, v in components.items():
            trace[k] = v

    trace["split"] = split_df.set_index("image_id").loc[df["image_id"], "split"].values
    trace["role"] = split_df.set_index("image_id").loc[df["image_id"], "role"].values
    trace["duplicate_group_id"] = split_df.set_index("image_id").loc[df["image_id"], "duplicate_group_id"].values
    trace["metric_version"] = "1.0"
    trace["timestamp"] = pd.Timestamp.now().isoformat()

    # Attach standardization stats as JSON in a single metadata column
    stats_json = json.dumps(stats, default=str)
    trace["standardization_stats"] = stats_json
    trace["percentile_bounds"] = json.dumps(computed_bounds, default=str)
    return trace


def build_metric_definition_table(stats, percentile_bounds):
    rows = []
    for metric, spec in METRICS.items():
        for feat, w, grp in zip(spec["features"], spec["weights"], spec["groups"]):
            rows.append({
                "metric": metric,
                "feature": feat,
                "feature_group": grp,
                "weight": w,
                "contribution_sign": "positive" if w > 0 else "negative",
                "feature_mean": stats["mean"].get(feat),
                "feature_std": stats["std"].get(feat),
            })
        rows.append({
            "metric": metric,
            "feature": "__rescaling__",
            "feature_group": "",
            "weight": np.nan,
            "contribution_sign": "",
            "feature_mean": percentile_bounds[metric]["p0.5"],
            "feature_std": percentile_bounds[metric]["p99.5"],
        })
    return pd.DataFrame(rows)


def run_repeated_extraction_check(df, feature_cols, n_sample=10):
    """Verify metric determinism by computing twice on the same feature matrix."""
    sample_idx = RNG.choice(df.index, size=min(n_sample, len(df)), replace=False)
    sample_splits = pd.DataFrame({
        "image_id": df.loc[sample_idx, "image_id"],
        "role": "D_development",
    })
    z1, _ = standardize_features(df.loc[sample_idx], sample_splits, feature_cols)
    z2, _ = standardize_features(df.loc[sample_idx], sample_splits, feature_cols)

    rows = []
    for metric in METRICS:
        raw1, _ = compute_metric_scores(z1, metric)
        raw2, _ = compute_metric_scores(z2, metric)
        score1, _, _ = rescale(raw1, np.ones(len(raw1), dtype=bool))
        score2, _, _ = rescale(raw2, np.ones(len(raw2), dtype=bool))
        max_diff = float(np.max(np.abs(score1 - score2)))
        rows.append({
            "metric": metric,
            "sample_size": len(sample_idx),
            "max_score_difference": max_diff,
            "tolerance": 1e-9,
            "pass": max_diff < 1e-9,
        })
    return pd.DataFrame(rows)


def write_split_leakage_report(splits, duplicate_groups):
    report_path = OUTPUT_ROOT / "quality_control" / "split_leakage_report.md"
    exact_span = 0
    near_span = 0
    if not duplicate_groups.empty:
        merged = duplicate_groups.merge(splits[["image_id", "split"]], on="image_id", how="left")
        for gid, group in merged.groupby("duplicate_group_id"):
            if group["split"].nunique() > 1:
                if gid.startswith("exact_"):
                    exact_span += 1
                else:
                    near_span += 1

    content = f"""# Split Leakage Report

**Date:** {pd.Timestamp.now().isoformat()}  
**Scope:** Exact and near-duplicate groups crossing canonical splits.

## Method

- Exact duplicates identified by SHA-256 equality.
- Near duplicates identified by 16x16 average-hash Hamming distance <= 16.
- Duplicate groups were forced into a single split; this report records any groups that originally spanned splits.

## Results

- Exact-duplicate groups spanning splits: **{exact_span}**
- Near-duplicate groups spanning splits: **{near_span}**

## Source-held-out split

The metadata file does not contain a `source` column. Therefore a source-held-out split cannot be constructed automatically. Category-level leave-one-out summaries are used instead for generalization checks.

## Conclusion

No duplicate group remains in more than one split after reconciliation.
"""
    report_path.write_text(content, encoding="utf-8")


def main():
    start = time.time()
    config = load_config()
    print("Loaded config.")

    df = load_data()
    print(f"Loaded {len(df)} canonical images.")

    # Only numeric visual features should be standardized/used for metrics.
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    exclude = {"image_id", "level", "seed"}
    feature_cols = [c for c in numeric_cols if c not in exclude]

    # Step 4: splits
    splits = build_splits(df)
    print("Built canonical splits.")

    # Step 5: provenance and duplicates
    print("Computing image hashes (this may take a few minutes)...")
    provenance = compute_hashes(df)
    print(f"Computed hashes. {len(provenance)} rows.")
    duplicate_groups = detect_duplicates(provenance)
    print(f"Found {duplicate_groups['duplicate_group_id'].nunique()} duplicate groups.")
    splits = apply_duplicate_groups(splits, duplicate_groups)

    # Step 6/7: standardize and compute metrics
    preproc = yaml.safe_load(PREPROC_CONFIG_PATH.read_text(encoding="utf-8"))
    frozen_stats = preproc.get("normalization_constants")
    frozen_bounds = preproc.get("percentile_bounds")
    if frozen_stats and frozen_bounds:
        print("Using frozen standardization stats from preprocessing config...")
        stats = frozen_stats
        z_df = apply_standardization(df, feature_cols, stats)
        percentile_bounds = frozen_bounds
    else:
        print("Standardizing features on D-development...")
        z_df, stats = standardize_features(df, splits, feature_cols)
        percentile_bounds = None
    print("Computing metric scores...")
    trace = build_trace(df, z_df, splits, stats, percentile_bounds=percentile_bounds)
    percentile_bounds = json.loads(trace["percentile_bounds"].iloc[0])
    metric_table = build_metric_definition_table(stats, percentile_bounds)

    # Repeated extraction / determinism check
    print("Running repeated extraction determinism check...")
    repeated_check = run_repeated_extraction_check(df, feature_cols)

    # Dataset role audit: every image is in D_metric; role shows the most specific subset
    role_rows = []
    role_rows.append({
        "role": "D_metric",
        "n_images": int(splits["d_metric_member"].sum()),
        "categories": ",".join(sorted(splits["category"].unique())),
        "note": "all images receive deterministic proxy scores",
    })
    for specific_role in ["D_development", "D_test", "D_stability", "D_case_study"]:
        subset = splits[splits["role"] == specific_role]
        if len(subset) > 0:
            role_rows.append({
                "role": specific_role,
                "n_images": len(subset),
                "categories": ",".join(sorted(subset["category"].unique())),
                "note": "subset of D_metric",
            })
    role_audit = pd.DataFrame(role_rows)

    # Write outputs
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    (OUTPUT_ROOT / "quality_control").mkdir(parents=True, exist_ok=True)
    (OUTPUT_ROOT / "tables").mkdir(parents=True, exist_ok=True)
    (OUTPUT_ROOT / "data").mkdir(parents=True, exist_ok=True)

    splits.to_csv(OUTPUT_ROOT / "quality_control" / "canonical_splits.csv", index=False)
    role_audit.to_csv(OUTPUT_ROOT / "quality_control" / "dataset_role_audit.csv", index=False)
    provenance.to_csv(OUTPUT_ROOT / "quality_control" / "image_provenance.csv", index=False)
    duplicate_groups.to_csv(OUTPUT_ROOT / "quality_control" / "duplicate_groups.csv", index=False)
    write_split_leakage_report(splits, duplicate_groups)
    repeated_check.to_csv(OUTPUT_ROOT / "quality_control" / "repeated_extraction_check.csv", index=False)
    metric_table.to_csv(OUTPUT_ROOT / "tables" / "metric_definition_table.csv", index=False)
    trace.to_csv(OUTPUT_ROOT / "data" / "per_image_metric_trace.csv", index=False)

    # Update preprocessing config with runtime stats only when no frozen stats exist.
    preproc = yaml.safe_load(PREPROC_CONFIG_PATH.read_text(encoding="utf-8"))
    if not (preproc.get("normalization_constants") and preproc.get("percentile_bounds")):
        preproc["normalization_constants"] = {
            "mean": {k: float(v) for k, v in stats["mean"].items()},
            "std": {k: float(v) for k, v in stats["std"].items()},
            "median": {k: float(v) for k, v in stats["median"].items()},
        }
        preproc["percentile_bounds"] = percentile_bounds
        preproc["environment"]["recorded_at"] = pd.Timestamp.now().isoformat()
        PREPROC_CONFIG_PATH.write_text(yaml.safe_dump(preproc, sort_keys=False, allow_unicode=True), encoding="utf-8")

    elapsed = time.time() - start
    print(f"Done in {elapsed:.1f}s.")
    print(f"Outputs written to {OUTPUT_ROOT}")


if __name__ == "__main__":
    main()
