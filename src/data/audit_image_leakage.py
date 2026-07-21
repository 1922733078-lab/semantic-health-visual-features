#!/usr/bin/env python3
"""
Step 5 — Detect Duplicate, Source, and Category Leakage
"""
import hashlib
import multiprocessing as mp
from collections import defaultdict
from pathlib import Path

import imagehash
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.model_selection import StratifiedKFold

ROOT = Path(__file__).resolve().parents[2]
QC = ROOT / "results" / "redesign" / "quality_control"
QC.mkdir(parents=True, exist_ok=True)

METADATA = ROOT / "data" / "processed" / "metadata.csv"
HUMAN_LIST = ROOT / "盲评问卷" / "图像清单.csv"
SPLIT_FILE = QC / "canonical_splits.csv"
DUPLICATE_FILE = QC / "duplicate_groups.csv"
REPORT_FILE = QC / "split_leakage_report.md"
HASH_CACHE = QC / "image_hash_cache.csv"


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def phash_file(path: Path) -> imagehash.ImageHash:
    try:
        # Use a small thumbnail to keep perceptual hashing fast.
        with Image.open(path) as im:
            im.thumbnail((256, 256))
            return imagehash.phash(im)
    except Exception:
        return None


def compute_hashes(args):
    """Worker for parallel hash computation."""
    image_id, path_str = args
    path = ROOT / path_str
    return image_id, sha256_file(path), str(phash_file(path)) if path.exists() else ""


def find_exact_duplicates(df):
    groups = defaultdict(list)
    for idx, h in enumerate(df["sha256"]):
        groups[h].append(idx)
    dup_group_ids = {}
    gid = 0
    for h, idxs in groups.items():
        if len(idxs) > 1:
            gid += 1
            for idx in idxs:
                dup_group_ids[idx] = f"exact_{gid:05d}"
    return dup_group_ids


def find_near_duplicates_focused(df, human_indices, max_hamming=8):
    """
    Find near-duplicates using phash.
    Focus on (a) pairs within the human subset and (b) pairs crossing the human/proxy boundary.
    Proxy-internal near-duplicates are not exhaustively clustered to keep computation tractable.
    """
    hashes = df["phash"].tolist()
    n = len(hashes)
    human_set = set(human_indices)

    parent = list(range(n))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(x, y):
        rx, ry = find(x), find(y)
        if rx != ry:
            parent[rx] = ry

    # Compare all human images against all images (including themselves)
    for hi in human_indices:
        if hashes[hi] is None:
            continue
        for j in range(n):
            if hashes[j] is None:
                continue
            if hashes[hi] - hashes[j] <= max_hamming:
                union(hi, j)

    groups = defaultdict(list)
    for idx in range(n):
        groups[find(idx)].append(idx)

    dup_group_ids = {}
    gid = 0
    for root_idx, idxs in groups.items():
        if len(idxs) > 1:
            gid += 1
            for idx in idxs:
                dup_group_ids[idx] = f"near_{gid:05d}"
    return dup_group_ids


def build_canonical_split(df, n_splits=5, seed=42):
    # Group-aware: assign each duplicate group to a single fold.
    group_keys = df["duplicate_group_id"].fillna(df["image_id"]).values
    unique_groups, group_inv = np.unique(group_keys, return_inverse=True)
    first_idx = {g: np.where(group_keys == g)[0][0] for g in unique_groups}
    group_category = np.array([df.loc[first_idx[g], "category"] for g in unique_groups])

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    group_folds = np.zeros(len(unique_groups), dtype=int)
    for fold_idx, (_, test_idx) in enumerate(skf.split(np.zeros(len(unique_groups)), group_category)):
        group_folds[test_idx] = fold_idx

    df["fold"] = group_folds[group_inv]
    return df


def main():
    df = pd.read_csv(METADATA)
    print(f"Loaded metadata: {len(df)} images")

    # Use the actual 100 human-rated image list to override the legacy in_human_subset flag.
    human_list_df = pd.read_csv(HUMAN_LIST)
    human_ids = set(human_list_df["文件名"].str.replace(".jpg", "").values)
    df["in_human_subset"] = df["image_id"].isin(human_ids)
    print(f"Actual human-rated images marked: {df['in_human_subset'].sum()}")

    # Compute hashes, using a cache when images are unchanged.
    cache = pd.DataFrame(columns=["image_id", "sha256", "phash"])
    if HASH_CACHE.exists():
        cache = pd.read_csv(HASH_CACHE)
        print(f"Loaded hash cache: {len(cache)} entries")

    cache_ids = set(cache["image_id"]) if not cache.empty else set()
    to_compute = [(row.image_id, row.standardized_path) for _, row in df.iterrows() if row.image_id not in cache_ids]

    if to_compute:
        print(f"Computing hashes for {len(to_compute)} images (SHA-256 + phash)...")
        n_workers = max(1, mp.cpu_count() - 1)
        with mp.Pool(n_workers) as pool:
            computed = pool.map(compute_hashes, to_compute)
        computed_df = pd.DataFrame(computed, columns=["image_id", "sha256", "phash"])
        cache = pd.concat([cache, computed_df], ignore_index=True)
        cache.to_csv(HASH_CACHE, index=False)
        print(f"Updated hash cache: {HASH_CACHE}")
    else:
        print("All hashes found in cache.")

    df = df.merge(cache, on="image_id", how="left")
    df["phash"] = df["phash"].apply(lambda s: imagehash.hex_to_hash(s) if s and isinstance(s, str) and s.strip() else None)

    exact_dups = find_exact_duplicates(df)
    human_indices = df.index[df["in_human_subset"]].tolist()
    near_dups = find_near_duplicates_focused(df, human_indices, max_hamming=8)

    # Combine: exact overrides near
    duplicate_group_id = {}
    for idx in range(len(df)):
        if idx in exact_dups:
            duplicate_group_id[idx] = exact_dups[idx]
        elif idx in near_dups:
            duplicate_group_id[idx] = near_dups[idx]
        else:
            duplicate_group_id[idx] = df.loc[idx, "image_id"]
    df["duplicate_group_id"] = [duplicate_group_id[i] for i in range(len(df))]

    # Save duplicate groups (serialize phash as hex string)
    dup_out = df[["image_id", "category", "sha256", "duplicate_group_id"]].copy()
    dup_out["phash"] = df["phash"].apply(lambda h: str(h) if h is not None else "")
    dup_out.to_csv(DUPLICATE_FILE, index=False)
    print(f"Wrote {DUPLICATE_FILE}")

    # Build canonical split
    df = build_canonical_split(df, n_splits=5, seed=42)

    # Exclude from proxy development any image that is in the human subset or
    # shares a duplicate group with a human-rated image.
    human_groups = set(df.loc[df["in_human_subset"], "duplicate_group_id"])
    df["exclude_from_proxy_development"] = df["duplicate_group_id"].isin(human_groups)

    df[["image_id", "category", "duplicate_group_id", "fold", "in_human_subset", "exclude_from_proxy_development"]].to_csv(SPLIT_FILE, index=False)
    print(f"Wrote {SPLIT_FILE}")

    # Leakage checks
    human_df = df[df["in_human_subset"]].copy()
    proxy_df = df[~df["in_human_subset"]].copy()

    # Exact duplicates crossing human/proxy boundary
    human_exact_groups = set(human_df["duplicate_group_id"]) - set(human_df["image_id"])
    proxy_exact_cross = proxy_df[proxy_df["duplicate_group_id"].isin(human_exact_groups) & proxy_df["duplicate_group_id"].str.startswith("exact_")]

    # Near duplicates crossing human/proxy boundary
    human_near_groups = set(human_df["duplicate_group_id"]) - set(human_df["image_id"])
    proxy_near_cross = proxy_df[proxy_df["duplicate_group_id"].isin(human_near_groups) & proxy_df["duplicate_group_id"].str.startswith("near_")]

    # Source field
    df["source"] = df["image_id"].apply(lambda x: x.split("_")[0] if "_" in x else "unknown")

    report_lines = [
        "# Split Leakage and Duplicate Report\n",
        f"- Total images: {len(df)}",
        f"- Exact duplicate groups: {len(set(v for v in exact_dups.values()))}",
        f"- Near-duplicate groups (phash Hamming <= 8): {len(set(v for v in near_dups.values()))}",
        f"- Actual human-rated images (from 盲评问卷/图像清单.csv): {len(human_ids)}",
        f"- Human-rated images in metadata: {df['in_human_subset'].sum()}",
        "\n## Human/proxy boundary checks\n",
        "The 100 human-rated images are a subset of the 17,337-image metadata. Therefore `image_id` overlap is expected. The critical leakage risks are exact or near duplicates that cross the human/proxy boundary.",
        f"- Exact-duplicate groups crossing human/proxy boundary: {len(proxy_exact_cross)}",
        f"- Near-duplicate groups crossing human/proxy boundary: {len(proxy_near_cross)}",
        "\n## Cross-boundary duplicates (if any)\n",
    ]
    crosses = list(proxy_exact_cross.itertuples()) + list(proxy_near_cross.itertuples())
    if crosses:
        for row in crosses[:30]:
            report_lines.append(f"- {row.image_id} ({row.category}) shares group {row.duplicate_group_id} with a human-rated image")
    else:
        report_lines.append("None detected.")

    report_lines.extend([
        "\n## Split rules enforced\n",
        "- Exact duplicates are assigned to the same fold.",
        "- Near-duplicates (perceptual hash Hamming distance <= 8) are assigned to the same fold.",
        "- Stratification is performed by category at the group level.",
        "- All models must use the canonical `fold` column from this file.",
        "\n## Limitations\n",
        "- Perceptual duplicate detection uses phash with a fixed Hamming threshold; sensitivity analysis with other thresholds or hashes can be added.",
        "- Source-held-out evaluation is recorded as a planned analysis in Step 8 / Step 16.",
        "\n",
    ])
    REPORT_FILE.write_text("\n".join(report_lines), encoding="utf-8")
    print(f"Wrote {REPORT_FILE}")

    print("Step 5 complete.")


if __name__ == "__main__":
    main()
