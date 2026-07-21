#!/usr/bin/env python3
"""Protocol v6: corrected second-renderer and mechanism-crossing audit."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import synthetic_benchmark.run_external_holdout_protocol_v4 as v4
import synthetic_benchmark.run_semantic_health_protocol_v5 as v5
from src.no_human.semantic_health import score_distribution_health


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/no_human/runs/run_20260720_external_protocol_v6"
V5_OUT = ROOT / "results/no_human/runs/run_20260720_semantic_health_v5"
SALIENCY_FIELDS = [
    "saliency_mean", "saliency_std", "fg_bg_ratio", "rule_of_thirds",
    "center_offset_x", "center_offset_y",
]
OPERATIONAL_DESCRIPTORS = list(v4.EXTERNAL_MAP)


def corrected_original_targets(frame: pd.DataFrame) -> pd.DataFrame:
    values, _ = v5.development_frozen_targets(frame)
    return frame.drop(columns=v5.TARGETS).merge(
        values, on="image_id", validate="one_to_one"
    )


def frozen_v5_score_components(features: pd.DataFrame):
    scalers = pd.read_csv(V5_OUT / "development_feature_scalers.csv")
    bounds = pd.read_csv(V5_OUT / "development_score_bounds.csv")
    scalers = scalers.loc[scalers["version"].eq("post_fix_spectral_residual")]
    bounds = bounds.loc[bounds["version"].eq("post_fix_spectral_residual")]
    scores = pd.DataFrame(index=features.index)
    raw_scores = pd.DataFrame(index=features.index)
    unbounded_scores = pd.DataFrame(index=features.index)
    for proxy, (names, weights) in v5.METRICS.items():
        raw = np.zeros(len(features), dtype=float)
        proxy_scalers = scalers.loc[scalers["proxy"].eq(proxy)].set_index("feature")
        for name, weight in zip(names, weights):
            row = proxy_scalers.loc[name]
            if not bool(row["retained"]):
                raise RuntimeError(f"v5 scaler unexpectedly dropped {proxy}/{name}")
            z = (features[name].to_numpy(float) - row["mean"]) / row["sample_std"]
            raw += np.clip(z, -4, 4) * weight
        bound = bounds.loc[bounds["proxy"].eq(proxy)].iloc[0]
        unbounded = (
            100 * (raw - bound["raw_p0_5"])
            / (bound["raw_p99_5"] - bound["raw_p0_5"])
        )
        raw_scores[proxy] = raw
        unbounded_scores[proxy] = unbounded
        scores[proxy] = np.clip(unbounded, 0, 100)
    return scores, raw_scores, unbounded_scores


def frozen_v5_scores(features: pd.DataFrame) -> pd.DataFrame:
    scores, _, _ = frozen_v5_score_components(features)
    return scores


def clustered_interval(frame: pd.DataFrame, prediction, target):
    prediction = np.asarray(prediction, dtype=float)
    outcome = frame[target].to_numpy(float)
    seeds = np.array(sorted(frame["seed"].unique()))
    groups = {
        seed: np.flatnonzero(frame["seed"].to_numpy() == seed) for seed in seeds
    }
    rng = np.random.default_rng(v4.RNG_SEED)
    draws = []
    for _ in range(v4.N_BOOT):
        selected = rng.choice(seeds, size=len(seeds), replace=True)
        index = np.concatenate([groups[seed] for seed in selected])
        draws.append(v4.rho(prediction[index], outcome[index]))
    low, high = np.quantile(draws, [0.025, 0.975])
    return v4.rho(prediction, outcome), float(low), float(high)


def corrected_controls(original, original_features, external, external_features):
    feature_columns = [
        column for column in original_features.select_dtypes(include=[np.number]).columns
        if column in external_features.columns
    ]
    base = original.merge(
        original_features[["image_id"] + feature_columns],
        on="image_id", validate="one_to_one",
    )
    ext = external.merge(
        external_features[["image_id"] + feature_columns],
        on="image_id", validate="one_to_one",
    )
    dev = base["split"].eq("G-dev").to_numpy()
    validation = base["split"].eq("G-validation").to_numpy()
    test = base["split"].eq("G-test").to_numpy()
    retained = [
        column for column in feature_columns
        if np.isfinite(base.loc[dev, column]).all()
        and base.loc[dev, column].std(ddof=1) > 1e-12
    ]
    rows = []
    for target in v4.TARGETS:
        candidates = []
        for alpha in [0.1, 1.0, 10.0, 100.0]:
            model = v5.ridge_fit(
                base.loc[dev, retained].to_numpy(float),
                base.loc[dev, target].to_numpy(float), alpha,
            )
            validation_rho = v4.rho(
                v5.ridge_predict(model, base.loc[validation, retained].to_numpy(float)),
                base.loc[validation, target],
            )
            candidates.append((validation_rho, alpha))
        validation_rho, alpha = max(candidates, key=lambda item: (item[0], -item[1]))
        train = dev | validation
        model = v5.ridge_fit(
            base.loc[train, retained].to_numpy(float),
            base.loc[train, target].to_numpy(float), alpha,
        )
        gtest_frame = base.loc[test].reset_index(drop=True)
        gtest_prediction = v5.ridge_predict(
            model, base.loc[test, retained].to_numpy(float)
        )
        external_prediction = v5.ridge_predict(
            model, ext[retained].to_numpy(float)
        )
        g_rho, g_low, g_high = clustered_interval(
            gtest_frame, gtest_prediction, target
        )
        x_rho, x_low, x_high = clustered_interval(
            ext.reset_index(drop=True), external_prediction, target
        )
        rows.append({
            "target": target,
            "n_candidate_numeric_features": len(feature_columns),
            "n_nonconstant_development_features": len(retained),
            "selected_alpha": alpha,
            "validation_rho": validation_rho,
            "gtest_rho": g_rho, "gtest_cluster_ci_lower": g_low,
            "gtest_cluster_ci_upper": g_high,
            "internal_opencv_rho": x_rho,
            "internal_opencv_cluster_ci_lower": x_low,
            "internal_opencv_cluster_ci_upper": x_high,
            "training_splits": "G-dev+G-validation after G-validation alpha selection",
            "scope": "pixel-side recoverability probe; not external generalization evidence",
        })
    sensitivity = pd.DataFrame(columns=[
        "target", "domain", "analysis_role", "note",
    ])
    (OUT / "positive_control_feature_list.txt").write_text(
        "\n".join(retained) + "\n", encoding="utf-8"
    )
    return pd.DataFrame(rows), sensitivity


def crossing_audit():
    external = pd.read_csv(OUT / "external_target_metadata.csv")
    features = pd.read_csv(OUT / "external_traditional_features.csv")
    descriptors = pd.read_csv(OUT / "operational_descriptors_external_generator.csv")
    external_frame = external.merge(features, on="image_id", validate="one_to_one").merge(
        descriptors, on="image_id", validate="one_to_one"
    ).reset_index(drop=True)
    original = corrected_original_targets(
        pd.read_csv(ROOT / "synthetic_benchmark/metadata/target_metadata.csv")
    )
    original = original.loc[original["split"].eq("G-test")]
    original_features = pd.read_csv(
        ROOT / "synthetic_benchmark/metadata/synthetic_traditional_features.csv"
    )
    original_descriptors = pd.read_csv(
        OUT / "operational_descriptors_original_gtest.csv"
    )
    original_frame = original.merge(
        original_features, on="image_id", validate="one_to_one"
    ).merge(original_descriptors, on="image_id", validate="one_to_one").reset_index(drop=True)
    intended = {
        **{name: target for name, (target, _) in v4.EXTERNAL_MAP.items()},
        "saliency_mean": "focal_hierarchy_index",
        "saliency_std": "focal_hierarchy_index",
        "fg_bg_ratio": "focal_hierarchy_index",
        "rule_of_thirds": "geometric_order_index",
        "center_offset_x": "geometric_order_index",
        "center_offset_y": "geometric_order_index",
    }
    rows = []
    for renderer, frame in [
        ("Pillow_G-test", original_frame),
        ("internal_OpenCV", external_frame),
    ]:
        for descriptor in OPERATIONAL_DESCRIPTORS + SALIENCY_FIELDS:
            for target in v4.TARGETS:
                observed, low, high = clustered_interval(frame, frame[descriptor], target)
                rows.append({
                    "renderer": renderer, "descriptor": descriptor,
                    "target": target,
                    "is_predeclared_nominal_pair": intended.get(descriptor) == target,
                    "n_images": len(frame),
                    "n_seed_clusters": frame["seed"].nunique(),
                    "rho": observed, "cluster_ci_lower": low,
                    "cluster_ci_upper": high,
                })
    pd.DataFrame(rows).to_csv(OUT / "mechanism_crossing_intervals.csv", index=False)

    health_rows = []
    for field in SALIENCY_FIELDS:
        values = external_frame[field]
        health_rows.append({
            "field": field, "n_images": len(values),
            "n_unique": values.nunique(dropna=False),
            "minimum": values.min(), "maximum": values.max(),
            "sample_std": values.std(ddof=1),
            "finite": bool(np.isfinite(values).all()),
            "passed_non_degenerate": bool(
                np.isfinite(values).all() and values.nunique(dropna=False) > 1
                and values.std(ddof=1) > 1e-12
            ),
        })
    health = pd.DataFrame(health_rows)
    if not health["passed_non_degenerate"].all():
        raise RuntimeError("second-renderer saliency semantic-health gate failed")
    health.to_csv(OUT / "external_feature_semantic_health.csv", index=False)

    final_scores, raw_scores, unbounded_scores = frozen_v5_score_components(
        features.reset_index(drop=True)
    )
    score_health_rows = []
    for proxy in v5.TARGET_MAP:
        audit = score_distribution_health(final_scores[proxy])
        score_health_rows.append({
            "proxy": proxy, **audit,
            "below_zero_before_final_clip_fraction": float(
                (unbounded_scores[proxy] < 0).mean()
            ),
            "above_100_before_final_clip_fraction": float(
                (unbounded_scores[proxy] > 100).mean()
            ),
            "passed_non_degenerate_final_score": bool(
                audit["finite"] and audit["n_unique"] > 1
                and audit["sample_std"] > 1e-12
            ),
        })
    pd.DataFrame(score_health_rows).to_csv(
        OUT / "external_composite_score_health.csv", index=False
    )
    raw_composites, _ = v4.composite_results(external_frame, raw_scores)
    raw_composites.to_csv(OUT / "raw_composite_external_holdout.csv", index=False)
    write_score_collapse_figure(raw_composites)

    balance_rows = []
    for renderer, frame in [
        ("Pillow_G-test", original_frame),
        ("internal_OpenCV", external_frame),
    ]:
        target_matrix = frame[v4.TARGETS].corr(method="spearman")
        off_diagonal = target_matrix.to_numpy()[
            ~np.eye(len(v4.TARGETS), dtype=bool)
        ]
        balance_rows.append({
            "renderer": renderer, "n_images": len(frame),
            "n_seed_clusters": frame["seed"].nunique(),
            "max_abs_off_diagonal_target_rho": float(np.abs(off_diagonal).max()),
            "interpretation": "factor-balance diagnostic only; does not establish natural-image generalization",
        })
    pd.DataFrame(balance_rows).to_csv(OUT / "renderer_factor_balance.csv", index=False)


def write_score_collapse_figure(raw_composites=None):
    if raw_composites is None:
        raw_composites = pd.read_csv(OUT / "raw_composite_external_holdout.csv")
    final = pd.read_csv(OUT / "composite_external_holdout.csv")
    merged = final[["proxy", "rho"]].merge(
        raw_composites[["proxy", "rho"]], on="proxy", suffixes=("_final", "_raw")
    )
    labels = ["Complexity", "Order", "Harmony", "Intensity", "Hierarchy"]
    x = np.arange(len(merged)); width = 0.36
    fig, ax = v4.plt.subplots(figsize=(9.0, 5.4))
    ax.bar(x - width/2, merged["rho_raw"], width, label="Raw weighted sum")
    ax.bar(x + width/2, merged["rho_final"], width, label="Clipped 0--100 score")
    ax.axhline(0, color="black", lw=0.8)
    ax.set_xticks(x, labels, rotation=20, ha="right")
    ax.set_ylabel("OpenCV nominal Spearman rho")
    ax.set_title("Frozen score bounds can erase cross-renderer rank signal")
    ax.legend(frameon=False); fig.tight_layout()
    fig.savefig(OUT / "external_score_boundary_collapse.png", dpi=300,
                metadata={"Software": "Matplotlib protocol-v6 renderer"})
    v4.plt.close(fig)


def refresh_metadata_and_hashes():
    metadata_path = OUT / "run_metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata.update({
        "protocol": "6.0-corrected-second-renderer-mechanism-audit",
        "run_id": "run_20260720_external_protocol_v6",
        "primary_target_scaling": "G-dev constants for Pillow; direct orthogonal levels for OpenCV",
        "proxy_scaling": "protocol-v5 G-dev constants applied unchanged",
        "saliency_algorithm": "in-repository spectral residual; no cv2.saliency dependency",
        "scope": "two internally controlled renderers; failure analysis, not external natural-image validation",
        "external_score_health": "feature maps are checked separately from final-score boundary collapse; raw and clipped composite associations are both released",
        "restricted_null_role": "not run and not used as a decision gate",
    })
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    rows = []
    for path in sorted(OUT.iterdir()):
        if path.is_file() and path.name != "artifact_hashes.csv":
            rows.append({
                "file": path.name,
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "bytes": path.stat().st_size,
            })
    pd.DataFrame(rows).to_csv(OUT / "artifact_hashes.csv", index=False)


def main():
    if not (V5_OUT / "development_feature_scalers.csv").exists():
        raise RuntimeError("run protocol v5 before protocol v6")
    v4.OUT = OUT
    v4.IMAGE_DIR = OUT / "images"
    v4.RNG_SEED = 26072006
    v4.proxy_scores = frozen_v5_scores
    v4.prepare_original_targets = corrected_original_targets
    v4.fit_controls = corrected_controls
    v4.main()
    crossing_audit()
    refresh_metadata_and_hashes()
    print(f"Protocol v6 complete: {OUT}")


if __name__ == "__main__":
    main()
