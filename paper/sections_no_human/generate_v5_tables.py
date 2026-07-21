#!/usr/bin/env python3
"""Generate manuscript tables only from corrected protocol-v5/v6 artefacts."""

from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
V5 = ROOT / "results/no_human/runs/run_20260720_semantic_health_v5"
V6 = ROOT / "results/no_human/runs/run_20260720_external_protocol_v6"
OUT = ROOT / "paper/sections_no_human/generated_tables_v5.tex"

PROXY = {
    "visual_complexity_proxy": "Complexity",
    "layout_order_proxy": "Order",
    "colour_harmony_proxy": "Harmony",
    "visual_intensity_proxy": "Intensity",
    "layout_hierarchy_proxy": "Hierarchy",
}
TARGET = {
    "structural_clutter_index": "Structural clutter",
    "geometric_order_index": "Geometric order",
    "palette_coherence_index": "Palette coherence",
    "visual_salience_index": "Visual salience",
    "focal_hierarchy_index": "Focal hierarchy",
}


def f3(value):
    return f"{value:.3f}"


def ci(row, lower="cluster_ci_lower", upper="cluster_ci_upper"):
    return f"[{f3(row[lower])}, {f3(row[upper])}]"


def main():
    required = [
        V5 / "pre_post_nominal_comparison.csv",
        V5 / "primary_cluster_results.csv",
        V5 / "paired_baseline_cluster.csv",
        V6 / "positive_control_cross_generator.csv",
        V6 / "external_metric_correlations.csv",
        V6 / "composite_external_holdout.csv",
        V6 / "raw_composite_external_holdout.csv",
    ]
    missing = [path for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError("missing corrected results: " + ", ".join(map(str, missing)))
    lines = []

    repair = pd.read_csv(required[0])
    lines += [
        r"\begin{table}[htbp]", r"\centering",
        r"\caption{Semantic-health repair on the primary G-test split. Retained weight is the fraction of absolute formula weight attached to nonconstant G-dev inputs. Intervals resample ten seeds as clusters.}",
        r"\label{tab:semantic_repair}", r"\small",
        r"\resizebox{\textwidth}{!}{%", r"\begin{tabular}{lrrrrr}",
        r"\toprule",
        r"Legacy composite & Pre retained & Post retained & Pre-fix $\rho$ & Post-fix $\rho$ & $\Delta\rho$ \\",
        r"\midrule",
    ]
    for _, row in repair.iterrows():
        lines.append(
            f"{PROXY[row['proxy']]} & {row['pre_fix_retained_abs_weight_fraction']:.2f} & "
            f"{row['post_fix_retained_abs_weight_fraction']:.2f} & "
            f"{f3(row['pre_fix_rho'])} & {f3(row['post_fix_rho'])} & {f3(row['rho_change'])} \\\\"
        )
    lines += [r"\bottomrule", r"\end{tabular}", r"}", r"\end{table}", ""]

    primary = pd.read_csv(required[1])
    lines += [
        r"\begin{table}[htbp]", r"\centering",
        r"\caption{Corrected G-test associations after G-dev-only target and feature scaling. All 25 associations are released; the table shows nominal pairs and the strongest non-target response.}",
        r"\label{tab:primary_v5}", r"\small",
        r"\resizebox{\textwidth}{!}{%", r"\begin{tabular}{llrrrr}",
        r"\toprule",
        r"Composite & Target & $\rho$ & Seed-cluster 95\% CI & Max other $|\rho|$ & Margin \\",
        r"\midrule",
    ]
    for _, row in primary.iterrows():
        lines.append(
            f"{PROXY[row['proxy']]} & {TARGET[row['target']]} & {f3(row['rho'])} & "
            f"{ci(row)} & {f3(row['max_abs_non_target_rho'])} & {f3(row['selectivity_margin'])} \\\\"
        )
    lines += [r"\bottomrule", r"\end{tabular}", r"}", r"\end{table}", ""]

    baselines = pd.read_csv(required[2])
    lines += [
        r"\begin{table}[htbp]", r"\centering",
        r"\caption{Elementary baselines selected on G-dev and locked before paired G-test evaluation. $\Delta\rho=\rho_{composite}-\rho_{baseline}$.}",
        r"\label{tab:baseline_v5}", r"\footnotesize",
        r"\resizebox{\textwidth}{!}{%", r"\begin{tabular}{llrrrr}",
        r"\toprule",
        r"Composite & Locked baseline & Composite $\rho$ & Baseline $\rho$ & $\Delta\rho$ & Seed-cluster 95\% CI \\",
        r"\midrule",
    ]
    for _, row in baselines.iterrows():
        name = row["baseline"].replace("_", " ").title()
        delta_ci = f"[{f3(row['cluster_delta_ci_lower'])}, {f3(row['cluster_delta_ci_upper'])}]"
        lines.append(
            f"{PROXY[row['proxy']]} & {name} & {f3(row['proposed_rho'])} & "
            f"{f3(row['baseline_rho'])} & {f3(row['delta_rho'])} & {delta_ci} \\\\"
        )
    lines += [r"\bottomrule", r"\end{tabular}", r"}", r"\end{table}", ""]

    controls = pd.read_csv(required[3])
    lines += [
        r"\begin{table}[htbp]", r"\centering",
        r"\caption{Pixel-side ridge recoverability probes. Alpha was selected on G-validation; G-dev-constant features were removed; intervals resample ten seeds. These are scope diagnostics, not perceptual validation.}",
        r"\label{tab:controls_v6}", r"\footnotesize",
        r"\resizebox{\textwidth}{!}{%", r"\begin{tabular}{lrrrrr}",
        r"\toprule",
        r"Target & Features & Pillow $\rho$ & Pillow 95\% CI & OpenCV $\rho$ & OpenCV 95\% CI \\",
        r"\midrule",
    ]
    for _, row in controls.iterrows():
        gci = f"[{f3(row['gtest_cluster_ci_lower'])}, {f3(row['gtest_cluster_ci_upper'])}]"
        xci = f"[{f3(row['internal_opencv_cluster_ci_lower'])}, {f3(row['internal_opencv_cluster_ci_upper'])}]"
        lines.append(
            f"{TARGET[row['target']]} & {int(row['n_nonconstant_development_features'])} & "
            f"{f3(row['gtest_rho'])} & {gci} & {f3(row['internal_opencv_rho'])} & {xci} \\\\"
        )
    lines += [r"\bottomrule", r"\end{tabular}", r"}", r"\end{table}", ""]

    metrics = pd.read_csv(required[4])
    original = metrics.loc[metrics["generator"].eq("original_G-test")].copy()
    external = metrics.loc[metrics["generator"].eq("internal_opencv")].copy()
    joined = original.merge(external, on=["metric", "target"], suffixes=("_g", "_x"))
    metric_names = {
        "canny_edge_density": "Canny edge fraction",
        "multiscale_lab_residual_entropy": "Lab residual entropy",
        "jpeg_bytes_per_pixel_q90": "JPEG bytes/pixel",
        "hasler_susstrunk_colourfulness": "Colourfulness",
        "global_luminance_cv": "Luminance CV",
        "horizontal_mirror_similarity": "Mirror similarity",
    }
    lines += [
        r"\begin{table}[htbp]", r"\centering",
        r"\caption{Pre-fixed operational descriptors on the two controlled renderers. Each cell reports Spearman $\rho$ and its ten-seed cluster interval.}",
        r"\label{tab:operational_v6}", r"\footnotesize",
        r"\resizebox{\textwidth}{!}{%", r"\begin{tabular}{llrr}",
        r"\toprule", r"Descriptor & Nominal target & Pillow G-test & Internal OpenCV \\",
        r"\midrule",
    ]
    for _, row in joined.iterrows():
        g = f"{f3(row['rho_g'])} [{f3(row['cluster_ci_lower_g'])}, {f3(row['cluster_ci_upper_g'])}]"
        x = f"{f3(row['rho_x'])} [{f3(row['cluster_ci_lower_x'])}, {f3(row['cluster_ci_upper_x'])}]"
        lines.append(f"{metric_names[row['metric']]} & {TARGET[row['target']]} & {g} & {x} \\\\ ")
    lines += [r"\bottomrule", r"\end{tabular}", r"}", r"\end{table}", ""]

    composites = pd.read_csv(required[5])
    raw_composites = pd.read_csv(required[6])
    composites = composites.merge(
        raw_composites[["proxy", "rho", "cluster_ci_lower", "cluster_ci_upper"]].rename(
            columns={
                "rho": "raw_rho",
                "cluster_ci_lower": "raw_cluster_ci_lower",
                "cluster_ci_upper": "raw_cluster_ci_upper",
            }
        ),
        on="proxy", validate="one_to_one",
    )
    lines += [
        r"\begin{table}[htbp]", r"\centering",
        r"\caption{Corrected legacy composites on the pairwise-orthogonal internal OpenCV renderer ($n=1{,}250$, ten seeds). Each association is followed by its seed-cluster 95\% interval.}",
        r"\label{tab:external_v6}", r"\footnotesize", r"\resizebox{\textwidth}{!}{%", r"\begin{tabular}{lrrr}",
        r"\toprule", r"Composite & Final score $\rho$ [95\% CI] & Raw sum $\rho$ [95\% CI] & Final margin \\",
        r"\midrule",
    ]
    for _, row in composites.iterrows():
        final = f"{f3(row['rho'])} {ci(row)}"
        raw = (
            f"{f3(row['raw_rho'])} "
            f"[{f3(row['raw_cluster_ci_lower'])}, {f3(row['raw_cluster_ci_upper'])}]"
        )
        lines.append(
            f"{PROXY[row['proxy']]} & {final} & {raw} & {f3(row['selectivity_margin'])} \\\\"
        )
    lines += [r"\bottomrule", r"\end{tabular}", r"}", r"\end{table}", ""]

    OUT.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {OUT}")


if __name__ == "__main__":
    main()
