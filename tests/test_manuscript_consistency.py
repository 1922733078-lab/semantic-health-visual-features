"""Regression checks connecting active prose, formulas, references, and frozen results."""

import re
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
PAPER = ROOT / "paper"


def active_tex():
    section_root = PAPER / "sections_no_human"
    paths = [
        PAPER / "main_no_human.tex",
        *(section_root / name for name in [
            "01-introduction.tex", "02-related_work.tex", "03-method.tex",
            "04-experiments.tex", "05-results.tex", "06-discussion.tex",
            "07-conclusion.tex", "generated_tables_v5.tex",
        ]),
    ]
    return "\n".join(path.read_text(encoding="utf-8") for path in paths)


def test_active_bibliography_is_bidirectionally_closed():
    text = active_tex()
    bib = (PAPER / "references.bib").read_text(encoding="utf-8")
    bib_keys = set(re.findall(r"@[^{]+\{([^,]+),", bib))
    cited = set()
    for match in re.finditer(r"\\cite\w*\{([^}]+)\}", text):
        cited.update(key.strip() for key in match.group(1).split(","))
    assert cited == bib_keys, {
        "missing_bibliography_entries": sorted(cited - bib_keys),
        "uncited_bibliography_entries": sorted(bib_keys - cited),
    }


def test_corrected_jvcir_author_metadata():
    bib = (PAPER / "references.bib").read_text(encoding="utf-8")
    assert "Yang, Shuai and Wang, Zibei and Wang, Guangao" in bib
    assert "Yan, Xingao and Shao, Feng and Chen, Hangwei and Jiang, Qiuping" in bib


def test_restricted_corpus_claims_are_withdrawn_from_active_paper():
    text = active_tex()
    for forbidden in ("17,337", "tab:stability", "tab:sensitivity", "tab:category", "tab:convergence"):
        assert forbidden not in text


def test_duplicate_edge_term_is_absent_from_active_proxy_algebra():
    from synthetic_benchmark.run_semantic_health_protocol_v5 import METRICS, BASELINE_CANDIDATES

    for features, _ in METRICS.values():
        assert "text_coverage" not in features
        assert len(features) == len(set(features))
    assert BASELINE_CANDIDATES["layout_hierarchy_proxy"][-1] == (
        "edge_coverage", "edge_density", 1.0
    )
    for candidates in BASELINE_CANDIDATES.values():
        signed_sources = [(source, direction) for _, source, direction in candidates]
        assert len(signed_sources) == len(set(signed_sources))


def test_external_descriptor_names_do_not_impersonate_published_methods():
    from synthetic_benchmark.run_external_holdout_protocol_v4 import EXTERNAL_MAP
    from synthetic_benchmark.run_external_holdout_protocol_v6 import OPERATIONAL_DESCRIPTORS

    expected = {
        "canny_edge_density",
        "multiscale_lab_residual_entropy",
        "jpeg_bytes_per_pixel_q90",
        "hasler_susstrunk_colourfulness",
        "global_luminance_cv",
        "horizontal_mirror_similarity",
    }
    assert set(EXTERNAL_MAP) == expected
    assert set(OPERATIONAL_DESCRIPTORS) == expected
    code = (ROOT / "synthetic_benchmark/run_external_holdout_protocol_v4.py").read_text()
    assert "rosenholtz_subband_entropy" not in code
    assert "rms_luminance_contrast" not in code


def test_primary_v5_numbers_are_generated_into_active_tables():
    results = pd.read_csv(
        ROOT / "results/no_human/runs/run_20260720_semantic_health_v5/primary_cluster_results.csv"
    )
    text = active_tex()
    for row in results.itertuples():
        assert f"{row.rho:.3f}" in text
        assert f"[{row.cluster_ci_lower:.3f}, {row.cluster_ci_upper:.3f}]" in text


def test_v6_composite_numbers_are_synchronised():
    root = ROOT / "results/no_human/runs/run_20260720_external_protocol_v6"
    final = pd.read_csv(root / "composite_external_holdout.csv")
    raw = pd.read_csv(root / "raw_composite_external_holdout.csv")
    text = active_tex()
    for row in final.itertuples():
        assert f"{row.rho:.3f}" in text
        assert f"[{row.cluster_ci_lower:.3f}, {row.cluster_ci_upper:.3f}]" in text
        assert f"{row.selectivity_margin:.3f}" in text
    for row in raw.itertuples():
        assert f"{row.rho:.3f}" in text
        assert f"[{row.cluster_ci_lower:.3f}, {row.cluster_ci_upper:.3f}]" in text


def test_paired_baseline_numbers_are_synchronised():
    results = pd.read_csv(
        ROOT / "results/no_human/runs/run_20260720_semantic_health_v5/paired_baseline_cluster.csv"
    )
    text = active_tex()
    for row in results.itertuples():
        assert f"{row.delta_rho:.3f}" in text
        assert f"[{row.cluster_delta_ci_lower:.3f}, {row.cluster_delta_ci_upper:.3f}]" in text


def test_v6_descriptor_table_is_synchronised():
    results = pd.read_csv(
        ROOT / "results/no_human/runs/run_20260720_external_protocol_v6/external_metric_correlations.csv"
    )
    text = active_tex()
    for metric, group in results.groupby("metric"):
        assert len(group) == 2
        for value in group["rho"]:
            assert f"{value:.3f}" in text


def test_v6_controls_are_scoped_and_synchronised():
    controls = pd.read_csv(
        ROOT / "results/no_human/runs/run_20260720_external_protocol_v6/positive_control_cross_generator.csv"
    )
    assert controls["n_nonconstant_development_features"].eq(38).all()
    assert controls["scope"].eq(
        "pixel-side recoverability probe; not external generalization evidence"
    ).all()
    text = active_tex()
    assert "These are scope diagnostics, not perceptual validation" in text
    for row in controls.itertuples():
        assert f"{row.gtest_rho:.3f}" in text
        assert f"[{row.gtest_cluster_ci_lower:.3f}, {row.gtest_cluster_ci_upper:.3f}]" in text
        assert f"{row.internal_opencv_rho:.3f}" in text
        assert (
            f"[{row.internal_opencv_cluster_ci_lower:.3f}, "
            f"{row.internal_opencv_cluster_ci_upper:.3f}]"
        ) in text
