#!/usr/bin/env python3
"""Generate the active package-relative Route-C claim/evidence ledger."""

from pathlib import Path
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/no_human/quality_control/claim_evidence_ledger.csv"
DATE = "2026-07-20"


def make(claim_id, location, claim, evidence, detail, run_id,
         status="supported", risk="high"):
    return {
        "claim_id": claim_id,
        "document_location": location,
        "exact_claim": claim,
        "evidence_file": evidence,
        "evidence_row_or_field": detail,
        "run_id": run_id,
        "support_status": status,
        "risk_level": risk,
        "checked_date": DATE,
    }


V5 = "tier_b_release/results/no_human/runs/run_20260720_semantic_health_v5"
V6 = "tier_b_release/results/no_human/runs/run_20260720_external_protocol_v6"

rows = [
    make("C001", "Abstract; Results RQ1",
         "The pre-fix saliency path was constant across 17,675 controlled images.",
         f"{V5}/pre_fix_synthetic_traditional_features.csv",
         "six constant fields; OpenCV pre-fix behaviour traced in audit", "v5+v6"),
    make("C002", "Method: semantic-health gates",
         "Every nonzero-weight post-fix input is finite and nonconstant on G-dev.",
         f"{V5}/semantic_health_gate.csv", "all active_gate_passed=True", "v5"),
    make("C003", "Results RQ1",
         "Repair changes nominal order rho by +0.047 and hierarchy rho by -0.263.",
         f"{V5}/pre_post_nominal_comparison.csv",
         "saliency-columns-only comparison", "v5"),
    make("C004", "Results RQ2",
         "All 25 G-test associations are released and none of the five names is selective for its nominal target.",
         f"{V5}/full_25_proxy_target_cluster.csv",
         "5x5 associations; primary margins in primary_cluster_results.csv", "v5"),
    make("C005", "Abstract; Results RQ2",
         "Order exceeds its G-dev-selected locked baseline by 0.045; the other four composites do not.",
         f"{V5}/paired_baseline_cluster.csv", "five paired G-test rows", "v5"),
    make("C006", "Method: leakage-controlled scaling",
         "Target feature and score constants are fitted on G-dev and frozen elsewhere.",
         f"{V5}/run_metadata.json", "three fit/selection split fields", "v5"),
    make("C007", "Results RQ3",
         "All six repaired saliency-dependent fields are finite and nonconstant on 1,250 OpenCV images.",
         f"{V6}/external_feature_semantic_health.csv",
         "six passed_non_degenerate rows", "v6"),
    make("C008", "Abstract; Results RQ3",
         "Frozen bounds collapse OpenCV complexity and intensity final scores to zero.",
         f"{V6}/external_composite_score_health.csv",
         "two failed final-score rows", "v6"),
    make("C009", "Abstract; Results RQ3",
         "Raw OpenCV complexity and intensity correlations are 0.895 and 0.674.",
         f"{V6}/raw_composite_external_holdout.csv",
         "visual_complexity_proxy and visual_intensity_proxy", "v6"),
    make("C010", "Results RQ3",
         "Five of six operational descriptors preserve the expected interval sign in both renderers.",
         f"{V6}/external_metric_correlations.csv",
         "twelve renderer-by-descriptor rows", "v6"),
    make("C011", "Results RQ3",
         "The complete mechanism audit contains 120 renderer-descriptor-target intervals.",
         f"{V6}/mechanism_crossing_intervals.csv", "120 rows", "v6"),
    make("C012", "Experiments: second renderer",
         "The OpenCV factor design is pairwise orthogonal.",
         f"{V6}/renderer_factor_balance.csv", "maximum off-diagonal rho=0", "v6"),
    make("C013", "Results: controls",
         "All 38 numeric features vary on G-dev and the five ridge probes are reported on both renderers.",
         f"{V6}/positive_control_cross_generator.csv", "five rows; 38 retained features", "v6"),
    make("C014", "Data and code availability",
         "The research object contains 16,425 Pillow and 1,250 OpenCV images.",
         "tier_b_release/public_demo/dataset_manifest.csv",
         "16,425 Pillow rows; v6 external_image_manifest.csv has 1,250 rows", "release"),
    make("C015", "Reproducibility outcome",
         "The Tier B object is covered by a complete path size and SHA-256 manifest.",
         "tier_b_release/manifests/release_manifest.csv", "17,806 payload rows", "release"),
    make("C016", "Reproducibility outcome",
         "A clean environment passed the full suite and independently reproduced v5 byte for byte.",
         "tier_b_release/logs/verification_protocol_v5_clean.log", "RESULT: PASS", "release"),
    make("C017", "Ethics statement",
         "The active study contains no human raters or human-subject data.",
         "documents/NO_HUMAN_SCOPE.md", "scope and exclusions", "release"),
    make("C018", "Discussion: Limits",
         "The second renderer is internal and does not establish natural-image or perceptual validity.",
         "manuscript_source/sections_no_human/06-discussion.tex", "Limits", "manuscript"),
    make("C019", "Generative-AI declaration",
         "ChatGPT and Codex supported preparation; authors verified outputs; no reported research image was generated or altered.",
         "manuscript_source/main_no_human.tex", "AI-use declaration", "manuscript"),
    make("C020", "Data and code availability",
         "A persistent repository DOI/URL is still required before submission.",
         "audits/pre_submission_audit.md", "Administrative holds", "release",
         status="qualified", risk="medium"),
]

OUT.parent.mkdir(parents=True, exist_ok=True)
pd.DataFrame(rows).to_csv(OUT, index=False)
print(f"Wrote {len(rows)} active claims to {OUT}")
