#!/usr/bin/env python3
"""Build the v5/v6 Route-C canonical-to-promoted artifact register."""

import csv
import hashlib
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/no_human/quality_control/route_c_canonical_artifact_register.csv"
SCOPE = "Route C semantic-health two-renderer failure analysis"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def dir_sha(path: Path) -> str:
    digest = hashlib.sha256()
    for file in sorted(p for p in path.rglob("*") if p.is_file()):
        rel = file.relative_to(path).as_posix()
        if "__pycache__" in file.parts or ".pytest_cache" in file.parts or file.suffix == ".pyc":
            continue
        digest.update(rel.encode("utf-8"))
        digest.update(file.read_bytes())
    return digest.hexdigest()


SPECS = [
    ("main_latex_source", "paper/main_no_human.tex", "manuscript_source/main_no_human.tex", "manuscript"),
    ("bibliography", "paper/references.bib", "manuscript_source/references.bib", "manuscript"),
    ("manuscript_sections", "paper/sections_no_human", "manuscript_source/sections_no_human", "manuscript"),
    ("pdf_output", "paper/main_no_human.pdf", "manuscript_output/main_no_human.pdf", "manuscript"),
    ("docx_output", "paper/main_no_human.docx", "manuscript_output/main_no_human.docx", "manuscript"),
    ("highlights", "paper/highlights_no_human.txt", "documents/highlights_no_human.txt", "manuscript"),
    ("cover_letter", "paper/cover_letter_no_human.md", "documents/cover_letter_no_human.md", "manuscript"),
    ("canonical_audit", "results/no_human/pre_submission_audit.md", "audits/pre_submission_audit.md", "audit"),
    ("claim_ledger", "results/no_human/quality_control/claim_evidence_ledger.csv", "quality_control/claim_evidence_ledger.csv", "audit"),
    ("corrective_issue_ledger", "results/no_human/quality_control/jvcir_corrective_issue_ledger.csv", "quality_control/jvcir_corrective_issue_ledger.csv", "audit"),
    ("feature_dictionary", "docs/FEATURE_DATA_DICTIONARY.md", "documents/FEATURE_DATA_DICTIONARY.md", "v5"),
    ("data_availability", "results/no_human/tier_b_release/docs/DATA_AVAILABILITY.md", "documents/DATA_AVAILABILITY.md", "release"),
    ("no_human_scope", "results/no_human/tier_b_release/docs/NO_HUMAN_SCOPE.md", "documents/NO_HUMAN_SCOPE.md", "release"),
    ("reproduction_protocol", "results/no_human/tier_b_release/docs/REPRODUCTION_PROTOCOL.md", "documents/REPRODUCTION_PROTOCOL.md", "release"),
    ("restricted_source_scope", "results/no_human/tier_b_release/docs/RESTRICTED_MAIN_CORPUS.md", "documents/RESTRICTED_MAIN_CORPUS.md", "release"),
    ("protocol_v5", "synthetic_benchmark/PROTOCOL_V5_SEMANTIC_HEALTH.md", "tier_b_release/synthetic_benchmark/PROTOCOL_V5_SEMANTIC_HEALTH.md", "v5"),
    ("protocol_v5_script", "synthetic_benchmark/run_semantic_health_protocol_v5.py", "tier_b_release/synthetic_benchmark/run_semantic_health_protocol_v5.py", "v5"),
    ("protocol_v5_outputs", "results/no_human/runs/run_20260720_semantic_health_v5", "tier_b_release/results/no_human/runs/run_20260720_semantic_health_v5", "v5"),
    ("protocol_v5_clean_log", "results/no_human/tier_b_release/logs/verification_protocol_v5_clean.log", "tier_b_release/logs/verification_protocol_v5_clean.log", "v5"),
    ("protocol_v6", "synthetic_benchmark/PROTOCOL_V6_EXTERNAL_HOLDOUT.md", "tier_b_release/synthetic_benchmark/PROTOCOL_V6_EXTERNAL_HOLDOUT.md", "v6"),
    ("protocol_v6_script", "synthetic_benchmark/run_external_holdout_protocol_v6.py", "tier_b_release/synthetic_benchmark/run_external_holdout_protocol_v6.py", "v6"),
    ("protocol_v6_outputs", "results/no_human/runs/run_20260720_external_protocol_v6", "tier_b_release/results/no_human/runs/run_20260720_external_protocol_v6", "v6"),
    ("protocol_v6_clean_log", "results/no_human/tier_b_release/logs/verification_protocol_v6_clean.log", "tier_b_release/logs/verification_protocol_v6_clean.log", "v6"),
    ("feature_extractor", "src/features/traditional_features.py", "tier_b_release/src/features/traditional_features.py", "v5"),
    ("semantic_health_library", "src/no_human/semantic_health.py", "tier_b_release/src/no_human/semantic_health.py", "v5"),
    ("frozen_splits", "synthetic_benchmark/splits/frozen_splits.csv", "tier_b_release/synthetic_benchmark/splits/frozen_splits.csv", "release"),
    ("requirements_lock", "requirements-lock.txt", "tier_b_release/requirements-lock.txt", "release"),
    ("tier_b_release", "results/no_human/tier_b_release", "tier_b_release", "release"),
    ("figure_benchmark_examples", "paper/figures_route_c/benchmark_examples.png", "figures/benchmark_examples.png", "manuscript"),
    ("figure_weight_retention", "results/no_human/runs/run_20260720_semantic_health_v5/semantic_health_weight_retention.png", "figures/semantic_health_weight_retention.png", "v5"),
    ("figure_pre_post", "results/no_human/runs/run_20260720_semantic_health_v5/pre_post_nominal_correlations.png", "figures/pre_post_nominal_correlations.png", "v5"),
    ("figure_score_collapse", "results/no_human/runs/run_20260720_external_protocol_v6/external_score_boundary_collapse.png", "figures/external_score_boundary_collapse.png", "v6"),
]


def main() -> None:
    now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    rows = []
    for role, source_rel, promoted_rel, group in SPECS:
        source = ROOT / source_rel
        if not source.exists():
            raise FileNotFoundError(source)
        rows.append({
            "role": role,
            "canonical_path": source_rel,
            "promoted_path": promoted_rel,
            "expected_sha256": (
                sha(source / "manifests/release_manifest.csv")
                if role == "tier_b_release"
                else dir_sha(source) if source.is_dir() else sha(source)
            ),
            "run_id": {
                "v5": "run_20260720_semantic_health_v5",
                "v6": "run_20260720_external_protocol_v6",
                "release": "tier_b_release_20260720_v3",
                "audit": "route_c_20260720_round7_candidate",
                "manuscript": "route_c_20260720_round7_candidate",
            }[group],
            "protocol_version": {"v5": "5.0", "v6": "6.0"}.get(group, "Route-C-v3"),
            "scope": SCOPE,
            "status": (
                "verified_manifest" if role == "tier_b_release"
                else "verified_directory" if source.is_dir() else "verified"
            ),
            "verified_utc": now,
        })
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {len(rows)} canonical Route-C roles to {OUT}")


if __name__ == "__main__":
    main()
