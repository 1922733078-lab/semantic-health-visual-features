#!/usr/bin/env python3
"""
Step 2 — Audit Human-Rating Provenance
Reconcile real_human_ratings.csv against the 30 per-rater Excel workbooks.
"""
import csv
import re
from collections import Counter
from pathlib import Path

import openpyxl

ROOT = Path(__file__).resolve().parents[2]
RATING_DIR = ROOT / "盲评问卷" / "ratings"
QC = ROOT / "results" / "redesign" / "quality_control"
DOC = ROOT / "data" / "documentation"
QC.mkdir(parents=True, exist_ok=True)
DOC.mkdir(parents=True, exist_ok=True)

DIMENSION_MAP = {
    "视觉复杂度": "complexity",
    "美感吸引力": "beauty",
    "秩序感": "order",
    "视觉层级清晰度": "hierarchy",
    "情感强度": "emotion",
}
DIMENSION_COLS = [
    "视觉复杂度 (1=极简, 7=极繁)",
    "美感吸引力 (1=毫无美感, 7=非常有美感)",
    "秩序感 (1=完全混乱, 7=非常有序)",
    "视觉层级清晰度 (1=分不清主次, 7=主次分明)",
    "情感强度 (1=平淡无感, 7=印象深刻)",
]


def load_csv_ratings():
    path = RATING_DIR / "real_human_ratings.csv"
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            rows.append({
                "image_id": r["image_id"],
                "category": r["category"],
                "rater_id": r["rater_id"],
                "dimension_en": DIMENSION_MAP[r["dimension"]],
                "dimension_cn": r["dimension"],
                "rating": int(r["rating"]),
            })
    return rows


def load_excel_ratings():
    all_rows = []
    for i in range(1, 31):
        rater_id = f"rater_{i:02d}"
        path = RATING_DIR / f"评分表_{rater_id}.xlsx"
        wb = openpyxl.load_workbook(path, read_only=True, data_only=True)
        ws = wb["评分表"]
        header = None
        for idx, row in enumerate(ws.iter_rows(values_only=True)):
            if idx == 0:
                header = row
                continue
            if row[0] is None:
                continue
            image_id = row[1]
            category = row[2]
            for col_idx, dim_cn_full in enumerate(DIMENSION_COLS, start=4):
                dim_en = DIMENSION_MAP[dim_cn_full.split()[0]]
                all_rows.append({
                    "image_id": image_id,
                    "category": category,
                    "rater_id": rater_id,
                    "dimension_en": dim_en,
                    "dimension_cn": dim_cn_full.split()[0],
                    "rating": int(row[col_idx]),
                })
    return all_rows


def build_key(r):
    return (r["image_id"], r["rater_id"], r["dimension_en"])


def reconcile(csv_rows, excel_rows):
    csv_dict = {build_key(r): r for r in csv_rows}
    excel_dict = {build_key(r): r for r in excel_rows}

    only_csv = sorted(csv_dict.keys() - excel_dict.keys())
    only_excel = sorted(excel_dict.keys() - csv_dict.keys())

    mismatches = []
    for key in sorted(csv_dict.keys() & excel_dict.keys()):
        r_csv = csv_dict[key]
        r_xls = excel_dict[key]
        if r_csv["rating"] != r_xls["rating"]:
            mismatches.append({
                "image_id": key[0],
                "rater_id": key[1],
                "dimension": key[2],
                "csv_rating": r_csv["rating"],
                "excel_rating": r_xls["rating"],
            })

    return only_csv, only_excel, mismatches


def run_quality_checks(rows):
    checks = {
        "total_records": len(rows),
        "unique_raters": len(set(r["rater_id"] for r in rows)),
        "unique_images": len(set(r["image_id"] for r in rows)),
        "unique_dimensions": len(set(r["dimension_en"] for r in rows)),
        "min_rating": min(r["rating"] for r in rows),
        "max_rating": max(r["rating"] for r in rows),
        "missing_ratings": sum(1 for r in rows if r["rating"] is None),
        "integer_only": all(isinstance(r["rating"], int) for r in rows),
    }
    category_counts = Counter(r["category"] for r in rows)
    dim_counts = Counter(r["dimension_en"] for r in rows)
    return checks, category_counts, dim_counts


def main():
    csv_rows = load_csv_ratings()
    excel_rows = load_excel_ratings()

    only_csv, only_excel, mismatches = reconcile(csv_rows, excel_rows)
    checks, cat_counts, dim_counts = run_quality_checks(csv_rows)

    # Reconciliation report
    recon_path = QC / "rating_reconciliation.csv"
    with open(recon_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["check", "count", "details"])
        writer.writerow(["csv_total", len(csv_rows), ""])
        writer.writerow(["excel_total", len(excel_rows), ""])
        writer.writerow(["keys_only_in_csv", len(only_csv), ";".join(f"{k[0]}/{k[1]}/{k[2]}" for k in only_csv[:10])])
        writer.writerow(["keys_only_in_excel", len(only_excel), ";".join(f"{k[0]}/{k[1]}/{k[2]}" for k in only_excel[:10])])
        writer.writerow(["rating_mismatches", len(mismatches), ""])
        for m in mismatches[:20]:
            writer.writerow(["mismatch", "", f"{m['image_id']}/{m['rater_id']}/{m['dimension']}: csv={m['csv_rating']} excel={m['excel_rating']}"])
    print(f"Wrote {recon_path}")

    # Missing and range checks
    missing_path = QC / "missing_and_range_checks.csv"
    with open(missing_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["metric", "value"])
        for k, v in checks.items():
            writer.writerow([k, v])
        for cat, cnt in sorted(cat_counts.items()):
            writer.writerow([f"records_category_{cat}", cnt])
        for dim, cnt in sorted(dim_counts.items()):
            writer.writerow([f"records_dimension_{dim}", cnt])
    print(f"Wrote {missing_path}")

    # Provenance document
    provenance_path = DOC / "human_rating_provenance.md"
    lines = [
        "# Human Rating Provenance Record\n",
        "## Collection summary\n",
        f"- Total individual ratings: {len(csv_rows)}",
        f"- Raters: {checks['unique_raters']}",
        f"- Images: {checks['unique_images']}",
        f"- Dimensions: {checks['unique_dimensions']}",
        f"- Categories: {len(cat_counts)}",
        f"- Rating scale: {checks['min_rating']} to {checks['max_rating']}",
        "\n## File inventory\n",
        "| File | Role | Note |",
        "|---|---|---|",
        "| `盲评问卷/ratings/real_human_ratings.csv` | Canonical long-format rating file | Aggregated from per-rater workbooks |",
    ]
    for i in range(1, 31):
        lines.append(f"| `盲评问卷/ratings/评分表_rater_{i:02d}.xlsx` | Processed per-rater workbook | Batch-generated structured workbook |")
    lines.append("| `盲评问卷/盲评问卷.xlsx` | Questionnaire template / protocol | Source of anchors and instructions |\n")

    lines.extend([
        "## Reconciliation results\n",
        f"- Records in CSV: {len(csv_rows)}",
        f"- Records reconstructed from Excel: {len(excel_rows)}",
        f"- Keys only in CSV: {len(only_csv)}",
        f"- Keys only in Excel: {len(only_excel)}",
        f"- Rating mismatches: {len(mismatches)}",
        "\n## Quality checks\n",
        f"- Integer ratings only: {checks['integer_only']}",
        f"- Missing ratings: {checks['missing_ratings']}",
        f"- Expected structure (30 raters × 100 images × 5 dimensions = 15,000): {len(csv_rows) == 15000}",
        "\n## Limitations\n",
        "- The earliest participant-submitted raw files are not available in this folder; only the processed per-rater workbooks and the aggregated CSV are retained.",
        "- If original platform exports or participant-submitted files exist elsewhere, they should be added to `data/documentation/human_rating_provenance.md` and hashed.",
        "\n",
    ])
    provenance_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {provenance_path}")

    if only_csv or only_excel or mismatches:
        print("WARNING: Reconciliation found discrepancies. See rating_reconciliation.csv.")
    else:
        print("Reconciliation: CSV and Excel workbooks match exactly.")

    print("Step 2 complete.")


if __name__ == "__main__":
    main()
