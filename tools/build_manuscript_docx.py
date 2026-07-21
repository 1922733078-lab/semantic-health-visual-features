#!/usr/bin/env python3
"""Build the complete submission DOCX from the canonical LaTeX manuscript."""

from __future__ import annotations

import argparse
from pathlib import Path
import re
import shutil
import subprocess
import tempfile


ROOT = Path(__file__).resolve().parents[1]
PAPER = ROOT / "paper"
PANDOC = Path("/Users/xuezihang/.local/bin/pandoc")
BUNDLED_PYTHON = Path(
    "/Users/xuezihang/.cache/codex-runtimes/codex-primary-runtime/"
    "dependencies/python/bin/python3"
)


def strip_resizebox_wrappers(path: Path) -> None:
    """Expose wide LaTeX tabular environments to Pandoc in the temporary tree."""
    lines = path.read_text(encoding="utf-8").splitlines()
    cleaned: list[str] = []
    skip_closing_brace = False
    for line in lines:
        if line.strip() == r"\resizebox{\textwidth}{!}{%":
            skip_closing_brace = True
            continue
        if skip_closing_brace and line.strip() == "}" and cleaned:
            if cleaned[-1].strip() == r"\end{tabular}":
                skip_closing_brace = False
                continue
        cleaned.append(line)
    if skip_closing_brace:
        raise RuntimeError("Unmatched resizebox wrapper in generated_tables_v5.tex")
    path.write_text("\n".join(cleaned) + "\n", encoding="utf-8")


def normalise_pandoc_tex(path: Path) -> None:
    """Replace LaTeX macros that Pandoc otherwise drops from visible text."""
    text = path.read_text(encoding="utf-8")
    text = re.sub(r"\\path\{([^{}]*)\}", r"\\texttt{\1}", text)
    path.write_text(text, encoding="utf-8")


def build(output: Path) -> None:
    if not PANDOC.exists():
        raise FileNotFoundError(PANDOC)
    if not BUNDLED_PYTHON.exists():
        raise FileNotFoundError(BUNDLED_PYTHON)

    with tempfile.TemporaryDirectory(prefix="route-c-docx-") as tmp:
        temporary_paper = Path(tmp) / "paper"
        shutil.copytree(PAPER, temporary_paper)
        for tex_path in temporary_paper.rglob("*.tex"):
            normalise_pandoc_tex(tex_path)
        strip_resizebox_wrappers(
            temporary_paper / "sections_no_human" / "generated_tables_v5.tex"
        )
        raw = Path(tmp) / "main_no_human.docx"
        resource_paths = [
            temporary_paper,
            temporary_paper / "figures_route_c",
            ROOT / "results/no_human/runs/run_20260720_semantic_health_v5",
            ROOT / "results/no_human/runs/run_20260720_external_protocol_v6",
        ]
        subprocess.run(
            [
                str(PANDOC),
                "main_no_human.tex",
                "--from=latex",
                "--to=docx",
                "--citeproc",
                "--bibliography=references.bib",
                "--resource-path=" + ":".join(str(p) for p in resource_paths),
                "-o",
                str(raw),
            ],
            cwd=temporary_paper,
            check=True,
        )
        subprocess.run(
            [
                str(BUNDLED_PYTHON),
                str(ROOT / "tools/postprocess_manuscript_docx.py"),
                str(raw),
            ],
            check=True,
        )
        output.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(raw, output)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=PAPER / "main_no_human.docx",
    )
    args = parser.parse_args()
    build(args.output.resolve())
    print(f"Wrote {args.output.resolve()}")


if __name__ == "__main__":
    main()
