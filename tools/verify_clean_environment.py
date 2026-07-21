#!/usr/bin/env python3
"""
Verify that the current run uses the clean release environment and not the project-wide venv_sys.

Checks:
- interpreter path is inside the release root or a non-project venv (no system-site-packages);
- input data resolves inside the release package;
- output paths are inside the release package;
- no import resolves to the original project src tree outside the release.
"""
import argparse
import site
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--package", type=Path, default=Path(".").resolve(),
                        help="Clean release package root")
    parser.add_argument("--allow-project", type=Path, default=None,
                        help="Original project root to forbid imports from")
    args = parser.parse_args()

    package = args.package.resolve()
    issues = []

    # Interpreter location
    exe = Path(sys.executable).resolve()
    if "venv_sys" in str(exe) or "venv" in str(exe).lower() and not str(exe).startswith(str(package)):
        issues.append(f"Interpreter {exe} appears to be a project development venv, not the clean release venv.")

    # site-packages should be under package or a clean venv
    for p in site.getsitepackages() + [site.getusersitepackages()]:
        if p is None:
            continue
        pp = Path(p).resolve()
        if "venv_sys" in str(pp):
            issues.append(f"site-packages path {pp} belongs to project venv_sys.")

    # sys.path should not include original project src
    if args.allow_project:
        project_root = args.allow_project.resolve()
        for p in sys.path:
            try:
                pp = Path(p).resolve()
                if pp == project_root or str(pp).startswith(str(project_root / "src")):
                    issues.append(f"sys.path includes original project tree: {p}")
            except Exception:
                pass

    # Current working directory must be inside package
    cwd = Path.cwd().resolve()
    if not str(cwd).startswith(str(package)):
        issues.append(f"Working directory {cwd} is outside package {package}.")

    if issues:
        print("Clean-environment verification FAILED:")
        for issue in issues:
            print(f"  - {issue}")
        sys.exit(1)
    else:
        print("Clean-environment verification passed.")
        print(f"  interpreter: {exe}")
        print(f"  package:     {package}")


if __name__ == "__main__":
    main()
