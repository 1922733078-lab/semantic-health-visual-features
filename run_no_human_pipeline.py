#!/usr/bin/env python3
"""
No-human-evaluation pipeline.

Runs the full label-free computational analysis end-to-end:
  1. Compute deterministic proxy metrics and canonical data artifacts.
  2. Benign-transformation stability analysis.
  3. Controlled directional-sensitivity analysis.
  4. Structural discriminability and cross-category generalization.
  5. Independent computational-agreement analyses.
  6. Robustness/ablation, explainability audit, and efficiency benchmark.

Usage:
    python run_no_human_pipeline.py
    python run_no_human_pipeline.py --clean-output
"""

import argparse
import json
import os
import platform
import shutil
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent
OUTPUT_ROOT = ROOT / "results" / "no_human"
RUNS_ROOT = OUTPUT_ROOT / "runs"
SRC_NO_HUMAN = ROOT / "src" / "no_human"

SCRIPTS = [
    ("compute_no_human_metrics.py", "Step 4-7: metrics and canonical artifacts"),
    ("run_stability_analysis.py", "Step 9: benign-transformation stability"),
    ("run_sensitivity_analysis.py", "Step 10: directional sensitivity"),
    ("run_generalization_analysis.py", "Steps 11-12: generalization"),
    ("run_independent_reference_analysis.py", "Step 13: independent references"),
    ("run_robustness_explainability_efficiency.py", "Steps 14-16: robustness/explainability/efficiency"),
]


def get_environment_info():
    try:
        import importlib.metadata as importlib_metadata
    except ImportError:
        import importlib_metadata

    packages = ["numpy", "pandas", "scipy", "matplotlib", "seaborn", "scikit-learn",
                "scikit-image", "opencv-python", "torch", "torchvision", "transformers",
                "xgboost", "lightgbm", "shap", "statsmodels", "pingouin", "tqdm",
                "Pillow", "umap-learn", "colorspacious", "easyocr"]
    versions = {}
    for pkg in packages:
        try:
            versions[pkg] = importlib_metadata.version(pkg)
        except Exception:
            versions[pkg] = "not_installed"
    return {
        "python_version": platform.python_version(),
        "os": platform.platform(),
        "cpu": platform.processor(),
        "machine": platform.machine(),
        "package_versions": versions,
        "timestamp": datetime.now().isoformat(),
    }


def write_environment_info(output_root):
    env = get_environment_info()
    (output_root / "manifests").mkdir(parents=True, exist_ok=True)
    with open(output_root / "manifests" / "environment.txt", "w", encoding="utf-8") as f:
        f.write(f"# Environment captured at {env['timestamp']}\n")
        f.write(f"python_version: {env['python_version']}\n")
        f.write(f"os: {env['os']}\n")
        f.write(f"cpu: {env['cpu']}\n")
        f.write(f"machine: {env['machine']}\n\n")
        f.write("package_versions:\n")
        for pkg, ver in env["package_versions"].items():
            f.write(f"  {pkg}: {ver}\n")
    return env


def hash_outputs(output_root):
    import hashlib
    rows = []
    for path in sorted(output_root.rglob("*")):
        if path.is_file():
            try:
                sha = hashlib.sha256(path.read_bytes()).hexdigest()
                rows.append({"path": str(path.relative_to(output_root)), "sha256": sha})
            except Exception:
                pass
    df = __import__("pandas").DataFrame(rows)
    df.to_csv(output_root / "manifests" / "output_hashes.csv", index=False)


def run_script(script_name, output_root, log_file, config_path=None):
    script_path = SRC_NO_HUMAN / script_name
    if not script_path.exists():
        print(f"  [SKIP] {script_path} not found")
        return False

    print(f"\n>> Running {script_name}")
    start = time.time()
    env = os.environ.copy()
    env["PYTHONPATH"] = str(ROOT)
    env["NO_HUMAN_OUTPUT_ROOT"] = str(output_root)
    if config_path is not None:
        env["NO_HUMAN_CONFIG_PATH"] = str(config_path)
    try:
        result = subprocess.run(
            [sys.executable, str(script_path)],
            cwd=ROOT,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            check=True,
        )
        elapsed = time.time() - start
        log_file.write(f"\n=== {script_name} ===\n")
        log_file.write(result.stdout)
        log_file.write(f"\nElapsed: {elapsed:.1f}s\n")
        log_file.write("STATUS: SUCCESS\n")
        print(f"  [OK] {script_name} in {elapsed:.1f}s")
        return True
    except subprocess.CalledProcessError as e:
        elapsed = time.time() - start
        log_file.write(f"\n=== {script_name} ===\n")
        log_file.write(e.stdout)
        log_file.write(f"\nElapsed: {elapsed:.1f}s\n")
        log_file.write("STATUS: FAILED\n")
        print(f"  [FAIL] {script_name} in {elapsed:.1f}s")
        return False


def snapshot_directory(path, exclude_prefixes=None):
    """Return a set of relative file paths existing under path."""
    files = set()
    exclude_prefixes = exclude_prefixes or []
    if path.exists():
        for p in path.rglob("*"):
            if p.is_file():
                rel = str(p.relative_to(path))
                if any(rel.startswith(pref) for pref in exclude_prefixes):
                    continue
                files.add(rel)
    return files


def main():
    parser = argparse.ArgumentParser(description="Run the no-human-evaluation pipeline")
    parser.add_argument("--clean-output", action="store_true",
                        help="Write results to a fresh timestamped subdirectory")
    parser.add_argument("--config", type=Path, default=None,
                        help="Path to no_human_design.yaml (default: configs/no_human_design.yaml)")
    args = parser.parse_args()

    output_root = OUTPUT_ROOT
    config_path = args.config or ROOT / "configs" / "no_human_design.yaml"

    canonical_snapshot = None
    if args.clean_output:
        timestamp = datetime.now(datetime.now().astimezone().tzinfo).strftime("%Y%m%d_%H%M%S")
        output_root = RUNS_ROOT / f"run_{timestamp}"
        output_root.mkdir(parents=True, exist_ok=True)
        canonical_snapshot = snapshot_directory(OUTPUT_ROOT, exclude_prefixes=["runs/"])
        print(f"Clean output mode: results will be written to {output_root}")
        print(f"Canonical directory snapshot captured; any new file in {OUTPUT_ROOT} will fail the run.")
    else:
        output_root.mkdir(parents=True, exist_ok=True)

    (output_root / "logs").mkdir(parents=True, exist_ok=True)
    (output_root / "manifests").mkdir(parents=True, exist_ok=True)

    log_path = output_root / "logs" / "clean_environment_reproduction.log"
    with open(log_path, "w", encoding="utf-8") as log_file:
        log_file.write(f"No-human pipeline started at {datetime.now().isoformat()}\n")
        log_file.write(f"Output root: {output_root}\n")
        log_file.write(f"Config path: {config_path}\n")
        log_file.write(f"Python: {sys.executable}\n\n")

        env = write_environment_info(output_root)
        log_file.write(f"Environment: {json.dumps(env, indent=2)}\n\n")

        # Load and, in clean mode, rewrite config to point to the run directory
        with open(config_path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)
        if args.clean_output:
            config["paths"]["output_root"] = str(output_root.relative_to(ROOT))
            config["paths"]["manifests"] = str((output_root / "manifests").relative_to(ROOT))
            config["paths"]["quality_control"] = str((output_root / "quality_control").relative_to(ROOT))
            config["paths"]["tables"] = str((output_root / "tables").relative_to(ROOT))
            config["paths"]["figures"] = str((output_root / "figures").relative_to(ROOT))
            config["paths"]["data"] = str((output_root / "data").relative_to(ROOT))
            config["paths"]["logs"] = str((output_root / "logs").relative_to(ROOT))
            run_config = output_root / "manifests" / "no_human_design_run.yaml"
            with open(run_config, "w", encoding="utf-8") as f:
                yaml.safe_dump(config, f, sort_keys=False, allow_unicode=True)
            config_path = run_config

        all_ok = True
        for script_name, description in SCRIPTS:
            log_file.write(f"\n# {description}\n")
            ok = run_script(script_name, output_root, log_file, config_path=config_path)
            if not ok:
                all_ok = False
                if script_name in ["compute_no_human_metrics.py"]:
                    print("Critical step failed; aborting pipeline.")
                    log_file.write("CRITICAL FAILURE: aborting pipeline.\n")
                    sys.exit(1)

        # Record output hashes
        try:
            hash_outputs(output_root)
            log_file.write("\nOutput hashes written to manifests/output_hashes.csv\n")
        except Exception as e:
            log_file.write(f"\nFailed to hash outputs: {e}\n")

        # Clean-output isolation check
        if args.clean_output and canonical_snapshot is not None:
            after_snapshot = snapshot_directory(OUTPUT_ROOT, exclude_prefixes=["runs/"])
            new_files = after_snapshot - canonical_snapshot
            if new_files:
                msg = "Clean-output isolation violated: the following new files appeared in the canonical output directory:\n"
                for fp in sorted(new_files):
                    msg += f"  - {fp}\n"
                log_file.write(f"\n{msg}")
                print(msg)
                sys.exit(3)
            else:
                log_file.write("\nClean-output isolation check passed.\n")

        # Promote successful clean run to current symlink
        if args.clean_output and all_ok:
            current_link = OUTPUT_ROOT / "current"
            if current_link.exists() or current_link.is_symlink():
                current_link.unlink()
            current_link.symlink_to(output_root.relative_to(OUTPUT_ROOT), target_is_directory=True)
            log_file.write(f"\nPromoted {output_root} to {current_link}\n")

        log_file.write(f"\nPipeline finished at {datetime.now().isoformat()}\n")
        log_file.write(f"Overall status: {'SUCCESS' if all_ok else 'PARTIAL_FAILURE'}\n")

    print(f"\nPipeline complete. Outputs in {output_root}")
    if not all_ok:
        print("Some non-critical steps failed; see log for details.")
        sys.exit(2)


if __name__ == "__main__":
    main()
