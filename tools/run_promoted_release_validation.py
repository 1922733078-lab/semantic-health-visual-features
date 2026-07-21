#!/usr/bin/env python3
"""Run and record the final protocol-v5/v6 Route-C validation sequence."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import os
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
PACKAGE = ROOT.parent / "99-投稿包/no_human"
TIER_B = PACKAGE / "tier_b_release"
LOG_DIR = PACKAGE / "logs"
PYTHON = ROOT / "venv/bin/python"
LOCK = ROOT / "requirements-lock.txt"
RELEASE_RUN_ID = "route_c_20260720_round7_candidate"


@dataclass(frozen=True)
class Check:
    name: str
    cwd: Path
    command: tuple[str, ...]


def now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run(check: Check, env: dict[str, str]) -> tuple[int, Path]:
    log_path = LOG_DIR / f"final_validation_{check.name}.log"
    print(f"[{now()}] START {check.name}", flush=True)
    with log_path.open("w", encoding="utf-8") as log:
        process = subprocess.Popen(
            check.command, cwd=check.cwd, env=env,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, bufsize=1,
        )
        assert process.stdout is not None
        for line in process.stdout:
            log.write(line); print(line, end="", flush=True)
        code = process.wait()
    print(f"[{now()}] END {check.name} exit={code}", flush=True)
    return code, log_path


def main() -> None:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    checks = [
        Check("00_canonical_full_suite", ROOT,
              (str(PYTHON), "-m", "pytest", "-q", "-p", "no:cacheprovider")),
        Check("01_feature_semantic_health", ROOT,
              (str(PYTHON), "-m", "pytest", "tests/test_feature_semantic_health.py",
               "-q", "-p", "no:cacheprovider")),
        Check("02_manuscript_consistency", ROOT,
              (str(PYTHON), "-m", "pytest", "tests/test_manuscript_consistency.py",
               "-q", "-p", "no:cacheprovider")),
        Check("03_synthetic_integrity", ROOT,
              (str(PYTHON), "-m", "pytest", "tests/test_synthetic_benchmark_integrity.py",
               "-q", "-p", "no:cacheprovider")),
        Check("04_tier_b_suite", TIER_B,
              (str(PYTHON), "-m", "pytest", "-q", "-p", "no:cacheprovider")),
        Check("05_tier_b_verifier", ROOT,
              (str(PYTHON), "tools/verify_tier_b_release.py", "--package", str(TIER_B))),
        Check("06_submission_verifier", ROOT,
              (str(PYTHON), "tools/verify_submission_manifest.py", "--package", str(PACKAGE),
               "--canonical-root", str(ROOT))),
    ]
    env = os.environ.copy(); env["PYTHONDONTWRITEBYTECODE"] = "1"
    records = []
    for check in checks:
        started = now(); code, log_path = run(check, env); finished = now()
        records.append({
            "started_utc": started, "finished_utc": finished,
            "check": check.name, "working_directory": str(check.cwd),
            "command": " ".join(check.command), "exit_code": code,
            "log_path": log_path.relative_to(PACKAGE).as_posix(),
            "lockfile_sha256": sha(LOCK), "release_run_id": RELEASE_RUN_ID,
        })
        if code != 0: break

    if len(records) == len(checks) and all(row["exit_code"] == 0 for row in records):
        # Intentional one-byte semantic-equivalent mutation: the verifier must
        # reject a payload whose manifest/canonical hash is stale, then pass
        # again after the exact original bytes are restored.
        target = PACKAGE / "documents/highlights_no_human.txt"
        original = target.read_bytes()
        target.write_bytes(original + b"\n")
        negative = Check(
            "07_negative_hash_substitution", ROOT,
            (str(PYTHON), "tools/verify_submission_manifest.py", "--package", str(PACKAGE),
             "--canonical-root", str(ROOT)),
        )
        started = now(); code, log_path = run(negative, env); finished = now()
        target.write_bytes(original)
        expected_rejection = code != 0
        records.append({
            "started_utc": started, "finished_utc": finished,
            "check": negative.name, "working_directory": str(ROOT),
            "command": " ".join(negative.command),
            "exit_code": 0 if expected_rejection else 1,
            "log_path": log_path.relative_to(PACKAGE).as_posix(),
            "lockfile_sha256": sha(LOCK), "release_run_id": RELEASE_RUN_ID,
        })
        final = Check(
            "08_post_restore_submission_verifier", ROOT,
            (str(PYTHON), "tools/verify_submission_manifest.py", "--package", str(PACKAGE),
             "--canonical-root", str(ROOT)),
        )
        started = now(); final_code, final_log = run(final, env); finished = now()
        records.append({
            "started_utc": started, "finished_utc": finished,
            "check": final.name, "working_directory": str(ROOT),
            "command": " ".join(final.command), "exit_code": final_code,
            "log_path": final_log.relative_to(PACKAGE).as_posix(),
            "lockfile_sha256": sha(LOCK), "release_run_id": RELEASE_RUN_ID,
        })

    record_path = LOG_DIR / "final_validation_command_record.csv"
    with record_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=records[0].keys())
        writer.writeheader(); writer.writerows(records)
    print(f"Wrote {record_path}")
    if len(records) != 9 or any(row["exit_code"] != 0 for row in records):
        sys.exit(1)


if __name__ == "__main__":
    main()
