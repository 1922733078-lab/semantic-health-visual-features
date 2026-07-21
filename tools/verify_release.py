#!/usr/bin/env python3
"""
Thin wrapper around tools/verify_tier_b_release.py for documentation consistency.

Usage:
    python tools/verify_release.py --package . --mode tier-b
"""
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

import verify_tier_b_release  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description="Verify a Tier B public release package")
    parser.add_argument("--package", type=Path, required=True)
    parser.add_argument("--mode", default="tier-b", help="Unused; accepted for command parity with docs")
    args = parser.parse_args()

    sys.argv = ["verify_tier_b_release.py", "--package", str(args.package)]
    verify_tier_b_release.main()


if __name__ == "__main__":
    main()
