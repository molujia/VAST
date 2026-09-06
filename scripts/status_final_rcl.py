#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = REPO_ROOT / "src"
for import_root in (REPO_ROOT, SOURCE_ROOT):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

from rcl_study.final_rcl_execution import (  # noqa: E402
    FinalRCLExecutionError,
    read_final_run_status,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Report durable progress for the final six-unit RCL run."
    )
    parser.add_argument("--run-root", required=True, type=Path)
    parser.add_argument(
        "--tmux-alive",
        choices=("auto", "true", "false"),
        default="auto",
        help="Use auto on Linux, or override for tests/remote diagnostics.",
    )
    return parser


def _tmux_alive(run_root: Path, mode: str) -> bool:
    if mode != "auto":
        return mode == "true"
    manifest = json.loads(
        (Path(run_root) / "run-manifest.json").read_text(encoding="utf-8")
    )
    session = str(manifest.get("tmux_session", ""))
    if not session:
        return False
    try:
        completed = subprocess.run(
            ("tmux", "has-session", "-t", session),
            check=False,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    except OSError:
        return False
    return completed.returncode == 0


def main() -> int:
    args = _parser().parse_args()
    try:
        alive = _tmux_alive(args.run_root, args.tmux_alive)
        status = read_final_run_status(run_root=args.run_root, tmux_alive=alive)
    except (FinalRCLExecutionError, OSError, ValueError) as exc:
        print(str(exc), file=sys.stderr)
        return 2
    print(json.dumps(status, allow_nan=False, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
