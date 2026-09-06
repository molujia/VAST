#!/usr/bin/env python3
"""Reject data, result, credential, and machine-path leakage from VAST."""

from __future__ import annotations

import argparse
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Iterable


FORBIDDEN_DIRECTORY_PARTS = frozenset(
    {
        "data",
        "datasets",
        "outputs",
        "output",
        "results",
        "scores",
        "candidate_pools",
        "query_plans",
        "feature_inventories",
        "checkpoints",
        "__pycache__",
        ".pytest_cache",
        ".pytest_tmp",
    }
)
FORBIDDEN_SUFFIXES = frozenset(
    {
        ".ckpt",
        ".csv",
        ".h5",
        ".hdf5",
        ".joblib",
        ".npy",
        ".npz",
        ".parquet",
        ".pickle",
        ".pkl",
        ".pt",
        ".pth",
        ".pyc",
        ".pyo",
        ".xlsx",
    }
)
TEXT_SUFFIXES = frozenset(
    {
        "",
        ".cfg",
        ".ini",
        ".json",
        ".md",
        ".py",
        ".rst",
        ".sh",
        ".toml",
        ".txt",
        ".yaml",
        ".yml",
    }
)
SECRET_PATTERNS = (
    re.compile(r"\bgh[pousr]_[A-Za-z0-9]{20,}\b"),
    re.compile(r"\bgithub_pat_[A-Za-z0-9_]{20,}\b"),
    re.compile(r"-----BEGIN (?:OPENSSH |RSA |EC |DSA )?PRIVATE KEY-----"),
    re.compile(
        r"(?i)\b(?:api[_-]?key|access[_-]?token|password|secret)\s*[:=]\s*"
        r"['\"]?(?!\$\{|<|example|changeme|none|null)[^\s'\"]{12,}"
    ),
)
ABSOLUTE_PATH_PATTERNS = (
    re.compile(r"(?i)(?<![\\/A-Za-z0-9_])[A-Z]:[\\/](?:Users|Documents and Settings)[\\/][^\s'\"`]+"),
    re.compile(r"(?<![A-Za-z0-9_])/(?:home|Users)/[A-Za-z0-9_.-]+(?:/[^\s'\"`]+)?"),
)


@dataclass(frozen=True, order=True)
class Finding:
    kind: str
    path: str
    detail: str


def _git_visible_files(root: Path) -> tuple[Path, ...] | None:
    if not (root / ".git").exists():
        return None
    completed = subprocess.run(
        [
            "git",
            "ls-files",
            "--cached",
            "--others",
            "--exclude-standard",
            "-z",
        ],
        cwd=root,
        check=False,
        capture_output=True,
    )
    if completed.returncode != 0:
        return None
    return tuple(
        root / item.decode("utf-8", errors="surrogateescape")
        for item in completed.stdout.split(b"\0")
        if item
    )


def _candidate_files(root: Path) -> Iterable[Path]:
    git_files = _git_visible_files(root)
    if git_files is not None:
        yield from git_files
        return
    for path in root.rglob("*"):
        if ".git" in path.relative_to(root).parts:
            continue
        if path.is_file():
            yield path


def _path_finding(relative: PurePosixPath) -> Finding | None:
    directory_parts = {part.lower() for part in relative.parts[:-1]}
    forbidden_parts = sorted(directory_parts.intersection(FORBIDDEN_DIRECTORY_PARTS))
    suffix = relative.suffix.lower()
    if forbidden_parts:
        return Finding("forbidden_path", str(relative), forbidden_parts[0])
    if suffix in FORBIDDEN_SUFFIXES:
        return Finding("forbidden_path", str(relative), suffix)
    return None


def _text_findings(path: Path, relative: PurePosixPath) -> list[Finding]:
    if path.suffix.lower() not in TEXT_SUFFIXES or path.stat().st_size > 2_000_000:
        return []
    try:
        text = path.read_text(encoding="utf-8")
    except (OSError, UnicodeError):
        return [Finding("unreadable_text", str(relative), "not valid UTF-8")]
    findings: list[Finding] = []
    for pattern in SECRET_PATTERNS:
        if pattern.search(text):
            findings.append(Finding("secret", str(relative), pattern.pattern))
            break
    for pattern in ABSOLUTE_PATH_PATTERNS:
        if pattern.search(text):
            findings.append(Finding("absolute_path", str(relative), pattern.pattern))
            break
    return findings


def scan_repository(root: Path) -> list[Finding]:
    """Return deterministic findings for all Git-visible repository files."""

    root = root.resolve()
    findings: list[Finding] = []
    for path in _candidate_files(root):
        try:
            relative = PurePosixPath(path.resolve().relative_to(root).as_posix())
        except (OSError, ValueError):
            findings.append(Finding("path_escape", str(path), "outside repository"))
            continue
        path_problem = _path_finding(relative)
        if path_problem is not None:
            findings.append(path_problem)
            continue
        findings.extend(_text_findings(path, relative))
    return sorted(set(findings))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", nargs="?", type=Path, default=Path.cwd())
    args = parser.parse_args()
    findings = scan_repository(args.root)
    if findings:
        for finding in findings:
            print(f"{finding.kind}: {finding.path}: {finding.detail}")
        return 1
    print("repository firewall passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
