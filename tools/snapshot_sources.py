#!/usr/bin/env python3
"""Build an auditable, data-free snapshot of the selected VAST implementation."""

from __future__ import annotations

import ast
import hashlib
import shutil
from collections import deque
from pathlib import Path, PurePosixPath
from typing import Iterable


INTERNAL_PACKAGES = frozenset(
    {"rcl_study", "scripts", "fixed_active_learning", "nexusrcl_rebuild"}
)
FORBIDDEN_PARTS = frozenset(
    {
        "reference",
        "verification",
        "reproduced",
        "scores",
        "candidate_pools",
        "query_plans",
        "feature_inventories",
        "outputs",
        "output",
        "data",
        "datasets",
        "__pycache__",
        ".pytest_cache",
    }
)
FORBIDDEN_SUFFIXES = frozenset({".pyc", ".pyo", ".pt", ".pth", ".ckpt"})


class SnapshotError(ValueError):
    """Raised when the requested snapshot is incomplete or unsafe."""


def _relative_path(value: str | Path) -> PurePosixPath:
    text = str(value).replace("\\", "/")
    path = PurePosixPath(text)
    if path.is_absolute() or ".." in path.parts or not path.parts:
        raise SnapshotError(f"snapshot path must be relative and contained: {value}")
    return path


def _assert_allowed(relative_path: PurePosixPath) -> None:
    lowered = {part.lower() for part in relative_path.parts}
    forbidden = sorted(lowered.intersection(FORBIDDEN_PARTS))
    if forbidden or relative_path.suffix.lower() in FORBIDDEN_SUFFIXES:
        reason = forbidden[0] if forbidden else relative_path.suffix
        raise SnapshotError(f"forbidden snapshot path ({reason}): {relative_path}")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _module_candidates(module_name: str) -> tuple[str, str]:
    stem = module_name.replace(".", "/")
    return f"{stem}.py", f"{stem}/__init__.py"


def _absolute_import_name(
    node: ast.ImportFrom, *, current_relative_path: str
) -> str | None:
    if node.level == 0:
        return node.module
    package_parts = list(PurePosixPath(current_relative_path).with_suffix("").parts[:-1])
    ascend = node.level - 1
    if ascend > len(package_parts):
        raise SnapshotError(
            f"invalid relative import in {current_relative_path}: level {node.level}"
        )
    base = package_parts[: len(package_parts) - ascend]
    if node.module:
        base.extend(node.module.split("."))
    return ".".join(base)


def _internal_imports(path: Path, relative_path: str) -> tuple[str, ...]:
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    except (OSError, UnicodeError, SyntaxError) as exc:
        raise SnapshotError(f"cannot parse Python source {relative_path}: {exc}") from exc
    modules: set[str] = set()
    for node in ast.walk(tree):
        names: Iterable[str]
        if isinstance(node, ast.Import):
            names = (alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            name = _absolute_import_name(node, current_relative_path=relative_path)
            names = () if name is None else (name,)
        else:
            continue
        for name in names:
            if name.split(".", 1)[0] in INTERNAL_PACKAGES:
                modules.add(name)
    return tuple(sorted(modules))


def collect_python_closure(
    source_root: Path,
    *,
    roots: Iterable[str],
    runtime_roots: Iterable[str] = (),
) -> tuple[str, ...]:
    """Return the deterministic internal import closure for declared Python roots."""

    source_root = source_root.resolve()
    queue: deque[str] = deque(
        str(_relative_path(item)) for item in (*tuple(roots), *tuple(runtime_roots))
    )
    selected: set[str] = set()
    while queue:
        relative = queue.popleft()
        if relative in selected:
            continue
        relative_path = _relative_path(relative)
        _assert_allowed(relative_path)
        source = source_root.joinpath(*relative_path.parts)
        if not source.is_file():
            raise SnapshotError(f"missing snapshot root: {relative}")
        selected.add(str(relative_path))
        for module_name in _internal_imports(source, str(relative_path)):
            candidates = _module_candidates(module_name)
            dependency = next(
                (
                    candidate
                    for candidate in candidates
                    if source_root.joinpath(*PurePosixPath(candidate).parts).is_file()
                ),
                None,
            )
            if dependency is None:
                raise SnapshotError(
                    f"missing internal dependency {module_name!r} imported by {relative}"
                )
            queue.append(dependency)
    return tuple(sorted(selected))


def copy_snapshot_file(
    source_root: Path,
    destination_root: Path,
    relative_source_path: str,
    *,
    destination_relative_path: str | None = None,
    role: str = "source",
) -> dict[str, str]:
    """Copy one allowed file and return its stable provenance row."""

    source_relative = _relative_path(relative_source_path)
    destination_relative = _relative_path(
        destination_relative_path or relative_source_path
    )
    _assert_allowed(source_relative)
    _assert_allowed(destination_relative)
    source = source_root.resolve().joinpath(*source_relative.parts)
    destination = destination_root.resolve().joinpath(*destination_relative.parts)
    if not source.is_file():
        raise SnapshotError(f"snapshot source does not exist: {source_relative}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, destination)
    source_sha = _sha256(source)
    destination_sha = _sha256(destination)
    if source_sha != destination_sha:
        raise SnapshotError(f"copied file hash mismatch: {source_relative}")
    return {
        "source_path": str(source_relative),
        "destination_path": str(destination_relative),
        "source_sha256": source_sha,
        "destination_sha256": destination_sha,
        "role": role,
    }
