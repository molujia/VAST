#!/usr/bin/env python3
"""Build an auditable, data-free snapshot of the selected VAST implementation."""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import re
import shutil
from collections import deque
from datetime import date
from pathlib import Path, PurePosixPath
from typing import Iterable


INTERNAL_PACKAGES = frozenset({"rcl_study", "scripts"})
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


PYTHON_ROOTS = (
    "rcl_study/__init__.py",
    "rcl_study/final_rcl_contract.py",
    "rcl_study/final_rcl_evaluation.py",
    "rcl_study/final_rcl_execution.py",
    "rcl_study/final_rcl_hdbscan_proxy.py",
    "rcl_study/final_rcl_oser.py",
    "rcl_study/final_rcl_oser_runner.py",
    "rcl_study/final_rcl_pairwise.py",
    "rcl_study/final_rcl_real_execution.py",
    "rcl_study/final_rcl_training.py",
    "scripts/__init__.py",
    "scripts/run_final_rcl_formal.py",
    "scripts/run_final_rcl_real_smoke.py",
    "scripts/status_final_rcl.py",
)
RUNTIME_ROOTS = (
    "rcl_study/service_continuous_neural_runner.py",
    "rcl_study/service_continuous_real_ranker_smoke.py",
)
FOCUSED_TEST_NAMES = (
    "test_final_rcl_contract.py",
    "test_final_rcl_evaluation.py",
    "test_final_rcl_execution.py",
    "test_final_rcl_hdbscan_proxy.py",
    "test_final_rcl_neural_request.py",
    "test_final_rcl_oser.py",
    "test_final_rcl_oser_runner.py",
    "test_final_rcl_pairwise.py",
    "test_final_rcl_real_execution.py",
    "test_final_rcl_training.py",
    "test_run_final_rcl_formal_cli.py",
    "test_run_final_rcl_real_smoke_cli.py",
)
PAIRWISE_BACKEND_FILES = (
    "half_supervise/src/nexusrcl_rebuild/__init__.py",
    "half_supervise/src/nexusrcl_rebuild/training/__init__.py",
    "half_supervise/src/nexusrcl_rebuild/training/pairwise_backend.py",
)
FROZEN_CONFIG = "configs/final_rcl/hdbscan_proxy_cvae_oser_seed42.json"
INVENTORY_PATH = "docs/provenance/source-inventory.json"
_AUTHORITY_HOME = "/" + "home/" + "wangrunzhou"
MACHINE_PATH_REPLACEMENTS = (
    (
        _AUTHORITY_HOME + "/anaconda3/envs/scheme3_dgl_py312/bin/python",
        "${TORCH_PYTHON}",
    ),
    (
        _AUTHORITY_HOME + "/0_warlock/KIWI_OLD/NexusRCL_rebuild",
        "${NEXUSRCL_REBUILD_ROOT}",
    ),
    (_AUTHORITY_HOME + "/25挑战赛/aiopschallenge2025-main", "${AIOPS25_ROOT}"),
    (_AUTHORITY_HOME + "/0_warlock/0_RCAbench/data", "${RCABENCH_ROOT}"),
    (_AUTHORITY_HOME + "/dataset/aiops2022-pre", "${AIOPS22_ROOT}"),
    (_AUTHORITY_HOME + "/0_warlock/RCA_LAB", "${RCL_WORKSPACE}"),
)
SRC_LAYOUT_ENTRYPOINTS = frozenset(
    {
        "scripts/run_final_rcl_formal.py",
        "scripts/run_final_rcl_real_smoke.py",
        "scripts/status_final_rcl.py",
    }
)


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


def sanitize_machine_paths(text: str) -> tuple[str, int]:
    """Replace known authority-machine locations with explicit runtime inputs."""

    replacement_count = 0
    sanitized = text
    for source, placeholder in MACHINE_PATH_REPLACEMENTS:
        occurrences = sanitized.count(source)
        if occurrences:
            sanitized = sanitized.replace(source, placeholder)
            replacement_count += occurrences
    return sanitized, replacement_count


def adapt_repository_layout(
    text: str, *, destination_relative_path: str
) -> tuple[str, int]:
    """Point executable scripts at VAST's ``src`` package directory."""

    normalized = str(_relative_path(destination_relative_path))
    if normalized not in SRC_LAYOUT_ENTRYPOINTS:
        return text, 0
    pattern = re.compile(
        r"REPO_ROOT = Path\(__file__\)\.resolve\(\)\.parents\[1\](\r?\n)"
        r"if str\(REPO_ROOT\) not in sys\.path:\1"
        r"    sys\.path\.insert\(0, str\(REPO_ROOT\)\)"
    )

    def replacement(match: re.Match[str]) -> str:
        newline = match.group(1)
        return (
            "REPO_ROOT = Path(__file__).resolve().parents[1]"
            f"{newline}SOURCE_ROOT = REPO_ROOT / \"src\""
            f"{newline}for import_root in (REPO_ROOT, SOURCE_ROOT):"
            f"{newline}    if str(import_root) not in sys.path:"
            f"{newline}        sys.path.insert(0, str(import_root))"
        )

    adapted, count = pattern.subn(replacement, text, count=1)
    if count != 1:
        raise SnapshotError(
            f"runtime entrypoint layout precondition missing: {normalized}"
        )
    return adapted, count


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
    sanitize_paths: bool = False,
) -> dict[str, object]:
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
    replacement_count = 0
    layout_replacement_count = 0
    newline_replacement_count = 0
    if sanitize_paths:
        try:
            original = source.read_bytes().decode("utf-8")
        except UnicodeError as exc:
            raise SnapshotError(
                f"cannot sanitize non-UTF-8 source: {source_relative}"
            ) from exc
        sanitized, replacement_count = sanitize_machine_paths(original)
        sanitized, layout_replacement_count = adapt_repository_layout(
            sanitized, destination_relative_path=str(destination_relative)
        )
        newline_replacement_count = sanitized.count("\r\n")
        without_crlf = sanitized.replace("\r\n", "\n")
        newline_replacement_count += without_crlf.count("\r")
        sanitized = without_crlf.replace("\r", "\n")
        destination.write_bytes(sanitized.encode("utf-8"))
    else:
        shutil.copyfile(source, destination)
    source_sha = _sha256(source)
    destination_sha = _sha256(destination)
    if not sanitize_paths and source_sha != destination_sha:
        raise SnapshotError(f"copied file hash mismatch: {source_relative}")
    row: dict[str, object] = {
        "source_path": str(source_relative),
        "destination_path": str(destination_relative),
        "source_sha256": source_sha,
        "destination_sha256": destination_sha,
        "role": role,
    }
    transforms: dict[str, int] = {}
    if replacement_count:
        transforms["machine_path_placeholders"] = replacement_count
    if layout_replacement_count:
        transforms["src_layout_entrypoint"] = layout_replacement_count
    if newline_replacement_count:
        transforms["line_endings_lf"] = newline_replacement_count
    if transforms:
        row["transforms"] = transforms
    return row


def _canonical_sha256(payload: object) -> str:
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def write_source_inventory(
    destination_root: Path,
    *,
    rows: Iterable[dict[str, object]],
    source_authorities: dict[str, str],
) -> Path:
    """Write the stable provenance manifest for mechanically copied files."""

    destination_root = destination_root.resolve()
    ordered_rows = sorted(
        (dict(row) for row in rows), key=lambda row: row["destination_path"]
    )
    destinations = [row["destination_path"] for row in ordered_rows]
    if len(destinations) != len(set(destinations)):
        raise SnapshotError("duplicate destination path in source inventory")
    payload: dict[str, object] = {
        "schema_version": "vast-source-inventory-v1",
        "snapshot_date": date(2026, 9, 6).isoformat(),
        "method_id": "hdbscan-proxy-cvae-compatible-oser-p02",
        "source_authorities": dict(sorted(source_authorities.items())),
        "file_count": len(ordered_rows),
        "files": ordered_rows,
    }
    payload["inventory_sha256"] = _canonical_sha256(payload)
    inventory_path = destination_root.joinpath(*PurePosixPath(INVENTORY_PATH).parts)
    inventory_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = inventory_path.with_suffix(".json.tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    temporary.replace(inventory_path)
    return inventory_path


def verify_source_inventory(
    destination_root: Path, *, inventory_path: Path | None = None
) -> dict[str, object]:
    """Verify the manifest identity and every copied destination byte-for-byte."""

    destination_root = destination_root.resolve()
    path = inventory_path or destination_root.joinpath(
        *PurePosixPath(INVENTORY_PATH).parts
    )
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise SnapshotError(f"cannot read source inventory: {exc}") from exc
    claimed = payload.get("inventory_sha256")
    unsigned = dict(payload)
    unsigned.pop("inventory_sha256", None)
    if claimed != _canonical_sha256(unsigned):
        raise SnapshotError("source inventory identity mismatch")
    rows = list(payload.get("files", ()))
    if payload.get("file_count") != len(rows):
        raise SnapshotError("source inventory file count mismatch")
    for row in rows:
        relative = _relative_path(row["destination_path"])
        _assert_allowed(relative)
        destination = destination_root.joinpath(*relative.parts)
        if not destination.is_file():
            raise SnapshotError(f"missing inventory destination: {relative}")
        actual = _sha256(destination)
        if actual != row.get("destination_sha256"):
            raise SnapshotError(f"destination hash mismatch: {relative}")
    return {
        "valid": True,
        "file_count": len(rows),
        "inventory_sha256": claimed,
    }


def _copy_python_closure(
    source_root: Path, destination_root: Path
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for relative in collect_python_closure(
        source_root, roots=PYTHON_ROOTS, runtime_roots=RUNTIME_ROOTS
    ):
        destination = f"src/{relative}" if relative.startswith("rcl_study/") else relative
        rows.append(
            copy_snapshot_file(
                source_root,
                destination_root,
                relative,
                destination_relative_path=destination,
                role=(
                    "final_method_dependency"
                    if relative.startswith("rcl_study/")
                    else "runtime_entrypoint"
                ),
                sanitize_paths=True,
            )
        )
    return rows


def _copy_hdbscan_authority(
    source_root: Path, destination_root: Path
) -> list[dict[str, object]]:
    authority = (
        source_root
        / "artifacts"
        / "rcl-active-learning-fixed-v1-20260906"
        / "src"
        / "fixed_active_learning"
    )
    if not authority.is_dir():
        raise SnapshotError(f"missing fixed HDBSCAN authority: {authority}")
    rows: list[dict[str, object]] = []
    for source in sorted(authority.rglob("*.py")):
        relative_inside = source.relative_to(authority).as_posix()
        if "__pycache__" in PurePosixPath(relative_inside).parts:
            continue
        source_relative = source.relative_to(source_root).as_posix()
        rows.append(
            copy_snapshot_file(
                source_root,
                destination_root,
                source_relative,
                destination_relative_path=f"src/fixed_active_learning/{relative_inside}",
                role="fixed_hdbscan_authority",
                sanitize_paths=True,
            )
        )
    return rows


def materialize_snapshot(
    source_root: Path, destination_root: Path
) -> dict[str, object]:
    """Copy the selected method closure and produce its provenance manifest."""

    source_root = source_root.resolve()
    destination_root = destination_root.resolve()
    rows = _copy_python_closure(source_root, destination_root)
    rows.extend(_copy_hdbscan_authority(source_root, destination_root))
    for relative in PAIRWISE_BACKEND_FILES:
        suffix = relative.removeprefix("half_supervise/src/")
        rows.append(
            copy_snapshot_file(
                source_root,
                destination_root,
                relative,
                destination_relative_path=f"src/{suffix}",
                role="minimal_pairwise_backend",
                sanitize_paths=True,
            )
        )
    rows.append(
        copy_snapshot_file(
            source_root,
            destination_root,
            FROZEN_CONFIG,
            role="frozen_method_config",
            sanitize_paths=True,
        )
    )
    for name in FOCUSED_TEST_NAMES:
        rows.append(
            copy_snapshot_file(
                source_root,
                destination_root,
                f"tests/{name}",
                role="focused_upstream_test",
                sanitize_paths=True,
            )
        )
    inventory_path = write_source_inventory(
        destination_root,
        rows=rows,
        source_authorities={
            "active_learning": "rcl-active-learning-fixed-v1-20260906",
            "integration_change": "integrate-hdbscan-proxy-cvae-oser-rcl",
            "pairwise_backend": "half_supervise/nexusrcl_rebuild",
        },
    )
    audit = verify_source_inventory(
        destination_root, inventory_path=inventory_path
    )
    return {
        **audit,
        "inventory_path": inventory_path.as_posix(),
    }


def main() -> int:
    repository_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, default=repository_root.parent)
    parser.add_argument("--destination-root", type=Path, default=repository_root)
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()
    if args.verify_only:
        audit = verify_source_inventory(args.destination_root)
        print(
            "source inventory valid: "
            f"{audit['file_count']} files, {audit['inventory_sha256']}"
        )
        return 0
    audit = materialize_snapshot(args.source_root, args.destination_root)
    print(
        f"snapshot materialized: {audit['file_count']} files; "
        f"inventory={audit['inventory_path']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
