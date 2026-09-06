from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from tools.snapshot_sources import (
    SnapshotError,
    collect_python_closure,
    copy_snapshot_file,
)


def _write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def test_collects_relative_imports_and_explicit_runtime_roots(tmp_path: Path) -> None:
    _write(
        tmp_path / "rcl_study/final_rcl_contract.py",
        "from .helper import method\n",
    )
    _write(tmp_path / "rcl_study/helper.py", "VALUE = 1\n")
    _write(
        tmp_path / "rcl_study/service_continuous_neural_runner.py",
        "from rcl_study.neural_helper import train\n",
    )
    _write(tmp_path / "rcl_study/neural_helper.py", "def train(): return None\n")

    selected = collect_python_closure(
        tmp_path,
        roots=("rcl_study/final_rcl_contract.py",),
        runtime_roots=("rcl_study/service_continuous_neural_runner.py",),
    )

    assert selected == (
        "rcl_study/final_rcl_contract.py",
        "rcl_study/helper.py",
        "rcl_study/neural_helper.py",
        "rcl_study/service_continuous_neural_runner.py",
    )


def test_collects_indented_runtime_imports(tmp_path: Path) -> None:
    _write(
        tmp_path / "scripts/run.py",
        "def run():\n    from rcl_study.worker import execute\n    return execute()\n",
    )
    _write(tmp_path / "rcl_study/worker.py", "def execute(): return 1\n")

    selected = collect_python_closure(tmp_path, roots=("scripts/run.py",))

    assert selected == ("rcl_study/worker.py", "scripts/run.py")


@pytest.mark.parametrize(
    "relative_path",
    (
        "scores/result.json",
        "candidate_pools/rcabench.json",
        "reference/query_plan.json",
        "src/pkg/__pycache__/module.pyc",
        "outputs/run/checkpoint.pt",
    ),
)
def test_copy_rejects_forbidden_source_path(
    tmp_path: Path, relative_path: str
) -> None:
    _write(tmp_path / relative_path, "forbidden\n")

    with pytest.raises(SnapshotError, match="forbidden"):
        copy_snapshot_file(tmp_path, tmp_path / "destination", relative_path)


def test_copy_inventory_hashes_source_and_destination(tmp_path: Path) -> None:
    source_root = tmp_path / "source"
    destination_root = tmp_path / "destination"
    content = "VALUE = 'stable'\n"
    _write(source_root / "rcl_study/component.py", content)

    row = copy_snapshot_file(
        source_root,
        destination_root,
        "rcl_study/component.py",
        destination_relative_path="src/rcl_study/component.py",
        role="final_method_dependency",
    )

    expected_sha = hashlib.sha256(
        (source_root / "rcl_study/component.py").read_bytes()
    ).hexdigest()
    assert row == {
        "source_path": "rcl_study/component.py",
        "destination_path": "src/rcl_study/component.py",
        "source_sha256": expected_sha,
        "destination_sha256": expected_sha,
        "role": "final_method_dependency",
    }
    assert (destination_root / "src/rcl_study/component.py").read_text(
        encoding="utf-8"
    ) == content


def test_closure_rejects_missing_internal_dependency(tmp_path: Path) -> None:
    _write(
        tmp_path / "rcl_study/entry.py",
        "from rcl_study.missing import value\n",
    )

    with pytest.raises(SnapshotError, match="missing internal dependency"):
        collect_python_closure(tmp_path, roots=("rcl_study/entry.py",))
