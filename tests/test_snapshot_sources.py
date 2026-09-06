from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from tools.snapshot_sources import (
    SnapshotError,
    adapt_repository_layout,
    collect_python_closure,
    copy_snapshot_file,
    sanitize_machine_paths,
    verify_source_inventory,
    write_source_inventory,
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


def test_source_inventory_round_trip_detects_destination_drift(tmp_path: Path) -> None:
    source_root = tmp_path / "source"
    destination_root = tmp_path / "destination"
    _write(source_root / "rcl_study/component.py", "VALUE = 1\n")
    row = copy_snapshot_file(
        source_root,
        destination_root,
        "rcl_study/component.py",
        destination_relative_path="src/rcl_study/component.py",
        role="final_method_dependency",
    )
    inventory_path = write_source_inventory(
        destination_root,
        rows=[row],
        source_authorities={
            "final_method": "integrate-hdbscan-proxy-cvae-oser-rcl",
            "active_learning": "rcl-active-learning-fixed-v1-20260906",
        },
    )

    audit = verify_source_inventory(destination_root, inventory_path=inventory_path)

    assert audit["valid"] is True
    assert audit["file_count"] == 1
    (destination_root / "src/rcl_study/component.py").write_text(
        "VALUE = 2\n", encoding="utf-8"
    )
    with pytest.raises(SnapshotError, match="destination hash mismatch"):
        verify_source_inventory(destination_root, inventory_path=inventory_path)


def test_machine_specific_paths_are_replaced_by_runtime_placeholders() -> None:
    authority_home = "/" + "home/" + "wangrunzhou"
    source = (
        f"workspace = '{authority_home}/0_warlock/RCA_LAB'\n"
        f"dataset = '{authority_home}/dataset/aiops2022-pre'\n"
    )

    sanitized, replacement_count = sanitize_machine_paths(source)

    assert sanitized == (
        "workspace = '${RCL_WORKSPACE}'\n"
        "dataset = '${AIOPS22_ROOT}'\n"
    )
    assert replacement_count == 2


def test_runtime_entrypoint_is_adapted_to_src_layout() -> None:
    source = (
        "REPO_ROOT = Path(__file__).resolve().parents[1]\n"
        "if str(REPO_ROOT) not in sys.path:\n"
        "    sys.path.insert(0, str(REPO_ROOT))\n"
    )

    adapted, replacement_count = adapt_repository_layout(
        source, destination_relative_path="scripts/run_final_rcl_formal.py"
    )

    assert adapted == (
        "REPO_ROOT = Path(__file__).resolve().parents[1]\n"
        "SOURCE_ROOT = REPO_ROOT / \"src\"\n"
        "for import_root in (REPO_ROOT, SOURCE_ROOT):\n"
        "    if str(import_root) not in sys.path:\n"
        "        sys.path.insert(0, str(import_root))\n"
    )
    assert replacement_count == 1


def test_sanitized_copy_normalizes_text_to_portable_lf(tmp_path: Path) -> None:
    source_root = tmp_path / "source"
    destination_root = tmp_path / "destination"
    source = source_root / "rcl_study/component.py"
    source.parent.mkdir(parents=True)
    source.write_bytes(b"VALUE = 1\r\nNEXT = 2\r\n")

    row = copy_snapshot_file(
        source_root,
        destination_root,
        "rcl_study/component.py",
        destination_relative_path="src/rcl_study/component.py",
        sanitize_paths=True,
    )

    assert (destination_root / "src/rcl_study/component.py").read_bytes() == (
        b"VALUE = 1\nNEXT = 2\n"
    )
    assert row["transforms"] == {"line_endings_lf": 2}
