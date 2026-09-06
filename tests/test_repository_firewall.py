from __future__ import annotations

from pathlib import Path

import pytest

from tools.repository_firewall import scan_repository


def _write(path: Path, content: str = "payload\n") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


@pytest.mark.parametrize(
    "relative_path",
    (
        "outputs/run/score.json",
        "data/rcabench/case.json",
        "model/checkpoint.pt",
        "src/pkg/__pycache__/module.pyc",
        "features/matrix.npy",
    ),
)
def test_blocks_forbidden_paths(tmp_path: Path, relative_path: str) -> None:
    _write(tmp_path / relative_path)

    findings = scan_repository(tmp_path)

    assert any(finding.kind == "forbidden_path" for finding in findings)


def test_blocks_token_private_key_and_machine_absolute_paths(tmp_path: Path) -> None:
    token = "github" + "_pat_" + "a" * 32
    windows_path = "C:" + "\\Users\\researcher\\dataset"
    linux_path = "/" + "home/researcher/dataset"
    private_key = "-----BEGIN " + "OPENSSH PRIVATE KEY-----"
    _write(
        tmp_path / "notes.txt",
        "\n".join((token, windows_path, linux_path, private_key)),
    )

    kinds = {finding.kind for finding in scan_repository(tmp_path)}

    assert kinds == {"secret", "absolute_path"}


def test_allows_scientific_hashes_placeholders_and_source_modules(tmp_path: Path) -> None:
    scientific_hash = "a" * 64
    _write(
        tmp_path / "configs/method.json",
        "{\n"
        f'  "profile_sha256": "{scientific_hash}",\n'
        '  "data_root": "${DATA_ROOT}"\n'
        "}\n",
    )
    _write(tmp_path / "src/rcl_study/datasets.py", "DATASET_IDS = ('rcabench',)\n")
    _write(
        tmp_path / "README.md",
        "Set git_personal_access_token in the process environment.\n",
    )

    assert scan_repository(tmp_path) == []


def test_git_metadata_is_not_scanned(tmp_path: Path) -> None:
    _write(tmp_path / ".git/config", "password=not-a-repository-file\n")

    assert scan_repository(tmp_path) == []
