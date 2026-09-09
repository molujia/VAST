"""Configuration helpers shared by preprocessing and training code."""

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class DatasetRoots:
    hd1_root: Path
    hd2_root: Path
    hd3_root: Path
    hd4_root: Path


@dataclass(frozen=True)
class RemoteWorkspace:
    host: str
    root: str
    conda_env: str


@dataclass(frozen=True)
class WorkspaceConfig:
    project_name: str
    local_root: Path
    remote: RemoteWorkspace
    datasets: DatasetRoots
    artifacts_dir: Path
    results_dir: Path


def load_workspace_config(path: Path) -> WorkspaceConfig:
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    project_root = path.resolve().parents[1]

    def _resolve_path(raw_value: Any) -> Path:
        raw_text = str(raw_value)
        if raw_text.startswith("/") or re.match(r"^[A-Za-z]:[\\/]", raw_text):
            return Path(raw_text)
        candidate = Path(raw_text)
        if candidate.is_absolute():
            return candidate
        return (project_root / candidate).resolve()

    return WorkspaceConfig(
        project_name=data["project_name"],
        local_root=_resolve_path(data["local"]["root"]),
        remote=RemoteWorkspace(
            host=data["remote"]["host"],
            root=data["remote"]["root"],
            conda_env=data["remote"]["conda_env"],
        ),
        datasets=DatasetRoots(
            hd1_root=_resolve_path(data["datasets"]["hd1_root"]),
            hd2_root=_resolve_path(data["datasets"]["hd2_root"]),
            hd3_root=_resolve_path(data["datasets"]["hd3_root"]),
            hd4_root=_resolve_path(data["datasets"]["hd4_root"]),
        ),
        artifacts_dir=_resolve_path(data["outputs"]["artifacts_dir"]),
        results_dir=_resolve_path(data["outputs"]["results_dir"]),
    )
