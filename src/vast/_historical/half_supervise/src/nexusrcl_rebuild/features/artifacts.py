"""Serializable artifacts for reusable window-level feature datasets."""

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence

from nexusrcl_rebuild.datasets.common import ensure_dir, write_csv, write_json, write_jsonl


@dataclass(frozen=True)
class WindowRecord:
    """Metadata for a single fault or normal window."""

    dataset: str
    window_id: str
    source_id: str
    window_kind: str
    day: str
    start_ts: int
    end_ts: int
    positive_ids: Sequence[str]
    positive_types: Sequence[str]
    positive_names: Sequence[str]
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "dataset": self.dataset,
            "window_id": self.window_id,
            "source_id": self.source_id,
            "window_kind": self.window_kind,
            "day": self.day,
            "start_ts": self.start_ts,
            "end_ts": self.end_ts,
            "duration_seconds": self.end_ts - self.start_ts,
            "positive_ids": list(self.positive_ids),
            "positive_types": list(self.positive_types),
            "positive_names": list(self.positive_names),
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True)
class WindowFeatureBundle:
    """Reusable dataset bundle for later training and evaluation."""

    dataset: str
    windows: Sequence[WindowRecord]
    entity_feature_rows: Sequence[Mapping[str, Any]]
    metadata: Mapping[str, Any]

    def to_summary_dict(self) -> Dict[str, Any]:
        return {
            "dataset": self.dataset,
            "window_count": len(self.windows),
            "entity_feature_row_count": len(self.entity_feature_rows),
            "metadata": dict(self.metadata),
        }


def _window_csv_rows(windows: Sequence[WindowRecord]) -> List[Dict[str, Any]]:
    rows = []
    for item in windows:
        payload = item.to_dict()
        rows.append(
            {
                "dataset": payload["dataset"],
                "window_id": payload["window_id"],
                "source_id": payload["source_id"],
                "window_kind": payload["window_kind"],
                "day": payload["day"],
                "start_ts": payload["start_ts"],
                "end_ts": payload["end_ts"],
                "duration_seconds": payload["duration_seconds"],
                "positive_ids": ";".join(payload["positive_ids"]),
                "positive_types": ";".join(payload["positive_types"]),
                "positive_names": ";".join(payload["positive_names"]),
                "metadata_json": json.dumps(payload["metadata"], ensure_ascii=False, sort_keys=True),
            }
        )
    return rows


def _entity_feature_fieldnames(rows: Sequence[Mapping[str, Any]]) -> List[str]:
    if not rows:
        return []

    preferred = [
        "dataset",
        "window_id",
        "source_id",
        "window_kind",
        "day",
        "start_ts",
        "end_ts",
        "entity_id",
        "entity_index",
        "entity_type",
        "entity_name",
        "is_positive",
    ]
    remaining = set()
    for row in rows:
        remaining.update(row.keys())
    ordered = [name for name in preferred if name in remaining]
    ordered.extend(sorted(name for name in remaining if name not in ordered))
    return ordered


def write_window_feature_bundle(output_dir: Path, bundle: WindowFeatureBundle) -> None:
    ensure_dir(output_dir)
    write_json(output_dir / "window_feature_manifest.json", bundle.to_summary_dict())
    write_json(output_dir / "metadata.json", dict(bundle.metadata))
    write_jsonl(output_dir / "windows.jsonl", [item.to_dict() for item in bundle.windows])
    write_csv(
        output_dir / "windows.csv",
        _window_csv_rows(bundle.windows),
        [
            "dataset",
            "window_id",
            "source_id",
            "window_kind",
            "day",
            "start_ts",
            "end_ts",
            "duration_seconds",
            "positive_ids",
            "positive_types",
            "positive_names",
            "metadata_json",
        ],
    )
    fieldnames = _entity_feature_fieldnames(bundle.entity_feature_rows)
    if fieldnames:
        write_csv(output_dir / "entity_features.csv", bundle.entity_feature_rows, fieldnames)


def write_window_feature_suite_manifest(
    output_root: Path,
    datasets: Sequence[str],
) -> Path:
    summary = {
        "datasets": {},
        "artifact_version": "hd1_hd3_hd4_casewise_v1",
    }
    for dataset in datasets:
        dataset_dir = output_root / str(dataset)
        window_manifest_path = dataset_dir / "window_feature_manifest.json"
        metadata_path = dataset_dir / "metadata.json"
        graph_dir = dataset_dir / "graph"
        window_manifest = {}
        metadata = {}
        if window_manifest_path.exists():
            window_manifest = json.loads(window_manifest_path.read_text(encoding="utf-8"))
        if metadata_path.exists():
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        summary["datasets"][str(dataset)] = {
            "artifact_dir": str(dataset_dir),
            "window_feature_manifest": str(window_manifest_path),
            "metadata_json": str(metadata_path),
            "windows_jsonl": str(dataset_dir / "windows.jsonl"),
            "windows_csv": str(dataset_dir / "windows.csv"),
            "entity_features_csv": str(dataset_dir / "entity_features.csv"),
            "graph_dir": str(graph_dir),
            "feature_version": metadata.get("feature_version"),
            "window_count": window_manifest.get("window_count"),
            "entity_feature_row_count": window_manifest.get("entity_feature_row_count"),
        }
    manifest_path = output_root / "benchmark_suite_manifest.json"
    write_json(manifest_path, summary)
    return manifest_path
