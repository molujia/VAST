"""CLI for building reusable window-level feature artifacts."""

import argparse
import shutil
from dataclasses import replace
from pathlib import Path
from typing import Optional, Sequence

from nexusrcl_rebuild.datasets.common import load_manifest
from nexusrcl_rebuild.settings import load_workspace_config

from rcl_study.datasets import resolve_dataset
from rcl_study.event_artifacts import (
    load_event_selection,
    write_event_selection_audit,
)

from .artifacts import write_window_feature_bundle, write_window_feature_suite_manifest
from .entities import load_entity_index
from .topology import load_topology_bundle
from .window_features import build_window_feature_bundle


GRAPH_FILES = [
    "entity_index.json",
    "topology.json",
    "service_service_edges.csv",
    "service_host_edges.csv",
    "host_host_edges.csv",
]


def _copy_graph_artifacts(source_dir: Path, target_dir: Path) -> None:
    target_dir.mkdir(parents=True, exist_ok=True)
    for file_name in GRAPH_FILES:
        source_path = source_dir / file_name
        if source_path.exists():
            shutil.copy2(source_path, target_dir / file_name)


def _slice_manifest(
    manifest,
    max_fault_windows: Optional[int],
    max_normal_windows: Optional[int],
):
    if max_fault_windows is None and max_normal_windows is None:
        return manifest
    return replace(
        manifest,
        cases=list(manifest.cases[:max_fault_windows]) if max_fault_windows is not None else list(manifest.cases),
        normal_windows=(
            list(manifest.normal_windows[:max_normal_windows])
            if max_normal_windows is not None
            else list(manifest.normal_windows)
        ),
    )


def build_window_feature_artifacts(
    config_path: Path,
    datasets: Sequence[str],
    manifest_root: Path,
    topology_root: Path,
    output_root: Path,
    max_fault_windows: Optional[int] = None,
    max_normal_windows: Optional[int] = None,
    max_log_files_per_day: Optional[int] = None,
    max_metric_files_per_day: Optional[int] = None,
    max_trace_files_per_day: Optional[int] = None,
    day_workers: int = 1,
    event_source_mode: str = "legacy_fallback",
    event_artifact_root: Optional[Path] = None,
    event_generator_id: str = "legacy_canonical",
    event_config_sha256: Optional[str] = None,
    event_fallback_reason: str = (
        "backward-compatible native metric-event generation"
    ),
) -> None:
    config = load_workspace_config(config_path)
    dataset_roots = {
        "hd1": config.datasets.hd1_root,
        "hd2": config.datasets.hd2_root,
        "hd3": config.datasets.hd3_root,
        "hd4": config.datasets.hd4_root,
    }

    if event_source_mode == "artifact" and len(datasets) != 1:
        raise ValueError(
            "artifact event selection requires exactly one dataset per build"
        )

    for dataset_name in datasets:
        manifest = load_manifest(manifest_root / dataset_name / "manifest.json")
        manifest = _slice_manifest(manifest, max_fault_windows, max_normal_windows)
        canonical_dataset_id = resolve_dataset(dataset_name).canonical_id
        expected_native_case_ids = {
            str(case.case_id) for case in manifest.cases
        }
        selection = load_event_selection(
            mode=event_source_mode,
            expected_dataset_id=canonical_dataset_id,
            expected_native_case_ids=expected_native_case_ids,
            artifact_root=event_artifact_root,
            expected_event_generator_id=event_generator_id,
            expected_config_sha256=event_config_sha256,
            fallback_reason=event_fallback_reason,
        )
        event_selection = {
            "mode": selection.mode,
            "generator_id": selection.generator_id,
            "events": selection.events,
            "fallback_reason": str(
                selection.audit.get("fallback_reason", "")
            ),
            "selection_audit": selection.audit,
        }
        graph_dir = topology_root / dataset_name
        entity_index = load_entity_index(graph_dir / "entity_index.json")
        topology = load_topology_bundle(graph_dir / "topology.json")
        bundle = build_window_feature_bundle(
            dataset_root=dataset_roots[dataset_name],
            manifest=manifest,
            entity_index=entity_index,
            topology=topology,
            scan_limits={
                "max_log_files_per_day": max_log_files_per_day,
                "max_metric_files_per_day": max_metric_files_per_day,
                "max_trace_files_per_day": max_trace_files_per_day,
            },
            day_workers=day_workers,
            event_selection=event_selection,
        )
        dataset_output = output_root / dataset_name
        write_window_feature_bundle(dataset_output, bundle)
        _copy_graph_artifacts(graph_dir, dataset_output / "graph")
        write_event_selection_audit(
            dataset_output / "event_source_selection.json",
            selection,
            extra={
                "method_id": "half_supervise",
                "legacy_dataset_id": dataset_name,
                "consumer_audit": dict(
                    bundle.metadata.get(
                        "event_source_selection", {}
                    ).get("consumer_audit", {})
                ),
            },
        )
    write_window_feature_suite_manifest(output_root, datasets)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build reusable window-level feature artifacts.")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/workspace.yaml"),
        help="Workspace config YAML path.",
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        choices=["hd1", "hd2", "hd3", "hd4"],
        default=["hd1", "hd2", "hd3", "hd4"],
        help="Datasets to process.",
    )
    parser.add_argument(
        "--manifest-root",
        type=Path,
        default=Path("artifacts/dataset_manifests_final"),
        help="Directory containing per-dataset manifest bundles.",
    )
    parser.add_argument(
        "--topology-root",
        type=Path,
        default=Path("artifacts/topology_artifacts_final"),
        help="Directory containing per-dataset entity-index and topology bundles.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("artifacts/window_feature_artifacts"),
        help="Directory to write reusable window feature bundles into.",
    )
    parser.add_argument(
        "--max-fault-windows",
        type=int,
        default=None,
        help="Optional debug cap on the number of fault windows per dataset.",
    )
    parser.add_argument(
        "--max-normal-windows",
        type=int,
        default=None,
        help="Optional debug cap on the number of normal windows per dataset.",
    )
    parser.add_argument(
        "--max-log-files-per-day",
        type=int,
        default=None,
        help="Optional debug cap on per-day log files scanned for each dataset.",
    )
    parser.add_argument(
        "--max-metric-files-per-day",
        type=int,
        default=None,
        help="Optional debug cap on per-day metric files scanned for each dataset.",
    )
    parser.add_argument(
        "--max-trace-files-per-day",
        type=int,
        default=None,
        help="Optional debug cap on per-day trace files scanned for each dataset.",
    )
    parser.add_argument(
        "--day-workers",
        type=int,
        default=1,
        help="Number of day-level worker processes to use inside each dataset build.",
    )
    parser.add_argument(
        "--event-source-mode",
        choices=["artifact", "legacy_fallback"],
        default="legacy_fallback",
        help="Use a validated canonical event artifact or the audited native fallback.",
    )
    parser.add_argument(
        "--event-artifact-root",
        type=Path,
        default=None,
        help="Validated canonical event artifact root for artifact mode.",
    )
    parser.add_argument(
        "--event-generator-id",
        default="legacy_canonical",
        help="Expected event generator ID recorded in the artifact manifest.",
    )
    parser.add_argument(
        "--event-config-sha256",
        default=None,
        help="Expected event generator configuration SHA-256.",
    )
    parser.add_argument(
        "--event-fallback-reason",
        default="backward-compatible native metric-event generation",
        help="Required audit reason when event-source-mode is legacy_fallback.",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    build_window_feature_artifacts(
        config_path=args.config,
        datasets=args.datasets,
        manifest_root=args.manifest_root,
        topology_root=args.topology_root,
        output_root=args.output_root,
        max_fault_windows=args.max_fault_windows,
        max_normal_windows=args.max_normal_windows,
        max_log_files_per_day=args.max_log_files_per_day,
        max_metric_files_per_day=args.max_metric_files_per_day,
        max_trace_files_per_day=args.max_trace_files_per_day,
        day_workers=args.day_workers,
        event_source_mode=args.event_source_mode,
        event_artifact_root=args.event_artifact_root,
        event_generator_id=args.event_generator_id,
        event_config_sha256=args.event_config_sha256,
        event_fallback_reason=args.event_fallback_reason,
    )
    for dataset_name in args.datasets:
        dataset_output = args.output_root / dataset_name
        print(
            "%s: manifest=%s features=%s"
            % (
                dataset_name,
                dataset_output / "window_feature_manifest.json",
                dataset_output / "entity_features.csv",
            )
        )
    print("suite_manifest=%s" % (args.output_root / "benchmark_suite_manifest.json"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
