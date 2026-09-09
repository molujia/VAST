"""CLI for entity-index and topology artifact generation."""

import argparse
from pathlib import Path
from typing import Sequence

from nexusrcl_rebuild.datasets.common import load_manifest
from nexusrcl_rebuild.settings import load_workspace_config

from .entities import build_entity_index, write_entity_index
from .topology import build_topology_artifacts, write_topology_bundle


def build_entity_indices_and_topologies(
    config_path: Path,
    datasets: Sequence[str],
    manifest_root: Path,
    output_root: Path,
) -> None:
    config = load_workspace_config(config_path)
    dataset_roots = {
        "hd1": config.datasets.hd1_root,
        "hd2": config.datasets.hd2_root,
        "hd3": config.datasets.hd3_root,
        "hd4": config.datasets.hd4_root,
    }

    for dataset_name in datasets:
        manifest = load_manifest(manifest_root / dataset_name / "manifest.json")
        entity_index = build_entity_index(dataset_roots[dataset_name], manifest)
        dataset_output = output_root / dataset_name
        write_entity_index(dataset_output / "entity_index.json", entity_index)
        topology = build_topology_artifacts(dataset_roots[dataset_name], manifest, entity_index)
        write_topology_bundle(dataset_output, topology)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build entity indices and topology artifacts.")
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
        "--output-root",
        type=Path,
        default=Path("artifacts/topology_artifacts"),
        help="Directory to write entity-index and topology artifacts into.",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    build_entity_indices_and_topologies(
        config_path=args.config,
        datasets=args.datasets,
        manifest_root=args.manifest_root,
        output_root=args.output_root,
    )
    for dataset_name in args.datasets:
        print(
            "%s: entity_index=%s topology=%s"
            % (
                dataset_name,
                args.output_root / dataset_name / "entity_index.json",
                args.output_root / dataset_name / "topology.json",
            )
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
