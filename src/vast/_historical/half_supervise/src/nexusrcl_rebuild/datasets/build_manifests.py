"""CLI entrypoint for dataset manifest construction."""

import argparse
from pathlib import Path
from typing import Dict, Sequence

from nexusrcl_rebuild.settings import load_workspace_config

from .common import ensure_dir, write_manifest_bundle
from .hd1 import build_hd1_manifest
from .hd2 import build_hd2_manifest
from .hd3 import build_hd3_manifest
from .hd4 import build_hd4_manifest


DATASET_BUILDERS = {
    "hd1": build_hd1_manifest,
    "hd2": build_hd2_manifest,
    "hd3": build_hd3_manifest,
    "hd4": build_hd4_manifest,
}


def build_all_manifests(
    config_path: Path,
    datasets: Sequence[str],
    output_root: Path,
    window_size_seconds: int,
    guard_band_seconds: int,
) -> Dict[str, object]:
    config = load_workspace_config(config_path)
    ensure_dir(output_root)

    dataset_roots = {
        "hd1": config.datasets.hd1_root,
        "hd2": config.datasets.hd2_root,
        "hd3": config.datasets.hd3_root,
        "hd4": config.datasets.hd4_root,
    }

    manifests = {}
    for dataset_name in datasets:
        builder = DATASET_BUILDERS[dataset_name]
        manifest = builder(
            dataset_root=dataset_roots[dataset_name],
            window_size_seconds=window_size_seconds,
            guard_band_seconds=guard_band_seconds,
        )
        write_manifest_bundle(output_root / dataset_name, manifest)
        manifests[dataset_name] = manifest
    return manifests


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build HD1/HD2/HD3/HD4 RCA manifests.")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/workspace.yaml"),
        help="Workspace config YAML path.",
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        choices=sorted(DATASET_BUILDERS),
        default=sorted(DATASET_BUILDERS),
        help="Datasets to rebuild.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=None,
        help="Output directory for manifest bundles. Defaults to <artifacts>/dataset_manifests.",
    )
    parser.add_argument(
        "--window-size-seconds",
        type=int,
        default=300,
        help="Normal window length for safe negative sampling.",
    )
    parser.add_argument(
        "--guard-band-seconds",
        type=int,
        default=20,
        help="Guard band before/after every fault interval.",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    config = load_workspace_config(args.config)
    output_root = args.output_root or (config.artifacts_dir / "dataset_manifests")
    manifests = build_all_manifests(
        config_path=args.config,
        datasets=args.datasets,
        output_root=output_root,
        window_size_seconds=args.window_size_seconds,
        guard_band_seconds=args.guard_band_seconds,
    )
    for name in args.datasets:
        manifest = manifests[name]
        print(
            "%s: cases=%d normal_windows=%d output=%s"
            % (name, len(manifest.cases), len(manifest.normal_windows), output_root / name)
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
