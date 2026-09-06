"""CPU-only label-free geometry smoke for combined active learning 2.0."""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
import hashlib
import json
from pathlib import Path
import sys
import time
from typing import Any

import numpy as np

from rcl_study.combined_active_learning_clustering import (
    build_effective_cluster_partition,
    dbscan_parameter_grid,
    evaluate_structure_gate,
    fit_native_dbscan,
    fit_native_hdbscan,
    fit_native_kmeans,
    fit_native_mutual_knn,
)
from rcl_study.combined_active_learning_representation import (
    FORMAL_REPRESENTATION_IDS,
    MODALITIES,
    build_formal_representation_candidates,
    fit_modality_standardizer,
    transform_modality_blocks,
)
from rcl_study.combined_active_learning_schemas import ACTIVE_LEARNING_SEEDS


DEFAULT_SAMPLE_SIZE = 160
FEATURE_ROOT = Path(
    "${NEXUSRCL_REBUILD_ROOT}/"
    "artifacts/window_feature_artifacts_hd134_stage1"
)
DATASET_CONFIG = {
    "rcabench": {"alias": "hd4", "target_clusters": 15},
    "aiops2022_pre": {"alias": "hd1", "target_clusters": 10},
}
GROUND_TRUTH_FIELD_NAMES = frozenset(
    (
        "fault_type",
        "fault_type_code",
        "root_cause",
        "root_cause_set",
        "label",
        "labels",
        "injection_name",
    )
)


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def select_evenly_spaced_case_ids(
    case_ids: Sequence[str],
    *,
    sample_size: int,
) -> tuple[str, ...]:
    """Select a fixed no-replacement geometry-smoke sample."""

    if isinstance(case_ids, (str, bytes, bytearray)) or not isinstance(
        case_ids, Sequence
    ):
        raise ValueError("case IDs must be an ordered sequence")
    normalized = tuple(str(case_id) for case_id in case_ids)
    if (
        not normalized
        or any(not case_id for case_id in normalized)
        or len(normalized) != len(set(normalized))
    ):
        raise ValueError("case IDs must be non-empty and unique")
    if (
        isinstance(sample_size, bool)
        or sample_size < 1
        or sample_size > len(normalized)
    ):
        raise ValueError("sample size must fit the candidate population")
    indices = np.linspace(0, len(normalized) - 1, num=sample_size, dtype=int)
    if len(set(int(index) for index in indices)) != sample_size:
        raise ValueError("evenly spaced sample unexpectedly repeated an index")
    return tuple(normalized[int(index)] for index in indices)


def _contains_ground_truth_field(value: Any) -> bool:
    if isinstance(value, Mapping):
        for key, item in value.items():
            if str(key).strip().lower() in GROUND_TRUTH_FIELD_NAMES:
                return True
            if _contains_ground_truth_field(item):
                return True
    elif isinstance(value, Sequence) and not isinstance(
        value, (str, bytes, bytearray)
    ):
        return any(_contains_ground_truth_field(item) for item in value)
    return False


def observable_candidate_case_ids(
    payload: Mapping[str, Any],
    *,
    expected_dataset_id: str,
) -> tuple[str, ...]:
    """Validate the label firewall and return only observable case keys."""

    if not isinstance(payload, Mapping):
        raise ValueError("observable candidate context must be a mapping")
    if payload.get("dataset_id") != expected_dataset_id:
        raise ValueError("observable candidate context dataset drifted")
    if payload.get("label_firewall") != (
        "private_annotation_oracle_never_enters_selector_candidates"
    ):
        raise ValueError("observable candidate label firewall is missing")
    rows = payload.get("observable_candidates")
    if not isinstance(rows, list) or not rows:
        raise ValueError("observable candidate context has no candidates")
    if _contains_ground_truth_field(rows):
        raise ValueError("ground-truth field entered observable candidate context")
    case_ids: list[str] = []
    for row in rows:
        if not isinstance(row, Mapping):
            raise ValueError("observable candidate row must be a mapping")
        case_id = row.get("case_id")
        if not isinstance(case_id, str) or not case_id:
            raise ValueError("observable candidate case ID is invalid")
        case_ids.append(case_id)
    if len(case_ids) != len(set(case_ids)):
        raise ValueError("observable candidate case IDs must be unique")
    return tuple(case_ids)


def filter_case_modalities_to_inventory(
    case_modalities: Mapping[str, Mapping[str, Mapping[str, Any]]],
    *,
    eligible_features_by_modality: Mapping[str, Sequence[str]],
) -> dict[str, dict[str, dict[str, Any]]]:
    """Remove every feature not approved by the frozen feature inventory."""

    if set(eligible_features_by_modality) != set(MODALITIES):
        raise ValueError("eligible feature inventory must contain five modalities")
    eligible = {
        modality: tuple(str(name) for name in eligible_features_by_modality[modality])
        for modality in MODALITIES
    }
    filtered: dict[str, dict[str, dict[str, Any]]] = {}
    for case_id, modalities in case_modalities.items():
        if set(modalities) != set(MODALITIES):
            raise ValueError("case modality record must contain five modalities")
        filtered_modalities: dict[str, dict[str, Any]] = {}
        for modality in MODALITIES:
            entry = modalities[modality]
            raw_names = tuple(str(name) for name in entry["feature_names"])
            raw_values = tuple(entry["values"])
            if len(raw_names) != len(raw_values) or len(raw_names) != len(set(raw_names)):
                raise ValueError("raw modality feature layout is invalid")
            raw_index = {name: index for index, name in enumerate(raw_names)}
            missing = tuple(name for name in eligible[modality] if name not in raw_index)
            if missing:
                raise ValueError(
                    f"eligible feature is missing from {modality}: {missing[0]}"
                )
            filtered_modalities[modality] = {
                "feature_names": eligible[modality],
                "values": tuple(
                    raw_values[raw_index[name]] for name in eligible[modality]
                ),
                "mask": entry["mask"],
                "coverage": entry["coverage"],
            }
        filtered[str(case_id)] = filtered_modalities
    return filtered


def _external_feature_loaders(workspace_root: Path):
    dependency_root = workspace_root / "half_supervise" / "src"
    if str(dependency_root) not in sys.path:
        sys.path.insert(0, str(dependency_root))
    from nexusrcl_rebuild.training.semisupervised import (  # noqa: PLC0415
        load_feature_bundle_tables,
    )
    from rcl_study.ordinary_multimodal_fusion import (  # noqa: PLC0415
        build_case_modality_records,
    )

    return load_feature_bundle_tables, build_case_modality_records


def _partition_summary(
    *,
    dataset_id: str,
    clusterer_id: str,
    active_learning_seed: int,
    result: Any,
) -> dict[str, Any]:
    partition = build_effective_cluster_partition(
        dataset_id=dataset_id,
        clusterer_id=clusterer_id,
        active_learning_seed=active_learning_seed,
        case_ids=result.case_ids,
        raw_labels=result.labels,
        source_geometry_sha256=result.geometry_sha256,
    )
    gate = evaluate_structure_gate(partition)
    return {
        "geometry_sha256": result.geometry_sha256,
        "raw_non_noise_cluster_count": len(
            {label for label in result.labels if label >= 0}
        ),
        "effective_cluster_count": partition.effective_cluster_count,
        "effective_coverage": partition.effective_coverage,
        "largest_effective_cluster_share": (
            partition.largest_effective_cluster_share
        ),
        "residual_count": len(partition.residual_case_ids),
        "structure_gate": gate.to_dict(),
    }


def _first_fitted_dbscan(
    *,
    dataset_id: str,
    candidate: Any,
    active_learning_seed: int,
):
    rejected: list[dict[str, Any]] = []
    for configuration in dbscan_parameter_grid():
        try:
            result = fit_native_dbscan(
                dataset_id=dataset_id,
                candidate=candidate,
                active_learning_seed=active_learning_seed,
                min_samples=configuration.min_samples,
                eps_quantile=configuration.eps_quantile,
                metric=configuration.metric,
            )
        except ValueError as exc:
            rejected.append(
                {
                    "configuration": configuration.to_dict(),
                    "reason": str(exc),
                }
            )
        else:
            return configuration, result, tuple(rejected)
    raise ValueError("DBSCAN smoke found no numerically fitted configuration")


def run_dataset_geometry_smoke(
    *,
    workspace_root: Path,
    dataset_id: str,
    sample_size: int,
    active_learning_seed: int,
) -> dict[str, Any]:
    """Build ten formal matrices and smoke all four clusterers on one dataset."""

    if dataset_id not in DATASET_CONFIG:
        raise ValueError("geometry smoke dataset is outside the formal protocol")
    if active_learning_seed not in ACTIVE_LEARNING_SEEDS:
        raise ValueError("geometry smoke seed is outside the formal protocol")
    context_path = (
        workspace_root
        / "inputs"
        / "authority_freeze"
        / dataset_id
        / "candidate-context.json"
    )
    context = _read_json(context_path)
    observable_ids = observable_candidate_case_ids(
        context,
        expected_dataset_id=dataset_id,
    )
    sampled_ids = select_evenly_spaced_case_ids(
        tuple(sorted(observable_ids)),
        sample_size=sample_size,
    )
    inventory_path = (
        workspace_root
        / "openspec"
        / "changes"
        / "evaluate-rcl-active-learning-combined-2-0"
        / "evidence"
        / "feature_inventory"
        / f"{dataset_id}.json"
    )
    inventory = _read_json(inventory_path)
    if inventory.get("dataset_id") != dataset_id:
        raise ValueError("feature inventory dataset drifted")
    eligible_features = inventory.get("eligible_features_by_modality")
    if not isinstance(eligible_features, Mapping):
        raise ValueError("feature inventory has no eligible feature mapping")

    load_tables, build_records = _external_feature_loaders(workspace_root)
    alias = str(DATASET_CONFIG[dataset_id]["alias"])
    tables = load_tables(FEATURE_ROOT, alias)
    raw_modalities = build_records(
        windows=tables.windows,
        entity_features=tables.entity_features,
        feature_columns=tables.feature_columns,
        case_ids=sampled_ids,
    )
    filtered = filter_case_modalities_to_inventory(
        raw_modalities,
        eligible_features_by_modality=eligible_features,
    )
    names = {
        modality: tuple(str(name) for name in eligible_features[modality])
        for modality in MODALITIES
    }
    standardizer = fit_modality_standardizer(
        fit_case_ids=sampled_ids,
        case_modalities=filtered,
        feature_names_by_modality=names,
    )
    standardized = transform_modality_blocks(
        artifact=standardizer,
        case_modalities=filtered,
        case_ids=sampled_ids,
    )
    candidates = build_formal_representation_candidates(standardized)
    if tuple(candidates) != FORMAL_REPRESENTATION_IDS:
        raise ValueError("geometry smoke did not build all ten formal representations")

    target_clusters = int(DATASET_CONFIG[dataset_id]["target_clusters"])
    representations: dict[str, Any] = {}
    for representation_id in FORMAL_REPRESENTATION_IDS:
        candidate = candidates[representation_id]
        kmeans = fit_native_kmeans(
            dataset_id=dataset_id,
            candidate=candidate,
            active_learning_seed=active_learning_seed,
            cluster_count=target_clusters,
        )
        dbscan_config, dbscan, dbscan_rejected = _first_fitted_dbscan(
            dataset_id=dataset_id,
            candidate=candidate,
            active_learning_seed=active_learning_seed,
        )
        hdbscan = fit_native_hdbscan(
            dataset_id=dataset_id,
            candidate=candidate,
            active_learning_seed=active_learning_seed,
            min_cluster_size=5,
            min_samples=3,
            cluster_selection_method="eom",
        )
        mutual_knn = fit_native_mutual_knn(
            dataset_id=dataset_id,
            candidate=candidate,
            active_learning_seed=active_learning_seed,
            neighbor_count=5,
            metric="euclidean",
        )
        dbscan_summary = _partition_summary(
            dataset_id=dataset_id,
            clusterer_id="dbscan",
            active_learning_seed=active_learning_seed,
            result=dbscan,
        )
        dbscan_summary.update(
            {
                "configuration": dbscan_config.to_dict(),
                "eps": dbscan.eps,
                "density_phase": dbscan.density_phase,
                "preceding_numerical_rejections": list(dbscan_rejected),
            }
        )
        representations[representation_id] = {
            "family": candidate.family,
            "matrix_sha256": candidate.matrix_sha256,
            "shape": list(candidate.matrix.shape),
            "formal_eligible": candidate.formal_eligible,
            "clusterers": {
                "kmeans": _partition_summary(
                    dataset_id=dataset_id,
                    clusterer_id="kmeans",
                    active_learning_seed=active_learning_seed,
                    result=kmeans,
                ),
                "dbscan": dbscan_summary,
                "hdbscan": _partition_summary(
                    dataset_id=dataset_id,
                    clusterer_id="hdbscan",
                    active_learning_seed=active_learning_seed,
                    result=hdbscan,
                ),
                "mutual_knn": _partition_summary(
                    dataset_id=dataset_id,
                    clusterer_id="mutual_knn",
                    active_learning_seed=active_learning_seed,
                    result=mutual_knn,
                ),
            },
        }
    return {
        "dataset_id": dataset_id,
        "candidate_context_path": str(context_path),
        "candidate_context_sha256": _file_sha256(context_path),
        "feature_inventory_path": str(inventory_path),
        "feature_inventory_sha256": _file_sha256(inventory_path),
        "feature_bundle_alias": alias,
        "feature_bundle_source_hashes": {
            name: _file_sha256(FEATURE_ROOT / alias / name)
            for name in ("windows.csv", "entity_features.csv")
        },
        "observable_candidate_count": len(observable_ids),
        "sample_size": sample_size,
        "sample_case_ids": list(standardized.case_ids),
        "active_learning_seed": active_learning_seed,
        "representation_count": len(representations),
        "geometry_run_count": len(representations) * 4,
        "representations": representations,
    }


def run_geometry_smoke(
    *,
    workspace_root: Path,
    output_path: Path,
    sample_size: int = DEFAULT_SAMPLE_SIZE,
    active_learning_seed: int = 41,
) -> dict[str, Any]:
    """Run the two-dataset smoke and atomically publish its evidence."""

    root = workspace_root.resolve()
    output = output_path.resolve()
    if not output.is_relative_to(root):
        raise ValueError("geometry smoke output must stay inside the 2.0 workspace")
    if output.exists():
        raise ValueError("geometry smoke output already exists and is immutable")
    started = time.monotonic()
    datasets = {
        dataset_id: run_dataset_geometry_smoke(
            workspace_root=root,
            dataset_id=dataset_id,
            sample_size=sample_size,
            active_learning_seed=active_learning_seed,
        )
        for dataset_id in DATASET_CONFIG
    }
    payload = {
        "schema_version": "rcl-active-learning-combined-2.0-geometry-smoke-v1",
        "status": "complete",
        "workspace": str(root),
        "sample_size_per_dataset": sample_size,
        "active_learning_seed": active_learning_seed,
        "representation_ids": list(FORMAL_REPRESENTATION_IDS),
        "clusterer_ids": ["kmeans", "dbscan", "hdbscan", "mutual_knn"],
        "dataset_count": len(datasets),
        "total_geometry_run_count": sum(
            dataset["geometry_run_count"] for dataset in datasets.values()
        ),
        "ground_truth_fields_read": [],
        "rcl_units_launched": 0,
        "gpu_used": False,
        "elapsed_seconds": time.monotonic() - started,
        "datasets": datasets,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(output.name + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(output)
    return payload


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--sample-size", type=int, default=DEFAULT_SAMPLE_SIZE)
    parser.add_argument("--active-learning-seed", type=int, default=41)
    args = parser.parse_args(argv)
    payload = run_geometry_smoke(
        workspace_root=args.workspace_root,
        output_path=args.output,
        sample_size=args.sample_size,
        active_learning_seed=args.active_learning_seed,
    )
    print(json.dumps({
        "status": payload["status"],
        "total_geometry_run_count": payload["total_geometry_run_count"],
        "output": str(args.output.resolve()),
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "DEFAULT_SAMPLE_SIZE",
    "filter_case_modalities_to_inventory",
    "main",
    "observable_candidate_case_ids",
    "run_dataset_geometry_smoke",
    "run_geometry_smoke",
    "select_evenly_spaced_case_ids",
]
