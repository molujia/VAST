"""Thin, frozen wrapper around the audited Combined 2.0 implementation."""

from __future__ import annotations

import json
import os
from pathlib import Path
import tempfile
from typing import Any, Mapping

import pandas as pd

from .config import FixedActiveLearningConfig
from .plan_contract import validate_query_plan


def _read_json_object(path: Path, label: str) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{label} must be a JSON object")
    return payload


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    serialized = json.dumps(
        payload, allow_nan=False, ensure_ascii=False, indent=2, sort_keys=True
    ) + "\n"
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=path.parent, delete=False
    ) as handle:
        handle.write(serialized)
        handle.flush()
        os.fsync(handle.fileno())
        temporary = Path(handle.name)
    os.replace(temporary, path)


def _filter_modalities(
    case_modalities: Mapping[str, Mapping[str, Mapping[str, Any]]],
    eligible_features: Mapping[str, list[str]],
    modalities: tuple[str, ...],
) -> dict[str, dict[str, dict[str, Any]]]:
    if set(eligible_features) != set(modalities):
        raise ValueError("feature inventory modality set drifted")
    filtered: dict[str, dict[str, dict[str, Any]]] = {}
    for case_id, blocks in case_modalities.items():
        if set(blocks) != set(modalities):
            raise ValueError(f"case {case_id} modality set drifted")
        current: dict[str, dict[str, Any]] = {}
        for modality in modalities:
            entry = blocks[modality]
            raw_names = tuple(str(name) for name in entry["feature_names"])
            raw_values = tuple(entry["values"])
            if len(raw_names) != len(raw_values) or len(raw_names) != len(set(raw_names)):
                raise ValueError(f"case {case_id} {modality} layout is invalid")
            index = {name: position for position, name in enumerate(raw_names)}
            approved = tuple(str(name) for name in eligible_features[modality])
            missing = [name for name in approved if name not in index]
            if missing:
                raise ValueError(f"eligible {modality} feature is missing: {missing[0]}")
            current[modality] = {
                "feature_names": approved,
                "values": tuple(raw_values[index[name]] for name in approved),
                "mask": entry["mask"],
                "coverage": entry["coverage"],
            }
        filtered[str(case_id)] = current
    return filtered


def build_global_pca_dim32(config: FixedActiveLearningConfig):
    """Rebuild the fixed representation from an external feature-bundle root."""
    if not config.data_root.is_dir():
        raise ValueError(f"external feature-bundle root is missing: {config.data_root}")
    pool = _read_json_object(config.candidate_pool_path, "candidate pool")
    inventory = _read_json_object(config.feature_inventory_path, "feature inventory")
    case_ids = tuple(str(item) for item in pool.get("case_ids", ()))
    if not case_ids or len(case_ids) != len(set(case_ids)) or case_ids != tuple(sorted(case_ids)):
        raise ValueError("candidate pool must be a non-empty sorted unique case list")
    if pool.get("dataset_id") != config.dataset_id or pool.get("ground_truth_included") is not False:
        raise ValueError("candidate pool identity or label firewall drifted")
    eligible = inventory.get("eligible_features_by_modality")
    if not isinstance(eligible, dict):
        raise ValueError("feature inventory lacks eligible_features_by_modality")

    from rcl_study.combined_active_learning_representation import (
        balance_modality_blocks,
        build_global_representation_candidates,
        fit_modality_standardizer,
        transform_modality_blocks,
    )
    from rcl_study.ordinary_multimodal_fusion import build_case_modality_records

    dataset_root = config.data_root / config.feature_alias
    metadata_path = dataset_root / "metadata.json"
    windows_path = dataset_root / "windows.csv"
    entity_features_path = dataset_root / "entity_features.csv"
    for required in (metadata_path, windows_path, entity_features_path):
        if not required.is_file():
            raise ValueError(f"feature-bundle file is missing: {required}")
    metadata = _read_json_object(metadata_path, "feature-bundle metadata")
    feature_columns = tuple(str(item) for item in metadata.get("all_feature_columns", ()))
    if not feature_columns or len(feature_columns) != len(set(feature_columns)):
        raise ValueError("metadata all_feature_columns is missing or duplicated")
    window_columns = {"window_id", "start_ts", "end_ts", "duration_seconds"}
    windows = pd.read_csv(
        windows_path,
        usecols=lambda column: str(column) in window_columns,
    )
    entity_columns = set(feature_columns) | {
        "window_id",
        "has_metric_signal",
        "has_log_signal",
        "has_trace_signal",
    }
    entity_features = pd.read_csv(
        entity_features_path,
        usecols=lambda column: str(column) in entity_columns,
    )
    raw = build_case_modality_records(
        windows=windows,
        entity_features=entity_features,
        feature_columns=feature_columns,
        case_ids=case_ids,
    )
    filtered = _filter_modalities(raw, eligible, config.modalities)
    standardizer = fit_modality_standardizer(
        fit_case_ids=case_ids,
        case_modalities=filtered,
        feature_names_by_modality={
            modality: tuple(eligible[modality]) for modality in config.modalities
        },
    )
    standardized = transform_modality_blocks(
        artifact=standardizer, case_modalities=filtered, case_ids=case_ids
    )
    candidates = build_global_representation_candidates(
        balance_modality_blocks(standardized)
    )
    candidate = candidates.get(config.representation_id)
    if candidate is None:
        raise ValueError("global_pca_dim32 representation was not generated")
    if candidate.matrix_sha256 != config.reference_representation_matrix_sha256:
        raise ValueError("global_pca_dim32 representation hash drifted")
    return candidate


def _run_rcabench(config: FixedActiveLearningConfig, seed: int, candidate):
    from rcl_study.combined_active_learning_acquisition import (
        build_equal_weight_round_robin_allocation,
        build_hdbscan_selector_queues,
        build_matched_cluster_query_plans,
        build_within_cluster_random_queues,
    )
    from rcl_study.combined_active_learning_clustering import (
        build_effective_cluster_partition,
        fit_native_hdbscan,
    )

    geometry = fit_native_hdbscan(
        dataset_id=config.dataset_id,
        candidate=candidate,
        active_learning_seed=seed,
        min_cluster_size=config.min_cluster_size,
        min_samples=config.min_samples,
        cluster_selection_method=config.cluster_selection_method,
    )
    partition = build_effective_cluster_partition(
        dataset_id=config.dataset_id,
        clusterer_id="hdbscan",
        active_learning_seed=seed,
        case_ids=geometry.case_ids,
        raw_labels=geometry.labels,
        source_geometry_sha256=geometry.geometry_sha256,
    )
    allocation = build_equal_weight_round_robin_allocation(partition, budget=30)
    queues = build_hdbscan_selector_queues(
        partition=partition, result=geometry, candidate=candidate
    )
    queues["within_cluster_random"] = build_within_cluster_random_queues(
        partition=partition,
        representation_matrix_sha256=candidate.matrix_sha256,
    )
    return build_matched_cluster_query_plans(
        partition=partition, allocation=allocation, selector_queues=queues
    )["center"]


def _run_aiops22(config: FixedActiveLearningConfig, seed: int, candidate):
    from rcl_study.combined_active_learning_tuning import generate_round_a_grid
    from rcl_study.combined_active_learning_tuning_structure import (
        build_seeded_center_plan,
        fit_configuration_geometry,
    )
    from rcl_study.combined_active_learning_winner_analysis import (
        build_tuning_anchored_matched_hdbscan_plans,
    )

    configurations = [
        item for item in generate_round_a_grid()
        if item.min_cluster_size == config.min_cluster_size
        and item.min_samples == config.min_samples
        and item.cluster_selection_method == config.cluster_selection_method
        and item.metric == "euclidean"
        and item.max_cluster_size is None
        and item.allow_single_cluster is False
    ]
    if len(configurations) != 1:
        raise ValueError("accepted AIOps22 HDBSCAN configuration is not unique")
    geometry = fit_configuration_geometry(configurations[0], candidate=candidate)
    frozen_center = build_seeded_center_plan(
        geometry,
        seed=seed,
        pool_case_ids=candidate.case_ids,
        budget=config.budget,
    )
    return build_tuning_anchored_matched_hdbscan_plans(
        fitted_geometry=geometry,
        candidate=candidate,
        active_learning_seed=seed,
        frozen_center_case_ids=frozen_center,
    ).plans["center"]


def run_fixed_active_learning(
    config: FixedActiveLearningConfig,
    *,
    seed: int,
    output_path: str | Path,
    verify_against_reference: bool = False,
) -> dict[str, Any]:
    if seed not in config.active_learning_seeds:
        raise ValueError("active-learning seed must be 41, 42, or 43")
    candidate = build_global_pca_dim32(config)
    plan_object = (
        _run_rcabench(config, seed, candidate)
        if config.dataset_id == "rcabench"
        else _run_aiops22(config, seed, candidate)
    )
    plan = plan_object.to_dict()
    validate_query_plan(plan)
    if verify_against_reference:
        reference_path = next(config.reference_plan_dir.glob(f"*seed{seed}.json"), None)
        if reference_path is None:
            raise ValueError("reference query plan is missing")
        reference = _read_json_object(reference_path, "reference query plan")
        validate_query_plan(reference)
        if plan["plan_sha256"] != reference["plan_sha256"]:
            raise ValueError("reproduced query plan differs from the frozen reference")
    _atomic_json(Path(output_path).resolve(), plan)
    return plan
