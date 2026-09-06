from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
import sys
from copy import deepcopy
from pathlib import Path
from typing import Any, Mapping

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from rcl_study.conservative_lofo_calibration import DATASETS, EXPECTED_EXCLUSIONS
from rcl_study.conservative_lofo_authority_scores import build_authority_score_bundle
from rcl_study.conservative_lofo_base_bridge import (
    fit_base_score_bridge,
    score_base_score_bridge,
)
from rcl_study.conservative_lofo_formal_calibration import (
    build_formal_calibration_execution,
    build_nested_calibration_folds,
)
from rcl_study.conservative_lofo_protocol import build_union_excluded_calibration_pool
from rcl_study.conservative_lofo_state import (
    ObservableStateSchema,
    build_candidate_state_rows,
    fit_fold_state_transform,
)
from rcl_study.ordinary_query_engine import (
    MappingAnnotationOracle,
    build_strategy_registry,
    run_ordinary_query_engine,
)
from rcl_study.ordinary_strategy_pipeline import (
    merge_fusion_and_candidate_context,
    semantic_sha256,
)


ARMS = ("oser_meta", "mm_dro", "cope_gate")


def _canonical(value: Any) -> str:
    return json.dumps(
        value, allow_nan=False, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    )


def _hash(value: Any) -> str:
    return hashlib.sha256(_canonical(value).encode("utf-8")).hexdigest()


def _read(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON input must be an object: {path}")
    return value


def _write(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def _ordinary_unit(manifest: Mapping[str, Any], dataset_id: str) -> dict[str, Any]:
    matches = [
        dict(unit)
        for unit in manifest.get("units", ())
        if unit.get("canonical_dataset_id") == dataset_id
        and unit.get("strategy_id") == "dbscan_coverage"
        and int(unit.get("active_learning_seed", -1)) == 42
    ]
    if len(matches) != 1:
        raise ValueError(f"expected exactly one DBSCAN seed-42 unit for {dataset_id}")
    unit = matches[0]
    expected = build_strategy_registry()["strategy_configs"]["dbscan_coverage"]
    if dict(unit.get("strategy_config", {})) != expected:
        raise ValueError(f"DBSCAN config drift for {dataset_id}")
    if int(unit.get("budget", -1)) != 30 or int(unit.get("split_seed", -1)) != 42:
        raise ValueError(f"DBSCAN budget/split drift for {dataset_id}")
    return unit


def _inventory(path: Path, dataset_id: str) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    source = _read(path)
    if source.get("canonical_dataset_id") != dataset_id or int(source.get("split_seed", -1)) != 42:
        raise ValueError(f"inventory dataset/split drift for {dataset_id}")
    records = tuple(dict(value) for value in source.get("records", ()))
    if len(records) != int(source.get("record_count", -1)):
        raise ValueError(f"inventory record count drift for {dataset_id}")
    by_case = {str(row.get("case_id", "")): row for row in records}
    if "" in by_case or len(by_case) != len(records):
        raise ValueError(f"inventory case identity drift for {dataset_id}")
    train = tuple(str(row["case_id"]) for row in records if row.get("split") == "outer_train")
    test = tuple(str(row["case_id"]) for row in records if row.get("split") == "outer_test")
    if len(train) != int(source.get("outer_train_count", -1)) or len(test) != int(
        source.get("outer_test_count", -1)
    ):
        raise ValueError(f"inventory split count drift for {dataset_id}")
    protocol_inventory = {
        "cases_by_id": {
            case_id: {
                "fault_type": str(row["fault_type"]),
                "incident_id": str(row.get("incident_id", case_id)),
            }
            for case_id, row in by_case.items()
        },
        "train_case_ids": train,
        "test_case_ids": test,
    }
    return protocol_inventory, by_case


def _actual_exclusions(dataset_id: str, inventory: Mapping[str, Any]) -> tuple[str, ...]:
    display = tuple(EXPECTED_EXCLUSIONS[dataset_id])
    available = {
        str(value["fault_type"]) for value in dict(inventory["cases_by_id"]).values()
    }
    if dataset_id != "rcabench" or not any(value.startswith("fault_type_code:") for value in available):
        return display
    codebook = _read(REPO_ROOT / "rcl_study" / "rcabench_fault_type_codebook.json")
    raw_to_code = {str(raw): str(code) for code, raw in codebook["code_to_raw_type"].items()}
    try:
        return tuple(f"fault_type_code:{raw_to_code[name]}" for name in display)
    except KeyError as exc:
        raise ValueError(f"RCABench exclusion cannot be resolved: {exc}") from exc


def _bundle_pool(
    dataset_id: str,
    raw_pool: Mapping[str, Any],
    queried_case_ids: tuple[str, ...],
    actual_excluded_fault_types: tuple[str, ...],
) -> dict[str, Any]:
    events = tuple(
        {
            "case_id": case_id,
            "role": "development_calibration_only",
            "fields": ("case_id", "observations", "topology", "root_cause", "fault_type"),
            "label_cost": 1,
        }
        for case_id in queried_case_ids
    )
    identity = {
        "schema_version": "conservative-lofo-union-excluded-calibration-v1",
        "dataset_id": dataset_id,
        "inventory_sha256": str(raw_pool["inventory_sha256"]),
        "excluded_fault_types": tuple(EXPECTED_EXCLUSIONS[dataset_id]),
        "excluded_case_ids": tuple(raw_pool["excluded_case_ids"]),
        "excluded_counts_by_fault_type": {
            display: int(raw_pool["excluded_counts_by_fault_type"].get(actual, 0))
            for display, actual in zip(
                EXPECTED_EXCLUSIONS[dataset_id], actual_excluded_fault_types
            )
        },
        "admitted_candidate_case_ids": tuple(raw_pool["admitted_candidate_case_ids"]),
        "unused_outer_test_case_ids": tuple(raw_pool["unused_outer_test_case_ids"]),
        "queried_development_case_ids": queried_case_ids,
        "label_access_events": events,
        "development_label_cost": 30,
    }
    return {**identity, "calibration_pool_sha256": _hash(identity)}


def _schema_from_payload(value: Mapping[str, Any]) -> ObservableStateSchema:
    return ObservableStateSchema(
        metric_fields=tuple(value.get("metric_fields", ())),
        log_fields=tuple(value.get("log_fields", ())),
        trace_fields=tuple(value.get("trace_fields", ())),
        topology_fields=tuple(value.get("topology_fields", ())),
        time_fields=tuple(value.get("time_fields", ())),
        candidate_fields=tuple(value.get("candidate_fields", ())),
        modality_presence_fields=dict(value.get("modality_presence_fields", {})),
        clip_value=float(value.get("clip_value", 20.0)),
    )


def _default_schema() -> ObservableStateSchema:
    return ObservableStateSchema(
        metric_fields=("metric_direction", "metric_magnitude", "metric_duration", "metric_sparsity"),
        log_fields=("log_intensity", "log_template_change", "log_relative_time"),
        trace_fields=("trace_latency", "trace_error", "trace_earliest_anomaly", "trace_hop_lag"),
        topology_fields=("topology_depth", "topology_width", "topology_direction_consistency"),
        time_fields=("relative_onset", "relative_peak", "relative_recovery"),
        candidate_fields=(
            "candidate_is_service", "candidate_reachability",
            "candidate_source_earliness", "candidate_explanation_coverage",
        ),
        modality_presence_fields={
            "metric": "has_metric_signal", "log": "has_log_signal",
            "trace": "has_trace_signal", "topology": "has_topology_signal",
            "time": "has_time_signal", "candidate": "has_candidate_signal",
        },
    )


def _number(row: Any, name: str, default: float = 0.0) -> float:
    value = getattr(row, name, default)
    try:
        result = float(value)
    except (TypeError, ValueError):
        return float(default)
    return result if math.isfinite(result) else float(default)


def _ratio(numerator: float, denominator: float) -> float:
    return 0.0 if abs(denominator) <= 1e-12 else numerator / denominator


def _rows_from_training_frame(frame: pd.DataFrame, feature_columns: tuple[str, ...]) -> list[dict[str, Any]]:
    rows = []
    for source in frame.itertuples(index=False):
        metric_mean = _number(source, "metric_window_mean")
        metric_max = abs(_number(source, "metric_abs_z_max"))
        metric_last = abs(_number(source, "metric_last_abs_z_max"))
        metric_series = max(0.0, _number(source, "metric_series_count"))
        metric_anomalous = max(0.0, _number(source, "metric_anomalous_kpi_count"))
        metric_samples = max(0.0, _number(source, "metric_sample_count"))
        metric_active = max(0.0, _number(source, "metric_event_active_timestamp_count"))
        topo_in = max(0.0, _number(source, "topo_in_degree"))
        topo_out = max(0.0, _number(source, "topo_out_degree"))
        topo_in_weight = abs(_number(source, "topo_in_weight"))
        topo_out_weight = abs(_number(source, "topo_out_weight"))
        has_metric = int(_number(source, "has_metric_signal") > 0)
        has_log = int(_number(source, "has_log_signal") > 0)
        has_trace = int(_number(source, "has_trace_signal") > 0)
        trace_rank = _number(source, "rel_win_rank_trace_latency_abs_z_max", 0.5)
        metric_rank = _number(source, "rel_win_rank_metric_abs_z_max", 0.5)
        modalities = max(0.0, min(3.0, _number(source, "modalities_present_count")))
        base_features = [_number(source, name) for name in feature_columns]
        rows.append({
            "case_id": str(source.window_id), "candidate_id": str(source.entity_id),
            "label": int(_number(source, "label")), "base_features": base_features,
            "metric_direction": 1.0 if metric_mean > 0 else (-1.0 if metric_mean < 0 else 0.0),
            "metric_magnitude": metric_max, "metric_duration": metric_active,
            "metric_sparsity": _ratio(metric_anomalous, max(metric_series, 1.0)),
            "log_intensity": _number(source, "log_error_count") + _number(source, "log_warn_count"),
            "log_template_change": _number(source, "log_message_entropy"),
            "log_relative_time": _number(source, "rel_win_rank_log_error_count", 0.5),
            "trace_latency": abs(_number(source, "trace_latency_abs_z_max")),
            "trace_error": _number(source, "trace_error_ratio"),
            "trace_earliest_anomaly": 1.0 - max(0.0, min(1.0, trace_rank)),
            "trace_hop_lag": abs(_number(source, "trace_server_latency_abs_z_gap")),
            "topology_depth": topo_in, "topology_width": topo_out,
            "topology_direction_consistency": _ratio(topo_out_weight, topo_in_weight + topo_out_weight + 1e-12),
            "relative_onset": _ratio(metric_active, max(metric_samples, 1.0)),
            "relative_peak": max(0.0, min(1.0, metric_rank)),
            "relative_recovery": _ratio(metric_last, max(metric_max, 1e-12)),
            "candidate_is_service": _number(source, "entity_is_service"),
            "candidate_reachability": _ratio(topo_out, topo_in + topo_out + 1e-12),
            "candidate_source_earliness": 1.0 - max(0.0, min(1.0, min(metric_rank, trace_rank))),
            "candidate_explanation_coverage": modalities / 3.0,
            "has_metric_signal": has_metric, "has_log_signal": has_log,
            "has_trace_signal": has_trace, "has_topology_signal": 1,
            "has_time_signal": int(has_metric or has_trace), "has_candidate_signal": 1,
        })
    return rows


def _materialize_source_with_backend(
    unit: Mapping[str, Any], selected_case_ids: tuple[str, ...], dataset_root: Path
) -> dict[str, Any]:
    from rcl_study.query_active_real_execution import run_deepening_query_only_t1_t2_unit

    runtime = deepcopy(dict(unit))
    namespace = dataset_root / "source-backend"
    runtime.update({
        "unit_id": f"formal-calibration-source.{unit['canonical_dataset_id']}.seed42",
        "selector_id": "fixed_budget_set", "selector_config": {},
        "selector_config_sha256": semantic_sha256({}),
        "selected_case_ids": list(selected_case_ids),
        "method_id": "ordinary_multimodal_active_learning",
        "namespace_root": str(namespace), "output_root": str(namespace / "backend"),
        "transform_root": str(namespace / "transform"), "selector_root": str(namespace / "selector"),
        "temp_root": str(namespace / "tmp"), "checkpoint_root": str(namespace / "checkpoints"),
        "score_root": str(namespace / "scores"), "marker_root": str(namespace / "markers"),
    })
    run_deepening_query_only_t1_t2_unit(runtime, dataset_root)
    backend = namespace / "backend"
    query = _read(backend / "views" / "T1" / "query_plan.json")
    feature_columns = tuple(str(value) for value in query["metadata"]["selected_feature_columns"])
    frame = pd.read_csv(backend / "views" / "T1" / "training_frame.csv")
    rows = _rows_from_training_frame(frame, feature_columns)
    return {
        "schema_version": "conservative-lofo-formal-state-source-v1",
        "dataset_id": str(unit["canonical_dataset_id"]),
        "base_feature_names": feature_columns,
        "observable_schema": {
            "metric_fields": _default_schema().metric_fields,
            "log_fields": _default_schema().log_fields,
            "trace_fields": _default_schema().trace_fields,
            "topology_fields": _default_schema().topology_fields,
            "time_fields": _default_schema().time_fields,
            "candidate_fields": _default_schema().candidate_fields,
            "modality_presence_fields": dict(_default_schema().modality_presence_fields),
            "clip_value": _default_schema().clip_value,
        },
        "rows": rows,
    }


def _source_payload(
    state_source_root: Path,
    unit: Mapping[str, Any],
    selected_case_ids: tuple[str, ...],
    dataset_root: Path,
) -> dict[str, Any]:
    path = state_source_root / str(unit["canonical_dataset_id"]) / "candidate_rows.json"
    source = _read(path) if path.is_file() else _materialize_source_with_backend(
        unit, selected_case_ids, dataset_root
    )
    if source.get("schema_version") != "conservative-lofo-formal-state-source-v1":
        raise ValueError("unexpected formal state-source schema")
    if source.get("dataset_id") != unit["canonical_dataset_id"]:
        raise ValueError("formal state-source dataset drift")
    return source


def _materialize_cases(
    dataset_id: str,
    selected_case_ids: tuple[str, ...],
    source: Mapping[str, Any],
    dataset_root: Path,
) -> tuple[dict[str, Any], dict[str, Any], str]:
    schema = _schema_from_payload(dict(source.get("observable_schema", {})))
    all_rows = [dict(value) for value in source.get("rows", ())]
    for row in all_rows:
        for field in schema.modality_presence_fields.values():
            if field not in row:
                raise ValueError(f"missing explicit modality presence field: {field}")
    selected_set = set(selected_case_ids)
    rows = [row for row in all_rows if str(row.get("case_id")) in selected_set]
    if {str(row.get("case_id")) for row in rows} != selected_set:
        raise ValueError(f"candidate-complete source coverage drift for {dataset_id}")
    transform = fit_fold_state_transform(rows, schema, selected_case_ids, ())
    artifact = build_candidate_state_rows(
        rows, schema, transform, selected_case_ids, "formal_calibration_candidate_complete_state"
    )
    state_lookup = {
        (member["case_id"], member["candidate_id"]): state
        for member, state in zip(artifact["membership"], artifact["state_rows"])
    }
    by_case: dict[str, list[dict[str, Any]]] = {case_id: [] for case_id in selected_case_ids}
    for row in rows:
        by_case[str(row["case_id"])].append(row)
    base_names = tuple(str(value) for value in source.get("base_feature_names", ()))
    if not base_names:
        raise ValueError("formal state source lacks base feature names")
    cases = {}
    for case_id in selected_case_ids:
        candidate_rows = by_case[case_id]
        candidate_ids = tuple(str(row.get("candidate_id", "")) for row in candidate_rows)
        if "" in candidate_ids or len(candidate_ids) != len(set(candidate_ids)):
            raise ValueError(f"candidate identity drift for {case_id}")
        positives = tuple(index for index, row in enumerate(candidate_rows) if int(row.get("label", 0)) == 1)
        if not positives:
            raise ValueError(f"queried case lacks a positive candidate: {case_id}")
        base_rows = [list(map(float, row.get("base_features", ()))) for row in candidate_rows]
        if any(len(values) != len(base_names) or not all(math.isfinite(value) for value in values) for values in base_rows):
            raise ValueError(f"base feature row drift for {case_id}")
        state_rows = [state_lookup[(case_id, candidate_id)] for candidate_id in candidate_ids]
        cases[case_id] = {
            "candidate_ids": candidate_ids,
            "positive_indices": positives,
            "base_feature_rows": base_rows,
            "observable_rows": candidate_rows,
            "states": [row["state_vector"] for row in state_rows],
            "state_masks": [row["state_mask"] for row in state_rows],
            "modality_masks": [row["modality_mask"] for row in state_rows],
            "modality_coverage": [row["coverage"] for row in state_rows],
            "evidence_supported": [any(int(value) for value in row["state_mask"]) for row in state_rows],
        }
    cases_identity = {
        "schema_version": "conservative-lofo-formal-cases-v1",
        "dataset_id": dataset_id,
        "base_feature_names": base_names,
        "observable_schema": schema.to_dict(),
        "cases": cases,
        "candidate_count": sum(len(case["candidate_ids"]) for case in cases.values()),
    }
    cases_payload = {**cases_identity, "cases_sha256": _hash(cases_identity)}
    _write(dataset_root / "cases.json", cases_payload)
    _write(dataset_root / "transform.json", transform)
    _write(dataset_root / "state_artifact.json", artifact)
    return cases_payload, transform, artifact["artifact_sha256"]


def _authority_rows_and_targets(
    cases: Mapping[str, Any],
    case_ids: tuple[str, ...],
    feature_names: tuple[str, ...],
) -> tuple[list[dict[str, Any]], dict[str, tuple[str, ...]], dict[str, tuple[str, ...]]]:
    rows: list[dict[str, Any]] = []
    targets: dict[str, tuple[str, ...]] = {}
    candidates_by_case: dict[str, tuple[str, ...]] = {}
    for case_id in case_ids:
        case = dict(cases[case_id])
        candidates = tuple(str(value) for value in case["candidate_ids"])
        base_rows = tuple(tuple(float(value) for value in row) for row in case["base_feature_rows"])
        positives = tuple(int(value) for value in case["positive_indices"])
        if len(candidates) != len(base_rows) or any(len(row) != len(feature_names) for row in base_rows):
            raise ValueError(f"authority candidate/base feature drift for {case_id}")
        candidates_by_case[case_id] = candidates
        targets[case_id] = tuple(candidates[index] for index in positives)
        for candidate_id, values in zip(candidates, base_rows):
            rows.append({
                "case_id": case_id,
                "candidate_id": candidate_id,
                **dict(zip(feature_names, values)),
            })
    return rows, targets, candidates_by_case


def _materialize_authority_score_bundles(
    *,
    dataset_id: str,
    selected_case_ids: tuple[str, ...],
    labels_by_case: Mapping[str, Mapping[str, str]],
    query_plan_sha256: str,
    cases_payload: Mapping[str, Any],
    output_root: Path,
) -> dict[str, dict[str, Any]]:
    cases = {str(key): dict(value) for key, value in dict(cases_payload["cases"]).items()}
    feature_names = tuple(str(value) for value in cases_payload["base_feature_names"])
    bindings: dict[str, dict[str, Any]] = {}
    folds = build_nested_calibration_folds(dataset_id, selected_case_ids, labels_by_case)
    for fold in folds:
        fold_id = str(fold["fold_id"])
        support_ids = tuple(str(value) for value in fold["support_case_ids"])
        evaluation_ids = tuple(str(value) for value in fold["validation_case_ids"])
        support_rows, support_targets, support_candidates = _authority_rows_and_targets(
            cases, support_ids, feature_names
        )
        evaluation_rows, evaluation_targets, evaluation_candidates = _authority_rows_and_targets(
            cases, evaluation_ids, feature_names
        )
        bridge = fit_base_score_bridge(
            dataset_id=dataset_id,
            training_rows=support_rows,
            targets_by_case=support_targets,
            feature_columns=feature_names,
            query_plan_sha256=query_plan_sha256,
            random_state=42,
        )
        support_artifact = score_base_score_bridge(
            bridge=bridge,
            rows=support_rows,
            targets_by_case=support_targets,
            artifact_role="support",
            expected_candidates_by_case=support_candidates,
        )
        evaluation_artifact = score_base_score_bridge(
            bridge=bridge,
            rows=evaluation_rows,
            targets_by_case=evaluation_targets,
            artifact_role="evaluation",
            expected_candidates_by_case=evaluation_candidates,
        )
        bundle = build_authority_score_bundle(
            dataset_id=dataset_id,
            fold_id=fold_id,
            fold_kind="pseudo_lofo",
            held_out_fault_type=str(fold["pseudo_held_out_fault_type"]),
            query_plan_sha256=query_plan_sha256,
            feature_order_sha256=bridge.feature_order_sha256,
            model_sha256=bridge.model_sha256,
            support_case_ids=support_ids,
            evaluation_case_ids=evaluation_ids,
            support_artifact=support_artifact,
            evaluation_artifact=evaluation_artifact,
        )
        relative_path = Path("authority_scores") / dataset_id / f"{fold_id}.json"
        _write(output_root / relative_path, bundle)
        bindings[fold_id] = {
            "fold_id": fold_id,
            "held_out_fault_type": str(fold["pseudo_held_out_fault_type"]),
            "support_case_ids": support_ids,
            "evaluation_case_ids": evaluation_ids,
            "authority_score_bundle_path": relative_path.as_posix(),
            "authority_score_bundle_sha256": bundle["authority_score_bundle_sha256"],
            "authority_model_sha256": bundle["model_sha256"],
            "authority_feature_order_sha256": bundle["feature_order_sha256"],
            "authority_score_artifact_sha256": dict(bundle["score_artifact_sha256"]),
        }
    return bindings


def _prepare_dataset(
    dataset_id: str,
    unit: Mapping[str, Any],
    state_source_root: Path,
    output_root: Path,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, dict[str, str]]]:
    inventory, inventory_rows = _inventory(Path(unit["inventory_path"]), dataset_id)
    actual_excluded = _actual_exclusions(dataset_id, inventory)
    empty_pool = build_union_excluded_calibration_pool(dataset_id, inventory, actual_excluded)
    fusion = _read(Path(unit["fusion_artifact_path"]))
    context = _read(Path(unit["candidate_context_path"]))
    merged = merge_fusion_and_candidate_context(fusion, context)
    admitted = set(empty_pool["admitted_candidate_case_ids"])
    candidates = [dict(row) for row in merged["candidates"] if row["case_id"] in admitted]
    answers = {
        case_id: dict(value)
        for case_id, value in merged["private_annotations"].items()
        if case_id in admitted
    }
    if {row["case_id"] for row in candidates} != admitted or set(answers) != admitted:
        raise ValueError(f"union-excluded candidate/context coverage drift for {dataset_id}")
    selection = run_ordinary_query_engine(
        candidates,
        annotation_oracle=MappingAnnotationOracle(answers),
        strategy_id="dbscan_coverage",
        active_learning_seed=42,
        config=dict(unit["strategy_config"]),
    )
    selected = tuple(str(value) for value in selection["selected_case_ids"])
    if len(selected) != 30 or set(selected) & set(empty_pool["excluded_case_ids"]):
        raise ValueError(f"union-excluded DBSCAN selection drift for {dataset_id}")
    labels = {case_id: answers[case_id] for case_id in selected}
    raw_pool = build_union_excluded_calibration_pool(
        dataset_id, inventory, actual_excluded, queried_case_ids=selected
    )
    pool = _bundle_pool(dataset_id, raw_pool, selected, actual_excluded)
    dataset_root = output_root / "datasets" / dataset_id
    query_identity = {
        "schema_version": "conservative-lofo-formal-query-plan-v1",
        "dataset_id": dataset_id,
        "strategy_id": "dbscan_coverage",
        "strategy_config": dict(unit["strategy_config"]),
        "active_learning_seed": 42,
        "budget": 30,
        "candidate_case_ids": tuple(row["case_id"] for row in candidates),
        "selected_case_ids": selected,
        "events": selection["events"],
        "engine_query_plan_sha256": selection["query_plan_sha256"],
        "screening_target_union_excluded": True,
    }
    query_plan = {**query_identity, "query_plan_sha256": _hash(query_identity)}
    labels_identity = {
        "schema_version": "conservative-lofo-formal-development-labels-v1",
        "dataset_id": dataset_id,
        "development_label_cost": 30,
        "labels_by_case": labels,
        "selected_case_ids": selected,
    }
    labels_payload = {**labels_identity, "labels_sha256": _hash(labels_identity)}
    _write(dataset_root / "query_plan.json", query_plan)
    _write(dataset_root / "labels.json", labels_payload)
    _write(dataset_root / "pool.json", pool)
    source = _source_payload(state_source_root, unit, selected, dataset_root)
    cases, transform, _ = _materialize_cases(dataset_id, selected, source, dataset_root)
    authority_bindings = _materialize_authority_score_bundles(
        dataset_id=dataset_id,
        selected_case_ids=selected,
        labels_by_case=labels,
        query_plan_sha256=query_plan["query_plan_sha256"],
        cases_payload=cases,
        output_root=output_root,
    )
    screening_ids = tuple(str(value) for value in raw_pool["excluded_case_ids"])
    execution_dataset = {
        "dataset_id": dataset_id,
        "queried_case_ids": selected,
        "fit_case_ids": selected,
        "labels_by_case": labels,
        "screening_target_case_ids": screening_ids,
        "formal_training_case_ids": screening_ids,
        "query_plan_sha256": query_plan["query_plan_sha256"],
        "cases_sha256": cases["cases_sha256"],
        "transform_sha256": transform["transform_sha256"],
        "authority_score_bundles_by_fold": authority_bindings,
    }
    return execution_dataset, pool, labels


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ordinary-run-root", type=Path, required=True)
    parser.add_argument("--state-source-root", type=Path, required=True)
    parser.add_argument("--profiles", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--code-sha256", required=True)
    args = parser.parse_args()
    if len(args.code_sha256) != 64 or any(value not in "0123456789abcdef" for value in args.code_sha256.lower()):
        raise ValueError("code-sha256 must be 64 lowercase hex characters")
    ordinary_manifest = _read(args.ordinary_run_root / "ordinary_strategy_manifest.json")
    profiles_source = _read(args.profiles)
    profiles = dict(profiles_source.get("arms", {}))
    if tuple(profiles) != ARMS or any(len(tuple(profiles[arm])) != 4 for arm in ARMS):
        raise ValueError("formal calibration requires four frozen profiles per arm")
    args.output_root.mkdir(parents=True, exist_ok=True)
    execution_inputs = {}
    pools = {}
    labels = {}
    for dataset_id in DATASETS:
        execution_inputs[dataset_id], pools[dataset_id], labels[dataset_id] = _prepare_dataset(
            dataset_id,
            _ordinary_unit(ordinary_manifest, dataset_id),
            args.state_source_root,
            args.output_root,
        )
    prepare_identity = {
        "schema_version": "conservative-lofo-calibration-prepare-input-v1",
        "pools_by_dataset": pools,
        "queried_labels_by_dataset": labels,
        "formal_training_case_ids_by_dataset": {
            dataset: execution_inputs[dataset]["formal_training_case_ids"] for dataset in DATASETS
        },
        "profiles_by_arm": profiles,
    }
    prepare_input = {**prepare_identity, "input_sha256": _hash(prepare_identity)}
    prepare_path = args.output_root / "prepare_input.json"
    _write(prepare_path, prepare_input)
    subprocess.run(
        [
            sys.executable,
            str(REPO_ROOT / "scripts" / "prepare_conservative_lofo_calibration.py"),
            "--input", str(prepare_path), "--output-root", str(args.output_root),
        ],
        cwd=REPO_ROOT,
        check=True,
    )
    manifest = _read(args.output_root / "manifest.json")
    execution = build_formal_calibration_execution(
        calibration_manifest=manifest,
        datasets_by_id=execution_inputs,
        code_sha256=args.code_sha256,
        output_root=str(args.output_root.resolve()),
    )
    execution_path = args.output_root / "execution.json"
    if execution_path.exists() and _read(execution_path) != execution:
        raise ValueError("formal calibration output root contains a conflicting execution")
    _write(execution_path, execution)
    print(json.dumps({
        "status": "prepared",
        "execution_sha256": execution["execution_sha256"],
        "development_label_cost": 60,
        "unit_count": 12,
    }, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
