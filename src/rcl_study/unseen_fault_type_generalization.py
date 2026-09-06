"""Utilities for leave-one-fault-type query-only RCL diagnostics."""

from __future__ import annotations

import hashlib
import json
import math
import os
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence


HIT_METRICS = ("hit_at_1", "hit_at_3", "hit_at_5")
RANKING_METRICS = (*HIT_METRICS, "mrr")
FAULT_TYPE_FIELDS = ("fault_type", "fault_type_oracle", "true_fault_type")
CHANGE_ID = "evaluate-unseen-category-rcl-generalization"
REMOTE_WORKSPACE = "${RCL_WORKSPACE}"
REMOTE_OUTPUT_PREFIX = (
    REMOTE_WORKSPACE + "/outputs/rcl_study/unseen_fault_type_generalization"
)
SPLIT_SEED = 42
DEFAULT_ACTIVE_LEARNING_SEEDS = (42, 43, 44)
DEFAULT_SELECTOR_ID = "sequential_proxy_mode_query"
DEFAULT_SELECTOR_FAMILY = "sequential_proxy_mode"
DEFAULT_SELECTOR_CONFIG = {
    "embedding_field": "embedding",
    "numeric_fields": ["inner_boundary_uncertainty", "baseline_score"],
    "categorical_fields": ["cluster_id", "time_bucket"],
    "score_field": "inner_boundary_uncertainty",
    "proxy_mode_count": 14,
    "max_per_proxy_mode": 5,
    "seed_budget": 8,
    "metric_ad_required": False,
}
UNKNOWN_MODE_SIGNATURE_RESERVE_SELECTOR_ID = "unknown_mode_signature_reserve"
UNKNOWN_MODE_SIGNATURE_RESERVE_SELECTOR_FAMILY = "unknown_mode_signature_reserve"
UNKNOWN_MODE_SIGNATURE_RESERVE_CONFIG = {
    "embedding_field": "embedding",
    "numeric_fields": ["inner_boundary_uncertainty", "baseline_score", "scheme1_rank"],
    "categorical_fields": ["cluster_id", "time_bucket"],
    "score_field": "inner_boundary_uncertainty",
    "proxy_mode_count": 14,
    "max_per_proxy_mode": 5,
    "unknown_mode_reserve_budget": 8,
    "unknown_mode_weight": 0.40,
    "coverage_weight": 0.35,
    "utility_weight": 0.20,
    "mode_novelty_weight": 0.05,
    "metric_ad_required": False,
    "metric_ad_fields": [
        "metric_ad_anomaly_direction",
        "metric_ad_duration",
        "metric_ad_metric_family_count",
        "metric_ad_propagation_width",
        "metric_ad_sparsity",
        "metric_ad_robust_z_mean",
        "metric_ad_robust_z_peak",
        "metric_ad_start_slope",
        "metric_ad_peak_lag",
        "metric_ad_recovery_slope",
    ],
}
DATASET_SOURCE_PATHS = {
    "rcabench": "${RCABENCH_ROOT}",
    "aiops2022_pre": "${AIOPS22_ROOT}",
}


def _json_bytes(payload: Any) -> bytes:
    return json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    ).encode("utf-8")


def semantic_sha256(payload: Any) -> str:
    return hashlib.sha256(_json_bytes(payload)).hexdigest()


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _write_json_atomic(path: Path | str, payload: Mapping[str, Any]) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name("%s.tmp-%d" % (destination.name, os.getpid()))
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2, default=str)
        + "\n",
        encoding="utf-8",
    )
    os.replace(str(temporary), str(destination))


def _safe_token(value: Any, context: str = "token") -> str:
    text = str(value).strip()
    if not text or "/" in text or "\\" in text or text in {".", ".."}:
        raise ValueError("unsafe %s: %r" % (context, value))
    return "".join(char if char.isalnum() or char in ("-", "_", ".") else "_" for char in text)


def _require_output_root(value: Any, *, allow_local_output_root: bool = False) -> str:
    root = str(value).rstrip("/\\")
    if not root:
        raise ValueError("output_root must not be empty")
    if allow_local_output_root:
        return root
    if not root.startswith(REMOTE_OUTPUT_PREFIX + "/"):
        raise ValueError("unseen output_root must be below %s" % REMOTE_OUTPUT_PREFIX)
    return root


def _selector_config_sha256(selector_id: str, selector_config: Mapping[str, Any]) -> str:
    return semantic_sha256(
        {
            "selector_id": str(selector_id),
            "selector_config": deepcopy(dict(selector_config)),
        }
    )


def _clean_allowed_input_contract(selector_id: str) -> str:
    selector = str(selector_id)
    if selector == UNKNOWN_MODE_SIGNATURE_RESERVE_SELECTOR_ID:
        return "label_free_metric_signature_unknown_mode_reserve"
    if selector == DEFAULT_SELECTOR_ID:
        return "label_free_plus_budgeted_revealed_labels"
    return "label_free_only"


def _unit_output_root(
    *,
    base_root: str,
    stage: str,
    dataset: str,
    seed: int,
    unit_id: str,
) -> str:
    return "/".join(
        [
            str(base_root).rstrip("/"),
            _safe_token(stage, "stage"),
            _safe_token(dataset, "dataset"),
            "seed%d" % int(seed),
            _safe_token(unit_id, "unit_id"),
        ]
    )


def _case_id(row: Mapping[str, Any]) -> str:
    text = str(row.get("case_id", "")).strip()
    if not text:
        raise ValueError("case metadata row missing case_id")
    return text


def fault_type_from_case(row: Mapping[str, Any]) -> str:
    """Return a stable true fault-type value from canonical metadata."""

    for field in FAULT_TYPE_FIELDS:
        value = str(row.get(field, "")).strip()
        if value:
            return value
    raise ValueError("case %s missing non-empty fault_type" % _case_id(row))


def _ordered_unique(values: Sequence[Any], context: str) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    for value in values:
        text = str(value).strip()
        if not text:
            raise ValueError("%s contains empty case_id" % context)
        if text in seen:
            raise ValueError("%s contains duplicate case_id: %s" % (context, text))
        seen.add(text)
        result.append(text)
    return result


def _fault_type_support_threshold(total_t1_cases: int) -> int:
    if int(total_t1_cases) <= 0:
        return 1
    return max(1, min(10, int(math.ceil(int(total_t1_cases) * 0.05))))


def build_fault_type_inventory(
    *,
    canonical_dataset_id: str,
    cases: Sequence[Mapping[str, Any]],
    outer_train_case_ids: Sequence[Any],
    outer_test_case_ids: Sequence[Any],
    t2_case_ids: Sequence[Any],
    budget: int,
) -> dict[str, Any]:
    """Build per-fault-type counts and eligibility flags for diagnostics."""

    if int(budget) <= 0:
        raise ValueError("budget must be positive")
    rows = [deepcopy(dict(row)) for row in cases]
    if not rows:
        raise ValueError("fault_type inventory requires cases")
    by_case: dict[str, dict[str, Any]] = {}
    for row in rows:
        case_id = _case_id(row)
        if case_id in by_case:
            raise ValueError("duplicate case metadata row: %s" % case_id)
        row["fault_type"] = fault_type_from_case(row)
        by_case[case_id] = row
    outer_train = _ordered_unique(outer_train_case_ids, "outer_train_case_ids")
    outer_test = _ordered_unique(outer_test_case_ids, "outer_test_case_ids")
    t2_ids = _ordered_unique(t2_case_ids, "t2_case_ids")
    for context, ids in (
        ("outer_train_case_ids", outer_train),
        ("outer_test_case_ids", outer_test),
        ("t2_case_ids", t2_ids),
    ):
        missing = [case_id for case_id in ids if case_id not in by_case]
        if missing:
            raise ValueError("%s references unknown cases: %s" % (context, missing[:5]))
    train_set = set(outer_train)
    test_set = set(outer_test)
    t2_set = set(t2_ids)
    total_by_type: dict[str, dict[str, Any]] = {}
    for case_id, row in sorted(by_case.items()):
        fault_type = str(row["fault_type"])
        bucket = total_by_type.setdefault(
            fault_type,
            {
                "fault_type": fault_type,
                "outer_train_count": 0,
                "outer_test_t1_count": 0,
                "t2_eligible_count": 0,
                "total_candidate_count": 0,
                "outer_train_case_ids": [],
                "outer_test_t1_case_ids": [],
                "t2_case_ids": [],
                "all_case_ids": [],
            },
        )
        bucket["total_candidate_count"] += 1
        bucket["all_case_ids"].append(case_id)
        if case_id in train_set:
            bucket["outer_train_count"] += 1
            bucket["outer_train_case_ids"].append(case_id)
        if case_id in test_set:
            bucket["outer_test_t1_count"] += 1
            bucket["outer_test_t1_case_ids"].append(case_id)
        if case_id in t2_set:
            bucket["t2_eligible_count"] += 1
            bucket["t2_case_ids"].append(case_id)
    support_threshold = _fault_type_support_threshold(len(outer_test))
    for bucket in total_by_type.values():
        failed: list[str] = []
        if int(bucket["outer_train_count"]) < 1:
            failed.append("no_outer_train_candidate")
        if int(bucket["outer_test_t1_count"]) < support_threshold:
            failed.append("insufficient_t1_support")
        nonheldout_pool = len(outer_train) - int(bucket["outer_train_count"])
        bucket["nonheldout_outer_train_candidate_count"] = nonheldout_pool
        if nonheldout_pool < int(budget):
            failed.append("insufficient_nonheldout_budget_pool")
        eligible = not failed
        bucket["support_threshold_t1_cases"] = support_threshold
        bucket["eligible"] = eligible
        bucket["low_support"] = not eligible
        bucket["failed_conditions"] = failed
    inventory_identity = {
        "schema_version": "rcl-unseen-fault-type-inventory-v1",
        "canonical_dataset_id": str(canonical_dataset_id),
        "budget": int(budget),
        "outer_train_case_ids": outer_train,
        "outer_test_case_ids": outer_test,
        "t2_case_ids": t2_ids,
        "fault_types": total_by_type,
    }
    return {
        **inventory_identity,
        "admission_counts": {
            "outer_train_fault_cases": len(outer_train),
            "outer_test_t1_fault_cases": len(outer_test),
            "t2_eligible_fault_cases": len(t2_ids),
            "total_candidate_fault_cases": len(rows),
        },
        "eligible_fault_types": [
            fault_type
            for fault_type, bucket in sorted(total_by_type.items())
            if bool(bucket["eligible"])
        ],
        "low_support_fault_types": [
            fault_type
            for fault_type, bucket in sorted(total_by_type.items())
            if bool(bucket["low_support"])
        ],
        "inventory_sha256": semantic_sha256(inventory_identity),
    }


def _unseen_unit(
    *,
    base_root: str,
    stage: str,
    dataset: str,
    seed: int,
    budget: int,
    selector_id: str,
    selector_family: str,
    selector_config: Mapping[str, Any],
    oracle_only: bool,
    method_id: str,
    inventory_sha256: str,
    held_out_fault_type: str | None = None,
) -> dict[str, Any]:
    held = str(held_out_fault_type or "").strip()
    unit_id = (
        "leave_one.%s.%s.seed%d" % (dataset, _safe_token(held, "held_out_fault_type"), seed)
        if held
        else "baseline.%s.seed%d" % (dataset, seed)
    )
    unit = {
        "unit_id": unit_id,
        "stage": str(stage),
        "canonical_dataset_id": str(dataset),
        "source_path": DATASET_SOURCE_PATHS[str(dataset)],
        "mode": "run_unseen_fault_type_query_t1_t2_unit",
        "method_id": str(method_id),
        "selector_id": str(selector_id),
        "selector_family": str(selector_family),
        "selector_config": deepcopy(dict(selector_config)),
        "selector_config_sha256": _selector_config_sha256(
            selector_id,
            selector_config,
        ),
        "allowed_input_contract": (
            "diagnostic_forced_fault_type_exclusion"
            if oracle_only
            else _clean_allowed_input_contract(selector_id)
        ),
        "oracle_only": bool(oracle_only),
        "budget": int(budget),
        "budget_unit": "unique_outer_training_fault_cases",
        "active_learning_seed": int(seed),
        "seed": int(seed),
        "split_seed": SPLIT_SEED,
        "normal_policy": "fault_only",
        "evaluation_views": ["T1", "T2"],
        "fault_type_inventory_sha256": str(inventory_sha256),
    }
    if held:
        unit["held_out_fault_type"] = held
        unit["diagnostic_protocol"] = {
            "oracle_use": "force_absence_of_true_fault_type_from_budget",
            "official_score_view": "T1",
            "diagnostic_score_view": "T2",
        }
    unit["output_root"] = _unit_output_root(
        base_root=base_root,
        stage=stage,
        dataset=dataset,
        seed=seed,
        unit_id=unit_id,
    )
    return unit


def build_unseen_manifest(
    *,
    run_id: str,
    output_root: str,
    inventories: Mapping[str, Mapping[str, Any]],
    active_learning_seeds: Sequence[Any] = DEFAULT_ACTIVE_LEARNING_SEEDS,
    budget: int = 30,
    selector_id: str = DEFAULT_SELECTOR_ID,
    selector_family: str = DEFAULT_SELECTOR_FAMILY,
    selector_config: Mapping[str, Any] | None = None,
    max_held_out_types_per_dataset: int | None = None,
    allow_local_output_root: bool = False,
) -> dict[str, Any]:
    """Build a matched-baseline plus leave-one-fault-type diagnostic manifest."""

    run_id_text = _safe_token(run_id, "run_id")
    root = _require_output_root(output_root, allow_local_output_root=allow_local_output_root)
    seeds = [int(seed) for seed in active_learning_seeds]
    if not seeds or len(seeds) != len(set(seeds)):
        raise ValueError("active_learning_seeds must be unique and non-empty")
    if int(budget) <= 0:
        raise ValueError("budget must be positive")
    max_held_out = (
        None
        if max_held_out_types_per_dataset is None
        else int(max_held_out_types_per_dataset)
    )
    if max_held_out is not None and max_held_out <= 0:
        raise ValueError("max_held_out_types_per_dataset must be positive when set")
    config = deepcopy(dict(selector_config or DEFAULT_SELECTOR_CONFIG))
    units: list[dict[str, Any]] = []
    normalized_inventories: dict[str, dict[str, Any]] = {}
    for dataset, inventory_raw in sorted(inventories.items()):
        dataset_id = str(dataset)
        if dataset_id not in DATASET_SOURCE_PATHS:
            raise ValueError("unsupported unseen dataset: %s" % dataset_id)
        inventory = deepcopy(dict(inventory_raw))
        if str(inventory.get("canonical_dataset_id")) != dataset_id:
            raise ValueError("inventory dataset mismatch: %s" % dataset_id)
        eligible = sorted(str(value) for value in inventory.get("eligible_fault_types") or [])
        if max_held_out is not None:
            eligible = eligible[:max_held_out]
        normalized_inventories[dataset_id] = inventory
        for seed in seeds:
            units.append(
                _unseen_unit(
                    base_root=root,
                    stage="baseline_replay",
                    dataset=dataset_id,
                    seed=seed,
                    budget=int(budget),
                    selector_id=selector_id,
                    selector_family=selector_family,
                    selector_config=config,
                    oracle_only=False,
                    method_id="current_label_free_reference",
                    inventory_sha256=str(inventory.get("inventory_sha256", "")),
                )
            )
            for fault_type in eligible:
                units.append(
                    _unseen_unit(
                        base_root=root,
                        stage="leave_one_fault_type",
                        dataset=dataset_id,
                        seed=seed,
                        budget=int(budget),
                        selector_id=selector_id,
                        selector_family=selector_family,
                        selector_config=config,
                        oracle_only=True,
                        method_id="leave_one_fault_type_diagnostic",
                        inventory_sha256=str(inventory.get("inventory_sha256", "")),
                        held_out_fault_type=fault_type,
                    )
                )
    identity = {
        "schema_version": "rcl-unseen-fault-type-generalization-manifest-v1",
        "change_id": CHANGE_ID,
        "run_id": run_id_text,
        "output_root": root,
        "split_seed": SPLIT_SEED,
        "budget": int(budget),
        "active_learning_seeds": seeds,
        "selector_id": str(selector_id),
        "selector_family": str(selector_family),
        "selector_config": config,
        "selector_config_sha256": _selector_config_sha256(selector_id, config),
        "max_held_out_types_per_dataset": max_held_out,
        "fault_type_inventory": normalized_inventories,
        "units": units,
        "expected_unit_count": len(units),
        "completion_proofs": {
            "COMPLETED.json": root + "/COMPLETED.json",
            "all.done": root + "/all.done",
            ".failed": root + "/.failed",
        },
    }
    manifest = {**identity, "manifest_sha256": semantic_sha256(identity)}
    validate_unseen_manifest(manifest, allow_local_output_root=allow_local_output_root)
    return manifest


def build_label_free_mitigation_manifest_from_diagnostic_report(
    *,
    run_id: str,
    output_root: str,
    diagnostic_report: Mapping[str, Any],
    active_learning_seeds: Sequence[Any] = DEFAULT_ACTIVE_LEARNING_SEEDS,
    budget: int = 30,
    selector_id: str = UNKNOWN_MODE_SIGNATURE_RESERVE_SELECTOR_ID,
    selector_family: str = UNKNOWN_MODE_SIGNATURE_RESERVE_SELECTOR_FAMILY,
    selector_config: Mapping[str, Any] | None = None,
    max_held_out_types_per_dataset: int | None = None,
    allow_local_output_root: bool = False,
) -> dict[str, Any]:
    """Build a label-free mitigation manifest for exposed datasets/types only."""

    report = deepcopy(dict(diagnostic_report))
    inventories = _require_mapping(report.get("inventory"), "diagnostic_report.inventory")
    threats = _require_mapping(report.get("dataset_threats"), "diagnostic_report.dataset_threats")
    max_held_out = (
        None
        if max_held_out_types_per_dataset is None
        else int(max_held_out_types_per_dataset)
    )
    if max_held_out is not None and max_held_out <= 0:
        raise ValueError("max_held_out_types_per_dataset must be positive when set")
    selected_inventories: dict[str, dict[str, Any]] = {}
    for dataset, threat_raw in sorted(threats.items()):
        threat = _require_mapping(threat_raw, "dataset_threats.%s" % dataset)
        if not bool(threat.get("exposed_to_unseen_fault_type_threat")):
            continue
        material_types = [
            str(value)
            for value in _require_sequence(
                threat.get("material_degraded_fault_types"),
                "dataset_threats.%s.material_degraded_fault_types" % dataset,
            )
            if str(value).strip()
        ]
        if not material_types:
            raise ValueError("exposed dataset has no material degraded fault types: %s" % dataset)
        inventory = deepcopy(
            dict(_require_mapping(inventories.get(dataset), "inventory.%s" % dataset))
        )
        eligible = [str(value) for value in inventory.get("eligible_fault_types") or []]
        missing = sorted(set(material_types) - set(eligible))
        if missing:
            raise ValueError(
                "material degraded fault types are not eligible in inventory for %s: %s"
                % (dataset, missing)
            )
        eligible_material = [
            fault_type for fault_type in eligible if fault_type in set(material_types)
        ]
        if max_held_out is not None:
            eligible_material = eligible_material[:max_held_out]
        inventory["eligible_fault_types"] = eligible_material
        selected_inventories[str(dataset)] = inventory
    if not selected_inventories:
        raise ValueError("diagnostic report exposes no dataset-level unseen fault-type threat")
    config = deepcopy(dict(selector_config or UNKNOWN_MODE_SIGNATURE_RESERVE_CONFIG))
    manifest = build_unseen_manifest(
        run_id=run_id,
        output_root=output_root,
        inventories=selected_inventories,
        active_learning_seeds=active_learning_seeds,
        budget=int(budget),
        selector_id=str(selector_id),
        selector_family=str(selector_family),
        selector_config=config,
        allow_local_output_root=allow_local_output_root,
    )
    manifest["source_diagnostic_report_sha256"] = str(
        report.get("diagnostic_report_sha256", "")
    )
    manifest["mitigation_protocol"] = {
        "schema_version": "rcl-unseen-label-free-mitigation-protocol-v1",
        "selector_id": str(selector_id),
        "selector_family": str(selector_family),
        "clean_selector_label_access": _clean_allowed_input_contract(str(selector_id)),
        "held_out_type_source": "material_degraded_types_from_diagnostic_report",
        "oracle_only_notice": (
            "leave-one mitigation stress units still use true fault_type only to "
            "force absence from the budget; the selector itself is label-free"
        ),
    }
    manifest["manifest_sha256"] = semantic_sha256(
        {key: value for key, value in manifest.items() if key != "manifest_sha256"}
    )
    return manifest


def validate_unseen_manifest(
    manifest: Mapping[str, Any],
    *,
    allow_local_output_root: bool = False,
) -> dict[str, Any]:
    payload = deepcopy(dict(manifest))
    if payload.get("schema_version") != "rcl-unseen-fault-type-generalization-manifest-v1":
        raise ValueError("unexpected unseen manifest schema")
    root = _require_output_root(
        payload.get("output_root"),
        allow_local_output_root=allow_local_output_root,
    )
    if int(payload.get("split_seed", 0)) != SPLIT_SEED:
        raise ValueError("unseen split_seed must remain 42")
    units = [deepcopy(dict(unit)) for unit in payload.get("units") or []]
    if not units:
        raise ValueError("unseen manifest requires units")
    if int(payload.get("expected_unit_count", -1)) != len(units):
        raise ValueError("unseen expected_unit_count mismatch")
    unit_ids = [str(unit.get("unit_id", "")) for unit in units]
    if any(not unit_id for unit_id in unit_ids) or len(unit_ids) != len(set(unit_ids)):
        raise ValueError("unseen unit IDs must be unique and non-empty")
    for unit in units:
        output_root = str(unit.get("output_root", ""))
        if not output_root.startswith(root + "/") and not output_root.startswith(root + "\\"):
            raise ValueError("unseen unit output_root escapes manifest root")
        if int(unit.get("split_seed", 0)) != SPLIT_SEED:
            raise ValueError("unseen unit split_seed must remain 42")
        stage = str(unit.get("stage", ""))
        oracle_only = bool(unit.get("oracle_only"))
        held = str(unit.get("held_out_fault_type", "")).strip()
        if stage == "baseline_replay":
            if oracle_only or held:
                raise ValueError("baseline replay must be clean and have no held_out_fault_type")
        elif stage == "leave_one_fault_type":
            if not oracle_only or not held:
                raise ValueError("leave-one unit must be oracle_only and declare held_out_fault_type")
        else:
            raise ValueError("unknown unseen stage: %s" % stage)
    return {
        "valid": True,
        "run_root": root,
        "unit_count": len(units),
        "datasets": sorted(
            {str(unit.get("canonical_dataset_id")) for unit in units}
        ),
    }


def format_unseen_progress_line(progress: Mapping[str, Any]) -> str:
    total = int(progress.get("total_units", 0))
    completed = int(progress.get("completed_units", 0))
    width = 20
    filled = 0 if total <= 0 else int(width * completed / total)
    bar = "[" + "=" * filled + "." * (width - filled) + "]"
    percent = 0.0 if total <= 0 else 100.0 * completed / total
    return (
        "%s %d/%d %5.1f%% phase=%s dataset=%s held_out=%s seed=%s failures=%d"
        % (
            bar,
            completed,
            total,
            percent,
            progress.get("phase", ""),
            progress.get("dataset", ""),
            progress.get("held_out_fault_type", ""),
            progress.get("active_learning_seed", ""),
            int(progress.get("failure_count", 0)),
        )
    )


def _progress_payload(
    *,
    phase: str,
    unit: Mapping[str, Any] | None,
    completed_units: int,
    total_units: int,
    failure_count: int,
    recent_log_path: str,
) -> dict[str, Any]:
    return {
        "schema_version": "rcl-unseen-fault-type-progress-v1",
        "phase": str(phase),
        "dataset": "" if unit is None else str(unit.get("canonical_dataset_id", "")),
        "held_out_fault_type": "" if unit is None else str(unit.get("held_out_fault_type", "")),
        "active_learning_seed": "" if unit is None else str(unit.get("active_learning_seed", "")),
        "completed_units": int(completed_units),
        "total_units": int(total_units),
        "failure_count": int(failure_count),
        "recent_log_path": str(recent_log_path),
        "updated_at_utc": _utc_now(),
    }


def _call_unit_executor(unit_executor: Any, unit: Mapping[str, Any], run_root: Path) -> Any:
    try:
        return unit_executor(unit, run_root)
    except TypeError as exc:
        try:
            return unit_executor(unit)
        except TypeError:
            raise exc


def _write_failure_marker(
    run_root: Path,
    exc: BaseException,
    *,
    completed_units: int,
    total_units: int,
) -> None:
    _write_json_atomic(
        run_root / ".failed",
        {
            "schema_version": "rcl-unseen-fault-type-failure-v1",
            "status": "failed",
            "error_type": type(exc).__name__,
            "error_message": str(exc),
            "completed_units": int(completed_units),
            "total_units": int(total_units),
            "failed_at_utc": _utc_now(),
        },
    )


def _finite_float(value: Any, context: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError("%s must be a finite number" % context) from exc
    if not math.isfinite(result):
        raise ValueError("%s must be a finite number" % context)
    return result


def _nonnegative_int(value: Any, context: str) -> int:
    if isinstance(value, bool):
        raise ValueError("%s must be a non-negative integer" % context)
    try:
        result = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError("%s must be a non-negative integer" % context) from exc
    if result < 0:
        raise ValueError("%s must be a non-negative integer" % context)
    return result


def _require_mapping(value: Any, context: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError("%s must be an object" % context)
    return deepcopy(dict(value))


def _require_sequence(value: Any, context: str) -> list[Any]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise ValueError("%s must be a sequence" % context)
    return list(value)


def _validate_per_case_evidence(
    row_raw: Mapping[str, Any],
    *,
    expected_dataset_id: str,
    context: str,
) -> dict[str, Any]:
    row = _require_mapping(row_raw, context)
    case_id = str(row.get("case_id", "")).strip()
    if not case_id:
        raise ValueError("%s.case_id must not be empty" % context)
    dataset_id = str(
        row.get("dataset_id") or row.get("canonical_dataset_id") or ""
    ).strip()
    if dataset_id != str(expected_dataset_id):
        raise ValueError(
            "%s.dataset_id must equal %s" % (context, expected_dataset_id)
        )
    if not str(row.get("fault_type", "")).strip():
        raise ValueError("%s.fault_type must not be empty" % context)
    targets = [str(target) for target in _require_sequence(row.get("targets"), context + ".targets")]
    if not any(str(target).strip() for target in targets):
        raise ValueError("%s.targets must not be empty" % context)
    ranking = [
        str(candidate)
        for candidate in _require_sequence(row.get("ranking"), context + ".ranking")
    ]
    first_rank = _nonnegative_int(
        row.get("first_matching_rank", row.get("first_target_rank")),
        context + ".first_matching_rank",
    )
    expected_rank = _first_matching_rank(ranking, targets)
    if first_rank != expected_rank:
        raise ValueError(
            "%s.first_matching_rank mismatch: %d != %d"
            % (context, first_rank, expected_rank)
        )
    reciprocal_rank = _finite_float(row.get("reciprocal_rank"), context + ".reciprocal_rank")
    expected_rr = 0.0 if first_rank == 0 else 1.0 / float(first_rank)
    if abs(reciprocal_rank - expected_rr) > 1e-9:
        raise ValueError("%s.reciprocal_rank mismatch" % context)
    for threshold in (1, 3, 5):
        metric = "hit_at_%d" % threshold
        expected_hit = bool(first_rank and first_rank <= threshold)
        if bool(row.get(metric)) != expected_hit:
            raise ValueError("%s.%s mismatch" % (context, metric))
    return row


def _validate_metric_block(
    block_raw: Any,
    *,
    expected_dataset_id: str,
    context: str,
) -> dict[str, Any]:
    block = _require_mapping(block_raw, context)
    denominator = _nonnegative_int(block.get("denominator"), context + ".denominator")
    case_ids = [str(case_id) for case_id in _require_sequence(block.get("case_ids"), context + ".case_ids")]
    if len(case_ids) != denominator:
        raise ValueError("%s.case_ids length must equal denominator" % context)
    hit_counts = _require_mapping(block.get("hit_counts"), context + ".hit_counts")
    for metric in HIT_METRICS:
        count = _nonnegative_int(hit_counts.get(metric), context + ".hit_counts." + metric)
        if count > denominator:
            raise ValueError("%s.hit_counts.%s exceeds denominator" % (context, metric))
        score = _finite_float(block.get(metric), context + "." + metric)
        expected_score = 0.0 if denominator == 0 else count / float(denominator)
        if abs(score - expected_score) > 1e-9:
            raise ValueError("%s.%s does not match hit_counts/denominator" % (context, metric))
    mrr = _finite_float(block.get("mrr"), context + ".mrr")
    mrr_numerator = _finite_float(block.get("mrr_numerator"), context + ".mrr_numerator")
    expected_mrr = 0.0 if denominator == 0 else mrr_numerator / float(denominator)
    if abs(mrr - expected_mrr) > 1e-9:
        raise ValueError("%s.mrr does not match mrr_numerator/denominator" % context)
    per_case = [
        _validate_per_case_evidence(
            _require_mapping(row, "%s.per_case[%d]" % (context, index)),
            expected_dataset_id=expected_dataset_id,
            context="%s.per_case[%d]" % (context, index),
        )
        for index, row in enumerate(
            _require_sequence(block.get("per_case"), context + ".per_case")
        )
    ]
    if len(per_case) != denominator:
        raise ValueError("%s.per_case length must equal denominator" % context)
    if [str(row["case_id"]) for row in per_case] != case_ids:
        raise ValueError("%s.per_case case_ids must match metric case_ids" % context)
    return {
        "denominator": denominator,
        "case_ids": case_ids,
        "hit_counts": {metric: int(hit_counts[metric]) for metric in HIT_METRICS},
        "mrr_numerator": mrr_numerator,
        "mrr": mrr,
    }


def validate_unseen_unit_result(
    unit: Mapping[str, Any],
    result: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate required unseen-unit scoring evidence before success markers."""

    unit_payload = deepcopy(dict(unit))
    payload = deepcopy(dict(result))
    context = "unit_result.%s" % str(unit_payload.get("unit_id", ""))
    if str(payload.get("unit_id")) != str(unit_payload.get("unit_id")):
        raise ValueError("%s.unit_id mismatch" % context)
    dataset_id = str(unit_payload.get("canonical_dataset_id", "")).strip()
    if str(payload.get("canonical_dataset_id", "")).strip() != dataset_id:
        raise ValueError("%s.canonical_dataset_id mismatch" % context)
    if str(payload.get("stage", "")) != str(unit_payload.get("stage", "")):
        raise ValueError("%s.stage mismatch" % context)
    if bool(payload.get("oracle_only")) != bool(unit_payload.get("oracle_only")):
        raise ValueError("%s.oracle_only mismatch" % context)
    if int(payload.get("budget", -1)) != int(unit_payload.get("budget", -2)):
        raise ValueError("%s.budget mismatch" % context)
    if int(payload.get("active_learning_seed", -1)) != int(
        unit_payload.get("active_learning_seed", -2)
    ):
        raise ValueError("%s.active_learning_seed mismatch" % context)
    if not str(payload.get("query_plan_sha256", "")).strip():
        raise ValueError("%s.query_plan_sha256 must not be empty" % context)
    if not _require_sequence(payload.get("queried_case_ids"), context + ".queried_case_ids"):
        raise ValueError("%s.queried_case_ids must not be empty" % context)
    metrics = _require_mapping(payload.get("metrics"), context + ".metrics")
    metric_summary = {
        view: _validate_metric_block(
            metrics.get(view),
            expected_dataset_id=dataset_id,
            context="%s.metrics.%s" % (context, view),
        )
        for view in ("T1", "T2")
    }
    evidence_by_view = _require_mapping(
        payload.get("per_case_ranking_evidence_by_view"),
        context + ".per_case_ranking_evidence_by_view",
    )
    for view in ("T1", "T2"):
        rows = [
            _validate_per_case_evidence(
                _require_mapping(row, "%s.per_case_ranking_evidence_by_view.%s[%d]" % (context, view, index)),
                expected_dataset_id=dataset_id,
                context="%s.per_case_ranking_evidence_by_view.%s[%d]" % (context, view, index),
            )
            for index, row in enumerate(
                _require_sequence(
                    evidence_by_view.get(view),
                    "%s.per_case_ranking_evidence_by_view.%s" % (context, view),
                )
            )
        ]
        expected_ids = metric_summary[view]["case_ids"]
        if [str(row["case_id"]) for row in rows] != expected_ids:
            raise ValueError("%s per-case evidence IDs must match metrics.%s" % (context, view))
    held = str(unit_payload.get("held_out_fault_type", "")).strip()
    if held:
        result_held = str(payload.get("held_out_fault_type", held)).strip()
        if result_held != held:
            raise ValueError("%s.held_out_fault_type mismatch" % context)
        slices = _require_mapping(payload.get("fault_type_slices"), context + ".fault_type_slices")
        for slice_name in ("held_out", "non_held_out"):
            slice_payload = _require_mapping(
                slices.get(slice_name),
                "%s.fault_type_slices.%s" % (context, slice_name),
            )
            expected_fault_type = held if slice_name == "held_out" else "__not_%s" % held
            if str(slice_payload.get("fault_type", "")).strip() != expected_fault_type:
                raise ValueError("%s.fault_type_slices.%s.fault_type mismatch" % (context, slice_name))
            for view in ("T1", "T2"):
                _validate_metric_block(
                    slice_payload.get(view),
                    expected_dataset_id=dataset_id,
                    context="%s.fault_type_slices.%s.%s" % (context, slice_name, view),
                )
            for view in ("T1", "T2"):
                combined = (
                    int(slices["held_out"][view]["denominator"])
                    + int(slices["non_held_out"][view]["denominator"])
                )
                if combined != metric_summary[view]["denominator"]:
                    raise ValueError("%s fault_type slice denominators must sum to metrics.%s" % (context, view))
    return {
        "valid": True,
        "unit_id": str(payload["unit_id"]),
        "metric_summary": metric_summary,
    }


def execute_unseen_manifest(
    manifest: Mapping[str, Any],
    unit_executor: Any,
    *,
    allow_local_output_root: bool = False,
) -> dict[str, Any]:
    """Execute unseen-category diagnostic units with fail-closed markers."""

    validation = validate_unseen_manifest(
        manifest,
        allow_local_output_root=allow_local_output_root,
    )
    run_root = Path(str(manifest["output_root"]))
    run_root.mkdir(parents=True, exist_ok=True)
    units = [deepcopy(dict(unit)) for unit in manifest.get("units") or []]
    results: list[dict[str, Any]] = []
    recent_log_path = str(run_root / "logs" / "tmux-runner.log")
    try:
        for marker in ("COMPLETED.json", "all.done", ".failed"):
            path = run_root / marker
            if path.exists():
                path.unlink()
        start = _progress_payload(
            phase="starting",
            unit=None,
            completed_units=0,
            total_units=len(units),
            failure_count=0,
            recent_log_path=recent_log_path,
        )
        _write_json_atomic(run_root / "progress.json", start)
        print(format_unseen_progress_line(start), flush=True)
        for index, unit in enumerate(units, start=1):
            running = _progress_payload(
                phase=str(unit.get("stage", "")),
                unit=unit,
                completed_units=index - 1,
                total_units=len(units),
                failure_count=0,
                recent_log_path=recent_log_path,
            )
            _write_json_atomic(run_root / "progress.json", running)
            print(format_unseen_progress_line(running), flush=True)
            result = deepcopy(dict(_call_unit_executor(unit_executor, unit, run_root)))
            if str(result.get("unit_id")) != str(unit.get("unit_id")):
                raise ValueError("unseen unit result unit_id mismatch")
            result["result_validation"] = validate_unseen_unit_result(unit, result)
            results.append(result)
            _write_json_atomic(
                run_root / "unit_results" / _safe_token(unit["unit_id"]) / "result.json",
                result,
            )
            done = _progress_payload(
                phase=str(unit.get("stage", "")),
                unit=unit,
                completed_units=index,
                total_units=len(units),
                failure_count=0,
                recent_log_path=recent_log_path,
            )
            _write_json_atomic(run_root / "progress.json", done)
            print(format_unseen_progress_line(done), flush=True)
        summary = {
            "schema_version": "rcl-unseen-fault-type-diagnostic-summary-v1",
            "change_id": CHANGE_ID,
            "run_id": manifest.get("run_id"),
            "manifest_sha256": manifest.get("manifest_sha256"),
            "manifest_validation": validation,
            "unit_count": len(results),
            "expected_unit_count": int(manifest.get("expected_unit_count", len(units))),
            "units": {
                str(result["unit_id"]): result
                for result in sorted(results, key=lambda row: str(row["unit_id"]))
            },
            "completed_at_utc": _utc_now(),
        }
        _write_json_atomic(run_root / "unseen_diagnostic_summary.json", summary)
        completion = {
            "schema_version": "rcl-unseen-fault-type-completion-v1",
            "status": "complete",
            "manifest_sha256": manifest.get("manifest_sha256"),
            "summary_path": str((run_root / "unseen_diagnostic_summary.json").resolve()),
            "summary_sha256": semantic_sha256(summary),
            "unit_count": len(results),
        }
        _write_json_atomic(run_root / "COMPLETED.json", completion)
        (run_root / "all.done").write_text(_utc_now() + "\n", encoding="utf-8")
        final = _progress_payload(
            phase="complete",
            unit=None,
            completed_units=len(units),
            total_units=len(units),
            failure_count=0,
            recent_log_path=recent_log_path,
        )
        _write_json_atomic(run_root / "progress.json", final)
        print(format_unseen_progress_line(final), flush=True)
        return completion
    except Exception as exc:
        for marker in ("COMPLETED.json", "all.done"):
            path = run_root / marker
            if path.exists():
                path.unlink()
        _write_failure_marker(
            run_root,
            exc,
            completed_units=len(results),
            total_units=len(units),
        )
        failed = _progress_payload(
            phase="failed",
            unit=None,
            completed_units=len(results),
            total_units=len(units),
            failure_count=1,
            recent_log_path=recent_log_path,
        )
        _write_json_atomic(run_root / "progress.json", failed)
        print(format_unseen_progress_line(failed), flush=True)
        raise
def _require_outer_train_fault_candidate(row: Mapping[str, Any]) -> dict[str, Any]:
    candidate = deepcopy(dict(row))
    if str(candidate.get("split")) != "outer_train":
        raise ValueError("query candidate must belong to outer_train")
    if str(candidate.get("case_kind")) != "fault":
        raise ValueError("query candidate must be a fault case")
    candidate["case_id"] = _case_id(candidate)
    candidate["fault_type"] = fault_type_from_case(candidate)
    return candidate


def _labels_for_case(
    authoritative_labels: Mapping[str, Sequence[Any]],
    case_id: str,
) -> list[str]:
    targets = [
        str(target).strip()
        for target in authoritative_labels.get(case_id, ())
        if str(target).strip()
    ]
    if not targets:
        raise ValueError("queried case has no authoritative label: %s" % case_id)
    return targets


def build_leave_one_fault_type_query_plan(
    *,
    canonical_dataset_id: str,
    ordered_candidates: Sequence[Mapping[str, Any]],
    authoritative_labels: Mapping[str, Sequence[Any]],
    held_out_fault_type: str,
    budget: int,
    active_learning_seed: int,
    split_seed: int,
    selector_id: str,
    selector_config: Mapping[str, Any],
    trainer_config_hash: str,
    scorer_config_hash: str,
    target_pair_artifact_hash: str,
) -> dict[str, Any]:
    """Build an oracle-only budget plan that excludes one true fault type."""

    held_out = str(held_out_fault_type).strip()
    if not held_out:
        raise ValueError("held_out_fault_type must not be empty")
    if int(budget) <= 0:
        raise ValueError("budget must be positive")
    rows = [_require_outer_train_fault_candidate(row) for row in ordered_candidates]
    seen: set[str] = set()
    for row in rows:
        case_id = str(row["case_id"])
        if case_id in seen:
            raise ValueError("ordered candidates contain duplicate case_id: %s" % case_id)
        seen.add(case_id)
    nonheldout = [row for row in rows if str(row["fault_type"]) != held_out]
    if len(nonheldout) < int(budget):
        raise ValueError(
            "cannot fill budget after excluding held-out fault type %s: %d < %d"
            % (held_out, len(nonheldout), int(budget))
        )
    selected = [deepcopy(dict(row)) for row in nonheldout[: int(budget)]]
    selected_ids = [str(row["case_id"]) for row in selected]
    annotations = [
        {
            "case_id": case_id,
            "annotation_source": "simulated_manual_ground_truth",
            "targets": _labels_for_case(authoritative_labels, case_id),
        }
        for case_id in selected_ids
    ]
    identity = {
        "schema_version": "rcl-unseen-fault-type-query-plan-v1",
        "canonical_dataset_id": str(canonical_dataset_id),
        "held_out_fault_type": held_out,
        "selector_id": str(selector_id),
        "selector_config": deepcopy(dict(selector_config)),
        "active_learning_seed": int(active_learning_seed),
        "split_seed": int(split_seed),
        "budget": int(budget),
        "selected_case_ids": selected_ids,
        "trainer_config_hash": str(trainer_config_hash),
        "scorer_config_hash": str(scorer_config_hash),
        "target_pair_artifact_hash": str(target_pair_artifact_hash),
        "oracle_only": True,
    }
    return {
        **identity,
        "selected_cases": selected,
        "annotations": annotations,
        "diagnostics": {
            "excluded_held_out_candidate_count": len(rows) - len(nonheldout),
            "available_nonheldout_candidate_count": len(nonheldout),
            "ordered_candidate_count": len(rows),
        },
        "query_plan_sha256": semantic_sha256(identity),
    }


def _first_matching_rank(ranking: Sequence[Any], targets: Sequence[Any]) -> int:
    target_set = {str(target) for target in targets if str(target).strip()}
    if not target_set:
        raise ValueError("targets must not be empty")
    for index, candidate in enumerate(ranking, start=1):
        if str(candidate) in target_set:
            return index
    return 0


def _ranking_lookup(
    ranking_rows: Sequence[Mapping[str, Any]],
) -> dict[str, dict[str, Any]]:
    lookup: dict[str, dict[str, Any]] = {}
    for row_raw in ranking_rows:
        row = deepcopy(dict(row_raw))
        case_id = _case_id(row)
        if case_id in lookup:
            raise ValueError("duplicate ranking row for case_id: %s" % case_id)
        row["fault_type"] = fault_type_from_case(row)
        lookup[case_id] = row
    if not lookup:
        raise ValueError("ranking_rows must not be empty")
    return lookup


def _empty_metric_block(case_ids: Sequence[str]) -> dict[str, Any]:
    return {
        "denominator": 0,
        "evaluation_fault_cases": 0,
        "case_ids": list(case_ids),
        "target_case_ids": list(case_ids),
        "hit_at_1": 0.0,
        "hit_at_3": 0.0,
        "hit_at_5": 0.0,
        "mrr": 0.0,
        "mrr_numerator": 0.0,
        "hit_counts": {metric: 0 for metric in HIT_METRICS},
        "per_case": [],
    }


def _metrics_for_case_ids(
    case_ids: Sequence[Any],
    ranking_lookup: Mapping[str, Mapping[str, Any]],
    *,
    canonical_dataset_id: str,
) -> dict[str, Any]:
    ids = [str(case_id) for case_id in case_ids]
    if not ids:
        return _empty_metric_block(ids)
    hit_counts = {metric: 0 for metric in HIT_METRICS}
    mrr_numerator = 0.0
    per_case: list[dict[str, Any]] = []
    for case_id in ids:
        if case_id not in ranking_lookup:
            raise ValueError("missing ranking for %s" % case_id)
        row = dict(ranking_lookup[case_id])
        ranking = row.get("ranking")
        targets = row.get("targets")
        if not isinstance(ranking, Sequence) or isinstance(ranking, (str, bytes)):
            raise ValueError("ranking must be a sequence for %s" % case_id)
        if not isinstance(targets, Sequence) or isinstance(targets, (str, bytes)):
            raise ValueError("targets must be a sequence for %s" % case_id)
        rank = _first_matching_rank(ranking, targets)
        reciprocal_rank = 0.0 if rank == 0 else 1.0 / float(rank)
        mrr_numerator += reciprocal_rank
        flags = {
            "hit_at_1": bool(rank and rank <= 1),
            "hit_at_3": bool(rank and rank <= 3),
            "hit_at_5": bool(rank and rank <= 5),
        }
        for metric, hit in flags.items():
            if hit:
                hit_counts[metric] += 1
        per_case.append(
            {
                "case_id": case_id,
                "dataset_id": str(canonical_dataset_id),
                "fault_type": str(row["fault_type"]),
                "targets": [str(target) for target in targets],
                "ranking": [str(candidate) for candidate in ranking],
                "first_matching_rank": rank,
                "reciprocal_rank": reciprocal_rank,
                **flags,
            }
        )
    denominator = len(ids)
    return {
        "denominator": denominator,
        "evaluation_fault_cases": denominator,
        "case_ids": ids,
        "target_case_ids": ids,
        "hit_at_1": hit_counts["hit_at_1"] / denominator,
        "hit_at_3": hit_counts["hit_at_3"] / denominator,
        "hit_at_5": hit_counts["hit_at_5"] / denominator,
        "mrr": mrr_numerator / denominator,
        "mrr_numerator": mrr_numerator,
        "hit_counts": hit_counts,
        "per_case": per_case,
    }


def score_fault_type_slices(
    *,
    canonical_dataset_id: str,
    outer_train_case_ids: Sequence[Any],
    outer_test_case_ids: Sequence[Any],
    queried_case_ids: Sequence[Any],
    ranking_rows: Sequence[Mapping[str, Any]],
    held_out_fault_type: str,
    extra_t2_case_ids: Sequence[Any] | None = None,
) -> dict[str, Any]:
    """Score aggregate T1/T2 plus held-out and non-held-out fault-type slices."""

    dataset_id = str(canonical_dataset_id).strip()
    if not dataset_id:
        raise ValueError("canonical_dataset_id must not be empty")
    held_out = str(held_out_fault_type).strip()
    if not held_out:
        raise ValueError("held_out_fault_type must not be empty")
    queried = {str(case_id) for case_id in queried_case_ids}
    t1_ids = [str(case_id) for case_id in outer_test_case_ids if str(case_id) not in queried]
    t2_tail = (
        [str(case_id) for case_id in extra_t2_case_ids]
        if extra_t2_case_ids is not None
        else [str(case_id) for case_id in outer_train_case_ids]
    )
    t2_ids = list(t1_ids) + [case_id for case_id in t2_tail if case_id not in queried]
    lookup = _ranking_lookup(ranking_rows)
    for view_name, ids in (("T1", t1_ids), ("T2", t2_ids)):
        overlap = sorted(queried.intersection(ids))
        if overlap:
            raise ValueError(
                "queried budget case appears in %s scoring partition: %s"
                % (view_name, overlap)
            )
    aggregate = {
        "T1": _metrics_for_case_ids(
            t1_ids,
            lookup,
            canonical_dataset_id=dataset_id,
        ),
        "T2": _metrics_for_case_ids(
            t2_ids,
            lookup,
            canonical_dataset_id=dataset_id,
        ),
    }

    def ids_by_fault(ids: Sequence[str], *, held: bool) -> list[str]:
        return [
            case_id
            for case_id in ids
            if (str(lookup[case_id]["fault_type"]) == held_out) is held
        ]

    slices = {
        "held_out": {
            "fault_type": held_out,
            "T1": _metrics_for_case_ids(
                ids_by_fault(t1_ids, held=True),
                lookup,
                canonical_dataset_id=dataset_id,
            ),
            "T2": _metrics_for_case_ids(
                ids_by_fault(t2_ids, held=True),
                lookup,
                canonical_dataset_id=dataset_id,
            ),
        },
        "non_held_out": {
            "fault_type": "__not_%s" % held_out,
            "T1": _metrics_for_case_ids(
                ids_by_fault(t1_ids, held=False),
                lookup,
                canonical_dataset_id=dataset_id,
            ),
            "T2": _metrics_for_case_ids(
                ids_by_fault(t2_ids, held=False),
                lookup,
                canonical_dataset_id=dataset_id,
            ),
        },
    }
    warnings: list[dict[str, Any]] = []
    for metric in RANKING_METRICS:
        t1_value = float(aggregate["T1"][metric])
        t2_value = float(aggregate["T2"][metric])
        diff = abs(t1_value - t2_value)
        if diff >= 0.05:
            warnings.append(
                {
                    "reason": "absolute_metric_gap_at_least_threshold",
                    "metric": metric,
                    "T1": t1_value,
                    "T2": t2_value,
                    "absolute_difference": diff,
                }
            )
    return {
        "schema_version": "rcl-unseen-fault-type-sliced-score-v1",
        "held_out_fault_type": held_out,
        "partitions": {"T1": t1_ids, "T2": t2_ids},
        "queried_case_ids": sorted(queried),
        "T1": aggregate["T1"],
        "T2": aggregate["T2"],
        "slices": slices,
        "per_case_ranking_evidence": {
            "T1": aggregate["T1"]["per_case"],
            "T2": aggregate["T2"]["per_case"],
        },
        "t1_t2_warnings": warnings,
    }


def _metric_block_from_per_case(
    rows_raw: Sequence[Mapping[str, Any]],
    *,
    case_ids: Sequence[str] | None = None,
) -> dict[str, Any]:
    rows = [deepcopy(dict(row)) for row in rows_raw]
    if case_ids is not None:
        wanted = {str(case_id) for case_id in case_ids}
        rows = [row for row in rows if str(row.get("case_id")) in wanted]
    denominator = len(rows)
    hit_counts = {metric: 0 for metric in HIT_METRICS}
    mrr_numerator = 0.0
    ids: list[str] = []
    for index, row in enumerate(rows):
        context = "per_case[%d]" % index
        case_id = str(row.get("case_id", "")).strip()
        if not case_id:
            raise ValueError("%s.case_id must not be empty" % context)
        ids.append(case_id)
        reciprocal_rank = _finite_float(
            row.get("reciprocal_rank"),
            "%s.reciprocal_rank" % context,
        )
        mrr_numerator += reciprocal_rank
        for metric in HIT_METRICS:
            if bool(row.get(metric)):
                hit_counts[metric] += 1
    if denominator == 0:
        return _empty_metric_block([])
    return {
        "denominator": denominator,
        "evaluation_fault_cases": denominator,
        "case_ids": ids,
        "target_case_ids": ids,
        "hit_at_1": hit_counts["hit_at_1"] / denominator,
        "hit_at_3": hit_counts["hit_at_3"] / denominator,
        "hit_at_5": hit_counts["hit_at_5"] / denominator,
        "mrr": mrr_numerator / denominator,
        "mrr_numerator": mrr_numerator,
        "hit_counts": hit_counts,
        "per_case": rows,
    }


def _held_out_metric_from_unit(
    unit_result: Mapping[str, Any],
    *,
    fault_type: str,
    view: str,
) -> dict[str, Any]:
    view_name = str(view)
    held = str(fault_type)
    slices = unit_result.get("fault_type_slices")
    if isinstance(slices, Mapping):
        held_slice = slices.get("held_out")
        if isinstance(held_slice, Mapping) and str(held_slice.get("fault_type")) == held:
            return deepcopy(dict(held_slice[view_name]))
    evidence_by_view = _require_mapping(
        unit_result.get("per_case_ranking_evidence_by_view"),
        "unit_result.per_case_ranking_evidence_by_view",
    )
    rows = [
        deepcopy(dict(row))
        for row in _require_sequence(
            evidence_by_view.get(view_name),
            "unit_result.per_case_ranking_evidence_by_view.%s" % view_name,
        )
        if str(dict(row).get("fault_type")) == held
    ]
    return _metric_block_from_per_case(rows)


def _metric_delta(
    baseline_metric: Mapping[str, Any],
    forced_metric: Mapping[str, Any],
) -> dict[str, Any]:
    baseline = deepcopy(dict(baseline_metric))
    forced = deepcopy(dict(forced_metric))
    result: dict[str, Any] = {
        "baseline": {
            metric: float(baseline[metric])
            for metric in RANKING_METRICS
        },
        "leave_one": {
            metric: float(forced[metric])
            for metric in RANKING_METRICS
        },
        "baseline_denominator": int(baseline.get("denominator", 0)),
        "leave_one_denominator": int(forced.get("denominator", 0)),
        "baseline_hit_counts": deepcopy(dict(baseline.get("hit_counts", {}))),
        "leave_one_hit_counts": deepcopy(dict(forced.get("hit_counts", {}))),
        "baseline_mrr_numerator": float(baseline.get("mrr_numerator", 0.0)),
        "leave_one_mrr_numerator": float(forced.get("mrr_numerator", 0.0)),
    }
    for metric in RANKING_METRICS:
        base_value = float(baseline[metric])
        forced_value = float(forced[metric])
        result["%s_absolute_drop" % metric] = base_value - forced_value
    base_mrr = float(baseline["mrr"])
    absolute_mrr_drop = float(result["mrr_absolute_drop"])
    result["mrr_relative_drop"] = (
        0.0 if base_mrr <= 0.0 else absolute_mrr_drop / base_mrr
    )
    hit_drop_count = sum(
        1
        for metric in HIT_METRICS
        if float(result["%s_absolute_drop" % metric]) >= 0.05
    )
    mrr_drop_gate = (
        absolute_mrr_drop >= 0.05
        and float(result["mrr_relative_drop"]) >= 0.10
    )
    hit_drop_gate = hit_drop_count >= 2
    result["hit_metric_drop_count_at_least_0p05"] = hit_drop_count
    result["material_degraded"] = bool(mrr_drop_gate or hit_drop_gate)
    result["material_reasons"] = [
        reason
        for reason, enabled in (
            ("mrr_drop_abs_ge_0p05_rel_ge_0p10", mrr_drop_gate),
            ("two_or_more_hit_metrics_drop_ge_0p05", hit_drop_gate),
        )
        if enabled
    ]
    return result


def _units_by_id(summary: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    units = {
        str(unit_id): deepcopy(dict(unit_result))
        for unit_id, unit_result in _require_mapping(
            summary.get("units"),
            "summary.units",
        ).items()
    }
    if not units:
        raise ValueError("summary.units must not be empty")
    return units


def build_unseen_diagnostic_report(
    *,
    manifest: Mapping[str, Any],
    inventory_by_dataset: Mapping[str, Mapping[str, Any]],
    summary: Mapping[str, Any],
) -> dict[str, Any]:
    """Build paired leave-one-fault-type degradation diagnostics."""

    manifest_payload = deepcopy(dict(manifest))
    summary_payload = deepcopy(dict(summary))
    expected = int(manifest_payload.get("expected_unit_count", summary_payload.get("expected_unit_count", -1)))
    if int(summary_payload.get("unit_count", -1)) != expected:
        raise ValueError("summary unit_count does not match expected_unit_count")
    units = _units_by_id(summary_payload)
    baseline_by_dataset_seed: dict[tuple[str, int], dict[str, Any]] = {}
    leave_one_units: list[dict[str, Any]] = []
    for unit in units.values():
        dataset = str(unit.get("canonical_dataset_id", "")).strip()
        seed = int(unit.get("active_learning_seed", -1))
        stage = str(unit.get("stage", ""))
        if stage == "baseline_replay":
            baseline_by_dataset_seed[(dataset, seed)] = unit
        elif stage == "leave_one_fault_type":
            leave_one_units.append(unit)
    paired: dict[str, dict[str, dict[str, Any]]] = {}
    warnings: list[dict[str, Any]] = []
    dataset_accumulator: dict[str, dict[str, Any]] = {}
    for unit in sorted(
        leave_one_units,
        key=lambda row: (
            str(row.get("canonical_dataset_id", "")),
            int(row.get("active_learning_seed", 0)),
            str(row.get("held_out_fault_type", "")),
        ),
    ):
        dataset = str(unit["canonical_dataset_id"])
        seed = int(unit["active_learning_seed"])
        seed_key = str(seed)
        held = str(unit.get("held_out_fault_type", "")).strip()
        if not held:
            raise ValueError("leave-one unit missing held_out_fault_type: %s" % unit.get("unit_id"))
        baseline = baseline_by_dataset_seed.get((dataset, seed))
        if baseline is None:
            raise ValueError("missing matched baseline for %s seed %d" % (dataset, seed))
        view_deltas = {
            view: _metric_delta(
                _held_out_metric_from_unit(baseline, fault_type=held, view=view),
                _held_out_metric_from_unit(unit, fault_type=held, view=view),
            )
            for view in ("T1", "T2")
        }
        paired.setdefault(dataset, {}).setdefault(seed_key, {})[held] = {
            "dataset": dataset,
            "active_learning_seed": seed,
            "fault_type": held,
            "baseline_unit_id": str(baseline["unit_id"]),
            "leave_one_unit_id": str(unit["unit_id"]),
            "T1": view_deltas["T1"],
            "T2": view_deltas["T2"],
            "official_material_degraded": bool(view_deltas["T1"]["material_degraded"]),
            "diagnostic_t2_material_degraded": bool(view_deltas["T2"]["material_degraded"]),
        }
        if bool(view_deltas["T1"]["material_degraded"]) != bool(
            view_deltas["T2"]["material_degraded"]
        ):
            warnings.append(
                {
                    "reason": "t1_t2_material_gate_disagreement",
                    "dataset": dataset,
                    "active_learning_seed": seed,
                    "fault_type": held,
                    "T1_material_degraded": bool(view_deltas["T1"]["material_degraded"]),
                    "T2_material_degraded": bool(view_deltas["T2"]["material_degraded"]),
                    "T1_denominator": int(view_deltas["T1"]["leave_one_denominator"]),
                    "T2_denominator": int(view_deltas["T2"]["leave_one_denominator"]),
                }
            )
        for metric in RANKING_METRICS:
            t1_value = float(view_deltas["T1"]["leave_one"][metric])
            t2_value = float(view_deltas["T2"]["leave_one"][metric])
            difference = abs(t1_value - t2_value)
            if difference >= 0.05:
                warnings.append(
                    {
                        "reason": "held_out_metric_t1_t2_gap_at_least_0p05",
                        "dataset": dataset,
                        "active_learning_seed": seed,
                        "fault_type": held,
                        "metric": metric,
                        "T1": t1_value,
                        "T2": t2_value,
                        "T1_denominator": int(view_deltas["T1"]["leave_one_denominator"]),
                        "T2_denominator": int(view_deltas["T2"]["leave_one_denominator"]),
                        "absolute_difference": difference,
                    }
                )
        accumulator = dataset_accumulator.setdefault(
            dataset,
            {
                "weighted_drop_numerator": 0.0,
                "weighted_drop_denominator": 0,
                "material_fault_types": set(),
                "observations": 0,
            },
        )
        denominator = int(view_deltas["T1"]["leave_one_denominator"])
        accumulator["weighted_drop_numerator"] += float(view_deltas["T1"]["mrr_absolute_drop"]) * denominator
        accumulator["weighted_drop_denominator"] += denominator
        accumulator["observations"] += 1
        if bool(view_deltas["T1"]["material_degraded"]):
            accumulator["material_fault_types"].add(held)
    dataset_threats: dict[str, dict[str, Any]] = {}
    inventories = {
        str(dataset): deepcopy(dict(inventory))
        for dataset, inventory in inventory_by_dataset.items()
    }
    for dataset in sorted(inventories):
        inventory = inventories[dataset]
        accumulator = dataset_accumulator.get(
            dataset,
            {
                "weighted_drop_numerator": 0.0,
                "weighted_drop_denominator": 0,
                "material_fault_types": set(),
                "observations": 0,
            },
        )
        denominator = int(accumulator["weighted_drop_denominator"])
        weighted_drop = (
            0.0
            if denominator == 0
            else float(accumulator["weighted_drop_numerator"]) / denominator
        )
        material_fault_types = sorted(str(value) for value in accumulator["material_fault_types"])
        exposed = bool(len(material_fault_types) >= 2 or weighted_drop >= 0.05)
        dataset_threats[dataset] = {
            "dataset": dataset,
            "eligible_fault_types": [str(value) for value in inventory.get("eligible_fault_types", [])],
            "low_support_fault_types": [str(value) for value in inventory.get("low_support_fault_types", [])],
            "material_degraded_fault_types": material_fault_types,
            "material_degraded_fault_type_count": len(material_fault_types),
            "weighted_mean_held_out_t1_mrr_drop": weighted_drop,
            "weighted_denominator": denominator,
            "paired_observation_count": int(accumulator["observations"]),
            "exposed_to_unseen_fault_type_threat": exposed,
            "exposure_reasons": [
                reason
                for reason, enabled in (
                    ("at_least_two_eligible_fault_types_material_degraded", len(material_fault_types) >= 2),
                    ("eligible_weighted_mean_mrr_drop_ge_0p05", weighted_drop >= 0.05),
                )
                if enabled
            ],
        }
    identity = {
        "schema_version": "rcl-unseen-fault-type-diagnostic-report-v1",
        "change_id": CHANGE_ID,
        "manifest_sha256": manifest_payload.get("manifest_sha256"),
        "summary_sha256": summary_payload.get("summary_sha256"),
        "unit_count": len(units),
        "expected_unit_count": expected,
        "inventory": inventories,
        "paired_comparisons": paired,
        "dataset_threats": dataset_threats,
        "t1_t2_disagreement_warnings": warnings,
        "mitigation_required": any(
            bool(threat["exposed_to_unseen_fault_type_threat"])
            for threat in dataset_threats.values()
        ),
        "clean_mitigation_status": (
            "required"
            if any(
                bool(threat["exposed_to_unseen_fault_type_threat"])
                for threat in dataset_threats.values()
            )
            else "not_required"
        ),
        "oracle_only_diagnostic_notice": (
            "leave-one fault-type units use true fault_type only to define the stress test; "
            "they are not publication-clean acquisition methods"
        ),
        "generated_at_utc": _utc_now(),
    }
    return {
        **identity,
        "diagnostic_report_sha256": semantic_sha256(identity),
    }


def _read_json_object(path: Path | str, context: str) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError("%s must be a JSON object" % context)
    return dict(payload)


def validate_unseen_completion_bundle(
    run_root: Path | str,
) -> dict[str, Any]:
    """Validate a completed unseen diagnostic run root before aggregation."""

    root = Path(run_root)
    required_files = {
        "manifest": root / "unseen_manifest.json",
        "inventory": root / "fault_type_inventory.json",
        "summary": root / "unseen_diagnostic_summary.json",
        "completion": root / "COMPLETED.json",
        "all_done": root / "all.done",
    }
    missing = [name for name, path in required_files.items() if not path.exists()]
    if missing:
        raise ValueError("unseen completion bundle missing required files: %s" % missing)
    if (root / ".failed").exists():
        raise ValueError("unseen completion bundle has failure marker: %s" % (root / ".failed"))
    manifest = _read_json_object(required_files["manifest"], "unseen manifest")
    inventory = _read_json_object(required_files["inventory"], "fault_type inventory")
    summary = _read_json_object(required_files["summary"], "unseen diagnostic summary")
    completion = _read_json_object(required_files["completion"], "unseen completion")
    manifest_sha = str(manifest.get("manifest_sha256", ""))
    if str(summary.get("manifest_sha256", "")) != manifest_sha:
        raise ValueError("summary manifest_sha256 mismatch")
    if str(completion.get("manifest_sha256", "")) != manifest_sha:
        raise ValueError("completion manifest_sha256 mismatch")
    units = _units_by_id(summary)
    expected = int(manifest.get("expected_unit_count", -1))
    if len(units) != expected or int(summary.get("unit_count", -1)) != expected:
        raise ValueError("completed unit count mismatch")
    manifest_units = {
        str(unit.get("unit_id")): deepcopy(dict(unit))
        for unit in _require_sequence(manifest.get("units"), "manifest.units")
    }
    if set(manifest_units) != set(units):
        raise ValueError("summary units do not match manifest units")
    for unit_id, unit_result in units.items():
        result_path = root / "unit_results" / _safe_token(unit_id, "unit_id") / "result.json"
        if not result_path.exists():
            raise ValueError("missing unit result file: %s" % result_path)
        on_disk = _read_json_object(result_path, "unit result")
        if str(on_disk.get("unit_id")) != unit_id:
            raise ValueError("unit result file unit_id mismatch: %s" % unit_id)
        validate_unseen_unit_result(manifest_units[unit_id], unit_result)
        if not bool(dict(unit_result.get("result_validation") or {}).get("valid")):
            raise ValueError("unit result_validation is not valid: %s" % unit_id)
    return {
        "valid": True,
        "run_root": str(root),
        "manifest": manifest,
        "inventory": inventory,
        "summary": summary,
        "completion": completion,
        "unit_count": len(units),
    }


def write_unseen_diagnostic_report(
    run_root: Path | str,
) -> dict[str, Any]:
    """Validate, aggregate, and write the unseen diagnostic report artifacts."""

    root = Path(run_root)
    bundle = validate_unseen_completion_bundle(root)
    report = build_unseen_diagnostic_report(
        manifest=bundle["manifest"],
        inventory_by_dataset=bundle["inventory"],
        summary=bundle["summary"],
    )
    report_path = root / "reports" / "unseen_diagnostic_report.json"
    _write_json_atomic(report_path, report)
    rows: list[str] = [
        "dataset,active_learning_seed,fault_type,t1_baseline_mrr,t1_leave_one_mrr,"
        "t1_mrr_absolute_drop,t1_mrr_relative_drop,t1_denominator,material_degraded"
    ]
    for dataset, seed_map in sorted(report["paired_comparisons"].items()):
        for seed, fault_map in sorted(seed_map.items(), key=lambda item: int(item[0])):
            for fault_type, comparison in sorted(fault_map.items()):
                t1 = comparison["T1"]
                rows.append(
                    ",".join(
                        [
                            str(dataset),
                            str(seed),
                            str(fault_type).replace(",", "_"),
                            "%.12g" % float(t1["baseline"]["mrr"]),
                            "%.12g" % float(t1["leave_one"]["mrr"]),
                            "%.12g" % float(t1["mrr_absolute_drop"]),
                            "%.12g" % float(t1["mrr_relative_drop"]),
                            str(int(t1["leave_one_denominator"])),
                            str(bool(t1["material_degraded"])).lower(),
                        ]
                    )
                )
    csv_path = root / "reports" / "paired_comparisons.csv"
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    csv_path.write_text("\n".join(rows) + "\n", encoding="utf-8")
    return {
        "valid": True,
        "report_path": str(report_path),
        "csv_path": str(csv_path),
        "diagnostic_report_sha256": report["diagnostic_report_sha256"],
        "mitigation_required": bool(report["mitigation_required"]),
        "dataset_threats": report["dataset_threats"],
    }


__all__ = [
    "build_fault_type_inventory",
    "build_label_free_mitigation_manifest_from_diagnostic_report",
    "build_leave_one_fault_type_query_plan",
    "build_unseen_diagnostic_report",
    "build_unseen_manifest",
    "execute_unseen_manifest",
    "fault_type_from_case",
    "format_unseen_progress_line",
    "score_fault_type_slices",
    "semantic_sha256",
    "validate_unseen_completion_bundle",
    "validate_unseen_manifest",
    "validate_unseen_unit_result",
    "write_unseen_diagnostic_report",
]
