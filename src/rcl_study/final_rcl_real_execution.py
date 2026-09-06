from __future__ import annotations

import hashlib
import json
import math
from copy import deepcopy
from pathlib import Path
from typing import Any, Mapping, Sequence


_DATASETS = ("rcabench", "aiops2022_pre")
_ARMS = (
    "hdbscan_query_cvae_oser",
    "oracle_full_cvae_oser",
    "hdbscan_query_pairwise",
)
_COMPLETE_ARMS = {
    "hdbscan_query_cvae_oser",
    "oracle_full_cvae_oser",
}


class FinalRCLRealExecutionError(RuntimeError):
    """Raised when bounded real execution evidence is not trustworthy."""


def _semantic_hash(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    ).hexdigest()


def _positive_int(value: Any, context: str) -> int:
    if isinstance(value, bool):
        raise FinalRCLRealExecutionError(f"{context} must be a positive integer")
    try:
        result = int(value)
    except (TypeError, ValueError) as exc:
        raise FinalRCLRealExecutionError(
            f"{context} must be a positive integer"
        ) from exc
    if result <= 0 or result != value:
        raise FinalRCLRealExecutionError(f"{context} must be a positive integer")
    return result


def _ids(value: Any, context: str) -> list[str]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise FinalRCLRealExecutionError(f"{context} must be a sequence")
    result = [str(item).strip() for item in value]
    if not result or "" in result or len(result) != len(set(result)):
        raise FinalRCLRealExecutionError(f"{context} must be unique and nonempty")
    return result


def build_final_smoke_manifest(
    *,
    registry: Mapping[str, Any],
    split_adapters: Mapping[str, Mapping[str, Any]],
    run_root: str | Path,
    oracle_train_limit: int,
    test_case_limit: int,
    cvae_optimizer_steps: int,
) -> dict[str, Any]:
    """Build six bounded real-data branches that can never become formal evidence."""

    if not isinstance(registry, Mapping):
        raise FinalRCLRealExecutionError("formal registry must be a mapping")
    frozen_registry = deepcopy(dict(registry))
    if (
        frozen_registry.get("schema_version") != "final-rcl-formal-registry-v1"
        or frozen_registry.get("seed") != 42
        or frozen_registry.get("clusterer_id") != "hdbscan"
        or frozen_registry.get("outer_router_enabled") is not False
        or frozen_registry.get("unit_count") != 6
    ):
        raise FinalRCLRealExecutionError("formal registry cannot own final smoke")
    registry_units = [dict(row) for row in frozen_registry.get("units", ())]
    expected_pairs = {
        (dataset_id, arm_id)
        for dataset_id in _DATASETS
        for arm_id in _ARMS
    }
    observed_pairs = {
        (str(row.get("dataset_id", "")), str(row.get("arm_id", "")))
        for row in registry_units
    }
    if (
        len(registry_units) != 6
        or len(observed_pairs) != 6
        or observed_pairs != expected_pairs
        or any(row.get("seed") != 42 for row in registry_units)
    ):
        raise FinalRCLRealExecutionError("smoke requires the exact six-unit registry")
    if set(split_adapters) != set(_DATASETS):
        raise FinalRCLRealExecutionError("smoke requires both fixed split adapters")

    oracle_limit = _positive_int(oracle_train_limit, "oracle smoke train limit")
    test_limit = _positive_int(test_case_limit, "smoke test case limit")
    optimizer_steps = _positive_int(cvae_optimizer_steps, "CVAE optimizer steps")
    root = Path(run_root)
    units: list[dict[str, Any]] = []
    for registry_unit in registry_units:
        dataset_id = str(registry_unit["dataset_id"])
        arm_id = str(registry_unit["arm_id"])
        split = deepcopy(dict(split_adapters[dataset_id]))
        if (
            split.get("schema_version") != "final-rcl-fixed-split-adapter-v1"
            or split.get("dataset_id") != dataset_id
            or split.get("split_seed") != 42
            or split.get("fit_test_overlap_count") != 0
        ):
            raise FinalRCLRealExecutionError(
                f"{dataset_id} fixed split adapter drifted"
            )
        outer_train = tuple(
            _ids(split.get("outer_train_case_ids"), "outer train cases")
        )
        outer_test = tuple(
            _ids(split.get("outer_test_case_ids"), "outer test cases")
        )
        query_ids = tuple(
            _ids(split.get("query_case_ids"), "HDBSCAN query cases")
        )
        if (
            len(query_ids) != 30
            or not set(query_ids) <= set(outer_train)
            or set(outer_train) & set(outer_test)
            or split.get("outer_train_case_count") != len(outer_train)
            or split.get("outer_test_case_count") != len(outer_test)
            or oracle_limit > len(outer_train)
            or test_limit > len(outer_test)
        ):
            raise FinalRCLRealExecutionError(
                f"{dataset_id} smoke split ownership drifted"
            )
        oracle = arm_id == "oracle_full_cvae_oser"
        fit_ids = outer_train[:oracle_limit] if oracle else outer_train
        supervised_ids = fit_ids if oracle else query_ids
        evaluation_ids = outer_test[:test_limit]
        if set(fit_ids) & set(evaluation_ids):
            raise FinalRCLRealExecutionError("smoke fit/test overlap is forbidden")
        unit = {
            "schema_version": "final-rcl-real-smoke-unit-v1",
            "unit_id": str(registry_unit["unit_id"]),
            "dataset_id": dataset_id,
            "arm_id": arm_id,
            "seed": 42,
            "evidence_role": "smoke_only",
            "promotable_to_formal": False,
            "training_mode": "oracle_full" if oracle else "query_only",
            "fit_case_ids": fit_ids,
            "supervised_case_ids": supervised_ids,
            "evaluation_case_ids": evaluation_ids,
            "cvae_enabled": arm_id in _COMPLETE_ARMS,
            "oser_enabled": arm_id in _COMPLETE_ARMS,
            "cvae_optimizer_steps": optimizer_steps,
            "run_directory": str(root / "units" / str(registry_unit["unit_id"])),
            "split_sha256": str(split.get("split_sha256", "")),
            "query_plan_sha256": (
                None if oracle else str(split.get("query_plan_sha256", ""))
            ),
        }
        units.append({**unit, "unit_smoke_sha256": _semantic_hash(unit)})

    identity = {
        "schema_version": "final-rcl-six-branch-smoke-v1",
        "evidence_role": "smoke_only",
        "promotable_to_formal": False,
        "seed": 42,
        "run_root": str(root),
        "registry_sha256": str(frozen_registry.get("registry_sha256", "")),
        "unit_count": 6,
        "units": units,
    }
    return {**identity, "smoke_manifest_sha256": _semantic_hash(identity)}


def _factor_values(record: Mapping[str, Any], factor: str, *, mask: bool) -> list[float]:
    container = record.get("mask" if mask else "state", {})
    if not isinstance(container, Mapping):
        raise FinalRCLRealExecutionError("candidate state or mask is invalid")
    raw = container.get(factor, ())
    if isinstance(raw, (str, bytes)) or not isinstance(raw, Sequence) or not raw:
        raise FinalRCLRealExecutionError(f"candidate {factor} vector is invalid")
    try:
        values = [float(item) for item in raw]
    except (TypeError, ValueError) as exc:
        raise FinalRCLRealExecutionError(
            f"candidate {factor} vector must be numeric"
        ) from exc
    if not all(math.isfinite(item) for item in values):
        raise FinalRCLRealExecutionError(f"candidate {factor} vector must be finite")
    return values


def _state_distance(left: Mapping[str, Any], right: Mapping[str, Any]) -> float:
    differences: list[float] = []
    for factor in ("mechanism", "propagation", "context"):
        left_state = _factor_values(left, factor, mask=False)
        right_state = _factor_values(right, factor, mask=False)
        left_mask = _factor_values(left, factor, mask=True)
        right_mask = _factor_values(right, factor, mask=True)
        if not (
            len(left_state)
            == len(right_state)
            == len(left_mask)
            == len(right_mask)
        ):
            raise FinalRCLRealExecutionError("candidate state widths drifted")
        for left_value, right_value, left_seen, right_seen in zip(
            left_state, right_state, left_mask, right_mask
        ):
            if left_seen > 0.0 and right_seen > 0.0:
                scale = max(1.0, abs(left_value), abs(right_value))
                differences.append(abs(left_value - right_value) / scale)
    if not differences:
        return 1.0
    return min(1.0, max(0.0, math.fsum(differences) / len(differences)))


def build_all_candidate_compatible_target_plan(
    *,
    source_case_id: str,
    source_root_service_id: str,
    candidate_records: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Record a deterministic compatibility/rejection decision for every service."""

    source_case = str(source_case_id).strip()
    source_root = str(source_root_service_id).strip()
    if not source_case or not source_root:
        raise FinalRCLRealExecutionError("source case and root service are required")
    if isinstance(candidate_records, (str, bytes)) or not isinstance(
        candidate_records, Sequence
    ):
        raise FinalRCLRealExecutionError("candidate records must be a sequence")
    records: dict[str, dict[str, Any]] = {}
    for raw in candidate_records:
        row = deepcopy(dict(raw))
        candidate_id = str(row.get("candidate_id", "")).strip()
        if (
            str(row.get("case_id", "")).strip() != source_case
            or not candidate_id
            or candidate_id in records
        ):
            raise FinalRCLRealExecutionError("candidate coverage or ownership drifted")
        records[candidate_id] = row
    if source_root not in records:
        raise FinalRCLRealExecutionError("source-root observable state is absent")
    source = records[source_root]
    source_mechanism_observed = any(
        item > 0.0 for item in _factor_values(source, "mechanism", mask=True)
    )
    provisional: list[dict[str, Any]] = []
    for candidate_id in sorted(records):
        target = records[candidate_id]
        reasons: list[str] = []
        if not any(
            item > 0.0 for item in _factor_values(target, "context", mask=True)
        ):
            reasons.append("missing_target_context")
        if source_mechanism_observed and not any(
            item > 0.0 for item in _factor_values(target, "mechanism", mask=True)
        ):
            reasons.append("missing_target_observability")
        if not any(
            item > 0.0 for item in _factor_values(target, "propagation", mask=True)
        ):
            reasons.append("missing_target_propagation")
        compatibility = math.exp(-_state_distance(source, target))
        provisional.append(
            {
                "target_service_id": candidate_id,
                "hard_eligible": not reasons,
                "hard_rejection_reasons": reasons,
                "compatibility": compatibility,
                "unnormalized_target_weight": (
                    max(compatibility, 1e-6) if not reasons else 0.0
                ),
                "compatibility_sha256": _semantic_hash(
                    {
                        "source_case_id": source_case,
                        "source_root_service_id": source_root,
                        "target_service_id": candidate_id,
                        "source_state_sha256": _semantic_hash(source),
                        "target_state_sha256": _semantic_hash(target),
                        "compatibility": compatibility,
                        "hard_rejection_reasons": reasons,
                    }
                ),
            }
        )
    total = math.fsum(row["unnormalized_target_weight"] for row in provisional)
    if total <= 0.0:
        raise FinalRCLRealExecutionError("no hard-compatible target service exists")
    normalized = [
        {
            key: value
            for key, value in row.items()
            if key != "unnormalized_target_weight"
        }
        | {"target_weight": row["unnormalized_target_weight"] / total}
        for row in provisional
    ]
    plan_sha = _semantic_hash(
        {
            "schema_version": "final-rcl-all-candidate-compatible-plan-v1",
            "source_case_id": source_case,
            "source_root_service_id": source_root,
            "targets": normalized,
        }
    )
    for row in normalized:
        row["target_weight_plan_sha256"] = plan_sha
    return normalized


def build_final_oser_handshake_request(
    *,
    dataset_id: str,
    training_mode: str,
    supervised_case_ids: Sequence[Any],
    real_label_records: Sequence[Mapping[str, Any]],
    synthetic_family_rows: Sequence[Mapping[str, Any]],
    training_cases: Mapping[str, Mapping[str, Any]],
    inference_cases: Mapping[str, Mapping[str, Any]],
    base_score_artifact: Mapping[str, Any],
    profile: Mapping[str, Any],
    state_transform_sha256: str,
    artifact_role: str,
) -> dict[str, Any]:
    """Seal a JSON-only OSER request for the separate Torch environment."""

    dataset = str(dataset_id)
    mode = str(training_mode)
    supervised = tuple(_ids(supervised_case_ids, "OSER supervised cases"))
    if dataset not in _DATASETS or mode not in {"query_only", "oracle_full"}:
        raise FinalRCLRealExecutionError("OSER dataset or training mode is invalid")
    labels = [deepcopy(dict(row)) for row in real_label_records]
    if [str(row.get("case_id", "")).strip() for row in labels] != list(supervised):
        raise FinalRCLRealExecutionError("OSER real label membership drifted")
    if any(not str(row.get("fault_type", "")).strip() for row in labels):
        raise FinalRCLRealExecutionError("OSER real fault-type label is absent")
    synthetic = [deepcopy(dict(row)) for row in synthetic_family_rows]
    synthetic_keys: list[str] = []
    for row in synthetic:
        row_hash = str(row.get("synthetic_row_sha256", "")).strip()
        parent = str(row.get("source_case_id", "")).strip()
        if (
            len(row_hash) != 64
            or any(character not in "0123456789abcdef" for character in row_hash)
            or parent not in set(supervised)
            or row.get("query_budget_cost") != 0
        ):
            raise FinalRCLRealExecutionError("OSER synthetic family ownership drifted")
        synthetic_keys.append("synthetic:" + row_hash)
    if len(synthetic_keys) != len(set(synthetic_keys)):
        raise FinalRCLRealExecutionError("OSER synthetic family IDs are duplicated")
    train = deepcopy(dict(training_cases))
    expected_training_keys = (*supervised, *synthetic_keys)
    if tuple(train) != expected_training_keys:
        raise FinalRCLRealExecutionError("OSER training case order or coverage drifted")
    inference = deepcopy(dict(inference_cases))
    if not inference or set(inference) & set(train):
        raise FinalRCLRealExecutionError("OSER inference ownership drifted")
    base = deepcopy(dict(base_score_artifact))
    frozen_profile = deepcopy(dict(profile))
    if (
        base.get("schema_version")
        != "conservative-lofo-base-score-artifact-v1"
        or frozen_profile.get("profile_id") != "oser-p02"
        or len(str(state_transform_sha256)) != 64
        or not str(artifact_role).strip()
    ):
        raise FinalRCLRealExecutionError("OSER score/profile/transform closure drifted")
    identity = {
        "schema_version": "final-rcl-oser-handshake-request-v1",
        "dataset_id": dataset,
        "training_mode": mode,
        "supervised_case_ids": supervised,
        "supervised_case_limit": len(supervised),
        "real_label_records": labels,
        "synthetic_family_rows": synthetic,
        "training_cases": train,
        "inference_cases": inference,
        "base_score_artifact": base,
        "profile": frozen_profile,
        "state_transform_sha256": str(state_transform_sha256),
        "artifact_role": str(artifact_role),
        "seed": 42,
        "outer_query_population": "real_cases_only",
    }
    return {**identity, "request_sha256": _semantic_hash(identity)}


def validate_final_smoke_report(
    *,
    smoke_manifest: Mapping[str, Any],
    unit_results: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Validate all six bounded branches while preserving smoke-only ownership."""

    manifest = deepcopy(dict(smoke_manifest))
    supplied_manifest_sha = manifest.pop("smoke_manifest_sha256", None)
    if (
        manifest.get("schema_version") != "final-rcl-six-branch-smoke-v1"
        or manifest.get("evidence_role") != "smoke_only"
        or manifest.get("promotable_to_formal") is not False
        or manifest.get("unit_count") != 6
        or supplied_manifest_sha != _semantic_hash(manifest)
    ):
        raise FinalRCLRealExecutionError("invalid smoke-only manifest")
    manifest_units = [dict(row) for row in manifest.get("units", ())]
    expected = {str(row.get("unit_id", "")): row for row in manifest_units}
    if len(manifest_units) != 6 or len(expected) != 6 or "" in expected:
        raise FinalRCLRealExecutionError("smoke manifest unit coverage drifted")
    if isinstance(unit_results, (str, bytes)) or not isinstance(unit_results, Sequence):
        raise FinalRCLRealExecutionError("smoke unit results must be a sequence")
    results = [deepcopy(dict(row)) for row in unit_results]
    observed = {str(row.get("unit_id", "")): row for row in results}
    if len(results) != 6 or len(observed) != 6 or set(observed) != set(expected):
        raise FinalRCLRealExecutionError("all six smoke branches are required")
    for unit_id, result in observed.items():
        unit = expected[unit_id]
        arm_id = str(unit.get("arm_id", ""))
        if result.get("evidence_role") != "smoke_only":
            raise FinalRCLRealExecutionError("smoke result cannot be promoted")
        if (
            result.get("status") != "completed"
            or result.get("dataset_id") != unit.get("dataset_id")
            or result.get("arm_id") != arm_id
            or result.get("training_mode") != unit.get("training_mode")
            or result.get("fit_test_overlap_count") != 0
            or result.get("candidate_complete_rankings") is not True
            or result.get("finite_rankings") is not True
            or result.get("real_training_case_count")
            != len(unit.get("supervised_case_ids", ()))
            or result.get("evaluation_case_count")
            != len(unit.get("evaluation_case_ids", ()))
        ):
            raise FinalRCLRealExecutionError(f"smoke branch failed validation: {unit_id}")
        oser_status = str(result.get("oser_status", ""))
        if arm_id in _COMPLETE_ARMS:
            if oser_status not in {"active", "inactive_fallback", "declared_fallback"}:
                raise FinalRCLRealExecutionError(
                    f"OSER activity/fallback evidence is absent: {unit_id}"
                )
        elif oser_status != "not_applicable":
            raise FinalRCLRealExecutionError(
                f"pairwise smoke cannot claim OSER activity: {unit_id}"
            )
    identity = {
        "schema_version": "final-rcl-six-branch-smoke-report-v1",
        "status": "passed",
        "evidence_role": "smoke_only",
        "promotable_to_formal": False,
        "passed_unit_count": 6,
        "smoke_manifest_sha256": supplied_manifest_sha,
        "unit_result_sha256s": {
            unit_id: _semantic_hash(result)
            for unit_id, result in sorted(observed.items())
        },
    }
    return {**identity, "smoke_report_sha256": _semantic_hash(identity)}


__all__ = [
    "FinalRCLRealExecutionError",
    "build_all_candidate_compatible_target_plan",
    "build_final_oser_handshake_request",
    "build_final_smoke_manifest",
    "validate_final_smoke_report",
]
