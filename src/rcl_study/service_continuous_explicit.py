from __future__ import annotations

import hashlib
import json
import math
import re
from copy import deepcopy
from collections.abc import Mapping, Sequence
from typing import Any


class ExplicitContinuityValidationError(ValueError):
    """Raised when deterministic service continuity violates its contract."""


_SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")
_EXPLICIT_ARMS = ("explicit_arbitrary", "explicit_compatible")
_INTERPOLATION_STEPS = (0.25, 0.50, 0.75, 1.00)
_BOUNDED_OBSERVABLE_FIELDS = (
    "amplitude",
    "duration",
    "lag",
    "recovery",
    "modality_metric_coverage",
    "modality_log_coverage",
    "modality_trace_coverage",
    "propagation_depth",
    "propagation_width",
    "propagation_direction_consistency",
    "propagation_attenuation",
)
_COVERAGE_FIELDS = frozenset(
    {
        "modality_metric_coverage",
        "modality_log_coverage",
        "modality_trace_coverage",
        "propagation_direction_consistency",
    }
)


def _semantic_hash(value: Any) -> str:
    payload = json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _finite(value: Any, context: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ExplicitContinuityValidationError(f"{context} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise ExplicitContinuityValidationError(f"{context} must be finite")
    return 0.0 if result == 0.0 else result


def _ordered_ids(value: Any, context: str) -> list[str]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise ExplicitContinuityValidationError(f"{context} must be a sequence")
    result = [str(item).strip() for item in value]
    if any(not item for item in result) or len(result) != len(set(result)):
        raise ExplicitContinuityValidationError(
            f"{context} must contain unique nonempty IDs"
        )
    return result


def _arm_config(arm_id: str, target_policy: str) -> dict[str, Any]:
    identity = {
        "schema_version": "service-continuous-explicit-arm-config-v1",
        "arm_id": arm_id,
        "generator_id": "deterministic_service_continuity_v1",
        "target_policy": target_policy,
        "interpolation_steps": list(_INTERPOLATION_STEPS),
        "mechanism_policy": "preserve_exact",
        "propagation_policy": "target_conditioned_linear_transfer",
        "context_policy": "target_conditioned_linear_transfer",
        "label_policy": "queried_source_fault_to_target_root",
        "same_fault_type_partner_required": False,
        "training_only": True,
    }
    return {**identity, "config_sha256": _semantic_hash(identity)}


def build_explicit_arm_registry() -> dict[str, Any]:
    arms = {
        "explicit_arbitrary": _arm_config("explicit_arbitrary", "arbitrary"),
        "explicit_compatible": _arm_config("explicit_compatible", "compatible"),
    }
    identity = {
        "schema_version": "service-continuous-explicit-arm-registry-v1",
        "arm_ids": list(_EXPLICIT_ARMS),
        "arms": arms,
    }
    return {**identity, "registry_sha256": _semantic_hash(identity)}


def validate_explicit_arm_config(
    arm_id: str, config: Mapping[str, Any]
) -> dict[str, Any]:
    arm = str(arm_id).strip()
    if arm not in _EXPLICIT_ARMS or not isinstance(config, Mapping):
        raise ExplicitContinuityValidationError("explicit arm configuration is unknown")
    expected = build_explicit_arm_registry()["arms"][arm]
    if deepcopy(dict(config)) != expected:
        raise ExplicitContinuityValidationError(
            "explicit arm configuration closure drift"
        )
    return {
        "valid": True,
        "arm_id": arm,
        "config_sha256": expected["config_sha256"],
    }


def _vector_state(value: Any, context: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ExplicitContinuityValidationError(f"{context} must be a mapping")
    schema = str(value.get("schema_version", "")).strip()
    raw_names = value.get("feature_names")
    raw_values = value.get("values")
    state_hash = str(value.get("state_sha256", ""))
    if not schema:
        raise ExplicitContinuityValidationError(f"{context} schema must be nonempty")
    if isinstance(raw_names, (str, bytes)) or not isinstance(raw_names, Sequence):
        raise ExplicitContinuityValidationError(f"{context} feature_names must be a sequence")
    if isinstance(raw_values, (str, bytes)) or not isinstance(raw_values, Sequence):
        raise ExplicitContinuityValidationError(f"{context} values must be a sequence")
    names = [str(name).strip() for name in raw_names]
    values = [_finite(item, f"{context}.values") for item in raw_values]
    if (
        not names
        or any(not name for name in names)
        or len(names) != len(set(names))
        or len(names) != len(values)
    ):
        raise ExplicitContinuityValidationError(f"{context} vector layout drift")
    if not _SHA256_PATTERN.fullmatch(state_hash):
        raise ExplicitContinuityValidationError(f"{context} state hash is invalid")
    return {
        "schema_version": schema,
        "feature_names": names,
        "values": values,
        "state_sha256": state_hash,
    }


def _interpolate_state(
    source: Mapping[str, Any],
    target: Mapping[str, Any],
    alpha: float,
    *,
    state_kind: str,
) -> dict[str, Any]:
    if source["feature_names"] != target["feature_names"]:
        raise ExplicitContinuityValidationError(
            f"{state_kind} source/target feature layout drift"
        )
    values = [
        (1.0 - alpha) * source_value + alpha * target_value
        for source_value, target_value in zip(source["values"], target["values"])
    ]
    identity = {
        "schema_version": "service-continuous-explicit-interpolated-state-v1",
        "state_kind": state_kind,
        "feature_names": list(source["feature_names"]),
        "values": values,
        "interpolation_alpha": alpha,
        "source_state_sha256": source["state_sha256"],
        "target_state_sha256": target["state_sha256"],
    }
    return {**identity, "state_sha256": _semantic_hash(identity)}


def _observable_block(
    value: Any, context: str, *, enforce_coverage_range: bool = True
) -> dict[str, dict[str, Any]]:
    if not isinstance(value, Mapping) or set(value) != set(_BOUNDED_OBSERVABLE_FIELDS):
        raise ExplicitContinuityValidationError(
            f"{context} observable field layout drift"
        )
    result: dict[str, dict[str, Any]] = {}
    for field in _BOUNDED_OBSERVABLE_FIELDS:
        raw = value[field]
        if not isinstance(raw, Mapping) or set(raw) != {"value", "mask"}:
            raise ExplicitContinuityValidationError(
                f"{context}.{field} must contain value and mask"
            )
        mask = raw.get("mask")
        if isinstance(mask, bool) or not isinstance(mask, int) or mask not in (0, 1):
            raise ExplicitContinuityValidationError(
                f"{context}.{field} mask must be 0 or 1"
            )
        observed = _finite(raw.get("value"), f"{context}.{field}.value")
        if mask == 0 and observed != 0.0:
            raise ExplicitContinuityValidationError(
                f"{context}.{field} missing values must be zero"
            )
        if (
            enforce_coverage_range
            and field in _COVERAGE_FIELDS
            and mask == 1
            and not 0.0 <= observed <= 1.0
        ):
            raise ExplicitContinuityValidationError(
                f"{context}.{field} coverage must be within [0, 1]"
            )
        result[field] = {"value": observed, "mask": mask}
    return result


def fit_fold_local_observable_bounds(
    rows: Sequence[Mapping[str, Any]],
    *,
    fit_case_ids: Sequence[Any],
    held_out_case_ids: Sequence[Any],
) -> dict[str, Any]:
    if isinstance(rows, (str, bytes)) or not isinstance(rows, Sequence):
        raise ExplicitContinuityValidationError("observable bound rows must be a sequence")
    fit_ids = [str(value).strip() for value in fit_case_ids]
    held_out = [str(value).strip() for value in held_out_case_ids]
    if (
        not fit_ids
        or any(not value for value in fit_ids + held_out)
        or len(fit_ids) != len(set(fit_ids))
        or len(held_out) != len(set(held_out))
    ):
        raise ExplicitContinuityValidationError(
            "fit and held-out case IDs must be unique and nonempty"
        )
    if set(fit_ids) & set(held_out):
        raise ExplicitContinuityValidationError(
            "held-out cases cannot enter observable bound fitting"
        )
    by_case: dict[str, dict[str, dict[str, Any]]] = {}
    for index, raw in enumerate(rows):
        if not isinstance(raw, Mapping) or set(raw) != {"case_id", "observables"}:
            raise ExplicitContinuityValidationError(
                f"observable bound row {index} layout drift"
            )
        case_id = str(raw.get("case_id", "")).strip()
        if not case_id:
            raise ExplicitContinuityValidationError(
                "observable bound case IDs must be unique and nonempty"
            )
        if case_id not in set(fit_ids):
            continue
        if case_id in by_case:
            raise ExplicitContinuityValidationError(
                "observable bound case IDs must be unique and nonempty"
            )
        by_case[case_id] = _observable_block(
            raw.get("observables"), f"observable bound row {case_id}"
        )
    if not set(fit_ids) <= set(by_case):
        raise ExplicitContinuityValidationError("fit observable rows are incomplete")

    fields: dict[str, Any] = {}
    for field in _BOUNDED_OBSERVABLE_FIELDS:
        values = [
            by_case[case_id][field]["value"]
            for case_id in fit_ids
            if by_case[case_id][field]["mask"] == 1
        ]
        if values:
            lower = min(values)
            upper = max(values)
            available = True
        else:
            lower = upper = 0.0
            available = False
        fields[field] = {
            "lower": lower,
            "upper": upper,
            "observed_count": len(values),
            "missing_count": len(fit_ids) - len(values),
            "available": available,
        }
    identity = {
        "schema_version": "service-continuous-explicit-fold-local-bounds-v1",
        "fit_case_ids": fit_ids,
        "held_out_case_ids": held_out,
        "held_out_overlap_count": 0,
        "fields": fields,
        "fit_case_ids_sha256": _semantic_hash(fit_ids),
    }
    return {**identity, "bounds_sha256": _semantic_hash(identity)}


def _validated_bounds(value: Any) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ExplicitContinuityValidationError("observable bounds must be a mapping")
    payload = deepcopy(dict(value))
    supplied_hash = str(payload.pop("bounds_sha256", ""))
    if supplied_hash != _semantic_hash(payload):
        raise ExplicitContinuityValidationError("observable bounds hash drift")
    if (
        payload.get("schema_version")
        != "service-continuous-explicit-fold-local-bounds-v1"
        or payload.get("held_out_overlap_count") != 0
    ):
        raise ExplicitContinuityValidationError("observable bounds protocol drift")
    fields = payload.get("fields")
    if not isinstance(fields, Mapping) or set(fields) != set(_BOUNDED_OBSERVABLE_FIELDS):
        raise ExplicitContinuityValidationError("observable bounds field layout drift")
    for field in _BOUNDED_OBSERVABLE_FIELDS:
        raw = fields[field]
        if not isinstance(raw, Mapping) or set(raw) != {
            "lower",
            "upper",
            "observed_count",
            "missing_count",
            "available",
        }:
            raise ExplicitContinuityValidationError("observable bound entry layout drift")
        lower = _finite(raw["lower"], f"bounds.{field}.lower")
        upper = _finite(raw["upper"], f"bounds.{field}.upper")
        if lower > upper or not isinstance(raw["available"], bool):
            raise ExplicitContinuityValidationError("observable bound interval drift")
    return {**payload, "bounds_sha256": supplied_hash}


def apply_bounded_observable_transform(
    proposal: Mapping[str, Any], bounds: Mapping[str, Any]
) -> dict[str, Any]:
    observables = _observable_block(
        proposal, "synthetic proposal", enforce_coverage_range=False
    )
    validated_bounds = _validated_bounds(bounds)
    transformed: dict[str, dict[str, Any]] = {}
    clipped: list[str] = []
    for field in _BOUNDED_OBSERVABLE_FIELDS:
        current = observables[field]
        interval = validated_bounds["fields"][field]
        if current["mask"] == 0:
            transformed[field] = {"value": 0.0, "mask": 0}
            continue
        if not interval["available"]:
            raise ExplicitContinuityValidationError(
                f"observable field {field} has no fold-local support"
            )
        bounded = min(
            float(interval["upper"]),
            max(float(interval["lower"]), float(current["value"])),
        )
        if bounded != current["value"]:
            clipped.append(field)
        transformed[field] = {"value": bounded, "mask": 1}
    identity = {
        "schema_version": "service-continuous-explicit-bounded-observables-v1",
        "observables": transformed,
        "clipped_field_count": len(clipped),
        "clipped_fields": clipped,
        "bounds_sha256": validated_bounds["bounds_sha256"],
    }
    return {**identity, "transform_sha256": _semantic_hash(identity)}


def generate_explicit_service_continuity(
    *,
    source_case: Mapping[str, Any],
    target_context: Mapping[str, Any],
    arm_config: Mapping[str, Any],
) -> dict[str, Any]:
    if not all(
        isinstance(value, Mapping)
        for value in (source_case, target_context, arm_config)
    ):
        raise ExplicitContinuityValidationError(
            "explicit generation inputs must be mappings"
        )
    arm_id = str(arm_config.get("arm_id", ""))
    validate_explicit_arm_config(arm_id, arm_config)
    source_id = str(source_case.get("case_id", "")).strip()
    target_id = str(target_context.get("target_service_id", "")).strip()
    target_level = str(target_context.get("target_entity_level", "")).strip()
    if not source_id or not target_id or target_level not in {"service", "host"}:
        raise ExplicitContinuityValidationError(
            "source case and target entity mapping must be valid"
        )
    label = source_case.get("queried_label")
    if not isinstance(label, Mapping) or set(label) != {
        "root_cause",
        "fault_type",
        "label_source",
        "budget_cost",
    }:
        raise ExplicitContinuityValidationError("queried label layout drift")
    source_root = str(label.get("root_cause", "")).strip()
    fault_type = str(label.get("fault_type", "")).strip()
    if (
        not source_root
        or not fault_type
        or label.get("label_source") != "queried_budget"
        or label.get("budget_cost") != 1
    ):
        raise ExplicitContinuityValidationError(
            "explicit generation requires one budget-owned queried label"
        )
    observation_case_ids = _ordered_ids(
        source_case.get("observation_case_ids"), "observation case IDs"
    )
    label_case_ids = _ordered_ids(
        source_case.get("label_case_ids"), "label case IDs"
    )
    if source_id not in observation_case_ids or set(label_case_ids) != {source_id}:
        raise ExplicitContinuityValidationError(
            "unqueried neighbor labels cannot enter explicit generation"
        )
    label_access_audit = {
        "label_case_ids": label_case_ids,
        "observation_only_case_ids": [
            case_id for case_id in observation_case_ids if case_id not in set(label_case_ids)
        ],
        "borrowed_label_count": 0,
    }
    raw_partners = source_case.get("same_fault_type_partner_case_ids", ())
    if isinstance(raw_partners, (str, bytes)) or not isinstance(raw_partners, Sequence):
        raise ExplicitContinuityValidationError(
            "same-fault-type partners must be a sequence"
        )
    partners = [str(value).strip() for value in raw_partners]
    if any(not value or value == source_id for value in partners):
        raise ExplicitContinuityValidationError("invalid same-fault-type partner IDs")

    mechanism = _vector_state(source_case.get("mechanism_state"), "source mechanism")
    source_propagation = _vector_state(
        source_case.get("propagation_state"), "source propagation"
    )
    source_context = _vector_state(source_case.get("context_state"), "source context")
    target_propagation = _vector_state(
        target_context.get("propagation_state"), "target propagation"
    )
    target_state = _vector_state(target_context.get("context_state"), "target context")
    target_weight = _finite(target_context.get("target_weight"), "target weight")
    compatibility = _finite(
        target_context.get("compatibility"), "target compatibility"
    )
    target_weight_plan_sha256 = str(
        target_context.get("target_weight_plan_sha256", "")
    )
    compatibility_sha256 = str(target_context.get("compatibility_sha256", ""))
    if not _SHA256_PATTERN.fullmatch(
        target_weight_plan_sha256
    ) or not _SHA256_PATTERN.fullmatch(compatibility_sha256):
        raise ExplicitContinuityValidationError(
            "target weight-plan and compatibility hashes are required"
        )
    if not 0.0 < target_weight <= 1.0 or not 0.0 <= compatibility <= 1.0:
        raise ExplicitContinuityValidationError(
            "target weight and compatibility must be normalized"
        )

    rows = []
    for alpha in _INTERPOLATION_STEPS:
        propagation_state = _interpolate_state(
            source_propagation,
            target_propagation,
            alpha,
            state_kind="propagation",
        )
        context_state = _interpolate_state(
            source_context,
            target_state,
            alpha,
            state_kind="context",
        )
        provenance_identity = {
            "schema_version": "service-continuous-explicit-provenance-v1",
            "arm_id": arm_id,
            "source_case_id": source_id,
            "queried_label": deepcopy(dict(label)),
            "queried_label_sha256": _semantic_hash(dict(label)),
            "target_service_id": target_id,
            "interpolation_alpha": alpha,
            "compatibility": compatibility,
            "target_weight": target_weight,
            "provisional_training_weight": target_weight
            / len(_INTERPOLATION_STEPS),
            "source_state_sha256": {
                "mechanism": mechanism["state_sha256"],
                "propagation": source_propagation["state_sha256"],
                "context": source_context["state_sha256"],
            },
            "target_state_sha256": {
                "propagation": target_propagation["state_sha256"],
                "context": target_state["state_sha256"],
            },
            "generated_state_sha256": {
                "propagation": propagation_state["state_sha256"],
                "context": context_state["state_sha256"],
            },
            "target_weight_plan_sha256": target_weight_plan_sha256,
            "compatibility_sha256": compatibility_sha256,
            "arm_config_sha256": arm_config["config_sha256"],
        }
        provenance = {
            **provenance_identity,
            "provenance_sha256": _semantic_hash(provenance_identity),
        }
        identity = {
            "schema_version": "service-continuous-explicit-synthetic-row-v1",
            "arm_id": arm_id,
            "source_case_id": source_id,
            "source_queried_root_cause": source_root,
            "target_service_id": target_id,
            "target_entity_level": target_level,
            "interpolation_alpha": alpha,
            "mechanism_state": deepcopy(mechanism),
            "propagation_state": propagation_state,
            "context_state": context_state,
            "synthetic_label": {
                "root_cause": target_id,
                "fault_type": fault_type,
                "label_source_case_id": source_id,
                "label_source": "queried_budget_transfer",
            },
            "target_weight": target_weight,
            "per_step_target_mass": target_weight / len(_INTERPOLATION_STEPS),
            "compatibility": compatibility,
            "soft_compatibility_used": arm_config["target_policy"] == "compatible",
            "query_budget_cost": 0,
            "same_fault_type_partner_used": False,
            "arm_config_sha256": arm_config["config_sha256"],
            "provenance": provenance,
        }
        rows.append({**identity, "synthetic_row_sha256": _semantic_hash(identity)})
    bundle_identity = {
        "schema_version": "service-continuous-explicit-generation-bundle-v1",
        "arm_id": arm_id,
        "source_case_id": source_id,
        "target_service_id": target_id,
        "same_fault_type_partner_required": False,
        "same_fault_type_partner_available": bool(partners),
        "label_access_audit": label_access_audit,
        "synthetic_row_count": len(rows),
        "synthetic_rows": rows,
    }
    return {
        **bundle_identity,
        "generation_bundle_sha256": _semantic_hash(bundle_identity),
    }


__all__ = [
    "ExplicitContinuityValidationError",
    "apply_bounded_observable_transform",
    "build_explicit_arm_registry",
    "fit_fold_local_observable_bounds",
    "generate_explicit_service_continuity",
    "validate_explicit_arm_config",
]
