from __future__ import annotations

import hashlib
import json
import math
from collections import Counter
from collections.abc import Mapping, Sequence
from copy import deepcopy
from typing import Any


class UnlabeledPoolValidationError(ValueError):
    """Raised when a Strict-LOFO unlabeled companion is unsafe or incomplete."""


_FACTORS = ("mechanism", "propagation", "context")
_RECORD_FIELDS = {"case_id", "candidate_id", "state", "mask"}
_MECHANISM_FIELDS = (
    "metric_direction",
    "metric_magnitude",
    "metric_duration",
    "metric_sparsity",
    "log_intensity",
    "log_template_change",
    "log_relative_time",
    "trace_latency",
    "trace_error",
    "trace_earliest_anomaly",
    "trace_hop_lag",
)
_PROPAGATION_FIELDS = (
    "topology_depth",
    "topology_width",
    "topology_direction_consistency",
    "relative_onset",
    "relative_peak",
    "relative_recovery",
)
_CONTEXT_VALUE_FIELDS = (
    "candidate_is_service",
    "candidate_reachability",
    "candidate_source_earliness",
    "candidate_explanation_coverage",
)
_PRESENCE_FIELDS = (
    "has_metric_signal",
    "has_log_signal",
    "has_trace_signal",
    "has_topology_signal",
    "has_time_signal",
    "has_candidate_signal",
)
OBSERVABLE_STATE_FIELDS = (
    *_MECHANISM_FIELDS,
    *_PROPAGATION_FIELDS,
    *_CONTEXT_VALUE_FIELDS,
    *_PRESENCE_FIELDS,
)
_FORBIDDEN_KEYS = {
    "fault_type",
    "held_out_fault_type",
    "is_positive",
    "label",
    "labels_by_case",
    "positive_index",
    "positive_indices",
    "positive_ids",
    "positive_ids_list",
    "root",
    "root_cause",
    "root_cause_id",
    "targets",
    "targets_by_case",
}


def _semantic_hash(value: Any) -> str:
    payload = json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _sha256(value: Any, context: str) -> str:
    result = str(value).strip().lower()
    if len(result) != 64 or any(character not in "0123456789abcdef" for character in result):
        raise UnlabeledPoolValidationError(f"{context} must be a SHA-256")
    return result


def _ids(value: Any, context: str) -> tuple[str, ...]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise UnlabeledPoolValidationError(f"{context} must be a sequence")
    result = tuple(str(item).strip() for item in value)
    if not result or any(not item for item in result) or len(result) != len(set(result)):
        raise UnlabeledPoolValidationError(f"{context} must contain unique nonempty IDs")
    return result


def _forbidden_keys(value: Any) -> list[str]:
    found: list[str] = []
    if isinstance(value, Mapping):
        for raw_key, item in value.items():
            key = str(raw_key).strip().lower()
            if key in _FORBIDDEN_KEYS:
                found.append(key)
            found.extend(_forbidden_keys(item))
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        for item in value:
            found.extend(_forbidden_keys(item))
    return found


def _finite_vector(value: Any, context: str, *, mask: bool) -> list[float]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise UnlabeledPoolValidationError(f"{context} must be a sequence")
    result = [float(item) for item in value]
    if not result or not all(math.isfinite(item) for item in result):
        raise UnlabeledPoolValidationError(f"{context} must be finite and nonempty")
    if mask and any(item not in (0.0, 1.0) for item in result):
        raise UnlabeledPoolValidationError(f"{context} must be binary")
    return result


def build_unlabeled_state_record(observable_row: Mapping[str, Any]) -> dict[str, Any]:
    """Reduce one legal observable row to the neural state/mask contract."""

    if not isinstance(observable_row, Mapping):
        raise UnlabeledPoolValidationError("observable row must be a mapping")
    row = dict(observable_row)
    expected_fields = {"case_id", "candidate_id", *OBSERVABLE_STATE_FIELDS}
    if set(row) != expected_fields:
        if _forbidden_keys(row):
            raise UnlabeledPoolValidationError(
                "observable row contains a forbidden ground-truth field"
            )
        raise UnlabeledPoolValidationError("observable row field closure drift")
    case_id = str(row["case_id"]).strip()
    candidate_id = str(row["candidate_id"]).strip()
    if not case_id or not candidate_id:
        raise UnlabeledPoolValidationError("observable row ownership drift")
    presence = {
        field: float(row[field]) for field in _PRESENCE_FIELDS
    }
    if any(value not in (0.0, 1.0) for value in presence.values()):
        raise UnlabeledPoolValidationError("observable presence mask must be binary")
    state = {
        "mechanism": _finite_vector(
            [row[field] for field in _MECHANISM_FIELDS],
            "mechanism state",
            mask=False,
        ),
        "propagation": _finite_vector(
            [row[field] for field in _PROPAGATION_FIELDS],
            "propagation state",
            mask=False,
        ),
        "context": _finite_vector(
            [row[field] for field in (*_CONTEXT_VALUE_FIELDS, *_PRESENCE_FIELDS)],
            "context state",
            mask=False,
        ),
    }
    masks = {
        "mechanism": (
            [presence["has_metric_signal"]] * 4
            + [presence["has_log_signal"]] * 3
            + [presence["has_trace_signal"]] * 4
        ),
        "propagation": (
            [presence["has_topology_signal"]] * 3
            + [presence["has_time_signal"]] * 3
        ),
        "context": [presence["has_candidate_signal"]] * 4 + [1.0] * 6,
    }
    return {
        "case_id": case_id,
        "candidate_id": candidate_id,
        "state": state,
        "mask": masks,
    }


def _normalize_records(
    records: Any,
    *,
    candidate_case_ids: tuple[str, ...],
    expected_candidates_per_case: int,
) -> list[dict[str, Any]]:
    if isinstance(records, (str, bytes)) or not isinstance(records, Sequence):
        raise UnlabeledPoolValidationError("records must be a sequence")
    if _forbidden_keys(records):
        raise UnlabeledPoolValidationError("records contain forbidden ground-truth fields")
    allowed_cases = set(candidate_case_ids)
    normalized: list[dict[str, Any]] = []
    ownership: set[tuple[str, str]] = set()
    dimensions: dict[str, int] = {}
    counts: Counter[str] = Counter()
    for raw in records:
        if not isinstance(raw, Mapping) or set(raw) != _RECORD_FIELDS:
            raise UnlabeledPoolValidationError("unlabeled record field closure drift")
        case_id = str(raw["case_id"]).strip()
        candidate_id = str(raw["candidate_id"]).strip()
        key = (case_id, candidate_id)
        if (
            not case_id
            or not candidate_id
            or case_id not in allowed_cases
            or key in ownership
        ):
            raise UnlabeledPoolValidationError("unlabeled record ownership drift")
        state = raw["state"]
        masks = raw["mask"]
        if not isinstance(state, Mapping) or set(state) != set(_FACTORS):
            raise UnlabeledPoolValidationError("state factor closure drift")
        if not isinstance(masks, Mapping) or set(masks) != set(_FACTORS):
            raise UnlabeledPoolValidationError("mask factor closure drift")
        normalized_state: dict[str, list[float]] = {}
        normalized_mask: dict[str, list[float]] = {}
        for factor in _FACTORS:
            values = _finite_vector(state[factor], f"{key}.{factor}", mask=False)
            factor_mask = _finite_vector(
                masks[factor], f"{key}.{factor}_mask", mask=True
            )
            if len(values) != len(factor_mask):
                raise UnlabeledPoolValidationError("state/mask dimension drift")
            if factor in dimensions and dimensions[factor] != len(values):
                raise UnlabeledPoolValidationError("factor width drift")
            dimensions[factor] = len(values)
            normalized_state[factor] = values
            normalized_mask[factor] = factor_mask
        ownership.add(key)
        counts[case_id] += 1
        normalized.append(
            {
                "case_id": case_id,
                "candidate_id": candidate_id,
                "state": normalized_state,
                "mask": normalized_mask,
            }
        )
    if set(counts) != allowed_cases or any(
        counts[case_id] != expected_candidates_per_case
        for case_id in candidate_case_ids
    ):
        raise UnlabeledPoolValidationError("candidate pool record coverage drift")
    return normalized


def build_unlabeled_candidate_companion(
    *,
    dataset_id: str,
    authority_request_sha256: str,
    membership_sha256: str,
    candidate_case_ids: Sequence[str],
    held_out_case_ids: Sequence[str],
    records: Sequence[Mapping[str, Any]],
    expected_candidates_per_case: int,
) -> dict[str, Any]:
    dataset = str(dataset_id).strip()
    if not dataset:
        raise UnlabeledPoolValidationError("dataset_id must be nonempty")
    authority_hash = _sha256(authority_request_sha256, "authority_request_sha256")
    membership_hash = _sha256(membership_sha256, "membership_sha256")
    candidates = _ids(candidate_case_ids, "candidate_case_ids")
    held_out = _ids(held_out_case_ids, "held_out_case_ids")
    if set(candidates) & set(held_out):
        raise UnlabeledPoolValidationError("held-out cases overlap the unlabeled candidate pool")
    expected = int(expected_candidates_per_case)
    if expected <= 0:
        raise UnlabeledPoolValidationError("expected_candidates_per_case must be positive")
    safe_records = _normalize_records(
        records,
        candidate_case_ids=candidates,
        expected_candidates_per_case=expected,
    )
    identity = {
        "schema_version": "service-continuous-unlabeled-candidate-companion-v1",
        "dataset_id": dataset,
        "authority_request_sha256": authority_hash,
        "membership_sha256": membership_hash,
        "candidate_case_ids": list(candidates),
        "candidate_case_ids_sha256": _semantic_hash(list(candidates)),
        "held_out_case_ids_sha256": _semantic_hash(list(held_out)),
        "expected_candidates_per_case": expected,
        "candidate_case_count": len(candidates),
        "candidate_row_count": len(safe_records),
        "records": safe_records,
        "label_access": "unlabeled_observations_only",
        "forbidden_field_count": 0,
        "held_out_overlap_count": 0,
    }
    return {**identity, "companion_sha256": _semantic_hash(identity)}


def validate_unlabeled_candidate_companion(
    companion: Mapping[str, Any],
    *,
    expected_authority_request_sha256: str,
    expected_membership_sha256: str,
    expected_candidate_case_ids: Sequence[str],
    expected_held_out_case_ids: Sequence[str],
    expected_candidates_per_case: int,
) -> dict[str, Any]:
    value = deepcopy(dict(companion))
    identity = {
        key: item for key, item in value.items() if key != "companion_sha256"
    }
    if value.get("companion_sha256") != _semantic_hash(identity):
        raise UnlabeledPoolValidationError("companion hash drift")
    candidates = _ids(expected_candidate_case_ids, "expected_candidate_case_ids")
    held_out = _ids(expected_held_out_case_ids, "expected_held_out_case_ids")
    if value.get("schema_version") != "service-continuous-unlabeled-candidate-companion-v1":
        raise UnlabeledPoolValidationError("companion schema drift")
    if value.get("authority_request_sha256") != _sha256(
        expected_authority_request_sha256, "expected_authority_request_sha256"
    ):
        raise UnlabeledPoolValidationError("authority request binding drift")
    if value.get("membership_sha256") != _sha256(
        expected_membership_sha256, "expected_membership_sha256"
    ):
        raise UnlabeledPoolValidationError("membership binding drift")
    if value.get("candidate_case_ids") != list(candidates):
        raise UnlabeledPoolValidationError("candidate membership drift")
    if value.get("candidate_case_ids_sha256") != _semantic_hash(list(candidates)):
        raise UnlabeledPoolValidationError("candidate membership hash drift")
    if value.get("held_out_case_ids_sha256") != _semantic_hash(list(held_out)):
        raise UnlabeledPoolValidationError("held-out membership hash drift")
    if set(candidates) & set(held_out):
        raise UnlabeledPoolValidationError("held-out cases overlap the candidate pool")
    records = _normalize_records(
        value.get("records"),
        candidate_case_ids=candidates,
        expected_candidates_per_case=int(expected_candidates_per_case),
    )
    if records != value.get("records"):
        raise UnlabeledPoolValidationError("companion record normalization drift")
    if _forbidden_keys(value.get("records")):
        raise UnlabeledPoolValidationError("companion contains forbidden fields")
    expected_rows = len(candidates) * int(expected_candidates_per_case)
    if (
        value.get("candidate_case_count") != len(candidates)
        or value.get("candidate_row_count") != expected_rows
        or value.get("expected_candidates_per_case")
        != int(expected_candidates_per_case)
        or value.get("forbidden_field_count") != 0
        or value.get("held_out_overlap_count") != 0
        or value.get("label_access") != "unlabeled_observations_only"
    ):
        raise UnlabeledPoolValidationError("companion audit closure drift")
    return {
        "valid": True,
        "candidate_case_count": len(candidates),
        "candidate_row_count": expected_rows,
        "held_out_overlap_count": 0,
        "forbidden_field_count": 0,
        "companion_sha256": value["companion_sha256"],
    }


__all__ = [
    "OBSERVABLE_STATE_FIELDS",
    "UnlabeledPoolValidationError",
    "build_unlabeled_state_record",
    "build_unlabeled_candidate_companion",
    "validate_unlabeled_candidate_companion",
]
