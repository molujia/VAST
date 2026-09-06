from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass, field
from statistics import median
from typing import Any, Mapping, Sequence


class StateValidationError(ValueError):
    """Raised when observable state is leaky, incomplete, or non-finite."""


_FIELD_TYPES = ("metric", "log", "trace", "topology", "time", "candidate")
_FORBIDDEN_EXACT = {
    "case_id",
    "case_id_semantics",
    "semantic_case_id",
    "candidate_id",
    "entity_id",
    "entity_name",
    "service_name",
    "raw_service_name",
    "dataset",
    "dataset_id",
    "fault_type",
    "fault_type_name",
    "failure_type",
    "injection_time",
    "injection_timestamp",
    "true_root_cause_anchor",
    "root_cause",
    "root_cause_id",
    "artificial_propagation_path",
    "label",
    "is_positive",
    "targets",
}
_FORBIDDEN_FRAGMENTS = (
    "fault_type",
    "root_cause",
    "injection_time",
    "artificial_propagation",
)


def _canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def _semantic_hash(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _identity_key(kind: str, value: str) -> str:
    return hashlib.sha256(f"{kind}:{value}".encode("utf-8")).hexdigest()


def _is_forbidden_field(name: str) -> bool:
    value = str(name).strip().lower()
    return value in _FORBIDDEN_EXACT or any(fragment in value for fragment in _FORBIDDEN_FRAGMENTS)


def _finite_float(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


@dataclass(frozen=True)
class ObservableStateSchema:
    metric_fields: tuple[str, ...]
    log_fields: tuple[str, ...]
    trace_fields: tuple[str, ...]
    topology_fields: tuple[str, ...]
    time_fields: tuple[str, ...]
    candidate_fields: tuple[str, ...]
    modality_presence_fields: Mapping[str, str] = field(default_factory=dict)
    clip_value: float = 20.0

    def __post_init__(self) -> None:
        groups = self.fields_by_type
        all_fields = self.feature_order
        if not all_fields or len(all_fields) != len(set(all_fields)):
            raise StateValidationError("observable-state feature names must be unique and nonempty")
        for name in all_fields:
            if not str(name).strip() or _is_forbidden_field(name):
                raise StateValidationError(f"forbidden observable-state feature: {name}")
        presence = dict(self.modality_presence_fields)
        if set(presence) != set(_FIELD_TYPES):
            raise StateValidationError("modality presence fields must cover every field type")
        if any(not str(value).strip() for value in presence.values()):
            raise StateValidationError("modality presence field names must be nonempty")
        if not math.isfinite(float(self.clip_value)) or float(self.clip_value) <= 0:
            raise StateValidationError("clip_value must be finite and positive")
        if any(not values for values in groups.values()):
            raise StateValidationError("every observable-state field type must be nonempty")

    @property
    def fields_by_type(self) -> dict[str, tuple[str, ...]]:
        return {
            "metric": tuple(self.metric_fields),
            "log": tuple(self.log_fields),
            "trace": tuple(self.trace_fields),
            "topology": tuple(self.topology_fields),
            "time": tuple(self.time_fields),
            "candidate": tuple(self.candidate_fields),
        }

    @property
    def feature_order(self) -> tuple[str, ...]:
        return tuple(
            field_name
            for field_type in _FIELD_TYPES
            for field_name in self.fields_by_type[field_type]
        )

    @property
    def field_type_by_name(self) -> dict[str, str]:
        return {
            field_name: field_type
            for field_type, values in self.fields_by_type.items()
            for field_name in values
        }

    def to_dict(self) -> dict[str, Any]:
        identity = {
            "schema_version": "conservative-lofo-observable-state-schema-v1",
            "fields_by_type": self.fields_by_type,
            "feature_order": self.feature_order,
            "modality_presence_fields": dict(self.modality_presence_fields),
            "clip_value": float(self.clip_value),
        }
        return {**identity, "schema_sha256": _semantic_hash(identity)}


def _normalize_rows(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    normalized = []
    seen: set[tuple[str, str]] = set()
    for raw in rows:
        row = dict(raw)
        case_id = str(row.get("case_id", "")).strip()
        candidate_id = str(row.get("candidate_id", "")).strip()
        if not case_id or not candidate_id:
            raise StateValidationError("observable-state row lacks case_id or candidate_id")
        key = (case_id, candidate_id)
        if key in seen:
            raise StateValidationError(f"duplicate observable-state candidate row: {key}")
        seen.add(key)
        row["case_id"] = case_id
        row["candidate_id"] = candidate_id
        normalized.append(row)
    if not normalized:
        raise StateValidationError("observable-state rows must be nonempty")
    return normalized


def _available(row: Mapping[str, Any], schema: ObservableStateSchema, field_type: str) -> bool:
    field_name = str(schema.modality_presence_fields[field_type])
    if field_name not in row:
        raise StateValidationError(f"missing explicit modality presence field: {field_name}")
    value = _finite_float(row[field_name])
    if value not in (0.0, 1.0):
        raise StateValidationError(f"modality presence must be binary: {field_name}")
    return bool(value)


def _robust_statistics(values: Sequence[float]) -> dict[str, Any]:
    if not values:
        return {
            "count": 0,
            "median": 0.0,
            "scale": 1.0,
            "scale_method": "empty_default",
        }
    ordered = sorted(float(value) for value in values)
    center = float(median(ordered))
    deviations = [abs(value - center) for value in ordered]
    mad = float(median(deviations))
    scale = 1.4826 * mad
    method = "mad"
    if scale <= 1e-9:
        lower = ordered[int((len(ordered) - 1) * 0.25)]
        upper = ordered[int((len(ordered) - 1) * 0.75)]
        scale = (upper - lower) / 1.349 if upper > lower else 0.0
        method = "iqr"
    if scale <= 1e-9:
        mean = sum(ordered) / len(ordered)
        scale = math.sqrt(sum((value - mean) ** 2 for value in ordered) / len(ordered))
        method = "std"
    if scale <= 1e-9:
        scale = 1.0
        method = "unit"
    return {
        "count": len(ordered),
        "median": center,
        "scale": float(scale),
        "scale_method": method,
    }


def fit_fold_state_transform(
    rows: Sequence[Mapping[str, Any]],
    schema: ObservableStateSchema,
    fit_case_ids: Sequence[Any],
    held_out_case_ids: Sequence[Any],
) -> dict[str, Any]:
    normalized = _normalize_rows(rows)
    fit_ids = tuple(str(value) for value in fit_case_ids)
    held_ids = tuple(str(value) for value in held_out_case_ids)
    if not fit_ids or len(fit_ids) != len(set(fit_ids)):
        raise StateValidationError("fit_case_ids must be unique and nonempty")
    if len(held_ids) != len(set(held_ids)):
        raise StateValidationError("held_out_case_ids must be unique")
    overlap = set(fit_ids) & set(held_ids)
    if overlap:
        raise StateValidationError("fold state transform has held-out overlap")
    available_case_ids = {row["case_id"] for row in normalized}
    if not set(fit_ids) <= available_case_ids:
        raise StateValidationError("fit_case_ids are absent from observable rows")
    fitted_rows = [row for row in normalized if row["case_id"] in set(fit_ids)]
    statistics: dict[str, Any] = {}
    for field_name in schema.feature_order:
        field_type = schema.field_type_by_name[field_name]
        values = []
        for row in fitted_rows:
            if not _available(row, schema, field_type):
                continue
            value = _finite_float(row.get(field_name))
            if value is not None:
                values.append(value)
        statistics[field_name] = _robust_statistics(values)
    schema_payload = schema.to_dict()
    identity = {
        "schema_version": "conservative-lofo-fold-state-transform-v1",
        "schema_sha256": schema_payload["schema_sha256"],
        "feature_order": schema.feature_order,
        "statistics": statistics,
        "fit_case_keys": tuple(sorted(_identity_key("case", value) for value in fit_ids)),
        "held_out_case_keys": tuple(sorted(_identity_key("case", value) for value in held_ids)),
        "fit_case_count": len(fit_ids),
        "held_out_case_count": len(held_ids),
        "held_out_overlap_count": 0,
        "clip_value": float(schema.clip_value),
    }
    return {**identity, "transform_sha256": _semantic_hash(identity)}


def _validate_transform(transform: Mapping[str, Any], schema: ObservableStateSchema) -> None:
    if transform.get("schema_version") != "conservative-lofo-fold-state-transform-v1":
        raise StateValidationError("unexpected fold state transform schema")
    if transform.get("schema_sha256") != schema.to_dict()["schema_sha256"]:
        raise StateValidationError("fold state transform schema ownership mismatch")
    if tuple(transform.get("feature_order", ())) != schema.feature_order:
        raise StateValidationError("fold state transform feature order drift")
    identity = {key: value for key, value in transform.items() if key != "transform_sha256"}
    if transform.get("transform_sha256") != _semantic_hash(identity):
        raise StateValidationError("fold state transform hash drift")


def _normalized_feature(
    row: Mapping[str, Any],
    field_name: str,
    statistics: Mapping[str, Any],
    clip_value: float,
) -> tuple[float, int]:
    value = _finite_float(row.get(field_name))
    if value is None:
        return 0.0, 0
    center = float(statistics["median"])
    scale = float(statistics["scale"])
    normalized = (value - center) / scale
    normalized = max(-clip_value, min(clip_value, normalized))
    return float(normalized), 1


def _cross_modal_corroboration(
    vector: Sequence[float],
    mask: Sequence[int],
    schema: ObservableStateSchema,
) -> float:
    index_by_feature = {name: index for index, name in enumerate(schema.feature_order)}
    activities = []
    for field_type in ("metric", "log", "trace"):
        indices = [index_by_feature[name] for name in schema.fields_by_type[field_type]]
        observed = [abs(float(vector[index])) for index in indices if mask[index] == 1]
        if observed:
            activities.append(sum(observed) / len(observed))
    if len(activities) < 2:
        return 0.0
    agreements = [
        1.0 / (1.0 + abs(left - right))
        for left_index, left in enumerate(activities)
        for right in activities[left_index + 1 :]
    ]
    return float(sum(agreements) / len(agreements))


def _raw_unit_value(row: Mapping[str, Any], field_name: str) -> float:
    value = _finite_float(row.get(field_name))
    return 0.0 if value is None else max(0.0, min(1.0, value))


def build_candidate_state_rows(
    rows: Sequence[Mapping[str, Any]],
    schema: ObservableStateSchema,
    transform: Mapping[str, Any],
    admitted_case_ids: Sequence[Any],
    artifact_role: str,
) -> dict[str, Any]:
    _validate_transform(transform, schema)
    normalized = _normalize_rows(rows)
    admitted = tuple(str(value) for value in admitted_case_ids)
    if not admitted or len(admitted) != len(set(admitted)):
        raise StateValidationError("admitted_case_ids must be unique and nonempty")
    admitted_set = set(admitted)
    selected = [row for row in normalized if row["case_id"] in admitted_set]
    if {row["case_id"] for row in selected} != admitted_set:
        raise StateValidationError("observable-state admitted case coverage drift")
    feature_manifest = [
        {
            "field_name": field_name,
            "field_type": schema.field_type_by_name[field_name],
            "index": index,
        }
        for index, field_name in enumerate(schema.feature_order)
    ]
    membership = []
    state_rows = []
    fields_by_type = schema.fields_by_type
    for row in selected:
        case_id = row["case_id"]
        candidate_id = row["candidate_id"]
        case_key = _identity_key("case", case_id)
        candidate_key = _identity_key("candidate", f"{case_id}\x1f{candidate_id}")
        membership.append(
            {
                "case_id": case_id,
                "candidate_id": candidate_id,
                "case_key": case_key,
                "candidate_key": candidate_key,
            }
        )
        modality_mask: dict[str, int] = {}
        coverage: dict[str, float] = {}
        state_vector = []
        state_mask = []
        for field_type in _FIELD_TYPES:
            declared_available = _available(row, schema, field_type)
            observed_count = 0
            for field_name in fields_by_type[field_type]:
                if declared_available:
                    value, observed = _normalized_feature(
                        row,
                        field_name,
                        transform["statistics"][field_name],
                        float(schema.clip_value),
                    )
                else:
                    value, observed = 0.0, 0
                state_vector.append(value)
                state_mask.append(observed)
                observed_count += observed
            modality_mask[field_type] = int(declared_available and observed_count > 0)
            coverage[field_type] = (
                observed_count / float(len(fields_by_type[field_type]))
                if declared_available
                else 0.0
            )
        corroboration = _cross_modal_corroboration(
            state_vector, state_mask, schema
        )
        if modality_mask["trace"] and modality_mask["topology"]:
            propagation_confidence = (
                0.50 * min(coverage["trace"], coverage["topology"])
                + 0.25
                * _raw_unit_value(row, "topology_direction_consistency")
                + 0.25 * corroboration
            )
            propagation_confidence = max(0.0, min(1.0, propagation_confidence))
        else:
            propagation_confidence = 0.0
        state_rows.append(
            {
                "case_key": case_key,
                "candidate_key": candidate_key,
                "state_vector": state_vector,
                "state_mask": state_mask,
                "modality_mask": modality_mask,
                "coverage": coverage,
                "cross_modal_corroboration": corroboration,
                "propagation_confidence": float(propagation_confidence),
            }
        )
    schema_payload = schema.to_dict()
    identity = {
        "schema_version": "conservative-lofo-observable-state-artifact-v1",
        "artifact_role": str(artifact_role),
        "schema_sha256": schema_payload["schema_sha256"],
        "transform_sha256": str(transform["transform_sha256"]),
        "feature_manifest": feature_manifest,
        "membership": membership,
        "state_rows": state_rows,
        "admitted_case_keys": tuple(sorted(_identity_key("case", value) for value in admitted)),
    }
    artifact = {**identity, "artifact_sha256": _semantic_hash(identity)}
    validate_observable_state_artifact(artifact, schema, transform)
    return artifact


def validate_observable_state_artifact(
    artifact: Mapping[str, Any],
    schema: ObservableStateSchema,
    transform: Mapping[str, Any],
) -> dict[str, Any]:
    _validate_transform(transform, schema)
    if artifact.get("schema_version") != "conservative-lofo-observable-state-artifact-v1":
        raise StateValidationError("unexpected observable-state artifact schema")
    if artifact.get("schema_sha256") != schema.to_dict()["schema_sha256"]:
        raise StateValidationError("observable-state schema ownership mismatch")
    if artifact.get("transform_sha256") != transform.get("transform_sha256"):
        raise StateValidationError("observable-state transform ownership mismatch")
    feature_manifest = list(artifact.get("feature_manifest", ()))
    if [row.get("field_name") for row in feature_manifest] != list(schema.feature_order):
        raise StateValidationError("observable-state feature manifest drift")
    if any(_is_forbidden_field(str(row.get("field_name", ""))) for row in feature_manifest):
        raise StateValidationError("observable-state feature manifest contains forbidden identity")
    membership = list(artifact.get("membership", ()))
    state_rows = list(artifact.get("state_rows", ()))
    if not membership or len(membership) != len(state_rows):
        raise StateValidationError("observable-state candidate membership is incomplete")
    member_keys = []
    for member, state in zip(membership, state_rows):
        if member.get("case_key") != state.get("case_key") or member.get(
            "candidate_key"
        ) != state.get("candidate_key"):
            raise StateValidationError("observable-state membership/state order drift")
        member_keys.append(str(member.get("candidate_key", "")))
        vector = list(state.get("state_vector", ()))
        mask = list(state.get("state_mask", ()))
        if len(vector) != len(schema.feature_order) or len(mask) != len(vector):
            raise StateValidationError("observable-state vector shape drift")
        if any(value not in (0, 1) for value in mask):
            raise StateValidationError("observable-state mask must be binary")
        if not all(math.isfinite(float(value)) for value in vector):
            raise StateValidationError("observable-state vector is non-finite")
        modality_mask = dict(state.get("modality_mask", {}))
        coverage = dict(state.get("coverage", {}))
        if set(modality_mask) != set(_FIELD_TYPES) or set(coverage) != set(_FIELD_TYPES):
            raise StateValidationError("observable-state modality audit is incomplete")
        offset = 0
        for field_type in _FIELD_TYPES:
            width = len(schema.fields_by_type[field_type])
            group_mask = mask[offset : offset + width]
            group_vector = vector[offset : offset + width]
            if modality_mask[field_type] not in (0, 1):
                raise StateValidationError("observable-state modality mask must be binary")
            if not 0.0 <= float(coverage[field_type]) <= 1.0:
                raise StateValidationError("observable-state coverage must lie in [0,1]")
            if modality_mask[field_type] == 0 and (
                any(group_mask) or any(float(value) != 0.0 for value in group_vector)
            ):
                raise StateValidationError("missing modality is not exact zero with mask zero")
            offset += width
        propagation = float(state.get("propagation_confidence", math.nan))
        corroboration = float(state.get("cross_modal_corroboration", math.nan))
        if not (math.isfinite(propagation) and 0.0 <= propagation <= 1.0):
            raise StateValidationError("propagation confidence is invalid")
        if not (math.isfinite(corroboration) and 0.0 <= corroboration <= 1.0):
            raise StateValidationError("cross-modal corroboration is invalid")
        if (not modality_mask["trace"] or not modality_mask["topology"]) and propagation != 0.0:
            raise StateValidationError("missing Trace/topology produced propagation confidence")
    if len(member_keys) != len(set(member_keys)):
        raise StateValidationError("observable-state candidate keys are duplicated")
    identity = {key: value for key, value in artifact.items() if key != "artifact_sha256"}
    if artifact.get("artifact_sha256") != _semantic_hash(identity):
        raise StateValidationError("observable-state artifact hash drift")
    return {
        "valid": True,
        "candidate_count": len(state_rows),
        "schema_sha256": artifact["schema_sha256"],
        "transform_sha256": artifact["transform_sha256"],
        "artifact_sha256": artifact["artifact_sha256"],
    }


def tensorize_observable_state_artifact(artifact: Mapping[str, Any]) -> dict[str, Any]:
    state_rows = list(artifact.get("state_rows", ()))
    if not state_rows:
        raise StateValidationError("cannot tensorize an empty observable-state artifact")
    width = len(state_rows[0].get("state_vector", ()))
    if width <= 0 or any(len(row.get("state_vector", ())) != width for row in state_rows):
        raise StateValidationError("observable-state tensor rows have inconsistent width")
    payload = {
        "schema_version": "conservative-lofo-observable-state-tensors-v1",
        "source_artifact_sha256": str(artifact.get("artifact_sha256", "")),
        "shape": [len(state_rows), width],
        "case_keys": [str(row["case_key"]) for row in state_rows],
        "candidate_keys": [str(row["candidate_key"]) for row in state_rows],
        "state_vector": [list(row["state_vector"]) for row in state_rows],
        "state_mask": [list(row["state_mask"]) for row in state_rows],
        "modality_mask": [dict(row["modality_mask"]) for row in state_rows],
        "coverage": [dict(row["coverage"]) for row in state_rows],
        "cross_modal_corroboration": [
            float(row["cross_modal_corroboration"]) for row in state_rows
        ],
        "propagation_confidence": [
            float(row["propagation_confidence"]) for row in state_rows
        ],
    }
    return {**payload, "tensor_sha256": _semantic_hash(payload)}


def summarize_modality_health(artifact: Mapping[str, Any]) -> dict[str, Any]:
    state_rows = list(artifact.get("state_rows", ()))
    if not state_rows:
        raise StateValidationError("cannot summarize empty observable-state rows")
    modalities = {}
    for field_type in _FIELD_TYPES:
        available = [int(row["modality_mask"][field_type]) for row in state_rows]
        coverages = [float(row["coverage"][field_type]) for row in state_rows]
        modalities[field_type] = {
            "available_count": sum(available),
            "missing_count": len(state_rows) - sum(available),
            "mean_coverage": sum(coverages) / len(coverages),
        }
    payload = {
        "schema_version": "conservative-lofo-modality-health-v1",
        "source_artifact_sha256": str(artifact.get("artifact_sha256", "")),
        "candidate_count": len(state_rows),
        "modalities": modalities,
    }
    return {**payload, "health_sha256": _semantic_hash(payload)}


def deterministic_identity_renaming(
    membership: Sequence[Mapping[str, Any]], seed: int = 42
) -> dict[str, Any]:
    if int(seed) != 42:
        raise StateValidationError("identity renaming seed must be 42")
    case_ids = sorted({str(row.get("case_id", "")) for row in membership})
    candidate_ids = sorted({str(row.get("candidate_id", "")) for row in membership})
    if not case_ids or "" in case_ids or "" in candidate_ids:
        raise StateValidationError("identity renaming membership is invalid")

    def renamed(prefix: str, value: str) -> str:
        digest = hashlib.sha256(f"42:{prefix}:{value}".encode("utf-8")).hexdigest()[:16]
        return f"{prefix}-{digest}"

    payload = {
        "schema_version": "conservative-lofo-identity-renaming-v1",
        "seed": 42,
        "case_ids": {value: renamed("case", value) for value in case_ids},
        "candidate_ids": {value: renamed("node", value) for value in candidate_ids},
    }
    return {**payload, "renaming_sha256": _semantic_hash(payload)}


def audit_identity_probe(artifact: Mapping[str, Any]) -> dict[str, Any]:
    feature_manifest = list(artifact.get("feature_manifest", ()))
    if any(_is_forbidden_field(str(row.get("field_name", ""))) for row in feature_manifest):
        raise StateValidationError("direct identity lookup is recoverable from feature manifest")
    membership = list(artifact.get("membership", ()))
    state_rows = list(artifact.get("state_rows", ()))
    if not membership or len(membership) != len(state_rows):
        raise StateValidationError("identity probe membership is incomplete")
    state_text = _canonical_json(state_rows)
    raw_identities = {
        str(row[field])
        for row in membership
        for field in ("case_id", "candidate_id")
        if str(row.get(field, ""))
    }
    if any(value in state_text for value in raw_identities):
        raise StateValidationError("direct identity lookup is recoverable from model state")
    vectors = [tuple(float(value) for value in row["state_vector"]) for row in state_rows]
    labels = [str(row["candidate_id"]) for row in membership]
    correct = 0
    if len(vectors) > 1:
        for index, vector in enumerate(vectors):
            neighbor = min(
                (other for other in range(len(vectors)) if other != index),
                key=lambda other: (
                    sum(
                        (left - right) ** 2
                        for left, right in zip(vector, vectors[other])
                    ),
                    other,
                ),
            )
            correct += int(labels[neighbor] == labels[index])
        accuracy = correct / len(vectors)
    else:
        accuracy = 0.0
    payload = {
        "schema_version": "conservative-lofo-identity-probe-v1",
        "source_artifact_sha256": str(artifact.get("artifact_sha256", "")),
        "candidate_count": len(vectors),
        "direct_lookup_recoverable": False,
        "nearest_neighbor_probe_accuracy": float(accuracy),
        "probe_definition": "leave_one_candidate_out_euclidean_nearest_neighbor",
    }
    return {**payload, "probe_sha256": _semantic_hash(payload)}


__all__ = [
    "ObservableStateSchema",
    "StateValidationError",
    "audit_identity_probe",
    "build_candidate_state_rows",
    "deterministic_identity_renaming",
    "fit_fold_state_transform",
    "summarize_modality_health",
    "tensorize_observable_state_artifact",
    "validate_observable_state_artifact",
]
